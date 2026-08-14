# distutils: language = c++
# cython: boundscheck=False, wraparound=False
"""Hard-forward differentiable sampling with the k=1 SIMPLE estimator.

The public :class:`DifferentiableSample` stores exact discrete assignments and
their packed one-hot encoding. Its :meth:`~DifferentiableSample.vjp` method
maps a downstream gradient on that one-hot encoding to SparC's ``GradBundle``
format. Gradients are with respect to normalized linear probabilities.
"""

import time

from libc.math cimport fabs, isfinite

import numpy as np
cimport numpy as cnp

from sparc._graph cimport (
    CompiledCircuit,
    LEAF_BERNOULLI,
    LEAF_CATEGORICAL,
    LEAF_GAUSSIAN,
    _flat_sample_node,
    _max_var_from_scope,
)
from sparc.eval cimport _sample_node
from sparc.grad cimport GradBundle, grad_arr
from sparc.nodes cimport (
    BernoulliInputNode,
    CategoricalInputNode,
    CircuitNode,
    FiniteDiscreteInputNode,
    GaussianInputNode,
    InternalNode,
    NODE_INPUT,
    NODE_PRODUCT,
    NODE_SUM,
    ProductNode,
    RandomState,
    SumNode,
)


cdef double PROB_TOL = 1e-6
cdef int TRACE_LEAF = 0
cdef int TRACE_SUM = 1
cdef int TRACE_CONT_LEAF = 2


cdef class DifferentiableSample:
    """A hard sample batch with a reusable SIMPLE vector-Jacobian product.

    Attributes:
        assignments: Read-only legacy ``int32`` assignments with shape
            ``(n_samples, max_var + 1)``.
        one_hot: Read-only packed hard one-hot values. Variables are packed in
            ascending order using ``offsets`` and ``cardinalities``.
        variables: Sorted variable IDs represented by ``one_hot``.
        cardinalities: Cardinality for each entry in ``variables``.
        offsets: Cumulative packed offsets with length ``len(variables) + 1``.
    """

    cdef readonly object assignments
    cdef readonly object one_hot
    cdef readonly object variables
    cdef readonly object cardinalities
    cdef readonly object offsets
    cdef object _events
    cdef Py_ssize_t _n_samples
    cdef Py_ssize_t _packed_width
    cdef bint _continuous

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "DifferentiableSample objects are created by sample("
            "differentiable=True)"
        )

    cdef void _initialize(
        self,
        object assignments,
        object one_hot,
        object variables,
        object cardinalities,
        object offsets,
        object events,
        bint continuous=False,
    ) except *:
        self.assignments = assignments
        self.one_hot = one_hot
        self.variables = variables
        self.cardinalities = cardinalities
        self.offsets = offsets
        self._events = events
        self._n_samples = assignments.shape[0]
        self._packed_width = one_hot.shape[1]
        self._continuous = continuous
        assignments.setflags(write=False)
        one_hot.setflags(write=False)
        variables.setflags(write=False)
        cardinalities.setflags(write=False)
        offsets.setflags(write=False)

    def vjp(self, object upstream):
        """Apply the frozen SIMPLE tape to a packed one-hot upstream gradient.

        ``upstream`` must have exactly the same shape as :attr:`one_hot`.
        Contributions are summed over samples and trace occurrences; no
        implicit averaging is applied.
        """
        cdef cnp.ndarray arr = np.asarray(upstream, dtype=np.float64, order="C")
        if self._continuous:
            return self._vjp_continuous(arr)
        if (
            arr.ndim != 2
            or arr.shape[0] != self._n_samples
            or arr.shape[1] != self._packed_width
        ):
            raise ValueError(
                "upstream must have shape "
                f"({self._n_samples}, {self._packed_width}), "
                f"got {(<object>arr).shape}"
            )

        cdef const double[:, ::1] up = arr
        cdef dict sum_grads = {}
        cdef dict cat_grads = {}
        cdef object event
        cdef object out_arr
        cdef object candidates_arr
        cdef object scope_offsets_arr
        cdef double[::1] out_view
        cdef const int[:, ::1] candidates
        cdef const Py_ssize_t[::1] scope_offsets
        cdef int kind
        cdef Py_ssize_t row
        cdef Py_ssize_t offset
        cdef Py_ssize_t card
        cdef Py_ssize_t i
        cdef Py_ssize_t j
        cdef double total
        cdef object node_id

        for event in self._events:
            kind = event[0]
            row = event[1]
            node_id = event[2]
            if kind == TRACE_LEAF:
                offset = event[3]
                card = event[4]
                out_arr = grad_arr(cat_grads, node_id, <size_t>card)
                out_view = out_arr
                for j in range(card):
                    out_view[j] += up[row, offset + j]
            else:
                candidates_arr = event[3]
                scope_offsets_arr = event[4]
                candidates = candidates_arr
                scope_offsets = scope_offsets_arr
                out_arr = grad_arr(
                    sum_grads, node_id, <size_t>candidates.shape[0]
                )
                out_view = out_arr
                for i in range(candidates.shape[0]):
                    total = 0.0
                    for j in range(candidates.shape[1]):
                        total += up[
                            row,
                            scope_offsets[j] + candidates[i, j],
                        ]
                    out_view[i] += total

        cdef GradBundle grads = GradBundle()
        grads.value = float("nan")
        grads.has_value = False
        grads.sum_grads = sum_grads
        grads.cat_grads = cat_grads
        grads.cont_grads = {}
        return grads

    cdef object _vjp_continuous(self, cnp.ndarray arr):
        cdef Py_ssize_t width = self.assignments.shape[1]
        if (
            arr.ndim != 2
            or arr.shape[0] != self._n_samples
            or arr.shape[1] != width
        ):
            raise ValueError(
                "upstream must have shape "
                f"({self._n_samples}, {width}), "
                f"got {(<object>arr).shape}"
            )
        cdef const double[:, ::1] up = arr
        cdef dict cont_grads = {}
        cdef object event
        cdef object out_arr
        cdef double[::1] out_view
        cdef Py_ssize_t row
        cdef int var
        cdef double eps
        cdef object node_id
        for event in self._events:
            if event[0] != TRACE_CONT_LEAF:
                continue
            row = event[1]
            node_id = event[2]
            var = event[3]
            eps = event[4]
            out_arr = grad_arr(cont_grads, node_id, 2)
            out_view = out_arr
            out_view[0] += up[row, var]
            out_view[1] += up[row, var] * eps
        cdef GradBundle grads = GradBundle()
        grads.value = float("nan")
        grads.has_value = False
        grads.sum_grads = {}
        grads.cat_grads = {}
        grads.cont_grads = cont_grads
        return grads

    def __repr__(self):
        return (
            "DifferentiableSample("
            f"assignments_shape={self.assignments.shape}, "
            f"one_hot_shape={self.one_hot.shape})"
        )

    def __reduce__(self):
        raise TypeError("DifferentiableSample tapes cannot be pickled")

    def __reduce_ex__(self, protocol):
        raise TypeError("DifferentiableSample tapes cannot be pickled")

    def __setstate__(self, state):
        raise TypeError("DifferentiableSample tape state cannot be restored")


cdef void _validate_probability_total(
    double total, str description
) except *:
    if not isfinite(total) or fabs(total - 1.0) > PROB_TOL:
        raise ValueError(f"{description} probabilities must sum to 1")


cdef dict _visit_object_node(
    CircuitNode node,
    dict colors,
    dict node_maps,
    dict id_owners,
    bint validate_live_parameters,
):
    cdef object object_id = id(node)
    cdef int color = int(colors.get(object_id, 0))
    cdef object owner
    cdef object node_id = int(node.id)
    cdef dict result
    cdef dict child_map
    cdef InternalNode internal
    cdef SumNode sum_node
    cdef FiniteDiscreteInputNode leaf
    cdef size_t n
    cdef size_t i
    cdef size_t j
    cdef int var
    cdef double value
    cdef double total
    cdef object key
    cdef object child_card

    if color == 1:
        raise ValueError("differentiable sampling does not support cyclic graphs")
    if color == 2:
        return node_maps[object_id]

    owner = id_owners.get(node_id)
    if owner is not None and owner is not node:
        raise ValueError(
            f"distinct nodes share id {node_id}; node IDs must be unique "
            "for differentiable sampling"
        )
    id_owners[node_id] = node
    colors[object_id] = 1

    if node.node_kind == NODE_INPUT:
        if not isinstance(node, FiniteDiscreteInputNode):
            raise TypeError(
                "differentiable sampling requires FiniteDiscreteInputNode "
                f"leaves, got {type(node).__name__}"
            )
        leaf = <FiniteDiscreteInputNode>node
        var = leaf.scope_var_c()
        if validate_live_parameters:
            n = leaf.support_size()
            if n == 0:
                raise ValueError(
                    f"leaf node {node_id} has an empty finite support"
                )
            total = 0.0
            for i in range(n):
                value = leaf.pmf_at(i)
                if not isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"leaf node {node_id} has invalid probability "
                        f"at outcome {i}"
                    )
                total += value
            _validate_probability_total(total, f"leaf node {node_id}")
            result = {var: int(n)}
        else:
            result = {var: 0}
    elif node.node_kind == NODE_SUM:
        sum_node = <SumNode>node
        n = sum_node.num_children()
        total = 0.0
        result = None
        for i in range(n):
            if validate_live_parameters:
                value = sum_node.parameter_at(i)
                if not isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"sum node {node_id} has invalid weight at child {i}"
                    )
                total += value
            child_map = _visit_object_node(
                sum_node.child_at(i),
                colors,
                node_maps,
                id_owners,
                validate_live_parameters,
            )
            if result is None:
                result = dict(child_map)
            elif result != child_map:
                raise ValueError(
                    f"sum node {node_id} is not smooth: all children must "
                    "have identical variable cardinalities"
                )
        if validate_live_parameters:
            _validate_probability_total(total, f"sum node {node_id}")
    elif node.node_kind == NODE_PRODUCT:
        internal = <InternalNode>node
        n = internal.num_children()
        result = {}
        for i in range(n):
            child_map = _visit_object_node(
                internal.child_at(i),
                colors,
                node_maps,
                id_owners,
                validate_live_parameters,
            )
            for key, child_card in child_map.items():
                if key in result:
                    raise ValueError(
                        f"product node {node_id} is not decomposable: "
                        f"variable {key} appears in multiple children"
                    )
                result[key] = child_card
    else:
        raise TypeError(
            f"unsupported node type for differentiable sampling: "
            f"{type(node).__name__}"
        )

    colors[object_id] = 2
    node_maps[object_id] = result
    return result


cdef tuple _validate_object_graph(
    CircuitNode root, bint validate_live_parameters
):
    cdef dict colors = {}
    cdef dict node_maps = {}
    cdef dict id_owners = {}
    cdef dict root_map = _visit_object_node(
        root,
        colors,
        node_maps,
        id_owners,
        validate_live_parameters,
    )
    return root_map, node_maps


cdef tuple _validate_compiled_graph(CompiledCircuit graph):
    cdef CircuitNode root = <CircuitNode>graph.node_objs[graph.root_index]
    cdef tuple validated = _validate_object_graph(root, False)
    cdef dict live_root_map = validated[0]
    cdef dict node_maps = validated[1]
    cdef dict flat_cards = {}
    cdef size_t n
    cdef size_t k
    cdef size_t start
    cdef size_t stop
    cdef size_t scope_start
    cdef size_t scope_stop
    cdef int var
    cdef int card
    cdef object old_card
    cdef CircuitNode live_node
    cdef dict computed_map
    cdef set stored_scope
    cdef double value
    cdef double total
    cdef object node_id

    for n in range(graph.n_nodes):
        node_id = int(graph.node_ids[n])
        live_node = <CircuitNode>graph.node_objs[n]
        computed_map = node_maps[id(live_node)]
        stored_scope = set()
        scope_start = graph.scope_vars_off[n]
        scope_stop = graph.scope_vars_off[n + 1]
        for k in range(scope_start, scope_stop):
            stored_scope.add(graph.scope_vars_flat[k])
        if set(computed_map) != stored_scope:
            raise ValueError(
                f"compiled node {node_id} has stale scope metadata; "
                "recompile the circuit"
            )
        if graph.kinds[n] == NODE_INPUT:
            var = graph.leaf_var[n]
            card = graph.leaf_card[n]
            if card <= 0:
                raise ValueError(
                    f"compiled leaf node {node_id} has an empty finite support"
                )
            old_card = flat_cards.get(var)
            if old_card is not None and int(old_card) != card:
                raise ValueError(
                    f"variable {var} has inconsistent cardinalities in the "
                    "compiled circuit"
                )
            flat_cards[var] = card
            start = graph.leaf_pmf_off[n]
            total = 0.0
            for k in range(<size_t>card):
                value = graph.leaf_pmf_flat[start + k]
                if not isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"compiled leaf node {node_id} has an invalid "
                        f"probability at outcome {k}"
                    )
                total += value
            _validate_probability_total(
                total, f"compiled leaf node {node_id}"
            )
        elif graph.kinds[n] == NODE_SUM:
            start = graph.child_off[n]
            stop = graph.child_off[n + 1]
            total = 0.0
            for k in range(start, stop):
                value = graph.sum_w_flat[k]
                if not isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"compiled sum node {node_id} has an invalid "
                        f"weight at child {k - start}"
                    )
                total += value
            _validate_probability_total(
                total, f"compiled sum node {node_id}"
            )

    if set(live_root_map) != set(flat_cards):
        raise ValueError(
            "compiled circuit topology changed; "
            "recompile the circuit"
        )
    return flat_cards, node_maps


cdef tuple _metadata_from_cards(dict cards):
    cdef list variables_list = sorted(cards)
    cdef Py_ssize_t n_variables = len(variables_list)
    cdef object variables = np.asarray(variables_list, dtype=np.int32)
    cdef object cardinalities = np.empty(n_variables, dtype=np.int32)
    cdef object offsets = np.empty(n_variables + 1, dtype=np.intp)
    cdef int[::1] cards_view = cardinalities
    cdef Py_ssize_t[::1] offsets_view = offsets
    cdef dict offset_by_var = {}
    cdef Py_ssize_t i
    cdef Py_ssize_t running = 0
    cdef int var
    offsets_view[0] = 0
    for i in range(n_variables):
        var = variables_list[i]
        cards_view[i] = int(cards[var])
        offset_by_var[var] = running
        running += cards_view[i]
        offsets_view[i + 1] = running
    return (
        variables,
        cardinalities,
        offsets,
        offset_by_var,
        variables_list,
        running,
    )


cdef inline size_t _draw_object_sum(
    SumNode node, RandomState rng
) except *:
    cdef size_t i
    cdef size_t chosen = node.num_children() - 1
    cdef double u = rng.next_double()
    cdef double cumulative = 0.0
    for i in range(node.num_children()):
        cumulative += node.parameter_at(i)
        if u < cumulative:
            chosen = i
            break
    return chosen


cdef void _sample_object_differentiable(
    CircuitNode node,
    RandomState rng,
    int* out,
    Py_ssize_t width,
    Py_ssize_t row,
    dict node_maps,
    dict offset_by_var,
    list events,
) except *:
    cdef FiniteDiscreteInputNode leaf
    cdef InternalNode internal
    cdef SumNode sum_node
    cdef size_t n_children
    cdef size_t i
    cdef size_t j
    cdef size_t selected
    cdef int var
    cdef int value
    cdef dict scope_map
    cdef list scope_vars
    cdef Py_ssize_t n_scope
    cdef object temp_arr
    cdef int[::1] temp
    cdef object candidates_arr
    cdef int[:, ::1] candidates
    cdef object scope_offsets_arr
    cdef Py_ssize_t[::1] scope_offsets

    if node.node_kind == NODE_INPUT:
        leaf = <FiniteDiscreteInputNode>node
        leaf.sample_into_c(rng, out)
        if isinstance(node, (CategoricalInputNode, BernoulliInputNode)):
            var = leaf.scope_var_c()
            events.append(
                (
                    TRACE_LEAF,
                    row,
                    int(node.id),
                    int(offset_by_var[var]),
                    int(leaf.support_size()),
                )
            )
        return

    if node.node_kind == NODE_PRODUCT:
        internal = <InternalNode>node
        for i in range(internal.num_children()):
            _sample_object_differentiable(
                internal.child_at(i),
                rng,
                out,
                width,
                row,
                node_maps,
                offset_by_var,
                events,
            )
        return

    sum_node = <SumNode>node
    n_children = sum_node.num_children()
    selected = _draw_object_sum(sum_node, rng)
    scope_map = node_maps[id(node)]
    scope_vars = sorted(scope_map)
    n_scope = len(scope_vars)
    candidates_arr = np.empty(
        (n_children, n_scope), dtype=np.int32
    )
    candidates = candidates_arr
    scope_offsets_arr = np.empty(n_scope, dtype=np.intp)
    scope_offsets = scope_offsets_arr
    for j in range(n_scope):
        scope_offsets[j] = int(offset_by_var[scope_vars[j]])

    for i in range(n_children):
        temp_arr = np.full(width, -1, dtype=np.int32)
        temp = temp_arr
        if i == selected:
            _sample_object_differentiable(
                sum_node.child_at(i),
                rng,
                &temp[0],
                width,
                row,
                node_maps,
                offset_by_var,
                events,
            )
        else:
            _sample_node(sum_node.child_at(i), rng, &temp[0])
        for j in range(n_scope):
            var = scope_vars[j]
            value = temp[var]
            if value < 0 or value >= int(scope_map[var]):
                raise RuntimeError(
                    f"sum node {node.id} child {i} sampled invalid value "
                    f"{value} for variable {var}"
                )
            candidates[i, j] = value

    for j in range(n_scope):
        out[scope_vars[j]] = candidates[selected, j]
    candidates_arr.setflags(write=False)
    scope_offsets_arr.setflags(write=False)
    events.append(
        (
            TRACE_SUM,
            row,
            int(node.id),
            candidates_arr,
            scope_offsets_arr,
        )
    )


cdef inline size_t _draw_flat_sum(
    CompiledCircuit graph, size_t node_index, RandomState rng
) noexcept nogil:
    cdef size_t start = graph.child_off[node_index]
    cdef size_t stop = graph.child_off[node_index + 1]
    cdef size_t k
    cdef size_t selected = stop - start - 1
    cdef double u = rng.next_double()
    cdef double cumulative = 0.0
    for k in range(start, stop):
        cumulative += graph.sum_w_flat[k]
        if u < cumulative:
            selected = k - start
            break
    return selected


cdef void _sample_flat_differentiable(
    CompiledCircuit graph,
    size_t node_index,
    RandomState rng,
    int* out,
    Py_ssize_t width,
    Py_ssize_t row,
    dict offset_by_var,
    dict cardinality_by_var,
    list events,
) except *:
    cdef int kind = graph.kinds[node_index]
    cdef size_t start
    cdef size_t stop
    cdef size_t k
    cdef size_t i
    cdef size_t j
    cdef size_t selected
    cdef size_t child_index
    cdef size_t scope_start
    cdef size_t scope_stop
    cdef int var
    cdef int value
    cdef int card
    cdef object temp_arr
    cdef int[::1] temp
    cdef object candidates_arr
    cdef int[:, ::1] candidates
    cdef object scope_offsets_arr
    cdef Py_ssize_t[::1] scope_offsets

    if kind == NODE_INPUT:
        _flat_sample_node(graph, node_index, rng, out)
        if (
            graph.leaf_kind[node_index] == LEAF_CATEGORICAL
            or graph.leaf_kind[node_index] == LEAF_BERNOULLI
        ):
            var = graph.leaf_var[node_index]
            card = graph.leaf_card[node_index]
            events.append(
                (
                    TRACE_LEAF,
                    row,
                    int(graph.node_ids[node_index]),
                    int(offset_by_var[var]),
                    card,
                )
            )
        return

    start = graph.child_off[node_index]
    stop = graph.child_off[node_index + 1]
    if kind == NODE_PRODUCT:
        for k in range(start, stop):
            _sample_flat_differentiable(
                graph,
                graph.children_flat[k],
                rng,
                out,
                width,
                row,
                offset_by_var,
                cardinality_by_var,
                events,
            )
        return

    selected = _draw_flat_sum(graph, node_index, rng)
    scope_start = graph.scope_vars_off[node_index]
    scope_stop = graph.scope_vars_off[node_index + 1]
    candidates_arr = np.empty(
        (stop - start, scope_stop - scope_start), dtype=np.int32
    )
    candidates = candidates_arr
    scope_offsets_arr = np.empty(
        scope_stop - scope_start, dtype=np.intp
    )
    scope_offsets = scope_offsets_arr
    for j in range(scope_stop - scope_start):
        var = graph.scope_vars_flat[scope_start + j]
        scope_offsets[j] = int(offset_by_var[var])

    for i in range(stop - start):
        child_index = graph.children_flat[start + i]
        temp_arr = np.full(width, -1, dtype=np.int32)
        temp = temp_arr
        if i == selected:
            _sample_flat_differentiable(
                graph,
                child_index,
                rng,
                &temp[0],
                width,
                row,
                offset_by_var,
                cardinality_by_var,
                events,
            )
        else:
            _flat_sample_node(graph, child_index, rng, &temp[0])
        for j in range(scope_stop - scope_start):
            var = graph.scope_vars_flat[scope_start + j]
            value = temp[var]
            if value < 0 or value >= int(cardinality_by_var[var]):
                raise RuntimeError(
                    f"compiled sum node {graph.node_ids[node_index]} child "
                    f"{i} sampled invalid value {value} for variable {var}"
                )
            candidates[i, j] = value

    for j in range(scope_stop - scope_start):
        var = graph.scope_vars_flat[scope_start + j]
        out[var] = candidates[selected, j]
    candidates_arr.setflags(write=False)
    scope_offsets_arr.setflags(write=False)
    events.append(
        (
            TRACE_SUM,
            row,
            int(graph.node_ids[node_index]),
            candidates_arr,
            scope_offsets_arr,
        )
    )


cdef object _materialize_result(
    object assignments,
    object variables,
    object cardinalities,
    object offsets,
    list variables_list,
    Py_ssize_t packed_width,
    list events,
):
    cdef Py_ssize_t n_samples = assignments.shape[0]
    cdef Py_ssize_t n_variables = len(variables_list)
    cdef object one_hot = np.zeros(
        (n_samples, packed_width), dtype=np.float64
    )
    cdef const int[:, ::1] assignment_view = assignments
    cdef double[:, ::1] one_hot_view = one_hot
    cdef const int[::1] cardinality_view = cardinalities
    cdef const Py_ssize_t[::1] offset_view = offsets
    cdef Py_ssize_t row
    cdef Py_ssize_t i
    cdef int var
    cdef int value
    for row in range(n_samples):
        for i in range(n_variables):
            var = variables_list[i]
            value = assignment_view[row, var]
            if value < 0 or value >= cardinality_view[i]:
                raise RuntimeError(
                    f"sampled value {value} for variable {var} is outside "
                    f"[0, {cardinality_view[i]})"
                )
            one_hot_view[row, offset_view[i] + value] = 1.0
    cdef DifferentiableSample result = DifferentiableSample.__new__(
        DifferentiableSample
    )
    result._initialize(
        assignments,
        one_hot,
        variables,
        cardinalities,
        offsets,
        events,
    )
    return result


def sample_differentiable(
    CircuitNode root,
    Py_ssize_t n_samples,
    object seed=None,
):
    """Draw unconditional hard samples and retain a SIMPLE VJP tape."""
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    cdef tuple validated = _validate_object_graph(root, True)
    cdef dict root_cards = validated[0]
    cdef dict node_maps = validated[1]
    cdef tuple metadata = _metadata_from_cards(root_cards)
    cdef object variables = metadata[0]
    cdef object cardinalities = metadata[1]
    cdef object offsets = metadata[2]
    cdef dict offset_by_var = metadata[3]
    cdef list variables_list = metadata[4]
    cdef Py_ssize_t packed_width = metadata[5]
    cdef Py_ssize_t width = variables_list[len(variables_list) - 1] + 1
    cdef object assignments = np.full(
        (n_samples, width), -1, dtype=np.int32
    )
    cdef int[:, ::1] assignment_view = assignments
    cdef unsigned long long rng_seed
    cdef Py_ssize_t row
    cdef list events = []
    if seed is None:
        rng_seed = <unsigned long long>time.time_ns()
    else:
        rng_seed = <unsigned long long>int(seed)
    cdef RandomState rng = RandomState(rng_seed)
    for row in range(n_samples):
        _sample_object_differentiable(
            root,
            rng,
            &assignment_view[row, 0],
            width,
            row,
            node_maps,
            offset_by_var,
            events,
        )
    return _materialize_result(
        assignments,
        variables,
        cardinalities,
        offsets,
        variables_list,
        packed_width,
        events,
    )


def sample_compiled_differentiable(
    CompiledCircuit graph,
    Py_ssize_t n_samples,
    object seed=None,
):
    """Draw differentiable samples from a compiled parameter snapshot."""
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    cdef tuple validated = _validate_compiled_graph(graph)
    cdef dict root_cards = validated[0]
    cdef tuple metadata = _metadata_from_cards(root_cards)
    cdef object variables = metadata[0]
    cdef object cardinalities = metadata[1]
    cdef object offsets = metadata[2]
    cdef dict offset_by_var = metadata[3]
    cdef list variables_list = metadata[4]
    cdef Py_ssize_t packed_width = metadata[5]
    cdef Py_ssize_t width = variables_list[len(variables_list) - 1] + 1
    cdef object assignments = np.full(
        (n_samples, width), -1, dtype=np.int32
    )
    cdef int[:, ::1] assignment_view = assignments
    cdef unsigned long long rng_seed
    cdef Py_ssize_t row
    cdef list events = []
    if seed is None:
        rng_seed = <unsigned long long>time.time_ns()
    else:
        rng_seed = <unsigned long long>int(seed)
    cdef RandomState rng = RandomState(rng_seed)
    for row in range(n_samples):
        _sample_flat_differentiable(
            graph,
            graph.root_index,
            rng,
            &assignment_view[row, 0],
            width,
            row,
            offset_by_var,
            root_cards,
            events,
        )
    return _materialize_result(
        assignments,
        variables,
        cardinalities,
        offsets,
        variables_list,
        packed_width,
        events,
    )


cdef void _sample_object_diff_cont(
    CircuitNode node,
    RandomState rng,
    double* out,
    Py_ssize_t row,
    list events,
) except *:
    cdef GaussianInputNode g
    cdef ProductNode p
    cdef SumNode s
    cdef size_t i
    cdef size_t idx
    cdef size_t n
    cdef double u
    cdef double cum
    cdef double eps
    cdef int var
    if node.node_kind == NODE_INPUT:
        g = <GaussianInputNode>node
        var = g.scope_var_c()
        eps = rng.next_normal()
        out[var] = g.mu + g.sigma * eps
        events.append((TRACE_CONT_LEAF, row, int(node.id), var, eps))
        return
    if node.node_kind == NODE_PRODUCT:
        p = <ProductNode>node
        n = p.num_children()
        for i in range(n):
            _sample_object_diff_cont(p.child_at(i), rng, out, row, events)
        return
    s = <SumNode>node
    n = s.num_children()
    u = rng.next_double()
    cum = 0.0
    idx = n - 1
    for i in range(n):
        cum += s.parameter_at(i)
        if u < cum:
            idx = i
            break
    _sample_object_diff_cont(s.child_at(idx), rng, out, row, events)


cdef void _sample_flat_diff_cont(
    CompiledCircuit graph,
    size_t n,
    RandomState rng,
    double* out,
    Py_ssize_t row,
    list events,
) except *:
    cdef int kind = graph.kinds[n]
    cdef size_t start
    cdef size_t stop
    cdef size_t k
    cdef size_t idx
    cdef size_t pbase
    cdef double u
    cdef double cum
    cdef double eps
    cdef double mu
    cdef double sigma
    cdef int var
    if kind == NODE_INPUT:
        var = graph.leaf_var[n]
        pbase = graph.leaf_param_off[n]
        mu = graph.leaf_param_flat[pbase]
        sigma = graph.leaf_param_flat[pbase + 1]
        eps = rng.next_normal()
        out[var] = mu + sigma * eps
        events.append(
            (TRACE_CONT_LEAF, row, int(graph.node_ids[n]), var, eps)
        )
        return
    start = graph.child_off[n]
    stop = graph.child_off[n + 1]
    if kind == NODE_PRODUCT:
        for k in range(start, stop):
            _sample_flat_diff_cont(
                graph, graph.children_flat[k], rng, out, row, events
            )
        return
    u = rng.next_double()
    cum = 0.0
    idx = graph.children_flat[stop - 1]
    for k in range(start, stop):
        cum += graph.sum_w_flat[k]
        if u < cum:
            idx = graph.children_flat[k]
            break
    _sample_flat_diff_cont(graph, idx, rng, out, row, events)


cdef object _materialize_continuous_result(
    object assignments, object events
):
    cdef object variables = np.arange(assignments.shape[1], dtype=np.int32)
    cdef object cardinalities = np.zeros(assignments.shape[1], dtype=np.int32)
    cdef object offsets = np.zeros(1, dtype=np.intp)
    cdef object one_hot = np.zeros(
        (assignments.shape[0], 0), dtype=np.float64
    )
    cdef DifferentiableSample result = DifferentiableSample.__new__(
        DifferentiableSample
    )
    result._initialize(
        assignments,
        one_hot,
        variables,
        cardinalities,
        offsets,
        events,
        True,
    )
    return result


def sample_differentiable_continuous(
    CircuitNode root,
    Py_ssize_t n_samples,
    object seed=None,
):
    """Draw reparameterized Gaussian samples and retain a VJP tape over (μ, σ)."""
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if root.scope.size() == 0:
        root.propagate_scope()
    cdef int max_var = _max_var_from_scope(root.scope)
    cdef size_t width = <size_t>(max_var + 1)
    cdef object assignments = np.full(
        (n_samples, width), np.nan, dtype=np.float64
    )
    cdef double[:, ::1] assignment_view = assignments
    cdef unsigned long long rng_seed
    cdef Py_ssize_t row
    cdef list events = []
    if seed is None:
        rng_seed = <unsigned long long>time.time_ns()
    else:
        rng_seed = <unsigned long long>int(seed)
    cdef RandomState rng = RandomState(rng_seed)
    for row in range(n_samples):
        _sample_object_diff_cont(
            root, rng, &assignment_view[row, 0], row, events
        )
    return _materialize_continuous_result(assignments, events)


def sample_compiled_differentiable_continuous(
    CompiledCircuit graph,
    Py_ssize_t n_samples,
    object seed=None,
):
    """Reparameterized differentiable samples from a compiled Gaussian circuit."""
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    cdef size_t width = <size_t>(graph.max_var + 1)
    cdef object assignments = np.full(
        (n_samples, width), np.nan, dtype=np.float64
    )
    cdef double[:, ::1] assignment_view = assignments
    cdef unsigned long long rng_seed
    cdef Py_ssize_t row
    cdef list events = []
    if seed is None:
        rng_seed = <unsigned long long>time.time_ns()
    else:
        rng_seed = <unsigned long long>int(seed)
    cdef RandomState rng = RandomState(rng_seed)
    for row in range(n_samples):
        _sample_flat_diff_cont(
            graph, graph.root_index, rng, &assignment_view[row, 0], row, events
        )
    return _materialize_continuous_result(assignments, events)
