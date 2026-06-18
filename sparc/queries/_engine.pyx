# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -O3
"""Shared machinery for pairwise (two-circuit) coupling queries.

This eliminates the ~80% copy-paste that used to be duplicated across the GCW /
CW / expectation modules: the pair-keyed memo, the gradient tape (append /
lookup / reverse replay), the per-side gradient arrays, and the structured
product-child scope matching all live here once. Each query subclasses
:class:`CoupleContext`, supplies its ``couple_value`` recursion and a handful of
:class:`TapeEntry` subclasses with their own ``backward`` math.

Two keying schemes are used deliberately, matching the original implementation:
the coupling memo / tape are keyed by Python object identity (unique across the
two circuits), while the returned gradient bundles are keyed by ``node.id`` so
callers can match gradients to nodes.
"""

from cython.operator cimport dereference as deref
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

import numpy as np

from sparc.nodes cimport CircuitNode, ProductNode

cdef size_t NO_TAPE_IDX = <size_t>(-1)


cdef inline void _ordered_key(
    CircuitNode P, CircuitNode Q, size_t* a, size_t* b
) noexcept:
    cdef size_t id_p = obj_id(P)
    cdef size_t id_q = obj_id(Q)
    if id_p <= id_q:
        a[0] = id_p
        b[0] = id_q
    else:
        a[0] = id_q
        b[0] = id_p


cdef class TapeEntry:
    """Base tape entry. Subclasses store forward state + override ``backward``."""

    cdef void backward(self, object ctx, double g) except *:
        pass


cdef class CoupleContext:
    def __cinit__(self):
        self.recording = False
        self.tape = []
        self.sum_grads0 = {}
        self.cat_grads0 = {}
        self.sum_grads1 = {}
        self.cat_grads1 = {}

    cdef bint memo_get(self, CircuitNode P, CircuitNode Q, double* out) noexcept:
        cdef size_t a
        cdef size_t b
        _ordered_key(P, Q, &a, &b)
        cdef unordered_map[size_t, unordered_map[size_t, double]].iterator outer = self.couple_memo.find(a)
        if outer == self.couple_memo.end():
            return False
        cdef unordered_map[size_t, double].iterator inner = deref(outer).second.find(b)
        if inner == deref(outer).second.end():
            return False
        out[0] = deref(inner).second
        return True

    cdef void memo_put(self, CircuitNode P, CircuitNode Q, double val) noexcept:
        cdef size_t a
        cdef size_t b
        _ordered_key(P, Q, &a, &b)
        self.couple_memo[a][b] = val

    cdef size_t lookup_pair_tape_idx(self, CircuitNode P, CircuitNode Q) noexcept:
        cdef size_t a
        cdef size_t b
        _ordered_key(P, Q, &a, &b)
        cdef unordered_map[size_t, unordered_map[size_t, size_t]].iterator outer = self.pair_to_tape.find(a)
        if outer == self.pair_to_tape.end():
            return NO_TAPE_IDX
        cdef unordered_map[size_t, size_t].iterator inner = deref(outer).second.find(b)
        if inner == deref(outer).second.end():
            return NO_TAPE_IDX
        return deref(inner).second

    cdef size_t append_tape(self, TapeEntry entry, CircuitNode P, CircuitNode Q) except *:
        cdef size_t idx = <size_t>len(self.tape)
        cdef size_t a
        cdef size_t b
        _ordered_key(P, Q, &a, &b)
        self.tape.append(entry)
        self.tape_adjoints.push_back(0.0)
        self.pair_to_tape[a][b] = idx
        return idx

    cdef object sum_grad_arr(self, int side, CircuitNode node, size_t n):
        cdef dict store = self.sum_grads0 if side == 0 else self.sum_grads1
        cdef object key = node.id
        cdef object arr = store.get(key)
        if arr is None:
            arr = np.zeros(n, dtype=np.float64)
            store[key] = arr
        return arr

    cdef object cat_grad_arr(self, int side, CircuitNode node, size_t n):
        cdef dict store = self.cat_grads0 if side == 0 else self.cat_grads1
        cdef object key = node.id
        cdef object arr = store.get(key)
        if arr is None:
            arr = np.zeros(n, dtype=np.float64)
            store[key] = arr
        return arr

    cdef void run_backward(self) except *:
        cdef ssize_t k
        cdef double g
        cdef TapeEntry entry
        for k in range(<ssize_t>len(self.tape) - 1, -1, -1):
            g = self.tape_adjoints[<size_t>k]
            if g == 0.0:
                continue
            entry = <TapeEntry>self.tape[k]
            entry.backward(self, g)

    cdef void reset_base(self):
        self.couple_memo.clear()
        self.tape = []
        self.tape_adjoints.clear()
        self.pair_to_tape.clear()
        self.sum_grads0 = {}
        self.cat_grads0 = {}
        self.sum_grads1 = {}
        self.cat_grads1 = {}


cdef object _scope_frozen(CircuitNode node):
    return frozenset(node.scope_as_list())


cdef void match_prod_children(
    ProductNode P,
    ProductNode Q,
    vector[int]& row_ind,
    vector[int]& col_ind,
    str query_name,
) except *:
    """Match product children by scope into a bijection (structured decomp.).

    Fills ``row_ind[i] = i`` and ``col_ind[i] = matched Q-child index``.
    """
    cdef size_t n = P.num_children()
    cdef size_t m = Q.num_children()
    cdef size_t i
    cdef size_t j
    cdef CircuitNode q_child
    cdef CircuitNode p_child
    cdef object scope_to_q
    cdef object p_key
    cdef object q_key
    cdef Py_ssize_t q_idx

    if n != m:
        raise ValueError(
            f"{query_name} incompatible: product nodes have different numbers "
            f"of children ({n} vs {m})"
        )
    row_ind.assign(n, 0)
    col_ind.assign(n, 0)
    scope_to_q = {}
    for j in range(m):
        q_child = Q.child_at(j)
        q_key = _scope_frozen(q_child)
        if q_key in scope_to_q:
            raise ValueError(
                f"{query_name} incompatible: duplicate child scope among Q "
                "product children"
            )
        scope_to_q[q_key] = j
    for i in range(n):
        p_child = P.child_at(i)
        p_key = _scope_frozen(p_child)
        if p_key not in scope_to_q:
            raise ValueError(
                f"{query_name} incompatible: no Q product child with scope "
                f"matching P child at index {i}"
            )
        q_idx = scope_to_q[p_key]
        del scope_to_q[p_key]
        row_ind[i] = <int>i
        col_ind[i] = <int>q_idx
    if len(scope_to_q) > 0:
        raise ValueError(
            f"{query_name} incompatible: Q product children with unmatched scopes"
        )
