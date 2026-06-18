from libcpp.vector cimport vector

from sparc.nodes cimport CircuitNode

# Leaf-kind tags for the flattened fast path. Built-in finite-discrete leaves
# get a concrete tag; anything else (custom subclasses, non-finite leaves) is
# tagged LEAF_FALLBACK and forces the owning graph to use the object path.
cdef enum LeafKind:
    LEAF_CATEGORICAL = 0
    LEAF_BERNOULLI = 1
    LEAF_INDICATOR = 2
    LEAF_LITERAL = 3
    LEAF_DISCRETE_LOGISTIC = 4
    LEAF_FALLBACK = 5


cdef class CompiledGraph:
    # --- topology (post-order; children appear before parents) ---------------
    cdef vector[int] kinds              # NODE_SUM / NODE_PRODUCT / NODE_INPUT
    cdef vector[size_t] child_off       # CSR offsets, size n_nodes + 1
    cdef vector[size_t] children_flat   # child node indices
    cdef vector[double] sum_w_flat      # aligned with children_flat (sum nodes)
    cdef vector[double] sum_logw_flat   # log of sum_w_flat

    # --- leaf descriptors (valid where kinds[n] == NODE_INPUT) ---------------
    cdef vector[int] leaf_kind          # LeafKind tag
    cdef vector[int] leaf_var           # scope variable, else -1
    cdef vector[int] leaf_card          # support size, else 0
    cdef vector[char] leaf_trainable    # 1 for categorical / bernoulli leaves
    cdef vector[size_t] leaf_pmf_off    # CSR offsets into pmf pools, size n + 1
    cdef vector[double] leaf_pmf_flat   # precomputed linear pmf
    cdef vector[double] leaf_logpmf_flat  # precomputed log pmf

    # --- identity / materialization ------------------------------------------
    cdef vector[size_t] node_ids        # node.id per index (gradient keys)
    cdef list node_objs                 # parallel python node objects (fallback)
    cdef size_t n_nodes
    cdef size_t root_index
    cdef bint has_fallback              # any LEAF_FALLBACK present
    cdef int max_var                    # largest scope variable index
    cdef readonly list variables        # sorted scope variables

    cdef void build(self, CircuitNode root) except *
    cdef void _classify_leaf(self, CircuitNode node, size_t n) except *
    cdef void _postorder(self, CircuitNode node, dict index_of, list order) except *


# nogil leaf math kept inline-able by consumers that read the pmf pools.
cdef double sp_graph_sigmoid(double x) noexcept nogil
