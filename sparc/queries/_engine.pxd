from libc.stdint cimport uint64_t
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from sparc.nodes cimport CircuitNode, ProductNode

cdef size_t NO_TAPE_IDX


cdef inline size_t obj_id(CircuitNode n) noexcept:
    return <size_t><void*>n


# Every coupled pair has exactly one node from circuit 1 (side 0) and one from
# circuit 2 (side 1); CW/expectation never swap sides, so packing the ordered
# (P.id, Q.id) pair into a single 64-bit key is lossless and lets a flat
# integer-keyed map replace the old nested pointer-keyed maps.
cdef inline uint64_t pair_key(CircuitNode P, CircuitNode Q) noexcept:
    return (<uint64_t>P.id << 32) | <uint64_t>Q.id


cdef class TapeEntry:
    cdef int side_P
    cdef int side_Q
    cdef CircuitNode P
    cdef CircuitNode Q

    cdef void backward(self, object ctx, double g) except *


cdef class CoupleContext:
    cdef unordered_map[uint64_t, double] couple_memo
    cdef bint recording
    cdef list tape
    cdef vector[double] tape_adjoints
    cdef unordered_map[uint64_t, size_t] pair_to_tape
    cdef dict sum_grads0
    cdef dict cat_grads0
    cdef dict sum_grads1
    cdef dict cat_grads1

    cdef bint memo_get(self, CircuitNode P, CircuitNode Q, double* out) noexcept
    cdef void memo_put(self, CircuitNode P, CircuitNode Q, double val) noexcept
    cdef size_t lookup_pair_tape_idx(self, CircuitNode P, CircuitNode Q) noexcept
    cdef size_t append_tape(self, TapeEntry entry, CircuitNode P, CircuitNode Q) except *
    cdef object sum_grad_arr(self, int side, CircuitNode node, size_t n)
    cdef object cat_grad_arr(self, int side, CircuitNode node, size_t n)
    cdef void run_backward(self) except *
    cdef void reset_base(self)


cdef void match_prod_children(
    ProductNode P,
    ProductNode Q,
    vector[int]& row_ind,
    vector[int]& col_ind,
    str query_name,
) except *
