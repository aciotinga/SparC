cdef class GradBundle:
    cdef public double value
    cdef public dict sum_grads
    cdef public dict cat_grads

cdef object grad_arr(dict store, object key, size_t n)
