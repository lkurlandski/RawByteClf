# bytes_to_str_utf8.pyx

from cpython.bytes cimport PyBytes_GET_SIZE, PyBytes_AS_STRING
from cpython.unicode cimport PyUnicode_FromKindAndData, PyUnicode_4BYTE_KIND
from libc.stdlib cimport malloc, free

cdef Py_UCS4* BYTE_TO_UTF8 = <Py_UCS4*>malloc(256 * sizeof(Py_UCS4))

# Initialize the lookup table
cdef int i
for i in range(256):
    BYTE_TO_UTF8[i] = i + 10752

def bytes_to_str_utf8(bytes b):
    cdef Py_ssize_t length = PyBytes_GET_SIZE(b)
    cdef Py_UCS4* result = <Py_UCS4*>malloc(length * sizeof(Py_UCS4))
    cdef Py_ssize_t i
    cdef char* byte_ptr = PyBytes_AS_STRING(b)

    for i in range(length):
        result[i] = BYTE_TO_UTF8[<unsigned char>byte_ptr[i]]

    string = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, result, length)
    free(result)
    return string