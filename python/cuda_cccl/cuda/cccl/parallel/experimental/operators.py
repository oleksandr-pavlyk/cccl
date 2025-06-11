import numpy as np

from ._bindings import OpKind

# OpKind.PLUS
# OpKind.MINUS
# OpKind.MULTIPLIES
# OpKind.DIVIDES
# OpKind.MODULUS

# OpKind.EQUAL_TO
# OpKind.NOT_EQUAL_TO
# OpKind.GREATER
# OpKind.LESS
# OpKind.GREATER_EQUAL
# OpKind.LESS_EQUAL

# OpKind.LOGICAL_AND
# OpKind.LOGICAL_OR
# OpKind.LOGICAL_NOT

# OpKind.BIT_AND
# OpKind.BIT_OR
# OpKind.BIT_XOR
# OpKind.BIT_NOT

# OpKind.IDENTITY
# OpKind.NEGATE

# OpKind.MINIMUM
# OpKind.MAXIMUM


class Plus:
    cccl_op_code = OpKind.PLUS
    has_identity = True

    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)
        if self.dtype.char not in "?bBhHiIlLqQfd":
            raise ValueError
        self.signature = (self.dtype, self.dtype, self.dtype)

    @staticmethod
    def op_impl(a, b):
        return a + b

    def __call__(self, a, b):
        return type(self).op_impl(a, b)

    @property
    def op_func(self):
        return type(self).op_impl

    def get_hinit(self, dtype):
        return np.zeros(tuple(), dtype=dtype)
