from math import sqrt
from operator import mul, add
from itertools import repeat
from functools import cached_property

from matrixx.vector_space import VectorSpace
from matrixx.immutable import Immutable
import matrixx.matrix as matrix


_ZERO = 0


class Vector(VectorSpace, Immutable):
    """
    Vector implementation. Index with V[i] (start at zero)
    Treated as vertical.

    Does NOT support vector*matrix multiplication
    (convert the vector to a matrix and do m*m)

    vector@vector returns dot product

    length_squared() returns the square of abs (memoised)
    length and __abs__() returns euclidean norm (memoised).
    norm(n) returns n-th norm
    cross(other) return cross product.
    project(basis) return projection onto basis.

    size: #entries
    """

    _IS_VECTOR = True

    def __init__(self, values, length=None):
        self._value = values
        self.size = len(values)

        if length is not None:
            self.length = length

    def __repr__(self):
        return f'({(", ".join(str(x) for x in self._value))})ᵗ'
        # add rounding
        # add horizontal vectors (?)
        # change to fstring

    def __getitem__(self, pos):
        return self._value[pos]

    def __add__(self, other):
        if other is _ZERO:
            # for sum()
            # this causes a warning but
            # I thought about it and "is 0" is what I want
            return self

        a = self._value
        b = other._value
        return Vector(tuple(map(add, a, b)))

    def __mul__(self, a):
        try:
            # pairwise mult
            return Vector(tuple(map(mul, self._value, a._value)))
        except AttributeError:  # TODO: which exception?
            # scalar mult
            return Vector(tuple(map(mul, self._value, repeat(a))))

    def __matmul__(self, other):  # v@v
        a = self._value
        b = other._value
        return sum(map(mul, a, b))

    def __rmatmul__(self, other):  # m@v multiplication
        a = other._value
        b = self._value
        c = tuple(sum(map(mul, a_row, b)) for a_row in a)
        return Vector(c)

    def __hash__(self):
        return hash(self._value)  # TODO: cache?

    @cached_property
    def length_squared(self):
        return self @ self

    @cached_property
    def length(self):  # another semi-memoised expensive function
        """
        Returns euclidean norm of vector.
        :return: int
        """
        return sqrt(self.length_squared)

    @cached_property
    def unit(self):
        return (1/self.length) * self

    def to_matrix(self, vert=True):
        """Converts Vector to Matrix."""
        if vert:
            return matrix.Matrix(tuple(zip(self._value)))
        return matrix.Matrix((self._value,))

    def to_tuple(self):
        return self._value

    def copy(self):
        return self

    @cached_property
    def orthant(self):
        res = 0
        for n in self._value:
            res *= 2
            if n > 0:
                res += 1
        return res

    def cross(self, other):
        """
        Cross Product. Only defined for 3 dimensional vectors.
        :param other: Vector
        :return: self⨯other
        """
        # Also exists for 7 dimensions... implement?

        a1, a2, a3 = self._value
        b1, b2, b3 = other._value

        s1 = a2 * b3 - a3 * b2
        s2 = a3 * b1 - a1 * b3
        s3 = a1 * b2 - a2 * b1

        return Vector((s1, s2, s3))

    def project(self, basis):
        """
        Return projection of vector in given basis.
        :param basis: iterable of Vectors
        :return: Vector
        """
        res = Vector([0, 0, 0])
        for base in basis:
            res += (self @ base) / (base @ base) * base  # inefficient TODO
        return res

    def crop(self, bound):
        """
        Shorten vector if it's longer than the given bound
        """
        if self.length_squared < bound**2:
            return self
        else:
            return bound * self.unit

    def permute(self, p):
        """
        Input: vector p with permutation
        Output: self permuted according to p
        """
        # TODO: should this just be done with perm matrix?
        return Vector(tuple(map(self._value.__getitem__, p._value)))
        # ok now that is some really bad code

    def limit(self, lim):
        return Vector([max(min(lim, value), -lim) for value in self._value])

    def limit_zero(self, lim):
        return Vector([max(min(lim, value), 0) for value in self._value])
