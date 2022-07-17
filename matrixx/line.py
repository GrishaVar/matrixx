from math import sqrt
from operator import mul, add
from itertools import repeat
from functools import cached_property

from matrixx.vector_space import VectorSpace
from matrixx.immutable import Immutable
import matrixx.matrix as matrix


_ZERO = 0


class Line(VectorSpace, Immutable):
    """
    Line in N dimensional space implementation. 
    Consists of 2 vectors, one is from the origin to the point where the line
    start from and the second is the direction from the previously mentioned
    point.
    """

    _IS_VECTOR = True

    def __init__(self, point, direction):
        self.point = point
        self.direction = direction

        if self.point.size != self.direction.size:
            raise ValueError(
                "Line source point and direction have different dimensions"
            )

    def __repr__(self):
        return f"{self.point} + n * {self.direction}"

    def __getitem__(self, pos):
        if pos == 0:
            return self.point
        elif pos == 1:
            return self.direction
        else:
            raise ValueError("Lines can only take index 1 or 2")

    def __add__(self, other):
        pass # create plane

    def __mul__(self, a):
        pass # 

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

    @cached_property
    def norm_1(self):
        return sum(map(abs, self._value))

    @cached_property
    def norm_inf(self):
        return max(map(abs, self._value))

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
        try:
            a1, a2, a3 = self._value
            b1, b2, b3 = other._value

            s1 = a2 * b3 - a3 * b2
            s2 = a3 * b1 - a1 * b3
            s3 = a1 * b2 - a2 * b1
        except ValueError:
            raise ValueError(
                "Cross Product only exist for vectors with 3 components."
            )

        return Vector((s1, s2, s3))

    @cached_property
    def perpendicular(self):
        """
        Perpendicular vector. Only defined for 2 dimensional vectors.
        Direction is arbitrary.
        """
        try:
            a1, a2 = self._value
        except ValueError:
            raise ValueError(
                "Perpendicular only implemented for vectors with 2 components."
            )

        return Vector((-a2, a1))

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
