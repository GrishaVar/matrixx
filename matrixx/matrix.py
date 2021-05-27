from operator import mul
from math import prod

from matrixx.vector_space import VectorSpace
import matrixx.vector


class Matrix(VectorSpace):  # TODO extract matrix and common to linalg module?
    """
    Matrix implementation. Index with M[i,j] (start at zero!)

    Supports addition (+), subtraction (-),
    negation (-m), scaling (*),
    matrix multiplication(@), powers (**).

    det(self) returns determinant (memoised).
    row_switch(self, i, j) elementary row operation swap.
    row_mult(self, i, m) elementary row operation multiply.
    row_add(self, i, j, m) elementary row operation add.

    decomp_LR() creates an LR decomposition (with scaling and partial pivoting)
    solve_LGS_via_LR(b) solves Ax=b for invertable A
    find_det_via_LR() finds the determinant of an n*n matrix
    find_inverse_via_LR() finds the inverse of an invertable matrix

    size: (#rows, #cols)
    is_square: #rows == #cols
    t: transpose
    """
    _IS_MATRIX = True

    def __init__(self, rows, det=None):  # assumes input is tuple of tuples!
        self._value = rows
        m = len(rows)
        n = len(rows[0])
        self.size = (m, n)
        self.is_square = m == n
        self._det = det
        self._LR = None

    @property
    def det(self):
        """
        Calculate and store determinant.
        :return: int
        """
        if self._det is None:  # kinda-sorta-memoised determinant
            v = self._value
            m, n = self.size
            if not self.is_square:
                self._det = 0
            elif m == 1:
                self._det = v[0][0]
            elif m == 2:
                self._det = (
                    v[0][0]*v[1][1] -
                    v[0][1]*v[1][0]
                )
            elif m == 3:
                # a(ei - fh) - b(di - fg) + c(dh - eg)
                self._det = (
                    v[0][0] * (v[1][1] * v[2][2] - v[1][2] * v[2][1]) -
                    v[0][1] * (v[1][0] * v[2][2] - v[1][2] * v[2][0]) +
                    v[0][2] * (v[1][0] * v[2][1] - v[1][1] * v[2][0])
                )
                # I would feel bad about doing this if it wasn't the best way
            else:
                self._det = self.find_det_via_LR()
        return self._det

    def __repr__(self):
        res_parts = []
        for row in self._value:
            res_parts.append('\t'.join(map(str, row)))
        res = ')\n('.join(res_parts)
        return '(' + res + ')'

    def __getitem__(self, pos):
        """index row with M[int] or value with M[int,int]. Index from 0."""
        # Inexcusable
        try:
            i, j = pos
            return self._value[i][j]
        except TypeError:  # pos not a tuple => requesting full row
            return self._value[pos]

    def __add__(self, other):
        if other is 0:
            # for sum()
            # this causes a warning but
            # I thought about it and "is 0" is what I want
            return self
        # size = self.size
        # if size != other.size:
        #     raise ValueError('Different Sizes!')
        m, n = self.size
        a = self._value
        b = other._value
        c = tuple(
            tuple(
                a[i][j] + b[i][j] for j in range(n)
            ) for i in range(m)
        )
        return Matrix(c)

    def __mul__(self, a):
        """Scalar multiplication."""
        # TODO: add pairwise
        m, n = self.size
        b = self._value
        c = tuple(
            tuple(
                a * b[i][j] for j in range(n)
            ) for i in range(m)
        )

        if (det := self._det) is not None:
            det *= a
        return Matrix(c, det)

    def __matmul__(self, other):
        if not other._IS_MATRIX:
            return NotImplemented

        if self.size[1] != other.size[0]:
            raise ValueError(f'Incompatible Sizes of multiplaction of {self} and {other}')
        # TODO make some toggleable thing to supress all checks

        a = self._value
        b = other._value
        c = tuple(
            tuple(
                sum(map(mul, a_row, b_col))
                for b_col in zip(*b)
            ) for a_row in a
        )  # I'm quite pleased with myself

        if not ((det := self._det) is None or (b_det := other._det) is None):
            det *= b_det

        return Matrix(c, det)

    def __pow__(self, other):
        # TODO: update to matmul (preferably without creating n Matrix objs)
        raise NotImplementedError('see todo')
        res = self
        for x in range(other-1):
            res *= self
        return res

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        if (res := self._hash) is None:
            res = self._hash = hash(self._value)
        return res

    @property
    def t(self):
        return Matrix(tuple(zip(*self._value)))

    def to_vector(self):
        """
        Converts single-column or single-row matrix into a vector.
        :return: Vector
        """
        m, n = self.size
        if m == 1:
            return vector.Vector(self._value[0])
        if n == 1:
            return vector.Vector(tuple(zip(*self._value))[0])
        raise ValueError('Incompatible size')

    def copy(self):
        return self

    def row_swap(self, i, j):
        """
        Swap row positions.
        :param i: index of row 1
        :param j: index of row 2
        """
        f = lambda x: i if x == j else (j if x == i else x)
        v = self._value
        return Matrix(tuple(v[f(k)] for k in range(self.size)))

    def row_mult(self, i, m):
        """
        Multiply a row by a scalar.
        :param i: index of row
        :param m: non-zero scalar
        """
        if m == 0:
            raise ValueError("m can't be zero!")
        v = self._value
        new = tuple(m * x for x in v[i])
        return Matrix(tuple(row if k != i else new for k, row in enumerate(v)))

    def row_addm(self, i, j, m):
        """
        Add a row to another (with scaling).
        :param i: index of row to be changed
        :param j: index of row to add
        :param m: non-zero scalar
        :return:
        """
        if m == 0:
            raise ValueError("m can't be zero!")
        v = self._value
        new = tuple(x + m*y for x, y in zip(v[i], v[j]))
        return Matrix(tuple(row if k != i else new for k, row in enumerate(v)))

    def decomp_LR(self):
        """
        Compute LU decomposition with scaling and pivoting
        PDA = LU
        O(1/3 n^3)

        Returns L, U matrices and P D vectors
        L is an lower tri matrix
        R is an upper tri matrix
        P is the pivoting vector (represents perm matrix)
        D is the scaling vector (represents dia matrix)
        f: number of flips in P
        """
        if self._LR is not None:
            return self._LR
        m, n = self.size
        if m != n:
            raise ValueError('LR Decomposition only works for square matrices.')
        A = self._value

        # Skalierung
        D = tuple(1/sum(map(abs, row)) for row in A)
        LR = [list(map(d.__mul__, row)) for d, row in zip(D, A)]

        P = list(range(n))
        f = 0  # yes, technically f is redundant, but recalculating the parity
        # of a permutation would require a whole new alg, might as well use
        # the information I have here. TODO: think about returning a list of
        # swaps instead of P and f. Then f is just the length of the list.
        for j in range(n-1):  # j is which col you're doing
            # Pivotisierung
            p = max((abs(row[j]), j+i) for i, row in enumerate(LR[j:]))[1]
            LR[p], LR[j] = LR[j], LR[p]
            if p != j:
                P[p], P[j] = P[j], P[p]
                f += 1

            # Elimination
            for i in range(j+1, n):  # i is each row in the j column
                LR[i][j] /= LR[j][j]
                for k in range(j+1, n):  # k is the columns right of i,j
                    LR[i][k] -= LR[i][j] * LR[j][k]

        L = tuple(
            tuple(
                LRij if i>j else (1 if i==j else 0)
                for j, LRij in enumerate(row)
            ) for i, row in enumerate(LR)
        )

        R = tuple(
            tuple(
                LRij if i<=j else 0
                for j, LRij in enumerate(row)
            ) for i, row in enumerate(LR)
        )

        self._LR = Matrix(L), Matrix(R), vector.Vector(P), vector.Vector(D), f
        # P is a list
        return self._LR

    def forward_insertion(self, b):
        """
        Takes a vector b and solves Ax = b
        assuming A (self) is a lower tri matrix
        O(1/2 n^2)

        returns vector x
        """
        A = self._value
        b = b._value

        x = []
        for i in range(len(b)):
            x.append((b[i] - sum(map(mul, A[i][:i], x))) / A[i][i])
        return vector.Vector(tuple(x))


    def backward_insertion(self, b):
        """
        Takes a vector b and solves Ax = b
        assuming A (self) is an upper tri matrix
        O(1/2 n^2)

        returns vector x
        """
        # TODO I'm pretty sure there should be a clean way to do backward
        # via forward... transpose and flip all vectors?

        n = b.size
        x = [0 for i in range(n)]
        A = self._value
        b = b._value

        for j in range(n-1, -1, -1):
            x[j] = (b[j] - sum(A[j][k]*x[k] for k in range(j+1, n))) / A[j][j]

        return vector.Vector(tuple(x))

    def solve_LGS_via_LR(self, b):
        """
        Takes a vector b and solves Ax = b
        assuming A (self) is invertable
        (aka there will be a unique solution)

        Returns vector x
        """
        L, R, P, D, _ = self.decomp_LR()
        # PDA = LR
        # Ax = b
        # PDAx = PDb
        # LRx = PDb
        # Ly = PDb and Rx = y
        PDb = (D * b).permute(P)
        y = L.forward_insertion(PDb)
        x = R.backward_insertion(y)
        return x

    def find_det_via_LR(self):
        """
        Finds determinant of self using LR decomposition
        does NOT update the det cache
        """
        _, R, _, D, f = self.decomp_LR()
        # PDA = LR
        # det P * det D * det A = det L * det R
        # det P * det D * det A = det R
        # (-1)^f * product(diagonal(D)) * det A = product(diagonal(R))
        # (-1)^f * prod D * det A = product(diagonal(R))
        r = prod(R[i][i] for i in range(D.size))
        d = prod(D._value)
        f = -1 if f%2 else 1  # (-1)^f
        return r / (f * d)

    def find_inverse_via_LR(self):
        res = []  # these are columns of the inverse
        m, n = self.size
        if m != n:
            return None
        # PDA = LR
        # AA' = I
        # Axi = ei where ei is the ith column of I
        for i in range(n):
            ei = vector.Vector(tuple(1 if i == j else 0 for j in range(n)))
            xi = self.solve_LGS_via_LR(ei)
            res.append(xi)
        return Matrix(tuple(zip(*res)))
