class VectorSpace:
    _IS_MATRIX = False
    _IS_VECTOR = False

    def __add__(self, other):
        raise NotImplementedError("Addition not implemented")

    def __radd__(self, other):
        # Vector Space addition is commutative
        return self + other

    def __mul__(self, other):
        raise NotImplementedError('Scaling not implemented')

    def __rmul__(self, other):
        # Vector Space scaling is commutative
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __eq__(self, other):
        raise NotImplementedError('Equality not implemented')

    def __abs__(self):
        raise NotImplementedError('Norm not implemented')
