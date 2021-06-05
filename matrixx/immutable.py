class Immutable:
    def __hash__(self, other):
        raise NotImplementedError('Hash not implemented')

    def __eq__(self, other):
        return hash(self) == hash(other)
