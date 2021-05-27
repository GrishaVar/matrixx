class Immutable:
    def __hash__(self, other):
        raise NotImplementedError('Hash not implemented')

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __setattr__(self, *args):
        raise AttributeError("Object is immutable.\
        Use _explicit_setattr if you know what you're doing")

    def _explicit_setattr(self, attr, value):
        object.__setattr__(self, attr, value)
