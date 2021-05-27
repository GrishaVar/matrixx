from math import cos, sin, sqrt

from matrixx.matrix import Matrix as M
from matrixx.vector import Vector as V


class V2:
    Z = V((0,0), 0)
    E = V((1,1), sqrt(2))
    I, J = V((1,0), 1), V((0,1), 1)


class V3:
    Z = V((0,0,0), 0)
    E = V((1,1,1), sqrt(3))
    I, J, K = V((1,0,0), 1), V((0,1,0), 1), V((0,0,1), 1)
    IJ, IK, JK = (V((1,1,0), sqrt(2)), V((1,0,1), sqrt(2)), V((0,1,1), sqrt(2)))


class M2:
    Z = M(((0,0), (0,0)), 0)
    E = M(((1,0), (0,1)), 1)

    @staticmethod
    def grower2(s):
        if s == 0:
            return M2.z
        if s == 1:
            return M2.e
        m = M((
            (s, 0),
            (0, s),
        ))
        m._det = s**2
        return m


class M3:
    z = M(((0,0,0), (0,0,0), (0,0,0)))
    z._det = 0

    e = M(((1,0,0), (0,1,0), (0,0,1)))
    e._det = 1

    @staticmethod
    def x_rot(a):  # right hand rule rotation!
        m = M((
            (1, 0, 0),
            (0, cos(a), -sin(a)),
            (0, sin(a), cos(a)),
        ))
        m._det = 1
        return m

    @staticmethod
    def y_rot(a):
        if a == 1:
            return M3.e
        m = M((
            (cos(a), 0, -sin(a)),
            (0, 1, 0),
            (sin(a), 0, cos(a)),
        ))
        m._det = 1
        return m

    @staticmethod
    def z_rot(a):
        if a == 1:
            return M3.e
        m = M((
            (cos(a), -sin(a), 0),
            (sin(a), cos(a), 0),
            (0, 0, 1),
        ))
        m._det = 1
        return m

    @staticmethod
    def grow(s):
        if s == 0:
            return M3.z
        if s == 1:
            return M3.e
        m = M((
            (s, 0, 0),
            (0, s, 0),
            (0, 0, s),
        ))
        m._det = s**3
        return m
