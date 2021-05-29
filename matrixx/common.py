from math import cos, sin, sqrt

from matrixx.matrix import Matrix as M
from matrixx.vector import Vector as V


class V2:
    Z = V((0, 0), 0)
    E = V((1, 1), sqrt(2))
    I, J = V((1, 0), 1), V((0, 1), 1)


class V3:
    Z = V((0, 0, 0), 0)
    E = V((1, 1, 1), sqrt(3))
    I, J, K = (
        V((1, 0, 0), 1),
        V((0, 1, 0), 1),
        V((0, 0, 1), 1),
    )
    IJ, IK, JK = (
        V((1, 1, 0), sqrt(2)),
        V((1, 0, 1), sqrt(2)),
        V((0, 1, 1), sqrt(2)),
    )


class M2:
    Z = M(((0, 0), (0, 0)), 0)
    E = M(((1, 0), (0, 1)), 1)

    @staticmethod
    def rot(a):  # rotates up / anticlockwise
        if not a:
            return M2.E
        return M((
            (cos(a), -sin(a)),
            (sin(a), cos(a)),
        ), 1)


class M3:
    Z = M(((0, 0, 0), (0, 0, 0), (0, 0, 0)), 0)
    E = M(((1, 0, 0), (0, 1, 0), (0, 0, 1)), 1)

    @staticmethod
    def x_rot(a):  # right hand rule rotation!
        if not a:
            return M3.E
        return M((
            (1, 0, 0),
            (0, cos(a), -sin(a)),
            (0, sin(a), cos(a)),
        ), 1)

    @staticmethod
    def y_rot(a):
        if not a:
            return M3.E
        return M((
            (cos(a), 0, -sin(a)),
            (0, 1, 0),
            (sin(a), 0, cos(a)),
        ), 1)

    @staticmethod
    def z_rot(a):
        if not a:
            return M3.E
        return M((
            (cos(a), -sin(a), 0),
            (sin(a), cos(a), 0),
            (0, 0, 1),
        ), 1)
