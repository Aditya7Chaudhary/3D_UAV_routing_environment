"""Geometric obstacle shapes used by the structured map generator."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


def euler_to_rot(euler: np.ndarray) -> np.ndarray:
    """ZYX quaternion composition matching the original C++ eulerToRot."""
    r, p, y = euler
    qx = np.array([np.cos(r / 2), np.sin(r / 2), 0.0, 0.0])
    qy = np.array([np.cos(p / 2), 0.0, np.sin(p / 2), 0.0])
    qz = np.array([np.cos(y / 2), 0.0, 0.0, np.sin(y / 2)])

    def qmul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ]
        )

    q = qmul(qz, qmul(qy, qx))
    w, x, y_, z = q
    return np.array(
        [
            [1 - 2 * (y_ * y_ + z * z), 2 * (x * y_ - z * w), 2 * (x * z + y_ * w)],
            [2 * (x * y_ + z * w), 1 - 2 * (x * x + z * z), 2 * (y_ * z - x * w)],
            [2 * (x * z - y_ * w), 2 * (y_ * z + x * w), 1 - 2 * (x * x + y_ * y_)],
        ]
    )


class Polyhedron:
    def __init__(self):
        self.h_poly = None  # (6, n_faces): point then normal
        self.A = None
        self.b0 = None
        self.bd = np.zeros(3)
        self.cpt = np.zeros(3)
        self.v = np.zeros(3)
        self.cloud = []

    def _build_planes(self):
        faces = self.h_poly.shape[1]
        self.A = np.zeros((faces, 3))
        self.b0 = np.zeros(faces)
        for i in range(faces):
            p = self.h_poly[:3, i]
            n = self.h_poly[3:, i]
            self.A[i] = n
            self.b0[i] = n.dot(p)

    def is_inside(self, pt: np.ndarray, epsilon: float = 1e-6) -> bool:
        if self.A is None or self.A.shape[0] == 0:
            return True
        return float(np.max(self.A @ pt - self.b0)) <= epsilon

    def random_init(self, cpt: np.ndarray, bound: np.ndarray, rng: np.random.Generator):
        n_verts = int(np.ceil(rng.uniform(4, 35)))
        verts = np.column_stack(
            [
                cpt[0] + rng.uniform(-bound[0], bound[0], n_verts),
                cpt[1] + rng.uniform(-bound[1], bound[1], n_verts),
                cpt[2] + rng.uniform(-bound[2], bound[2], n_verts),
            ]
        )
        hull = ConvexHull(verts)
        faces = hull.simplices
        h_poly = np.zeros((6, len(faces)))
        for i, tri in enumerate(faces):
            p0, p1, p2 = verts[tri]
            edge0 = p1 - p0
            edge1 = p2 - p1
            normal = np.cross(edge0, edge1)
            nrm = np.linalg.norm(normal)
            if nrm < 1e-12:
                normal = np.array([0.0, 0.0, 1.0])
            else:
                normal = normal / nrm
            # Orient outward relative to centroid
            if normal.dot(p1 - cpt) < 0:
                normal = -normal
            h_poly[:3, i] = p1
            h_poly[3:, i] = normal
        self.h_poly = h_poly
        self._build_planes()
        self.bd = bound.copy()
        self.cpt = cpt.copy()
        self.cloud = []

    def get_bd(self):
        return self.bd.copy()

    def get_center(self):
        return self.cpt.copy()

    def set_center(self, cpt):
        self.cpt = np.asarray(cpt, dtype=float)

    def get_vel(self):
        return self.v.copy()

    def set_vel(self, vel):
        self.v = np.asarray(vel, dtype=float)


class Cylinder:
    def __init__(self, cpt: np.ndarray, r: float, h: float):
        self.cpt = np.asarray(cpt, dtype=float)
        self.r = float(r)
        self.h = float(h)
        self.bd = np.array([r, r, h], dtype=float)
        self.v = np.zeros(3)
        self.cloud = []

    def is_inside(self, pt: np.ndarray) -> bool:
        if abs(pt[2] - self.cpt[2]) >= 0.5 * self.h:
            return False
        return float(np.linalg.norm(pt[:2] - self.cpt[:2])) <= self.r

    def get_bd(self):
        return self.bd.copy()

    def get_center(self):
        return self.cpt.copy()

    def set_center(self, cpt):
        self.cpt = np.asarray(cpt, dtype=float)

    def get_vel(self):
        return self.v.copy()

    def set_vel(self, vel):
        self.v = np.asarray(vel, dtype=float)


class Ellipsoid:
    def __init__(self):
        self.E = np.eye(3)
        self.Einv = np.eye(3)
        self.d = np.zeros(3)
        self.bd = np.zeros(3)
        self.v = np.zeros(3)
        self.cloud = []

    def init(self, cpt: np.ndarray, bound: np.ndarray, euler: np.ndarray):
        R = euler_to_rot(np.asarray(euler, dtype=float))
        coeff = np.diag(bound)
        self.E = R @ coeff @ R.T
        self.Einv = np.linalg.inv(self.E)
        self.d = np.asarray(cpt, dtype=float)
        self.bd = np.asarray(bound, dtype=float)
        self.cloud = []

    def is_inside(self, pt: np.ndarray) -> bool:
        return float(np.linalg.norm(self.Einv @ (pt - self.d))) <= 1.0

    def get_bd(self):
        return self.bd.copy()

    def get_center(self):
        return self.d.copy()

    def set_center(self, cpt):
        self.d = np.asarray(cpt, dtype=float)

    def get_vel(self):
        return self.v.copy()

    def set_vel(self, vel):
        self.v = np.asarray(vel, dtype=float)


class CircleGate:
    def __init__(self, cpt: np.ndarray, rect: np.ndarray, theta: float):
        self.cpt = np.asarray(cpt, dtype=float)
        self.rect = np.asarray(rect, dtype=float)
        self.theta = float(theta)
        c, s = np.cos(theta), np.sin(theta)
        self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        pt = self.R @ self.rect
        self.bd = np.array(
            [self.rect[0] + abs(pt[0]), self.rect[0] + abs(pt[1]), self.rect[2]]
        )
        self.v = np.zeros(3)
        self.cloud = []

    def is_inside(self, pt: np.ndarray) -> bool:
        diff = self.R.T @ (pt - self.cpt)
        if abs(diff[0]) >= self.rect[0]:
            return False
        if (diff[1] ** 2) / (self.rect[1] ** 2) + (diff[2] ** 2) / (self.rect[2] ** 2) >= 1.0:
            return False
        a = self.rect[1] - self.rect[0]
        b = self.rect[2] - self.rect[0]
        if a <= 0 or b <= 0:
            return True
        if (diff[1] ** 2) / (a * a) + (diff[2] ** 2) / (b * b) <= 1.0:
            return False
        return True

    def get_bd(self):
        return self.bd.copy()

    def get_center(self):
        return self.cpt.copy()

    def get_rect(self):
        return self.rect.copy()

    def get_rot(self):
        return self.R.copy()

    def set_center(self, cpt):
        self.cpt = np.asarray(cpt, dtype=float)

    def get_vel(self):
        return self.v.copy()

    def set_vel(self, vel):
        self.v = np.asarray(vel, dtype=float)


class RectGate:
    def __init__(self, cpt: np.ndarray, rect: np.ndarray, theta: float):
        self.cpt = np.asarray(cpt, dtype=float)
        self.rect = np.asarray(rect, dtype=float)
        self.theta = float(theta)
        c, s = np.cos(theta), np.sin(theta)
        self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.bd = np.array([self.rect[1], self.rect[1], self.rect[2]])
        self.v = np.zeros(3)
        self.cloud = []

    def is_inside(self, pt: np.ndarray) -> bool:
        diff = self.R.T @ (pt - self.cpt)
        if (
            abs(diff[2]) >= self.rect[2]
            or abs(diff[1]) >= self.rect[1]
            or abs(diff[0]) >= self.rect[0]
        ):
            return False
        if abs(diff[2]) <= self.rect[2] - self.rect[0] and abs(diff[1]) <= self.rect[
            1
        ] - self.rect[0]:
            return False
        return True

    def get_bd(self):
        return self.bd.copy()

    def get_center(self):
        return self.cpt.copy()

    def get_rect(self):
        return self.rect.copy()

    def get_rot(self):
        return self.R.copy()

    def set_center(self, cpt):
        self.cpt = np.asarray(cpt, dtype=float)

    def get_vel(self):
        return self.v.copy()

    def set_vel(self, vel):
        self.v = np.asarray(vel, dtype=float)
