"""Structured random obstacle map generator (port of struct_map_gen.hpp)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo_map import GeoMap
from .grid_map import GridMap, GridMapParams
from .shapes import CircleGate, Cylinder, Ellipsoid, Polyhedron, RectGate


@dataclass
class MapGenParams:
    cylinder_ratio: float = 0.01
    circle_ratio: float = 0.005
    gate_ratio: float = 0.005
    ellip_ratio: float = 0.005
    poly_ratio: float = 0.005
    w1: float = 0.02
    w2: float = 0.15
    w3: float = 0.6
    add_noise: bool = False


class StructMapGenerator:
    def __init__(self):
        self.cloud_map = np.zeros((0, 3), dtype=float)
        self.grid_map = GridMap()
        self.geo_map = GeoMap()
        self.mpa = GridMapParams()
        self.mgpa = MapGenParams()
        self.rng = np.random.default_rng(1)
        self.vel_h = np.array([0.1, 0.1, 0.0])
        self.cyl_list: list[Cylinder] = []
        self.cir_list: list[CircleGate] = []
        self.gate_list: list[RectGate] = []
        self.ellip_list: list[Ellipsoid] = []
        self.poly_list: list[Polyhedron] = []

    def init_params(self, mpa: GridMapParams):
        mpa.basic_mp.finalize()
        self.grid_map.init_map(mpa)
        self.mpa = mpa

    def change_res(self, res: float):
        self.mpa.resolution = res
        self.grid_map.init_map(self.mpa)

    def get_pc(self) -> np.ndarray:
        return self.cloud_map.copy()

    def get_pc_2d(self) -> np.ndarray:
        return self.grid_map.publish_2d_map()

    def clear(self):
        self.cloud_map = np.zeros((0, 3), dtype=float)
        self.grid_map.clear_all_occ()
        self.geo_map.clear_all()

    def _append_points(self, pts: list[np.ndarray]):
        if not pts:
            return
        arr = np.asarray(pts, dtype=float)
        self.cloud_map = (
            arr if self.cloud_map.size == 0 else np.vstack([self.cloud_map, arr])
        )

    def update_pts(self, geo_rep) -> int:
        bound = geo_rep.get_bd()
        cpt = geo_rep.get_center()
        res = self.mpa.resolution
        wid = np.ceil(bound / res).astype(int)

        cur = 0
        total_pts = []
        new_pts = []
        for r in range(-wid[0], wid[0]):
            rx = r * res
            for s in range(-wid[1], wid[1]):
                sy = s * res
                for t in range(-wid[2], wid[2]):
                    ob_pt = cpt + np.array([rx, sy, t * res])
                    if self.grid_map.is_occ(ob_pt) != 0:
                        continue
                    if not geo_rep.is_inside(ob_pt):
                        continue
                    self.grid_map.set_occ(ob_pt)
                    new_pts.append(ob_pt)
                    total_pts.append(ob_pt.copy())
                    geo_rep.cloud.append(ob_pt.copy())
                    cur += 1

        if self.mgpa.add_noise and total_pts:
            noise = self.rng.normal(0.0, self.mgpa.w1, size=(len(total_pts), 3))
            for i, ob_pt in enumerate(total_pts):
                if self.rng.random() >= self.mgpa.w1:
                    continue
                noisy = ob_pt + noise[i]
                if not self.grid_map.is_in_map(noisy):
                    continue
                self.grid_map.set_occ(noisy)
                new_pts.append(noisy)
                geo_rep.cloud.append(noisy.copy())

        self._append_points(new_pts)
        return cur

    def update_pts_light(self, geo_rep) -> int:
        rect = geo_rep.get_rect()
        cpt = geo_rep.get_center()
        rot = geo_rep.get_rot()
        res = self.mpa.resolution
        wid_num1 = int(np.ceil(rect[1] / res))
        wid_num2 = int(np.ceil(rect[2] / res))
        width = int(np.ceil(rect[0] / res))
        signs = np.array([[1, -1, 1, -1], [1, 1, -1, -1]], dtype=float)

        cur = 0
        total_pts = []
        new_pts = []
        for t in range(-width, width):
            tx = t * res
            for r in range(wid_num1):
                ry = r * res
                for s in range(wid_num2):
                    sz = s * res
                    for i in range(4):
                        local = np.array([tx, ry * signs[0, i], sz * signs[1, i]])
                        ob_pt = cpt + rot @ local
                        if not geo_rep.is_inside(ob_pt):
                            continue
                        if self.grid_map.is_occ(ob_pt) != 0:
                            continue
                        self.grid_map.set_occ(ob_pt)
                        new_pts.append(ob_pt)
                        total_pts.append(ob_pt.copy())
                        geo_rep.cloud.append(ob_pt.copy())
                        cur += 1

        if self.mgpa.add_noise and total_pts:
            for ob_pt in total_pts:
                if self.rng.random() >= self.mgpa.w1:
                    continue
                noisy = ob_pt + self.rng.normal(0.0, self.mgpa.w1, size=3)
                if not self.grid_map.is_in_map(noisy):
                    continue
                self.grid_map.set_occ(noisy)
                new_pts.append(noisy)
                geo_rep.cloud.append(noisy.copy())

        self._append_points(new_pts)
        return cur

    def traverse_pts(self, geo_reps):
        for geo_rep in geo_reps:
            self.update_pts(geo_rep)

    def reset_map(self):
        self.cloud_map = np.zeros((0, 3), dtype=float)
        self.traverse_pts(self.geo_map.polyhedron)
        self.traverse_pts(self.geo_map.cylinder)
        self.traverse_pts(self.geo_map.ellipsoid)
        self.traverse_pts(self.geo_map.circle_gate)
        self.traverse_pts(self.geo_map.rect_gate)

    def change_ratios(self, seed: float, if_dyn: bool = False, dt: float = 0.1):
        if if_dyn:
            self.dyn_generate(dt)
            return
        self.rng = np.random.default_rng(int(seed))
        # Match original: w2 * rand_w * rand_w where rand_w ~ U(w1, w2)
        rw = lambda: self.rng.uniform(self.mgpa.w1, self.mgpa.w2)
        self.mgpa.cylinder_ratio = self.mgpa.w2 * rw() * rw()
        self.mgpa.circle_ratio = self.mgpa.w2 * rw() * rw()
        self.mgpa.gate_ratio = self.mgpa.w2 * rw() * rw()
        self.mgpa.ellip_ratio = self.mgpa.w2 * rw() * rw()
        self.mgpa.poly_ratio = self.mgpa.w2 * rw() * rw()
        total = (
            self.mgpa.cylinder_ratio
            + self.mgpa.circle_ratio
            + self.mgpa.gate_ratio
            + self.mgpa.ellip_ratio
            + self.mgpa.poly_ratio
        )
        if total > self.mgpa.w2 * 0.5:
            scale = (self.mgpa.w2 * 0.5) / total
            self.mgpa.cylinder_ratio *= scale
            self.mgpa.circle_ratio *= scale
            self.mgpa.gate_ratio *= scale
            self.mgpa.ellip_ratio *= scale
            self.mgpa.poly_ratio *= scale
        self.generate(False)

    def random_uni_map_gen(self, mgpa: MapGenParams, seed: float, if_dyn: bool = False):
        self.mgpa = mgpa
        self.rng = np.random.default_rng(int(seed))
        self.generate(if_dyn)

    def _rand_w(self) -> float:
        return float(self.rng.uniform(self.mgpa.w1, self.mgpa.w2))

    def _rand_radiu(self) -> float:
        return float(self.rng.uniform(self.mgpa.w1, self.mgpa.w3))

    def _rand_h(self) -> float:
        z = self.mpa.basic_mp.map_size[2]
        return float(self.rng.uniform(0.5 * z, z))

    def _rand_theta(self) -> float:
        return float(self.rng.uniform(-np.pi, np.pi))

    def _sample_vel(self, dyn_mode: bool) -> np.ndarray:
        if not dyn_mode:
            return np.zeros(3)
        return np.array(
            [
                self.rng.uniform(-self.vel_h[0], self.vel_h[0]),
                self.rng.uniform(-self.vel_h[1], self.vel_h[1]),
                0.0,
            ]
        )

    def generate(self, dyn_mode: bool = False):
        self.grid_map.set_uni_rand(self.rng)
        all_grids = int(np.ceil(self.mpa.basic_mp.map_volume / (self.mpa.resolution**3)))
        cylinder_grids = int(np.ceil(all_grids * self.mgpa.cylinder_ratio))
        circle_grids = int(np.ceil(all_grids * self.mgpa.circle_ratio))
        gate_grids = int(np.ceil(all_grids * self.mgpa.gate_ratio))
        ellip_grids = int(np.ceil(all_grids * self.mgpa.ellip_ratio))
        poly_grids = int(np.ceil(all_grids * self.mgpa.poly_ratio))

        self.cyl_list.clear()
        self.cir_list.clear()
        self.gate_list.clear()
        self.ellip_list.clear()
        self.poly_list.clear()

        cur = 0
        while cur < cylinder_grids:
            cpt = self.grid_map.get_uni_rand_pos()
            h = self._rand_h()
            w = 0.2 + self._rand_w()
            cyl = Cylinder(cpt, w, h)
            cyl.set_vel(self._sample_vel(dyn_mode))
            cur += self.update_pts(cyl)
            self.geo_map.add(cyl)
            if dyn_mode:
                self.cyl_list.append(cyl)
        cylinder_grids = cur

        cur = 0
        while cur < circle_grids:
            cpt = self.grid_map.get_uni_rand_pos()
            theta = self._rand_theta()
            width = 0.1 + 0.2 * self._rand_radiu()
            bound = np.array(
                [width, width + self._rand_radiu(), width + self._rand_radiu()]
            )
            gate = CircleGate(cpt, bound, theta)
            gate.set_vel(self._sample_vel(dyn_mode))
            cur += self.update_pts_light(gate)
            self.geo_map.add(gate)
            if dyn_mode:
                self.cir_list.append(gate)
        circle_grids = cur

        cur = 0
        while cur < gate_grids:
            cpt = self.grid_map.get_uni_rand_pos()
            theta = self._rand_theta()
            width = 0.1 + 0.2 * self._rand_radiu()
            bound = np.array(
                [width, width + self._rand_radiu(), width + self._rand_radiu()]
            )
            gate = RectGate(cpt, bound, theta)
            gate.set_vel(self._sample_vel(dyn_mode))
            cur += self.update_pts_light(gate)
            self.geo_map.add(gate)
            if dyn_mode:
                self.gate_list.append(gate)
        gate_grids = cur

        cur = 0
        while cur < ellip_grids:
            cpt = self.grid_map.get_uni_rand_pos()
            euler = np.array([self._rand_theta(), self._rand_theta(), self._rand_theta()])
            bound = np.array([self._rand_radiu(), self._rand_radiu(), self._rand_radiu()])
            ellip = Ellipsoid()
            ellip.init(cpt, bound, euler)
            ellip.set_vel(self._sample_vel(dyn_mode))
            cur += self.update_pts(ellip)
            self.geo_map.add(ellip)
            if dyn_mode:
                self.ellip_list.append(ellip)
        ellip_grids = cur

        cur = 0
        while cur < poly_grids:
            cpt = self.grid_map.get_uni_rand_pos()
            bound = np.array([self._rand_radiu(), self._rand_radiu(), self._rand_radiu()])
            poly = Polyhedron()
            poly.random_init(cpt, bound, self.rng)
            poly.set_vel(self._sample_vel(dyn_mode))
            cur += self.update_pts(poly)
            self.geo_map.add(poly)
            if dyn_mode:
                self.poly_list.append(poly)
        poly_grids = cur

        print("++++++++++++++++++++++++++++++++++++++")
        print("+++ Finished generate random map ! +++")
        print("+++ The ratios for geometries are: +++")
        print(f"+++ cylinders  : {100 * cylinder_grids / max(all_grids, 1):.2f}%")
        print(f"+++ circles    : {100 * circle_grids / max(all_grids, 1):.2f}%")
        print(f"+++ gates      : {100 * gate_grids / max(all_grids, 1):.2f}%")
        print(f"+++ ellipsoids : {100 * ellip_grids / max(all_grids, 1):.2f}%")
        print(f"+++ polytopes  : {100 * poly_grids / max(all_grids, 1):.2f}%")
        print("++++++++++++++++++++++++++++++++++++++")

    def _update_moving_shape(self, shape, dt, bounds):
        x_l, x_h, y_l, y_h, z_l, z_h = bounds
        cur = shape.get_center()
        vel = shape.get_vel()
        nxt = cur + dt * vel
        nxt[2] = cur[2]

        def bounce(pos, v, lo, hi):
            if pos < lo:
                pos = lo
                v *= -1
            if pos > hi:
                pos = hi
                v *= -1
            return pos, v

        nxt[0], vel[0] = bounce(nxt[0], vel[0], x_l, x_h)
        nxt[1], vel[1] = bounce(nxt[1], vel[1], y_l, y_h)
        nxt[2], vel[2] = bounce(nxt[2], vel[2], z_l, z_h)
        shape.set_center(nxt)
        shape.set_vel(vel)
        dist = nxt - cur
        if shape.cloud:
            moved = np.asarray(shape.cloud, dtype=float) + dist
            shape.cloud = [p for p in moved]
            self.cloud_map = (
                moved
                if self.cloud_map.size == 0
                else np.vstack([self.cloud_map, moved])
            )

    def pcl2grid(self):
        count = 0
        for pt in self.cloud_map:
            if not self.grid_map.is_in_map(pt):
                continue
            self.grid_map.set_occ(pt)
            count += 1
        print(f"GRID OBS: {count}")

    def dyn_generate(self, dt: float):
        bounds = (-10, 10, -10, 10, 0, 5)
        for c in self.cir_list:
            self._update_moving_shape(c, dt, bounds)
        for c in self.ellip_list:
            self._update_moving_shape(c, dt, bounds)
        for c in self.cyl_list:
            self._update_moving_shape(c, dt, bounds)
        for c in self.poly_list:
            self._update_moving_shape(c, dt, bounds)
        for c in self.gate_list:
            self._update_moving_shape(c, dt, bounds)
        self.pcl2grid()
