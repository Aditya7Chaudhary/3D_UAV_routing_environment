"""Occupancy grid map (pure NumPy port of map_utils/grid_map.hpp)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BasicMapParams:
    map_origin: np.ndarray = field(default_factory=lambda: np.array([-20.0, -20.0, 0.0]))
    map_size: np.ndarray = field(default_factory=lambda: np.array([40.0, 40.0, 5.0]))
    min_range: np.ndarray = field(default_factory=lambda: np.zeros(3))
    max_range: np.ndarray = field(default_factory=lambda: np.zeros(3))
    map_volume: float = 0.0

    def finalize(self):
        self.map_origin = np.asarray(self.map_origin, dtype=float)
        self.map_size = np.asarray(self.map_size, dtype=float)
        self.min_range = self.map_origin.copy()
        self.max_range = self.map_origin + self.map_size
        self.map_volume = float(np.prod(self.map_size))


@dataclass
class GridMapParams:
    basic_mp: BasicMapParams = field(default_factory=BasicMapParams)
    resolution: float = 0.1
    inv_resolution: float = 10.0
    map_grid_size: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=int))
    map_grid_size_ytz: int = 0
    clamp_min_log: float = 0.01
    clamp_max_log: float = 0.99
    min_thrd: float = 0.80


class GridMap:
    def __init__(self):
        self.mp = GridMapParams()
        self.occupancy = np.zeros(0, dtype=float)
        self.obs_pts: list[np.ndarray] = []
        self._rng = np.random.default_rng(0)
        self._rand_low = np.zeros(3)
        self._rand_high = np.ones(3)

    def init_map(self, mpa: GridMapParams):
        self.mp = mpa
        self.mp.basic_mp.finalize()
        self.mp.inv_resolution = 1.0 / self.mp.resolution
        self.mp.map_grid_size = np.ceil(
            self.mp.basic_mp.map_size / self.mp.resolution
        ).astype(int)
        self.mp.map_grid_size_ytz = int(
            self.mp.map_grid_size[1] * self.mp.map_grid_size[2]
        )
        buffer_size = int(self.mp.map_grid_size[0] * self.mp.map_grid_size_ytz)
        self.occupancy = np.full(buffer_size, self.mp.clamp_min_log, dtype=float)
        self.obs_pts = []
        self.print_map_info()

    def print_map_info(self):
        print("++++++++++++++++++++++++++++++++++++++")
        print("+++++++ Grid Map Information +++++++++")
        print(f"+++ resolution : {self.mp.resolution}")
        print(f"+++ map volume : {self.mp.basic_mp.map_volume}")
        print(
            f"+++ origin     : {self.mp.basic_mp.map_origin[0]} "
            f"{self.mp.basic_mp.map_origin[1]} {self.mp.basic_mp.map_origin[2]}"
        )
        print(
            f"+++ size       : {self.mp.basic_mp.map_size[0]} "
            f"{self.mp.basic_mp.map_size[1]} {self.mp.basic_mp.map_size[2]}"
        )
        print("++++++++++++++++++++++++++++++++++++++")

    def set_uni_rand(self, rng: np.random.Generator):
        self._rng = rng
        self._rand_low = self.mp.basic_mp.min_range.copy()
        self._rand_high = self.mp.basic_mp.max_range.copy()

    def clear_all_occ(self):
        self.occupancy.fill(self.mp.clamp_min_log)
        self.obs_pts.clear()

    def pos_to_index(self, pos: np.ndarray) -> np.ndarray:
        return np.floor(
            (np.asarray(pos, dtype=float) - self.mp.basic_mp.map_origin)
            * self.mp.inv_resolution
        ).astype(int)

    def index_to_pos(self, idx: np.ndarray) -> np.ndarray:
        return self.mp.basic_mp.map_origin + (np.asarray(idx, dtype=float) + 0.5) * self.mp.resolution

    def get_buffer_cnt(self, idx: np.ndarray) -> int:
        return int(
            idx[0] * self.mp.map_grid_size_ytz
            + idx[1] * self.mp.map_grid_size[2]
            + idx[2]
        )

    def is_in_map_idx(self, idx: np.ndarray) -> bool:
        return bool(
            0 <= idx[0] < self.mp.map_grid_size[0]
            and 0 <= idx[1] < self.mp.map_grid_size[1]
            and 0 <= idx[2] < self.mp.map_grid_size[2]
        )

    def is_in_map(self, pos: np.ndarray) -> bool:
        pos = np.asarray(pos, dtype=float)
        mn, mx = self.mp.basic_mp.min_range, self.mp.basic_mp.max_range
        return bool(
            mn[0] <= pos[0] < mx[0]
            and mn[1] <= pos[1] < mx[1]
            and mn[2] <= pos[2] < mx[2]
        )

    def is_occ(self, pos: np.ndarray) -> int:
        idx = self.pos_to_index(pos)
        if not self.is_in_map_idx(idx):
            return -1
        return 1 if self.occupancy[self.get_buffer_cnt(idx)] > self.mp.min_thrd else 0

    def is_occ_idx(self, idx: np.ndarray) -> int:
        if not self.is_in_map_idx(idx):
            return -1
        return 1 if self.occupancy[self.get_buffer_cnt(idx)] > self.mp.min_thrd else 0

    def set_occ(self, pos: np.ndarray):
        idx = self.pos_to_index(pos)
        if not self.is_in_map_idx(idx):
            return
        self.occupancy[self.get_buffer_cnt(idx)] = self.mp.clamp_max_log
        self.obs_pts.append(np.asarray(pos, dtype=float))

    def set_occ_idx(self, idx: np.ndarray):
        if not self.is_in_map_idx(idx):
            return
        pos = self.index_to_pos(idx)
        self.occupancy[self.get_buffer_cnt(idx)] = self.mp.clamp_max_log
        self.obs_pts.append(pos)

    def get_grid_val(self, p: float) -> float:
        return np.floor(p * self.mp.inv_resolution) * self.mp.resolution + self.mp.resolution / 2.0

    def get_grid_center_pos(self, pos: np.ndarray) -> np.ndarray:
        return np.array(
            [self.get_grid_val(pos[0]), self.get_grid_val(pos[1]), self.get_grid_val(pos[2])]
        )

    def get_uni_rand_pos(self) -> np.ndarray:
        pos = self._rng.uniform(self._rand_low, self._rand_high)
        return self.get_grid_center_pos(pos)

    def fill_map(self, cloud: np.ndarray, inflated: float = 0.0):
        """Fill occupancy from an (N,3) point cloud, optionally inflated."""
        if cloud.size == 0:
            return
        if inflated > 0.0:
            step = int(np.ceil(inflated / self.mp.resolution))
            offsets = np.arange(-step, step + 1) * self.mp.resolution
            for pt in cloud:
                for dx in offsets:
                    for dy in offsets:
                        for dz in offsets:
                            infl = pt + np.array([dx, dy, dz])
                            if self.is_occ(infl) == 0:
                                self.set_occ(infl)
        else:
            for pt in cloud:
                if self.is_occ(pt) == 0:
                    self.set_occ(pt)

    def publish_map(self) -> np.ndarray:
        """Return centers of occupied voxels as (N,3)."""
        occupied = np.argwhere(self.occupancy.reshape(self.mp.map_grid_size) > self.mp.min_thrd)
        if occupied.size == 0:
            return np.zeros((0, 3))
        return np.array([self.index_to_pos(idx) for idx in occupied])

    def publish_2d_map(self) -> np.ndarray:
        """Project occupied columns to z=0 plane."""
        grid = self.occupancy.reshape(self.mp.map_grid_size)
        pts = []
        for x in range(self.mp.map_grid_size[0]):
            for y in range(self.mp.map_grid_size[1]):
                zs = np.where(grid[x, y, :] > self.mp.min_thrd)[0]
                if zs.size:
                    pos = self.index_to_pos(np.array([x, y, zs[0]]))
                    pts.append([pos[0], pos[1], 0.0])
        return np.asarray(pts, dtype=float) if pts else np.zeros((0, 3))

    def get_obs_pts(self) -> np.ndarray:
        if not self.obs_pts:
            return np.zeros((0, 3))
        return np.vstack(self.obs_pts)
