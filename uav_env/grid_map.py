"""
===============================================================================
3D UAV ROUTING ENVIRONMENT — OCCUPANCY GRID MAP
===============================================================================

PURPOSE
-------
This file represents the continuous 3D environment as a regular grid of
small cubic cells, also called voxels.  A voxel is marked occupied when it
contains an obstacle point.  Path planners can query this grid to decide
whether a candidate UAV position is free, occupied, or outside the map.

COORDINATE AND STORAGE CONVENTION
-------------------------------
``map_origin`` is the lower X/Y/Z corner of the map.  A world-space point is
converted to an integer ``[x, y, z]`` voxel index by subtracting that origin
and dividing by ``resolution``.  The index is flattened into one NumPy array
using X-major order, so no Python object is allocated for each voxel.

OCCUPANCY CONVENTION
--------------------
The stored values are probability-like log values retained from the original
map interface.  This port only writes either ``clamp_min_log`` (free) or
``clamp_max_log`` (occupied).  A value strictly above ``min_thrd`` is treated
as occupied.  Queries return 1 for occupied, 0 for free, and -1 outside the
map.

This file rasterises and queries obstacles; it does not generate obstacle
geometry or perform route planning.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BasicMapParams:
    """Store the physical dimensions of the map and its derived limits.

    ``map_origin`` and ``map_size`` are supplied by callers.  ``finalize()``
    calculates the lower/upper world-space limits and total physical volume
    needed by the generator.
    """
    map_origin: np.ndarray = field(default_factory=lambda: np.array([-20.0, -20.0, 0.0]))
    map_size: np.ndarray = field(default_factory=lambda: np.array([40.0, 40.0, 5.0]))
    min_range: np.ndarray = field(default_factory=lambda: np.zeros(3))
    max_range: np.ndarray = field(default_factory=lambda: np.zeros(3))
    map_volume: float = 0.0

    def finalize(self):
        """Convert parameters to arrays and rebuild values derived from bounds.

        ``max_range`` is intentionally an exclusive upper bound throughout
        this module.  A point at exactly ``origin + size`` does not belong to
        the final voxel.
        """
        # Normalisation ensures vector arithmetic works even if callers
        # initially provide Python lists or integer arrays.
        self.map_origin = np.asarray(self.map_origin, dtype=float)
        self.map_size = np.asarray(self.map_size, dtype=float)
        self.min_range = self.map_origin.copy()
        self.max_range = self.map_origin + self.map_size
        self.map_volume = float(np.prod(self.map_size))


@dataclass
class GridMapParams:
    """Configure voxel resolution, thresholds, and values derived at startup.

    ``map_grid_size`` and ``map_grid_size_ytz`` are populated by
    ``GridMap.init_map()``.  The latter is the number of Y-Z cells in one X
    slice and is used while flattening a three-dimensional index.
    """
    basic_mp: BasicMapParams = field(default_factory=BasicMapParams)
    resolution: float = 0.1
    inv_resolution: float = 10.0
    map_grid_size: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=int))
    map_grid_size_ytz: int = 0
    clamp_min_log: float = 0.01
    clamp_max_log: float = 0.99
    min_thrd: float = 0.80


class GridMap:
    """Manage occupancy storage plus convenient obstacle-point representations.

    ``occupancy`` is the compact one-dimensional voxel buffer. ``obs_pts``
    records points as they are inserted; this is useful during generation and
    is distinct from ``publish_map()``, which reconstructs every occupied
    voxel centre from the buffer.
    """
    def __init__(self):
        """Create an empty map that must later be configured with ``init_map``.

        The seeded generator makes random obstacle placement reproducible when
        the same seed is supplied by the structured-map generator.
        """
        self.mp = GridMapParams()
        self.occupancy = np.zeros(0, dtype=float)
        self.obs_pts: list[np.ndarray] = []
        self._rng = np.random.default_rng(0)
        self._rand_low = np.zeros(3)
        self._rand_high = np.ones(3)

    def init_map(self, mpa: GridMapParams):
        """Initialise the voxel buffer from map parameters and clear occupancy.

        The physical size is rounded upward to a whole number of cells.  This
        guarantees that the requested map extents are covered at the selected
        resolution.
        """
        self.mp = mpa
        self.mp.basic_mp.finalize()
        self.mp.inv_resolution = 1.0 / self.mp.resolution
        # Determine the count of resolution-sized voxels along X, Y, and Z.
        self.mp.map_grid_size = np.ceil(
            self.mp.basic_mp.map_size / self.mp.resolution
        ).astype(int)
        self.mp.map_grid_size_ytz = int(
            self.mp.map_grid_size[1] * self.mp.map_grid_size[2]
        )
        # Allocate one flat value per X-Y-Z combination; initially all are free.
        buffer_size = int(self.mp.map_grid_size[0] * self.mp.map_grid_size_ytz)
        self.occupancy = np.full(buffer_size, self.mp.clamp_min_log, dtype=float)
        self.obs_pts = []
        self.print_map_info()

    def print_map_info(self):
        """Print the active map geometry for command-line workflows."""
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
        """Configure reproducible uniform sampling over the map's world bounds.

        Sampled positions are snapped to centres by ``get_uni_rand_pos()`` so
        generated shape centres align with the voxel grid.
        """
        self._rng = rng
        self._rand_low = self.mp.basic_mp.min_range.copy()
        self._rand_high = self.mp.basic_mp.max_range.copy()

    def clear_all_occ(self):
        """Mark every voxel free and discard the cached obstacle points."""
        self.occupancy.fill(self.mp.clamp_min_log)
        self.obs_pts.clear()

    def pos_to_index(self, pos: np.ndarray) -> np.ndarray:
        """Convert a world-space position to the index of its containing voxel.

        The floor operation is important: it maps every location in the
        half-open interval of a voxel to the same integer cell.
        """
        return np.floor(
            (np.asarray(pos, dtype=float) - self.mp.basic_mp.map_origin)
            * self.mp.inv_resolution
        ).astype(int)

    def index_to_pos(self, idx: np.ndarray) -> np.ndarray:
        """Return the world-space centre of an integer voxel index.

        Adding 0.5 selects the centre rather than the lower voxel corner.
        """
        return self.mp.basic_mp.map_origin + (np.asarray(idx, dtype=float) + 0.5) * self.mp.resolution

    def get_buffer_cnt(self, idx: np.ndarray) -> int:
        """Flatten an ``[x, y, z]`` index using the buffer's X-major layout.

        One X increment skips an entire Y-Z plane; one Y increment skips a Z
        row; Z is the fastest-varying coordinate.
        """
        return int(
            idx[0] * self.mp.map_grid_size_ytz
            + idx[1] * self.mp.map_grid_size[2]
            + idx[2]
        )

    def is_in_map_idx(self, idx: np.ndarray) -> bool:
        """Return whether an integer voxel index lies within grid storage."""
        return bool(
            0 <= idx[0] < self.mp.map_grid_size[0]
            and 0 <= idx[1] < self.mp.map_grid_size[1]
            and 0 <= idx[2] < self.mp.map_grid_size[2]
        )

    def is_in_map(self, pos: np.ndarray) -> bool:
        """Return whether a world-space position lies in the half-open map."""
        pos = np.asarray(pos, dtype=float)
        mn, mx = self.mp.basic_mp.min_range, self.mp.basic_mp.max_range
        return bool(
            mn[0] <= pos[0] < mx[0]
            and mn[1] <= pos[1] < mx[1]
            and mn[2] <= pos[2] < mx[2]
        )

    def is_occ(self, pos: np.ndarray) -> int:
        """Classify a world-space point as occupied, free, or out of bounds.

        Returns:
            1: the containing voxel exceeds the occupancy threshold.
            0: the containing voxel is in bounds and free.
            -1: the position has no containing voxel in this map.
        """
        idx = self.pos_to_index(pos)
        if not self.is_in_map_idx(idx):
            return -1
        return 1 if self.occupancy[self.get_buffer_cnt(idx)] > self.mp.min_thrd else 0

    def is_occ_idx(self, idx: np.ndarray) -> int:
        """Classify an integer voxel index using the same codes as ``is_occ``."""
        if not self.is_in_map_idx(idx):
            return -1
        return 1 if self.occupancy[self.get_buffer_cnt(idx)] > self.mp.min_thrd else 0

    def set_occ(self, pos: np.ndarray):
        """Mark the voxel containing a world-space point as occupied.

        The original point—not the voxel centre—is also retained in ``obs_pts``
        so callers can access the points that were supplied during rasterising.
        """
        idx = self.pos_to_index(pos)
        if not self.is_in_map_idx(idx):
            return
        self.occupancy[self.get_buffer_cnt(idx)] = self.mp.clamp_max_log
        self.obs_pts.append(np.asarray(pos, dtype=float))

    def set_occ_idx(self, idx: np.ndarray):
        """Mark an integer voxel occupied and record its world-space centre."""
        if not self.is_in_map_idx(idx):
            return
        pos = self.index_to_pos(idx)
        self.occupancy[self.get_buffer_cnt(idx)] = self.mp.clamp_max_log
        self.obs_pts.append(pos)

    def get_grid_val(self, p: float) -> float:
        """Snap one coordinate to the centre of its resolution-sized voxel."""
        return np.floor(p * self.mp.inv_resolution) * self.mp.resolution + self.mp.resolution / 2.0

    def get_grid_center_pos(self, pos: np.ndarray) -> np.ndarray:
        """Snap all coordinates in ``pos`` to their voxel centres."""
        return np.array(
            [self.get_grid_val(pos[0]), self.get_grid_val(pos[1]), self.get_grid_val(pos[2])]
        )

    def get_uni_rand_pos(self) -> np.ndarray:
        """Sample a random in-bounds position and return its voxel centre."""
        pos = self._rng.uniform(self._rand_low, self._rand_high)
        return self.get_grid_center_pos(pos)

    def fill_map(self, cloud: np.ndarray, inflated: float = 0.0):
        """Fill occupancy from an ``(N, 3)`` cloud, optionally expanding it.

        Inflation samples an axis-aligned cube of neighbouring points around
        each cloud point at the map resolution.  It is deliberately a voxel
        approximation, not a Euclidean-radius sphere.
        """
        if cloud.size == 0:
            return
        if inflated > 0.0:
            # Convert the requested physical margin into whole voxel steps.
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
        """Return centres of every occupied voxel as an ``(N, 3)`` cloud.

        Reshaping recovers the logical X-Y-Z grid from flattened storage;
        ``argwhere`` then returns every index that crosses the threshold.
        """
        occupied = np.argwhere(self.occupancy.reshape(self.mp.map_grid_size) > self.mp.min_thrd)
        if occupied.size == 0:
            return np.zeros((0, 3))
        return np.array([self.index_to_pos(idx) for idx in occupied])

    def publish_2d_map(self) -> np.ndarray:
        """Project occupied X-Y columns onto the ``z = 0`` planning plane.

        Only the lowest occupied Z index in each X-Y column is needed to mark
        that column obstructed in the two-dimensional representation.
        """
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
        """Return points recorded while voxels were marked occupied."""
        if not self.obs_pts:
            return np.zeros((0, 3))
        return np.vstack(self.obs_pts)
