"""
===============================================================================
3D UAV ROUTING ENVIRONMENT — STRUCTURED MAP GENERATOR
===============================================================================

PURPOSE
-------
This file creates synthetic obstacle environments for UAV route-planning
experiments.  It randomly places cylinders, circular gates, rectangular
gates, ellipsoids, and convex polyhedra, then samples their solid regions at
the occupancy-grid resolution.  It also provides YAML-selectable corridor,
random-land, mountain, urban-block, and mixed environment presets.

HOW A MAP IS BUILT
------------------
1. Each shape family receives a target number of occupied voxels from its
   configured volume ratio.
2. A random centre, dimensions, and orientation are sampled for one shape.
3. Candidate voxel centres around that shape are tested with ``is_inside()``.
4. Accepted candidates are recorded in both ``GridMap`` and ``cloud_map``.
5. The source shape is kept in ``GeoMap`` for inspection or dynamic movement.

DYNAMIC MODE
------------
Dynamic shapes receive horizontal velocities.  ``dyn_generate()`` translates
their already-sampled point clouds and reverses velocity at fixed bounds.  It
does not regenerate their geometry.

ENVIRONMENT PRESETS
-------------------
``random_land`` randomly places a modest number of larger primitive obstacles.
``structured`` creates randomly arranged buildings and pillars spanning the
full map height. ``corridor`` fills the whole map with a connected 2.5D maze,
``mountains`` sums Gaussian hills, and ``blocks`` creates buildings.
Every preset keeps a clearance-bounded 6-connected free path from the origin
voxel ``(0, 0, 0)`` to the opposite corner so a drone planner has at least one
feasible route. Presets are selected through the ``environment`` YAML section.
===============================================================================
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from .geo_map import GeoMap
from .grid_map import GridMap, GridMapParams
from .shapes import CircleGate, Cylinder, Ellipsoid, Polyhedron, RectGate


@dataclass
class MapGenParams:
    """Configure total obstacle density, type weights, dimensions, and noise.

    ``occupancy_ratio`` sets the total occupied-voxel target. The ``*_ratio``
    values are relative weights that divide this target between primitive
    obstacle types. ``w1`` through ``w3`` define the distributions used for
    obstacle widths and radii; their historical names are retained to match
    the source implementation and existing YAML files.
    """
    occupancy_ratio: float = 0.5
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
    """Build static or moving shape-based environments and their voxel clouds.

    The class keeps three related representations in sync while generating:
    the flat grid for collision queries, a global cloud for export, and shape
    objects for type-aware/dynamic operations.
    """
    def __init__(self):
        """Create empty maps, default generation parameters, and seeded RNG."""
        self.cloud_map = np.zeros((0, 3), dtype=float)
        self.grid_map = GridMap()
        self.geo_map = GeoMap()
        self.mpa = GridMapParams()
        self.mgpa = MapGenParams()
        self.terrain_height_field: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.rng = np.random.default_rng(1)
        self.vel_h = np.array([0.1, 0.1, 0.0])
        self.cyl_list: list[Cylinder] = []
        self.cir_list: list[CircleGate] = []
        self.gate_list: list[RectGate] = []
        self.ellip_list: list[Ellipsoid] = []
        self.poly_list: list[Polyhedron] = []

    def init_params(self, mpa: GridMapParams):
        """Finalise map bounds and initialise the grid used for rasterisation."""
        mpa.basic_mp.finalize()
        self.grid_map.init_map(mpa)
        self.mpa = mpa
        self.terrain_height_field = None

    def change_res(self, res: float):
        """Reinitialise the grid using the current map bounds and new resolution."""
        self.mpa.resolution = res
        self.grid_map.init_map(self.mpa)
        self.terrain_height_field = None

    def get_pc(self) -> np.ndarray:
        """Return a copy of the generated 3D occupied-point cloud."""
        return self.cloud_map.copy()

    def get_pc_2d(self) -> np.ndarray:
        """Return the occupancy grid projected onto the ``z = 0`` plane."""
        return self.grid_map.publish_2d_map()

    def clear(self):
        """Clear grid, cloud, and registered shape collections before a new map.

        The per-family dynamic lists are intentionally not cleared here.  This
        preserves the ported behaviour; ``generate()`` clears them when it
        starts constructing a new set of dynamic obstacles.
        """
        self.cloud_map = np.zeros((0, 3), dtype=float)
        self.grid_map.clear_all_occ()
        self.geo_map.clear_all()
        self.terrain_height_field = None

    def _append_points(self, pts: list[np.ndarray]):
        """Append newly accepted obstacle points to the aggregate cloud.

        A batch is concatenated once per shape rather than once per point to
        avoid repeated allocation of increasingly large NumPy arrays.
        """
        if not pts:
            return
        arr = np.asarray(pts, dtype=float)
        self.cloud_map = (
            arr if self.cloud_map.size == 0 else np.vstack([self.cloud_map, arr])
        )

    def update_pts(self, geo_rep, max_points: int | None = None) -> int:
        """Rasterise a solid shape by testing voxels in its bounding region.

        Accepted voxel centres are stored in the grid, global cloud, and the
        shape's own cloud.  The return value is the number of non-noise points
        placed, which is used to meet each obstacle-family density target.
        """
        # The shape reports bounds large enough to cover its local geometry.
        # ``wid`` converts each physical half-extent into voxel-loop counts.
        bound = geo_rep.get_bd()
        cpt = geo_rep.get_center()
        res = self.mpa.resolution
        wid = np.ceil(bound / res).astype(int)

        cur = 0
        total_pts = []
        new_pts = []
        # Scan candidate voxel centres in the shape's axis-aligned bounding
        # box.  ``is_inside`` removes points that are outside the actual
        # curved, faceted, or rotated solid.
        for r in range(-wid[0], wid[0]):
            if max_points is not None and cur >= max_points:
                break
            rx = r * res
            for s in range(-wid[1], wid[1]):
                if max_points is not None and cur >= max_points:
                    break
                sy = s * res
                for t in range(-wid[2], wid[2]):
                    if max_points is not None and cur >= max_points:
                        break
                    ob_pt = cpt + np.array([rx, sy, t * res])
                    # A non-zero status includes occupied cells and cells
                    # outside the map, so neither is added again.
                    if self.grid_map.is_occ(ob_pt) != 0:
                        continue
                    if not geo_rep.is_inside(ob_pt):
                        continue
                    self.grid_map.set_occ(ob_pt)
                    new_pts.append(ob_pt)
                    total_pts.append(ob_pt.copy())
                    geo_rep.cloud.append(ob_pt.copy())
                    cur += 1

        # Optionally add jittered duplicate points after the main geometry is
        # sampled.  Noise never contributes to the ratio target ``cur``.
        if self.mgpa.add_noise and total_pts:
            noise = self.rng.normal(0.0, self.mgpa.w1, size=(len(total_pts), 3))
            for i, ob_pt in enumerate(total_pts):
                if self.rng.random() >= self.mgpa.w1:
                    continue
                noisy = ob_pt + noise[i]
                if self.grid_map.is_occ(noisy) != 1:
                    continue
                self.grid_map.set_occ(noisy)
                new_pts.append(noisy)
                geo_rep.cloud.append(noisy.copy())

        self._append_points(new_pts)
        return cur

    def update_pts_light(self, geo_rep, max_points: int | None = None) -> int:
        """Rasterise a rotated gate by sampling its local thickness and frame.

        This avoids scanning the larger axis-aligned world-space bounds that
        would be needed for a thin, rotated gate.
        """
        rect = geo_rep.get_rect()
        cpt = geo_rep.get_center()
        rot = geo_rep.get_rot()
        res = self.mpa.resolution
        wid_num1 = int(np.ceil(rect[1] / res))
        wid_num2 = int(np.ceil(rect[2] / res))
        width = int(np.ceil(rect[0] / res))
        # Four sign pairs enumerate the Y-Z quadrants of the local gate frame.
        signs = np.array([[1, -1, 1, -1], [1, 1, -1, -1]], dtype=float)

        cur = 0
        total_pts = []
        new_pts = []
        # Sample in local coordinates, then rotate each point into world space.
        # This is much cheaper than scanning empty cells around a thin gate.
        for t in range(-width, width):
            if max_points is not None and cur >= max_points:
                break
            tx = t * res
            for r in range(wid_num1):
                if max_points is not None and cur >= max_points:
                    break
                ry = r * res
                for s in range(wid_num2):
                    if max_points is not None and cur >= max_points:
                        break
                    sz = s * res
                    for i in range(4):
                        if max_points is not None and cur >= max_points:
                            break
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
                if self.grid_map.is_occ(noisy) != 1:
                    continue
                self.grid_map.set_occ(noisy)
                new_pts.append(noisy)
                geo_rep.cloud.append(noisy.copy())

        self._append_points(new_pts)
        return cur

    def traverse_pts(self, geo_reps):
        """Rasterise every shape in an iterable with the general solid sampler."""
        for geo_rep in geo_reps:
            self.update_pts(geo_rep)

    def reset_map(self):
        """Rebuild the aggregate cloud from geometry currently in ``GeoMap``.

        Shapes are traversed in a stable type order.  Existing grid occupancy
        is not cleared here, so duplicate or overlapping candidate points are
        skipped by ``update_pts``.
        """
        self.cloud_map = np.zeros((0, 3), dtype=float)
        self.traverse_pts(self.geo_map.polyhedron)
        self.traverse_pts(self.geo_map.cylinder)
        self.traverse_pts(self.geo_map.ellipsoid)
        self.traverse_pts(self.geo_map.circle_gate)
        self.traverse_pts(self.geo_map.rect_gate)

    def change_ratios(self, seed: float, if_dyn: bool = False, dt: float = 0.1):
        """Resample occupancy ratios from ``seed`` or advance a dynamic map.

        Ratios are capped together before generation to preserve a bounded
        total requested occupancy density.
        """
        if if_dyn:
            self.dyn_generate(dt)
            return
        # Re-seeding means a given seed always produces the same ratios and
        # the same subsequent shape-placement sequence.
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
        # Keep combined density bounded by scaling every family proportionally.
        if total > self.mgpa.w2 * 0.5:
            scale = (self.mgpa.w2 * 0.5) / total
            self.mgpa.cylinder_ratio *= scale
            self.mgpa.circle_ratio *= scale
            self.mgpa.gate_ratio *= scale
            self.mgpa.ellip_ratio *= scale
            self.mgpa.poly_ratio *= scale
        self.generate(False)

    def random_uni_map_gen(self, mgpa: MapGenParams, seed: float, if_dyn: bool = False):
        """Set caller-supplied parameters and generate a reproducible map."""
        self.mgpa = mgpa
        self.rng = np.random.default_rng(int(seed))
        self.generate(if_dyn)

    def _rand_w(self) -> float:
        """Sample a small width-like dimension from the configured interval."""
        return float(self.rng.uniform(self.mgpa.w1, self.mgpa.w2))

    def _rand_radiu(self) -> float:
        """Sample a radius-like dimension (name retained from source port)."""
        return float(self.rng.uniform(self.mgpa.w1, self.mgpa.w3))

    def _rand_h(self) -> float:
        """Sample a height between half and all of the map's Z extent."""
        z = self.mpa.basic_mp.map_size[2]
        return float(self.rng.uniform(0.5 * z, z))

    def _rand_theta(self) -> float:
        """Sample an orientation uniformly over one full rotation."""
        return float(self.rng.uniform(-np.pi, np.pi))

    def _sample_vel(self, dyn_mode: bool) -> np.ndarray:
        """Return zero velocity or a random horizontal dynamic-obstacle velocity."""
        if not dyn_mode:
            return np.zeros(3)
        # Vertical motion is deliberately disabled; vel_h[2] is retained in
        # configuration compatibility but dynamic sampling is planar.
        return np.array(
            [
                self.rng.uniform(-self.vel_h[0], self.vel_h[0]),
                self.rng.uniform(-self.vel_h[1], self.vel_h[1]),
                0.0,
            ]
        )

    def _clear_dynamic_lists(self):
        """Remove dynamic-shape references when replacing the whole environment."""
        self.cyl_list.clear()
        self.cir_list.clear()
        self.gate_list.clear()
        self.ellip_list.clear()
        self.poly_list.clear()

    def _valid_grid_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return in-bounds world-space centres for all X, Y, and Z voxels.

        ``GridMap`` may allocate a final partial cell when a physical size is
        not an exact multiple of resolution.  Filtering with ``is_in_map``
        prevents a terrain preset from placing points in that partial overhang.
        """
        dims = self.mpa.map_grid_size
        xs = np.array(
            [self.grid_map.index_to_pos(np.array([i, 0, 0]))[0] for i in range(dims[0])]
        )
        ys = np.array(
            [self.grid_map.index_to_pos(np.array([0, j, 0]))[1] for j in range(dims[1])]
        )
        zs = np.array(
            [self.grid_map.index_to_pos(np.array([0, 0, k]))[2] for k in range(dims[2])]
        )
        mx = self.mpa.basic_mp.max_range
        return xs[xs < mx[0]], ys[ys < mx[1]], zs[zs < mx[2]]

    def _replace_with_cloud(self, points: np.ndarray):
        """Replace map contents with canonical occupied voxels from ``points``.

        Terrain presets generate point clouds directly rather than parametric
        ``shapes`` objects.  Feeding them through ``fill_map`` applies the same
        bounds and duplicate handling as every other map source, then
        ``publish_map`` records one centre for each accepted occupied voxel.
        """
        self.grid_map.clear_all_occ()
        self.geo_map.clear_all()
        self._clear_dynamic_lists()
        self.grid_map.fill_map(np.asarray(points, dtype=float))
        self.cloud_map = self.grid_map.publish_map()

    def _occupied_bool(self) -> np.ndarray:
        """Return the in-bounds occupancy mask shaped ``(nx, ny, nz)``."""
        xs, ys, zs = self._valid_grid_axes()
        grid = self.grid_map.occupancy.reshape(self.mpa.map_grid_size) > self.mpa.min_thrd
        return grid[: len(xs), : len(ys), : len(zs)].copy()

    def _cloud_from_occupied(self, occupied: np.ndarray) -> np.ndarray:
        """Convert an occupancy mask into world-space voxel-centre points."""
        xs, ys, zs = self._valid_grid_axes()
        indices = np.argwhere(occupied)
        if indices.size == 0:
            return np.zeros((0, 3), dtype=float)
        return np.column_stack((xs[indices[:, 0]], ys[indices[:, 1]], zs[indices[:, 2]]))

    def _planning_endpoints(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Return start ``(0,0,0)`` and opposite-corner voxel indices."""
        xs, ys, zs = self._valid_grid_axes()
        if not len(xs) or not len(ys) or not len(zs):
            return (0, 0, 0), (0, 0, 0)
        return (0, 0, 0), (len(xs) - 1, len(ys) - 1, len(zs) - 1)

    def _planning_clearance(self) -> int:
        """Clearance in voxels, enough for a small drone around the reserved path."""
        return max(1, int(np.ceil(0.4 / max(self.mpa.resolution, 1e-9))))

    def _dilate_mask(self, mask: np.ndarray, radius: int) -> np.ndarray:
        """Expand a Boolean mask by Chebyshev radius ``radius`` voxels."""
        if radius <= 0:
            return mask.copy()
        footprint = np.ones((2 * radius + 1,) * mask.ndim, dtype=bool)
        return binary_dilation(mask, structure=footprint)

    def _free_path(self, occupied: np.ndarray, start, goal, clearance: int):
        """Return a 6-connected voxel path if one exists with the given clearance."""
        blocked = self._dilate_mask(occupied, clearance)
        blocked[start] = False
        blocked[goal] = False
        shape = occupied.shape
        if any(c < 0 or c >= shape[i] for i, c in enumerate(start)):
            return None
        if any(c < 0 or c >= shape[i] for i, c in enumerate(goal)):
            return None
        if blocked[start] or blocked[goal]:
            return None
        parent = {start: None}
        queue = deque([start])
        goal_t = tuple(goal)
        while queue:
            cur = queue.popleft()
            if cur == goal_t:
                path = []
                node = cur
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            x, y, z = cur
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                nxt = (x + dx, y + dy, z + dz)
                if nxt in parent:
                    continue
                if not (0 <= nxt[0] < shape[0] and 0 <= nxt[1] < shape[1] and 0 <= nxt[2] < shape[2]):
                    continue
                if blocked[nxt]:
                    continue
                parent[nxt] = cur
                queue.append(nxt)
        return None

    def _manhattan_segment(self, a, b) -> list[tuple[int, int, int]]:
        """Connect two voxels with axis-aligned steps."""
        x, y, z = a
        voxels = [(x, y, z)]
        while (x, y, z) != tuple(b):
            if x != b[0]:
                x += 1 if b[0] > x else -1
            elif y != b[1]:
                y += 1 if b[1] > y else -1
            else:
                z += 1 if b[2] > z else -1
            voxels.append((x, y, z))
        return voxels

    def _winding_path(self, start, goal, shape) -> list[tuple[int, int, int]]:
        """Build a reproducible winding polyline from start to the opposite corner."""
        nx, ny, nz = shape
        waypoints = [tuple(start)]
        n_wp = 3 if min(shape) >= 6 else 1
        for _ in range(n_wp):
            waypoints.append(
                (
                    int(self.rng.integers(0, nx)),
                    int(self.rng.integers(0, ny)),
                    int(self.rng.integers(0, nz)),
                )
            )
        waypoints.append(tuple(goal))
        voxels = []
        for a, b in zip(waypoints, waypoints[1:]):
            voxels.extend(self._manhattan_segment(a, b)[1 if voxels else 0 :])
        return voxels

    def _carve_voxels(self, occupied: np.ndarray, voxels, clearance: int) -> np.ndarray:
        """Clear a Chebyshev ball around each path voxel."""
        result = occupied.copy()
        nx, ny, nz = result.shape
        r = max(int(clearance), 0)
        for x, y, z in voxels:
            x0, x1 = max(0, x - r), min(nx, x + r + 1)
            y0, y1 = max(0, y - r), min(ny, y + r + 1)
            z0, z1 = max(0, z - r), min(nz, z + r + 1)
            result[x0:x1, y0:y1, z0:z1] = False
        return result

    def _protect_path(self, occupied: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Guarantee a free path from origin to the opposite corner.

        Existing maze-like free space is reused when it already connects the
        endpoints. Otherwise a winding tunnel is carved. The returned protected
        mask must stay free while occupancy is later fitted to the target.
        """
        start, goal = self._planning_endpoints()
        clearance = self._planning_clearance()
        result = occupied.copy()
        result[start] = False
        result[goal] = False
        path = self._free_path(result, start, goal, clearance)
        if path is None:
            carved = self._winding_path(start, goal, result.shape)
            result = self._carve_voxels(result, carved, clearance)
            path = carved
        protected = np.zeros_like(result)
        for x, y, z in path:
            protected[x, y, z] = True
        protected = self._dilate_mask(protected, clearance)
        protected[start] = True
        protected[goal] = True
        result[protected] = False
        return result, protected

    def _apply_occupancy_target(
        self,
        occupied: np.ndarray,
        target: int,
        protected_free: np.ndarray | None = None,
    ) -> np.ndarray:
        """Grow or peel solid obstacles until ``target`` voxels are occupied.

        Occupancy is adjusted only on the obstacle frontier so solids stay
        clustered. Protected free voxels — the reserved start-goal path —
        are never filled. Random independent voxel flips are avoided because
        they create a crowded, speckled look even at modest occupancy.
        """
        result = occupied.copy()
        protected = (
            np.zeros_like(result)
            if protected_free is None
            else np.asarray(protected_free, dtype=bool)
        )
        result[protected] = False
        capacity = int(result.size - protected.sum())
        target = min(max(int(target), 0), capacity)
        neighbourhood = np.ones((3, 3, 3), dtype=bool)

        while int(result.sum()) > target:
            surface = result & binary_dilation(~result, structure=neighbourhood)
            candidates = np.argwhere(surface if surface.any() else result)
            remaining = int(result.sum()) - target
            count = min(remaining, len(candidates))
            if count == 0:
                break
            selected = self.rng.choice(len(candidates), size=count, replace=False)
            result[tuple(candidates[selected].T)] = False
            result[protected] = False

        while int(result.sum()) < target:
            frontier = binary_dilation(result, structure=neighbourhood) & ~result & ~protected
            if not frontier.any():
                frontier = ~result & ~protected
            candidates = np.argwhere(frontier)
            remaining = target - int(result.sum())
            count = min(remaining, len(candidates))
            if count == 0:
                break
            selected = self.rng.choice(len(candidates), size=count, replace=False)
            result[tuple(candidates[selected].T)] = True
        return result

    def _finalize_for_planning(self, target: int, env_type: str):
        """Carve a start-goal route, restore the occupancy target, and sync maps."""
        occupied, protected = self._protect_path(self._occupied_bool())
        if env_type == "structured":
            occupied = self._refill_full_columns(occupied, target, protected)
        else:
            occupied = self._apply_occupancy_target(occupied, target, protected)
        points = self._cloud_from_occupied(occupied)
        self.grid_map.clear_all_occ()
        self.grid_map.fill_map(points)
        self.cloud_map = self.grid_map.publish_map()
        for shapes in (
            self.geo_map.cylinder,
            self.geo_map.circle_gate,
            self.geo_map.rect_gate,
            self.geo_map.ellipsoid,
            self.geo_map.polyhedron,
        ):
            for shape in shapes:
                shape.cloud = [point for point in shape.cloud if self.grid_map.is_occ(point) == 1]

    def _refill_full_columns(self, occupied: np.ndarray, target: int, protected: np.ndarray) -> np.ndarray:
        """Restore structured occupancy using whole-height pillars only."""
        result = occupied.copy()
        result[protected] = False
        nz = result.shape[2]
        columns = result.any(axis=2)
        result = np.repeat(columns[:, :, None], nz, axis=2)
        result[protected] = False
        current = int(result.sum())
        target = min(max(int(target), 0), int(result.size - protected.sum()))
        blocked_xy = protected.any(axis=2)
        if current < target:
            free_xy = np.argwhere(~columns & ~blocked_xy)
            need = (target - current + nz - 1) // nz
            if len(free_xy) and need:
                take = min(need, len(free_xy))
                pick = self.rng.choice(len(free_xy), size=take, replace=False)
                for i, j in free_xy[pick]:
                    result[i, j, :] = True
                result[protected] = False
        if int(result.sum()) > target:
            result = self._apply_occupancy_target(result, target, protected)
        elif int(result.sum()) < target:
            result = self._apply_occupancy_target(result, target, protected)
        return result

    def _planning_radius(self) -> float:
        """Sample a planning-friendly obstacle radius from the map size.

        Tiny historical ``w1``/``w2`` values created thousands of stick-like
        primitives at 20% occupancy. Scaling with map extent yields fewer,
        well-separated solids that still meet the voxel target.
        """
        xy = float(np.min(self.mpa.basic_mp.map_size[:2]))
        res = self.mpa.resolution
        maximum = max(res, 0.22 * xy)
        lo = min(max(3 * res, 0.05 * xy), maximum)
        hi = min(max(lo + res, 0.12 * xy), maximum)
        return float(self.rng.uniform(lo, hi)) if hi > lo else lo

    def _planning_height(self) -> float:
        """Sample an obstacle height that leaves some vertical fly-over room."""
        z = float(self.mpa.basic_mp.map_size[2])
        res = self.mpa.resolution
        lo = max(2 * res, 0.25 * z)
        hi = max(lo + res, 0.75 * z)
        return float(self.rng.uniform(lo, hi))

    def _fit_occupancy(self, points: np.ndarray, target: int) -> np.ndarray:
        """Adjust a generated cloud to an exact count of occupied map voxels."""
        xs, ys, zs = self._valid_grid_axes()
        shape = (len(xs), len(ys), len(zs))
        occupied = np.zeros(shape, dtype=bool)
        if points.size:
            indices = np.floor(
                (np.asarray(points, dtype=float) - self.mpa.basic_mp.map_origin)
                / self.mpa.resolution
            ).astype(int)
            valid = np.all((indices >= 0) & (indices < np.asarray(shape)), axis=1)
            occupied[tuple(indices[valid].T)] = True
        occupied = self._apply_occupancy_target(occupied, target)
        return self._cloud_from_occupied(occupied)

    def _height_field_cloud(self, heights: np.ndarray) -> np.ndarray:
        """Fill all voxels from map ground up to a height field.

        ``heights[i, j]`` is an absolute world-space Z height for the terrain
        surface at the X-Y voxel centre.  Filling below that surface makes hills
        and landforms solid obstacles rather than a single thin visual shell.
        """
        xs, ys, zs = self._valid_grid_axes()
        self.terrain_height_field = (
            xs.copy(),
            ys.copy(),
            np.asarray(heights, dtype=float).copy(),
        )
        points = []
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                # Include every valid Z centre no higher than this local terrain.
                for z in zs[zs <= heights[i, j]]:
                    points.append([x, y, z])
        return np.asarray(points, dtype=float) if points else np.zeros((0, 3))

    def _fit_height_field_cloud(self, heights: np.ndarray, target: int) -> np.ndarray:
        """Offset a height field to an exact voxel count without breaking columns."""
        xs, ys, zs = self._valid_grid_axes()
        heights = np.asarray(heights, dtype=float)
        if heights.shape != (len(xs), len(ys)):
            raise ValueError("Height field dimensions must match the map's X-Y grid.")
        if not len(zs):
            return np.zeros((0, 3), dtype=float)

        capacity = len(xs) * len(ys) * len(zs)
        target = min(max(int(target), 0), capacity)
        if target == 0:
            fitted = np.minimum(heights, np.nextafter(zs[0], -np.inf))
        elif target == capacity:
            fitted = np.full_like(heights, zs[-1])
        else:
            # Each threshold is the vertical offset at which one voxel centre
            # enters its column. Resolving ties per column keeps the exact
            # target while retaining the smooth field everywhere else.
            thresholds = zs[None, None, :] - heights[:, :, None]
            offset = np.partition(thresholds.ravel(), target - 1)[target - 1]
            below = thresholds < offset
            tied = thresholds == offset
            column_counts = below.sum(axis=2)
            tie_count = target - int(column_counts.sum())
            tie_columns = np.argwhere(tied)[:tie_count]
            column_counts[tie_columns[:, 0], tie_columns[:, 1]] += 1

            fitted = heights + offset
            nonempty = column_counts > 0
            fitted[nonempty] = np.maximum(
                    fitted[nonempty], zs[column_counts[nonempty] - 1]
            )
            partial = nonempty & (column_counts < len(zs))
            fitted[partial] = np.minimum(
                    fitted[partial],
                    np.nextafter(zs[column_counts[partial]], -np.inf),
            )
            fitted[column_counts == len(zs)] = zs[-1]

        return self._height_field_cloud(fitted)

    def _smooth_field(self, field: np.ndarray, passes: int) -> np.ndarray:
        """Smooth a random height field with repeated 3-by-3 neighbour averages."""
        result = field.copy()
        for _ in range(max(int(passes), 0)):
            # Edge padding keeps the field dimensions fixed at map boundaries.
            pad = np.pad(result, 1, mode="edge")
            result = sum(
                pad[dy : dy + result.shape[0], dx : dx + result.shape[1]]
                for dy in range(3)
                for dx in range(3)
            ) / 9.0
        return result

    def _random_land_cloud(self, options: dict, target: int | None = None) -> np.ndarray:
        """Generate solid terrain from a smoothed, seeded random height field.

        Options:
            base_height: Minimum terrain height above the map's Z origin.
            height_range: Added range between the lowest and highest terrain.
            smoothing_passes: More passes produce broader, gentler landforms.
        """
        xs, ys, _ = self._valid_grid_axes()
        base = float(options.get("base_height", 0.0))
        height_range = float(options.get("height_range", 1.0))
        raw = self.rng.normal(size=(len(xs), len(ys)))
        smooth = self._smooth_field(raw, int(options.get("smoothing_passes", 6)))
        span = float(smooth.max() - smooth.min())
        normalised = (smooth - smooth.min()) / span if span > 1e-12 else np.zeros_like(smooth)
        z0 = self.mpa.basic_mp.map_origin[2]
        z_max = self.mpa.basic_mp.max_range[2] - 0.5 * self.mpa.resolution
        heights = np.clip(z0 + base + height_range * normalised, z0, z_max)
        if target is not None:
            return self._fit_height_field_cloud(heights, target)
        return self._height_field_cloud(heights)

    def _mountain_cloud(self, options: dict, target: int | None = None) -> np.ndarray:
        """Generate separate, solid Gaussian mountain obstacles.

        Each hill has a bounded footprint so the valleys remain free space.
        Alternating smaller and larger peaks creates useful scale variation.
        When supplied, ``target`` adjusts only the mountain voxels; it never
        creates a ground layer across the map.
        """
        xs, ys, zs = self._valid_grid_axes()
        x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
        origin = self.mpa.basic_mp.map_origin
        maximum = self.mpa.basic_mp.max_range
        empty_height = np.nextafter(zs[0], -np.inf) if len(zs) else origin[2]
        heights = np.full_like(x_grid, empty_height)
        count = max(int(options.get("mountain_count", 5)), 1)
        peak_max = float(options.get("peak_height", 2.0))
        spread_min = float(options.get("spread_min", 1.5))
        spread_max = float(options.get("spread_max", 4.0))
        spread_mid = (spread_min + spread_max) / 2.0
        base_height = max(float(options.get("base_height", 0.0)), 0.0)
        for index in range(count):
            cx = self.rng.uniform(origin[0], maximum[0])
            cy = self.rng.uniform(origin[1], maximum[1])
            if index % 2 == 0:
                spread = self.rng.uniform(spread_mid, max(spread_mid, spread_max))
                peak = self.rng.uniform(0.75 * peak_max, peak_max)
            else:
                spread = self.rng.uniform(spread_min, max(spread_min, spread_mid))
                peak = self.rng.uniform(0.35 * peak_max, 0.65 * peak_max)
            profile = peak * np.exp(
                -((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * spread**2)
            )
            # Cut off the Gaussian tails so valleys do not become a thin,
            # map-wide floor layer.
            footprint = profile >= max(0.5 * self.mpa.resolution, 0.025 * peak)
            mountain_heights = origin[2] + base_height + profile
            heights[footprint] = np.maximum(heights[footprint], mountain_heights[footprint])

        z_max = maximum[2] - 0.5 * self.mpa.resolution
        heights = np.clip(heights, empty_height, z_max)
        occupied = zs[None, None, :] <= heights[:, :, None]
        if target is not None:
            occupied = self._apply_occupancy_target(occupied, target)
        self.terrain_height_field = None
        return self._cloud_from_occupied(occupied)

    def _blocks_cloud(self, options: dict) -> np.ndarray:
        """Generate solid rectangular buildings separated by open streets.

        Building positions, footprint dimensions, and heights are sampled in
        voxel units.  Overlap is allowed and simply joins adjacent buildings,
        yielding varied dense urban environments.
        """
        xs, ys, zs = self._valid_grid_axes()
        occupied = np.zeros((len(xs), len(ys), len(zs)), dtype=bool)
        res = self.mpa.resolution
        count = max(int(options.get("building_count", 20)), 1)
        min_width = max(1, int(np.ceil(float(options.get("min_width", 1.0)) / res)))
        max_width = max(min_width, int(np.ceil(float(options.get("max_width", 3.0)) / res)))
        min_height = max(1, int(np.ceil(float(options.get("min_height", 0.8)) / res)))
        max_height = max(min_height, int(np.ceil(float(options.get("max_height", 2.5)) / res)))
        for _ in range(count):
            wx = int(self.rng.integers(min_width, max_width + 1))
            wy = int(self.rng.integers(min_width, max_width + 1))
            wz = int(self.rng.integers(min_height, max_height + 1))
            # Limit starts so the sampled block is clipped only by map bounds.
            x0 = int(self.rng.integers(0, max(len(xs) - wx + 1, 1)))
            y0 = int(self.rng.integers(0, max(len(ys) - wy + 1, 1)))
            occupied[x0 : x0 + wx, y0 : y0 + wy, :wz] = True
        idx = np.argwhere(occupied)
        return np.array([[xs[i], ys[j], zs[k]] for i, j, k in idx], dtype=float)

    def _corridor_cloud(self, options: dict) -> np.ndarray:
        """Fill the whole map with a connected 2.5D maze of passages and walls.

        Passages start at the origin corner and reach the opposite XY corner so
        a drone can traverse the maze instead of a single straight corridor.
        Wall height defaults to the full map so the route cannot skip the maze
        by flying over it. Extra loops keep the maze from being a unique path.
        """
        xs, ys, zs = self._valid_grid_axes()
        nx, ny, nz = len(xs), len(ys), len(zs)
        if min(nx, ny, nz) == 0:
            return np.zeros((0, 3), dtype=float)
        res = self.mpa.resolution
        passage = max(2, int(np.round(float(options.get("opening_width", 2.0)) / res)))
        if "width" in options:
            passage = max(passage, int(np.round(float(options["width"]) / res / 4.0)))
        ratio = float(np.clip(self.mgpa.occupancy_ratio, 0.0, 1.0))
        default_wall = max(1, int(np.ceil(ratio * passage / max(1.0 - ratio, 1e-9))))
        wall = max(
            1,
            int(np.round(float(options.get("wall_thickness", default_wall * res)) / res)),
        )
        wall = min(wall, max(1, min(nx, ny) // 6))
        while 2 * passage + wall > min(nx, ny) and passage > 2:
            passage -= 1
        while 2 * passage + wall > min(nx, ny) and wall > 1:
            wall -= 1
        period = passage + wall
        ncx = max(2, (nx + wall) // period)
        ncy = max(2, (ny + wall) // period)
        while ncx > 2 and ncx * period - wall > nx:
            ncx -= 1
        while ncy > 2 and ncy * period - wall > ny:
            ncy -= 1

        visited = np.zeros((ncx, ncy), dtype=bool)
        east_open = np.zeros((ncx, ncy), dtype=bool)
        north_open = np.zeros((ncx, ncy), dtype=bool)
        stack = [(0, 0)]
        visited[0, 0] = True
        while stack:
            i, j = stack[-1]
            neighbors = []
            if i > 0 and not visited[i - 1, j]:
                neighbors.append((i - 1, j, "w"))
            if i < ncx - 1 and not visited[i + 1, j]:
                neighbors.append((i + 1, j, "e"))
            if j > 0 and not visited[i, j - 1]:
                neighbors.append((i, j - 1, "s"))
            if j < ncy - 1 and not visited[i, j + 1]:
                neighbors.append((i, j + 1, "n"))
            if not neighbors:
                stack.pop()
                continue
            ni, nj, direction = neighbors[int(self.rng.integers(len(neighbors)))]
            visited[ni, nj] = True
            if direction == "e":
                east_open[i, j] = True
            elif direction == "w":
                east_open[ni, nj] = True
            elif direction == "n":
                north_open[i, j] = True
            else:
                north_open[ni, nj] = True
            stack.append((ni, nj))

        extra = int(options.get("extra_openings", max(1, ncx * ncy // 6)))
        for _ in range(max(extra, 0)):
            i = int(self.rng.integers(0, ncx))
            j = int(self.rng.integers(0, ncy))
            if self.rng.random() < 0.5 and i < ncx - 1:
                east_open[i, j] = True
            elif j < ncy - 1:
                north_open[i, j] = True

        def cell_x_range(i: int) -> tuple[int, int]:
            start = i * period
            stop = nx if i == ncx - 1 else min(start + passage, nx)
            return start, max(stop, start + 1)

        def cell_y_range(j: int) -> tuple[int, int]:
            start = j * period
            stop = ny if j == ncy - 1 else min(start + passage, ny)
            return start, max(stop, start + 1)

        occupied = np.ones((nx, ny, nz), dtype=bool)
        for i in range(ncx):
            x0, x1 = cell_x_range(i)
            for j in range(ncy):
                y0, y1 = cell_y_range(j)
                occupied[x0:x1, y0:y1, :] = False
                if i < ncx - 1 and east_open[i, j]:
                    wx0 = x1
                    wx1 = cell_x_range(i + 1)[0]
                    occupied[wx0:wx1, y0:y1, :] = False
                if j < ncy - 1 and north_open[i, j]:
                    wy0 = y1
                    wy1 = cell_y_range(j + 1)[0]
                    occupied[x0:x1, wy0:wy1, :] = False
        occupied[0, 0, :] = False
        occupied[nx - 1, ny - 1, :] = False
        return self._cloud_from_occupied(occupied)

    def generate_environment(self, environment: dict | None, seed: float, if_dyn: bool = False):
        """Generate a YAML-selected environment preset with a reproducible seed.

        Args:
            environment: Mapping containing ``type`` and preset-specific options.
            seed: Seed used for every random choice in this generated map.
            if_dyn: Enable moving primitives for ``structured`` or ``mixed``.

        Valid types are ``structured``, ``corridor``, ``random_land``,
        ``mountains``, ``blocks``, and ``mixed``.  ``random_terrain``,
        ``mountain``, ``urban``, and ``city`` are accepted aliases. Other
        types raise an error rather than silently generating an unintended map.
        """
        options = dict(environment or {})
        env_type = str(options.get("type", "structured")).strip().lower()
        aliases = {
            "random_terrain": "random_land",
            "land": "random_land",
            "mountain": "mountains",
            "urban": "blocks",
            "city": "blocks",
        }
        # Accept common descriptive YAML names while using one implementation
        # for each environment family.
        env_type = aliases.get(env_type, env_type)
        self.rng = np.random.default_rng(int(seed))
        self.terrain_height_field = None
        valid_voxels = int(np.prod([len(axis) for axis in self._valid_grid_axes()]))
        if not np.isfinite(self.mgpa.occupancy_ratio) or not 0 <= self.mgpa.occupancy_ratio <= 1:
            raise ValueError("occupancy_ratio must be between 0 and 1.")
        target_occupancy = int(round(valid_voxels * self.mgpa.occupancy_ratio))
        if env_type == "structured":
            self._replace_with_cloud(self._structured_pillars(options))
        elif env_type == "corridor":
            self._replace_with_cloud(self._corridor_cloud(options))
        elif env_type == "random_land":
            self.clear()
            self._generate_planning_primitives(target_occupancy, if_dyn)
        elif env_type == "mountains":
            self._replace_with_cloud(self._mountain_cloud(options))
        elif env_type == "blocks":
            self._replace_with_cloud(
                self._fit_occupancy(self._blocks_cloud(options), target_occupancy)
            )
        elif env_type == "mixed":
            # Split the target between isolated mountain obstacles and primitives.
            terrain_target = target_occupancy // 2
            terrain = self._mountain_cloud(options, terrain_target)
            self._replace_with_cloud(terrain)
            remaining = target_occupancy - len(terrain)
            if remaining:
                self._generate_planning_primitives(remaining, if_dyn)
        else:
            valid = "structured, corridor, random_land, mountains, blocks, mixed"
            raise ValueError(f"Unknown environment type {env_type!r}. Choose one of: {valid}.")
        self._finalize_for_planning(target_occupancy, env_type)

    def _generate_planning_primitives(self, target: int, dyn_mode: bool):
        """Generate fewer, map-scaled primitive obstacles for planning presets."""
        axes = self._valid_grid_axes()
        valid_voxels = int(np.prod([len(axis) for axis in axes]))
        original_ratio = self.mgpa.occupancy_ratio
        self.mgpa.occupancy_ratio = target / max(valid_voxels, 1)
        try:
            self.generate(dyn_mode, planning_friendly=True)
        finally:
            self.mgpa.occupancy_ratio = original_ratio

    def generate(self, dyn_mode: bool = False, planning_friendly: bool = False):
        """Create each primitive family to its share of total target occupancy.

        Family ratios are normalized as weights, so their absolute sum does
        not change the configured total occupancy. Each shape is rasterised
        immediately, preventing later shapes from claiming occupied cells.
        ``planning_friendly`` scales primitive dimensions to the map so random
        presets use fewer, larger obstacles than the legacy shape generator.
        """
        self.grid_map.set_uni_rand(self.rng)
        axes = self._valid_grid_axes()
        all_grids = int(np.prod([len(axis) for axis in axes]))
        weights = np.asarray(
            [
                self.mgpa.cylinder_ratio,
                self.mgpa.circle_ratio,
                self.mgpa.gate_ratio,
                self.mgpa.ellip_ratio,
                self.mgpa.poly_ratio,
            ],
            dtype=float,
        )
        if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Primitive obstacle ratios must be finite, non-negative, and sum to a positive value.")
        if not np.isfinite(self.mgpa.occupancy_ratio) or not 0 <= self.mgpa.occupancy_ratio <= 1:
            raise ValueError("occupancy_ratio must be between 0 and 1.")
        target = int(round(all_grids * self.mgpa.occupancy_ratio))
        raw_targets = target * weights / weights.sum()
        family_targets = np.floor(raw_targets).astype(int)
        remainder = target - int(family_targets.sum())
        if remainder:
            priority = np.argsort(-(raw_targets - family_targets))
            family_targets[priority[:remainder]] += 1
        cylinder_grids, circle_grids, gate_grids, ellip_grids, poly_grids = family_targets

        self.cyl_list.clear()
        self.cir_list.clear()
        self.gate_list.clear()
        self.ellip_list.clear()
        self.poly_list.clear()

        # Each loop keeps placing shapes until this family reaches its target.
        # Collisions with earlier shapes reduce a shape's contribution because
        # occupied cells are never counted twice.
        cur = 0
        while cur < cylinder_grids:
            cpt = self.grid_map.get_uni_rand_pos()
            h = self._planning_height() if planning_friendly else self._rand_h()
            w = self._planning_radius() if planning_friendly else 0.2 + self._rand_w()
            cyl = Cylinder(cpt, w, h)
            cyl.set_vel(self._sample_vel(dyn_mode))
            cur += self.update_pts(cyl, cylinder_grids - cur)
            self.geo_map.add(cyl)
            if dyn_mode:
                self.cyl_list.append(cyl)
        cylinder_grids = cur

        cur = 0
        while cur < circle_grids:
            cpt = self.grid_map.get_uni_rand_pos()
            theta = self._rand_theta()
            radius = self._planning_radius() if planning_friendly else self._rand_radiu()
            width = 0.1 + 0.2 * radius
            bound = np.array(
                [
                    width,
                    width + (self._planning_radius() if planning_friendly else self._rand_radiu()),
                    width + (self._planning_radius() if planning_friendly else self._rand_radiu()),
                ]
            )
            gate = CircleGate(cpt, bound, theta)
            gate.set_vel(self._sample_vel(dyn_mode))
            cur += self.update_pts_light(gate, circle_grids - cur)
            self.geo_map.add(gate)
            if dyn_mode:
                self.cir_list.append(gate)
        circle_grids = cur

        cur = 0
        while cur < gate_grids:
            cpt = self.grid_map.get_uni_rand_pos()
            theta = self._rand_theta()
            radius = self._planning_radius() if planning_friendly else self._rand_radiu()
            width = 0.1 + 0.2 * radius
            bound = np.array(
                [
                    width,
                    width + (self._planning_radius() if planning_friendly else self._rand_radiu()),
                    width + (self._planning_radius() if planning_friendly else self._rand_radiu()),
                ]
            )
            gate = RectGate(cpt, bound, theta)
            gate.set_vel(self._sample_vel(dyn_mode))
            cur += self.update_pts_light(gate, gate_grids - cur)
            self.geo_map.add(gate)
            if dyn_mode:
                self.gate_list.append(gate)
        gate_grids = cur

        cur = 0
        while cur < ellip_grids:
            cpt = self.grid_map.get_uni_rand_pos()
            euler = np.array([self._rand_theta(), self._rand_theta(), self._rand_theta()])
            radius = self._planning_radius if planning_friendly else self._rand_radiu
            bound = np.array([radius(), radius(), radius()])
            ellip = Ellipsoid()
            ellip.init(cpt, bound, euler)
            ellip.set_vel(self._sample_vel(dyn_mode))
            cur += self.update_pts(ellip, ellip_grids - cur)
            self.geo_map.add(ellip)
            if dyn_mode:
                self.ellip_list.append(ellip)
        ellip_grids = cur

        cur = 0
        while cur < poly_grids:
            cpt = self.grid_map.get_uni_rand_pos()
            radius = self._planning_radius if planning_friendly else self._rand_radiu
            bound = np.array([radius(), radius(), radius()])
            poly = Polyhedron()
            poly.random_init(cpt, bound, self.rng)
            poly.set_vel(self._sample_vel(dyn_mode))
            cur += self.update_pts(poly, poly_grids - cur)
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

    def _structured_pillars(self, options: dict) -> np.ndarray:
        """Build randomly placed rectangular pillars spanning the full Z axis."""
        xs, ys, zs = self._valid_grid_axes()
        if not len(xs) or not len(ys) or not len(zs):
            return np.zeros((0, 3), dtype=float)
        if not np.isfinite(self.mgpa.occupancy_ratio) or not 0 <= self.mgpa.occupancy_ratio <= 1:
            raise ValueError("occupancy_ratio must be between 0 and 1.")

        target_columns = int(np.floor(len(xs) * len(ys) * self.mgpa.occupancy_ratio))
        occupied = np.zeros((len(xs), len(ys)), dtype=bool)
        res = self.mpa.resolution
        min_width = max(1, int(np.ceil(float(options.get("min_width", 0.6)) / res)))
        max_width = max(min_width, int(np.ceil(float(options.get("max_width", 2.0)) / res)))
        max_attempts = max(100, target_columns * 20)
        attempts = 0
        while int(occupied.sum()) < target_columns and attempts < max_attempts:
            attempts += 1
            wx = min(int(self.rng.integers(min_width, max_width + 1)), len(xs))
            wy = min(int(self.rng.integers(min_width, max_width + 1)), len(ys))
            x0 = int(self.rng.integers(0, len(xs) - wx + 1))
            y0 = int(self.rng.integers(0, len(ys) - wy + 1))
            region = occupied[x0 : x0 + wx, y0 : y0 + wy]
            available = np.argwhere(~region)
            remaining = target_columns - int(occupied.sum())
            selected_count = min(len(available), remaining)
            if selected_count:
                selected = self.rng.choice(len(available), size=selected_count, replace=False)
                region[tuple(available[selected].T)] = True
        columns = np.argwhere(occupied)
        return np.asarray(
            [[xs[i], ys[j], z] for i, j in columns for z in zs],
            dtype=float,
        )

    def _update_moving_shape(self, shape, dt, bounds):
        """Advance one shape, bounce it at bounds, and translate its point cloud.

        Args:
            shape: Any supported shape exposing centre, velocity, and cloud APIs.
            dt: Simulation time step in seconds.
            bounds: ``(x_low, x_high, y_low, y_high, z_low, z_high)`` limits.

        The position is clamped to a boundary before reversing velocity.  The
        stored per-shape cloud is translated by exactly the accepted movement
        distance so it remains aligned with the shape's centre.
        """
        x_l, x_h, y_l, y_h, z_l, z_h = bounds
        cur = shape.get_center()
        vel = shape.get_vel()
        nxt = cur + dt * vel
        # This environment restricts obstacle motion to the horizontal plane.
        nxt[2] = cur[2]

        def bounce(pos, v, lo, hi):
            """Clamp one coordinate at a limit and reverse outward velocity."""
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
        # Geometry need not be rasterised yet, so only move a non-empty cloud.
        if shape.cloud:
            moved = np.asarray(shape.cloud, dtype=float) + dist
            shape.cloud = [p for p in moved]
            self.cloud_map = (
                moved
                if self.cloud_map.size == 0
                else np.vstack([self.cloud_map, moved])
            )

    def pcl2grid(self):
        """Write the aggregate cloud back into the occupancy grid.

        This method does not reset the grid first.  It mirrors the original
        dynamic-update workflow, where moved points are inserted after their
        new positions are calculated.
        """
        count = 0
        for pt in self.cloud_map:
            if not self.grid_map.is_in_map(pt):
                continue
            self.grid_map.set_occ(pt)
            count += 1
        print(f"GRID OBS: {count}")

    def dyn_generate(self, dt: float):
        """Advance all dynamic families by ``dt`` within fixed simulation bounds.

        The fixed bounds are retained from the original environment port and
        are independent of arbitrary ``GridMap`` dimensions.
        """
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
