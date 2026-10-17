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
``structured`` keeps the original primitive-obstacle generator. ``corridor``
creates bounded passages with alternating openings. ``random_land`` creates a
smoothed random height field, ``mountains`` sums Gaussian hills, and ``blocks``
creates rectangular buildings. ``mixed`` lays structured obstacles over random
land.  Presets are selected through the ``environment`` YAML section.

The generator produces geometry and occupancy only; it does not choose UAV
start/goal positions or plan routes.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo_map import GeoMap
from .grid_map import GridMap, GridMapParams
from .shapes import CircleGate, Cylinder, Ellipsoid, Polyhedron, RectGate


@dataclass
class MapGenParams:
    """Configure target obstacle density, dimensions, and sampling noise.

    Each ``*_ratio`` is multiplied by the total number of map voxels to obtain
    a target count.  ``w1`` through ``w3`` define the distributions used for
    obstacle widths and radii; their historical names are retained to match
    the source implementation and existing YAML files.
    """
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

    def change_res(self, res: float):
        """Reinitialise the grid using the current map bounds and new resolution."""
        self.mpa.resolution = res
        self.grid_map.init_map(self.mpa)

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

    def update_pts(self, geo_rep) -> int:
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
            rx = r * res
            for s in range(-wid[1], wid[1]):
                sy = s * res
                for t in range(-wid[2], wid[2]):
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
                if not self.grid_map.is_in_map(noisy):
                    continue
                self.grid_map.set_occ(noisy)
                new_pts.append(noisy)
                geo_rep.cloud.append(noisy.copy())

        self._append_points(new_pts)
        return cur

    def update_pts_light(self, geo_rep) -> int:
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

    def _height_field_cloud(self, heights: np.ndarray) -> np.ndarray:
        """Fill all voxels from map ground up to a height field.

        ``heights[i, j]`` is an absolute world-space Z height for the terrain
        surface at the X-Y voxel centre.  Filling below that surface makes hills
        and landforms solid obstacles rather than a single thin visual shell.
        """
        xs, ys, zs = self._valid_grid_axes()
        points = []
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                # Include every valid Z centre no higher than this local terrain.
                for z in zs[zs <= heights[i, j]]:
                    points.append([x, y, z])
        return np.asarray(points, dtype=float) if points else np.zeros((0, 3))

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

    def _random_land_cloud(self, options: dict) -> np.ndarray:
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
        return self._height_field_cloud(heights)

    def _mountain_cloud(self, options: dict) -> np.ndarray:
        """Generate solid Gaussian hills over a flat base terrain.

        Options control the number, maximum height, and horizontal spread of
        hills.  Summing smooth Gaussian peaks creates mountain-like slopes
        without requiring external terrain assets.
        """
        xs, ys, _ = self._valid_grid_axes()
        x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
        origin = self.mpa.basic_mp.map_origin
        maximum = self.mpa.basic_mp.max_range
        heights = np.full_like(x_grid, origin[2] + float(options.get("base_height", 0.0)))
        count = max(int(options.get("mountain_count", 5)), 1)
        peak_max = float(options.get("peak_height", 2.0))
        spread_min = float(options.get("spread_min", 1.5))
        spread_max = float(options.get("spread_max", 4.0))
        for _ in range(count):
            cx = self.rng.uniform(origin[0], maximum[0])
            cy = self.rng.uniform(origin[1], maximum[1])
            peak = self.rng.uniform(0.4 * peak_max, peak_max)
            spread = self.rng.uniform(spread_min, max(spread_min, spread_max))
            # The exponent creates radial slopes that decay smoothly from a peak.
            heights += peak * np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * spread**2))
        z_max = maximum[2] - 0.5 * self.mpa.resolution
        return self._height_field_cloud(np.clip(heights, origin[2], z_max))

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
        """Generate a long corridor bounded by walls and alternating partitions.

        The corridor runs along global X.  Parallel Y walls define its width;
        cross-wall partitions create alternating openings, requiring a planner
        to follow a winding route instead of travelling straight through.
        """
        xs, ys, zs = self._valid_grid_axes()
        occupied = np.zeros((len(xs), len(ys), len(zs)), dtype=bool)
        res = self.mpa.resolution
        width = max(3, int(np.ceil(float(options.get("width", 6.0)) / res)))
        center = len(ys) // 2
        y_low = max(0, center - width // 2)
        y_high = min(len(ys) - 1, center + width // 2)
        wall_layers = min(len(zs), max(1, int(np.ceil(float(options.get("wall_height", 2.5)) / res))))
        # The two long walls form the outer boundary of the passage.
        occupied[:, y_low, :wall_layers] = True
        occupied[:, y_high, :wall_layers] = True
        spacing = max(1, int(np.ceil(float(options.get("partition_spacing", 6.0)) / res)))
        opening = max(1, int(np.ceil(float(options.get("opening_width", 1.5)) / res)))
        for number, x in enumerate(range(spacing, len(xs), spacing)):
            # Alternate opening positions near opposite walls to create turns.
            if number % 2 == 0:
                gap_start = y_low + 1
            else:
                gap_start = max(y_low + 1, y_high - opening)
            gap_end = min(gap_start + opening, y_high)
            occupied[x, y_low + 1 : y_high, :wall_layers] = True
            occupied[x, gap_start:gap_end, :wall_layers] = False
        idx = np.argwhere(occupied)
        return np.array([[xs[i], ys[j], zs[k]] for i, j, k in idx], dtype=float)

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
        if env_type == "structured":
            self.generate(if_dyn)
        elif env_type == "corridor":
            self._replace_with_cloud(self._corridor_cloud(options))
        elif env_type == "random_land":
            self._replace_with_cloud(self._random_land_cloud(options))
        elif env_type == "mountains":
            self._replace_with_cloud(self._mountain_cloud(options))
        elif env_type == "blocks":
            self._replace_with_cloud(self._blocks_cloud(options))
        elif env_type == "mixed":
            # Terrain is placed first so primitive shapes avoid its occupied voxels.
            self._replace_with_cloud(self._random_land_cloud(options))
            self.generate(if_dyn)
        else:
            valid = "structured, corridor, random_land, mountains, blocks, mixed"
            raise ValueError(f"Unknown environment type {env_type!r}. Choose one of: {valid}.")

    def generate(self, dyn_mode: bool = False):
        """Create each obstacle family until its target occupied-voxel count.

        The ratios translate to desired counts in the full voxel volume.  Each
        generated shape is rasterised immediately, preventing later shapes
        from claiming already occupied cells.
        """
        self.grid_map.set_uni_rand(self.rng)
        # Convert physical volume to an approximate count of cubic voxels.
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

        # Each loop keeps placing shapes until this family reaches its target.
        # Collisions with earlier shapes reduce a shape's contribution because
        # occupied cells are never counted twice.
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
