"""
===============================================================================
3D UAV ROUTING ENVIRONMENT — POINT-CLOUD VISUALISATION
===============================================================================

This file replaces the RViz-oriented visualisation path from the original
project.  It keeps the original lightweight point-cloud preview for backwards
compatible diagnostics, and also offers an occupancy-faithful surface renderer
for generated environments.  The surface renderer draws only exposed faces of
the existing occupied voxels; it never changes coordinates, occupancy, or
collision behaviour merely for presentation.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Muted colours deliberately favour shape and depth over a heat-map look.  The
# keys match the concrete shape collections on ``GeoMap``.
_LAYER_COLOURS = {
    "terrain": (0.38, 0.56, 0.34, 0.96),
    "blocks": (0.52, 0.57, 0.64, 0.98),
    "corridor": (0.62, 0.49, 0.35, 0.98),
    "occupancy": (0.45, 0.56, 0.68, 0.98),
    "cylinder": (0.25, 0.48, 0.72, 0.98),
    "circle_gate": (0.22, 0.62, 0.54, 0.98),
    "rect_gate": (0.76, 0.48, 0.25, 0.98),
    "ellipsoid": (0.57, 0.39, 0.67, 0.98),
    "polyhedron": (0.76, 0.62, 0.26, 0.98),
}


def occupancy_from_points(
    points: np.ndarray,
    map_origin: np.ndarray,
    map_size: np.ndarray,
    resolution: float,
) -> np.ndarray:
    """Return the occupied voxel mask represented by repository-standard points.

    This is a rendering helper, not a second collision representation.  It
    uses the same half-open bounds and floor-based indexing as :class:`GridMap`
    so a point cloud exported from an occupancy grid is displayed as the
    occupied cells it already represents.
    """
    origin = np.asarray(map_origin, dtype=float)
    size = np.asarray(map_size, dtype=float)
    res = float(resolution)
    dims = np.ceil(size / res).astype(int)
    occupied = np.zeros(dims, dtype=bool)
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return occupied
    pts = pts.reshape((-1, 3))
    idx = np.floor((pts - origin) / res).astype(int)
    valid = np.all((idx >= 0) & (idx < dims), axis=1)
    if np.any(valid):
        occupied[tuple(idx[valid].T)] = True
    return occupied


def voxel_surface_faces(
    occupied: np.ndarray,
    map_origin: np.ndarray,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build quads for exactly the exposed faces of an occupied voxel mask.

    The returned mesh is presentation-only, but it is exact at voxel
    resolution: each quad is a face of a cell marked occupied by the source
    map.  Shared interior faces are omitted, which makes blocks, walls, and
    height fields read as continuous solids rather than stacked point layers.
    ``face_kind`` contains ``-x, +x, -y, +y, -z, +z`` codes (0 through 5).
    """
    occ = np.asarray(occupied, dtype=bool)
    if occ.ndim != 3:
        raise ValueError("occupied must be a 3-D boolean array")
    if not np.any(occ):
        return np.zeros((0, 4, 3), dtype=float), np.zeros(0, dtype=np.int8)

    origin = np.asarray(map_origin, dtype=float)
    res = float(resolution)
    padded = np.pad(occ, 1, mode="constant", constant_values=False)
    centre = padded[1:-1, 1:-1, 1:-1]
    # Face-corner offsets are in voxel units.  Their order is stable so the
    # returned arrays are deterministic for a fixed occupancy grid.
    corners = (
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=float),
        np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=float),
        np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=float),
        np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=float),
        np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=float),
        np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float),
    )
    neighbour_slices = (
        (slice(0, -2), slice(1, -1), slice(1, -1)),
        (slice(2, None), slice(1, -1), slice(1, -1)),
        (slice(1, -1), slice(0, -2), slice(1, -1)),
        (slice(1, -1), slice(2, None), slice(1, -1)),
        (slice(1, -1), slice(1, -1), slice(0, -2)),
        (slice(1, -1), slice(1, -1), slice(2, None)),
    )
    face_batches: list[np.ndarray] = []
    kind_batches: list[np.ndarray] = []
    for kind, (offsets, neighbour_slice) in enumerate(zip(corners, neighbour_slices)):
        exposed = centre & ~padded[neighbour_slice]
        indices = np.argwhere(exposed)
        if indices.size == 0:
            continue
        lower_corners = origin + indices[:, None, :] * res
        face_batches.append(lower_corners + offsets[None, :, :] * res)
        kind_batches.append(np.full(len(indices), kind, dtype=np.int8))
    if not face_batches:
        return np.zeros((0, 4, 3), dtype=float), np.zeros(0, dtype=np.int8)
    return np.concatenate(face_batches), np.concatenate(kind_batches)


def surface_bounds(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return physical bounds of a surface mesh, or ``None`` when it is empty."""
    if np.asarray(faces).size == 0:
        return None
    vertices = np.asarray(faces, dtype=float).reshape((-1, 3))
    return vertices.min(axis=0), vertices.max(axis=0)


def _face_colours(base_colour: tuple[float, float, float, float], kinds: np.ndarray) -> np.ndarray:
    """Apply restrained directional shading without inventing new geometry."""
    base = np.asarray(base_colour, dtype=float)
    # Lower, side, and top faces receive different lightness values.  This
    # creates readable vertical faces even with Matplotlib's simple renderer.
    factors = np.array([0.76, 0.88, 0.70, 0.84, 0.58, 1.12], dtype=float)
    colours = np.tile(base, (len(kinds), 1))
    colours[:, :3] = np.clip(colours[:, :3] * factors[kinds, None], 0.0, 1.0)
    return colours


def _add_surface_layer(ax, occupied, origin, resolution, colour, vertical_exaggeration: float) -> int:
    """Add one exact voxel-surface layer and return its exposed-face count."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces, kinds = voxel_surface_faces(occupied, origin, resolution)
    if faces.size == 0:
        return 0
    display_faces = faces.copy()
    display_faces[:, :, 2] = origin[2] + (
        display_faces[:, :, 2] - origin[2]
    ) * vertical_exaggeration
    collection = Poly3DCollection(
        display_faces,
        facecolors=_face_colours(colour, kinds),
        edgecolors="none",
        linewidths=0.0,
        zsort="average",
    )
    ax.add_collection3d(collection)
    return len(faces)


def _add_height_field_layer(
    ax,
    height_field: tuple[np.ndarray, np.ndarray, np.ndarray],
    origin: np.ndarray,
    colour: tuple[float, float, float, float],
    vertical_exaggeration: float,
) -> int:
    """Render the generator's original smooth terrain height field.

    This is deliberately separate from collision geometry.  The terrain
    generator first defines this continuous field and then samples it into
    occupied voxels for planning.  Showing the field avoids visual terraces
    caused solely by that sampling while preserving the collision grid exactly.
    """
    xs, ys, heights = height_field
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
    display_heights = origin[2] + (np.asarray(heights) - origin[2]) * vertical_exaggeration
    ax.plot_surface(
        x_grid,
        y_grid,
        display_heights,
        rcount=len(xs),
        ccount=len(ys),
        color=colour,
        linewidth=0.0,
        antialiased=True,
        shade=True,
    )
    # Matplotlib triangulates its surface internally.  The count documents
    # the equivalent two-triangle mesh size without retaining a duplicate mesh.
    return max(len(xs) - 1, 0) * max(len(ys) - 1, 0) * 2


def _shape_masks(generator, full_occupancy: np.ndarray) -> dict[str, np.ndarray]:
    """Recover exact per-shape voxel layers from generated shape point clouds."""
    grid = generator.grid_map
    origin = grid.mp.basic_mp.map_origin
    size = grid.mp.basic_mp.map_size
    res = grid.mp.resolution
    masks: dict[str, np.ndarray] = {}
    for name in ("cylinder", "circle_gate", "rect_gate", "ellipsoid", "polyhedron"):
        shapes = getattr(generator.geo_map, name)
        if not shapes:
            continue
        clouds = [np.asarray(shape.cloud, dtype=float) for shape in shapes if shape.cloud]
        if clouds:
            masks[name] = occupancy_from_points(np.vstack(clouds), origin, size, res) & full_occupancy
    return masks


def plot_environment_matplotlib(
    generator,
    title: str = "Environment",
    save_path: str | Path | None = None,
    show: bool = True,
    environment_type: str | None = None,
    vertical_exaggeration: float = 1.0,
    azimuth: float = -58.0,
    elevation: float = 32.0,
) -> dict[str, int]:
    """Render a generated map as a surface faithful to its occupied voxels.

    ``generator`` remains the source of truth: this function only reads its
    grid and (when available) its already-rasterised source-shape clouds.  A
    mixed scene separates terrain from primitive obstacle families for colour
    only; both layers remain derived from the same occupancy grid.
    """
    import matplotlib.pyplot as plt

    if vertical_exaggeration <= 0:
        raise ValueError("vertical_exaggeration must be positive")
    grid = generator.grid_map
    map_params = grid.mp.basic_mp
    origin = map_params.map_origin
    size = map_params.map_size
    occupied = grid.occupancy.reshape(grid.mp.map_grid_size) > grid.mp.min_thrd
    if not np.any(occupied):
        raise ValueError("Cannot render an environment with no occupied voxels.")

    env = (environment_type or "").strip().lower()
    shape_masks = _shape_masks(generator, occupied)
    layers: list[tuple[str, np.ndarray, tuple[float, float, float, float]]] = []
    height_field = getattr(generator, "terrain_height_field", None)
    is_height_terrain = env in {"random_land", "mountains", "mixed"} and height_field is not None
    if is_height_terrain:
        # The smooth source field is added below.  Only mixed scenes need
        # additional primitive-obstacle layers here.
        if env == "mixed":
            layers.extend((name, mask, _LAYER_COLOURS[name]) for name, mask in shape_masks.items())
    elif env == "mixed" and shape_masks:
        primitive_occupancy = np.zeros_like(occupied)
        for mask in shape_masks.values():
            primitive_occupancy |= mask
        layers.append(("terrain", occupied & ~primitive_occupancy, _LAYER_COLOURS["terrain"]))
        layers.extend((name, mask, _LAYER_COLOURS[name]) for name, mask in shape_masks.items())
    elif shape_masks and env == "structured":
        layers.extend((name, mask, _LAYER_COLOURS[name]) for name, mask in shape_masks.items())
    else:
        base_name = env if env in _LAYER_COLOURS else "occupancy"
        layers.append((base_name, occupied, _LAYER_COLOURS[base_name]))

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    face_counts: dict[str, int] = {}
    if is_height_terrain:
        face_counts["terrain_height_field_triangles"] = _add_height_field_layer(
            ax, height_field, origin, _LAYER_COLOURS["terrain"], vertical_exaggeration
        )
    face_counts.update({
        name: _add_surface_layer(ax, mask, origin, grid.mp.resolution, colour, vertical_exaggeration)
        for name, mask, colour in layers
    })
    max_range = origin + size
    z_max = origin[2] + size[2] * vertical_exaggeration
    ax.set_xlim(origin[0], max_range[0])
    ax.set_ylim(origin[1], max_range[1])
    ax.set_zlim(origin[2], z_max)
    ax.set_box_aspect((size[0], size[1], size[2] * vertical_exaggeration))
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlabel("X (m)", labelpad=8)
    ax.set_ylabel("Y (m)", labelpad=8)
    z_label = "Z (m)" if vertical_exaggeration == 1.0 else f"Z (m, visual x{vertical_exaggeration:g})"
    ax.set_zlabel(z_label, labelpad=8)
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        print(f"Saved environment surface: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return face_counts


def plot_voxel_environment_matplotlib(
    points: np.ndarray,
    map_origin: np.ndarray,
    map_size: np.ndarray,
    resolution: float,
    title: str = "Environment",
    save_path: str | Path | None = None,
    show: bool = True,
    environment_type: str | None = None,
    vertical_exaggeration: float = 1.0,
) -> int:
    """Render an exported cloud as an exact occupancy-derived surface.

    Use this variant when only an exported point cloud is available.  It has no
    shape-family metadata, but the rendered faces still correspond exactly to
    occupied voxels represented by ``points``.
    """
    import matplotlib.pyplot as plt

    if vertical_exaggeration <= 0:
        raise ValueError("vertical_exaggeration must be positive")
    origin = np.asarray(map_origin, dtype=float)
    size = np.asarray(map_size, dtype=float)
    occupied = occupancy_from_points(points, origin, size, resolution)
    if not np.any(occupied):
        raise ValueError("Cannot render an environment with no occupied voxels.")
    env = (environment_type or "").strip().lower()
    colour = _LAYER_COLOURS.get(env, _LAYER_COLOURS["occupancy"])
    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    count = _add_surface_layer(ax, occupied, origin, resolution, colour, vertical_exaggeration)
    maximum = origin + size
    ax.set_xlim(origin[0], maximum[0])
    ax.set_ylim(origin[1], maximum[1])
    ax.set_zlim(origin[2], origin[2] + size[2] * vertical_exaggeration)
    ax.set_box_aspect((size[0], size[1], size[2] * vertical_exaggeration))
    ax.view_init(elev=32.0, azim=-58.0)
    ax.set_xlabel("X (m)", labelpad=8)
    ax.set_ylabel("Y (m)", labelpad=8)
    ax.set_zlabel("Z (m)", labelpad=8)
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        print(f"Saved environment surface: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return count


def plot_cloud_matplotlib(
    points: np.ndarray,
    title: str = "Point cloud",
    save_path: str | Path | None = None,
    show: bool = True,
    max_points: int = 80000,
):
    """Plot an ``(N, 3)`` cloud, optionally downsampling and saving a PNG.

    ``max_points`` caps plotting work only; it never alters the caller's
    source array.  Point colour represents Z height, making vertical geometry
    visible even in a dense scatter plot.
    """
    import matplotlib.pyplot as plt

    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        print("Empty point cloud — nothing to plot.")
        return
    # Downsample with a fixed seed so repeated previews of the same cloud use
    # the same displayed subset.
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=pts[:, 2], cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    fig.tight_layout()
    # Saving and displaying are independent: batch jobs can save and close the
    # figure by passing ``show=False``.
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def show_open3d(points: np.ndarray, title: str = "Point cloud"):
    """Open an interactive Open3D viewer with colours derived from Z height.

    Blue represents lower points and red represents higher points.  A small
    denominator offset handles a flat cloud where all Z values are identical.
    """
    import open3d as o3d

    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        print("Empty point cloud — nothing to show.")
        return
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    # Color by height
    z = pts[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
    colors = np.stack([z_norm, 0.4 * np.ones_like(z_norm), 1.0 - z_norm], axis=1)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([cloud], window_name=title)
