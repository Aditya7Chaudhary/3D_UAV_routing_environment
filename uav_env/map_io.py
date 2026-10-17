"""
===============================================================================
3D UAV ROUTING ENVIRONMENT — MAP INPUT/OUTPUT HELPERS
===============================================================================

PURPOSE
-------
This file converts the map formats used by the repository into the common
``(N, 3)`` NumPy point-cloud representation.  Every row is one obstacle point
in world coordinates ``[x, y, z]``.  Keeping this conversion separate lets
the environment run without ROS, rosbag, or PCL tooling.

SUPPORTED INPUTS
----------------
* Occupancy images are interpreted as 2D maps and extruded through map height.
* PCD files are read and written through Open3D.
* NPY files store the same floating-point coordinate array directly.

Image Y coordinates are flipped during conversion: image rows increase
downward from the top-left, while world Y increases upward from map origin.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def read_image_map(
    path: str | Path,
    map_size: np.ndarray,
    map_origin: np.ndarray,
    resolution: float,
    occ_th: float = 0.65,
    negate: bool = False,
) -> np.ndarray:
    """Convert a 2D occupancy image into a 3D extruded point cloud.

    Darker pixels are treated as free space by default (same as kr_param_map).
    The image is extruded along Z across the full map height.  Image rows are
    flipped when mapped to world Y so conventional top-left image coordinates
    align with the map's Cartesian coordinates.

    Args:
        path: Raster map to load.
        map_size: Physical X/Y/Z dimensions of the desired world.
        map_origin: World coordinate of the map's lower corner.
        resolution: Side length of one generated voxel.
        occ_th: Darkness-derived occupancy value required to create a point.
        negate: Invert brightness before applying the occupancy threshold.

    Returns:
        An ``(N, 3)`` array of occupied voxel centres, or an empty ``(0, 3)``
        array when no pixels pass the threshold.
    """
    path = Path(path)
    img = Image.open(path).convert("RGB")
    img_data = np.asarray(img, dtype=np.float64)
    dim_x = int(map_size[0] / resolution)
    dim_y = int(map_size[1] / resolution)
    dim_z = int(map_size[2] / resolution)
    # A source image need not match the requested voxel dimensions.  These
    # ratios map each voxel index to the corresponding source-image pixel.
    ratio_x = img.width / max(dim_x, 1)
    ratio_y = img.height / max(dim_y, 1)

    # Average RGB to a grayscale brightness, then turn darkness into an
    # occupancy score: black approaches 1 and white approaches 0.
    color_avg = img_data.mean(axis=2)
    if negate:
        color_avg = 255.0 - color_avg
    occ = (255.0 - color_avg) / 255.0

    points = []
    # Every occupied image cell is copied through every Z layer, creating a
    # vertical obstacle column across the complete configured map height.
    for m in range(dim_z):
        for j in range(dim_y):
            j_px = min(int(j * ratio_y), img.height - 1)
            for i in range(dim_x):
                i_px = min(int(i * ratio_x), img.width - 1)
                if occ[j_px, i_px] > occ_th:
                    # Match C++ indexToPos: origin + (idx + 0.5) * res
                    # with flipped Y: (i, dim_y - j - 1, m)
                    x = map_origin[0] + (i + 0.5) * resolution
                    y = map_origin[1] + (dim_y - j - 1 + 0.5) * resolution
                    z = map_origin[2] + (m + 0.5) * resolution
                    points.append([x, y, z])
    return np.asarray(points, dtype=float) if points else np.zeros((0, 3))


def read_pcd(path: str | Path) -> np.ndarray:
    """Load a PCD file via Open3D and return world points as an ``(N, 3)`` array.

    An empty PCD is normalised to a correctly shaped empty NumPy array so
    callers can process it without special handling.
    """
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        return np.zeros((0, 3))
    return np.asarray(cloud.points, dtype=float)


def write_pcd(path: str | Path, points: np.ndarray) -> None:
    """Write an ``(N, 3)`` array to a PCD file, creating parent folders.

    Open3D owns the PCD serialization details; this wrapper keeps conversion
    to its required vector type at the repository boundary.
    """
    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    o3d.io.write_point_cloud(str(path), cloud)


def save_npy(path: str | Path, points: np.ndarray) -> None:
    """Persist a point cloud as a NumPy ``.npy`` array, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(points, dtype=float))


def gen_random_cloud(bound: float = 50.0, num: int = 10000, seed: int = 0) -> np.ndarray:
    """Generate sparse random obstacle points while preserving two corridors.

    The first rejected region is a central X-Y corridor; the second is a Y-Z
    corridor.  The loop continues until exactly ``num`` accepted samples are
    collected, making the requested point count independent of rejections.
    """
    rng = np.random.default_rng(seed)
    pts = []
    # The historical mode uses quarter-bound half widths for both corridors.
    fx1 = fy1 = fy2 = fz2 = bound / 4.0
    while len(pts) < num:
        ax, ay, az = rng.uniform(-bound, bound, size=3)
        if abs(ax) < fx1 and abs(ay) < fy1:
            continue
        if abs(ay) < fy2 and abs(az) < fz2:
            continue
        pts.append([ax, ay, az])
    return np.asarray(pts, dtype=float)


def add_noise_to_cloud(
    cloud: np.ndarray,
    map_origin: np.ndarray,
    map_size: np.ndarray,
    noise_std: float = 0.05,
    noise_prob: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Append bounded Gaussian perturbations to a random subset of points.

    The original cloud is always retained; samples that leave the configured
    half-open map volume are discarded.  The RNG seed makes noise repeatable.
    """
    if cloud.size == 0 or noise_prob <= 0:
        return cloud
    rng = np.random.default_rng(seed)
    # Select points independently, then create one Gaussian offset per selected
    # point.  This models extra noisy returns rather than moving source points.
    mask = rng.random(len(cloud)) < noise_prob
    noisy = cloud[mask] + rng.normal(0.0, noise_std, size=(mask.sum(), 3))
    mn = map_origin
    mx = map_origin + map_size
    # Retain only jittered points that can be represented by the destination map.
    inside = (
        (noisy[:, 0] >= mn[0])
        & (noisy[:, 0] < mx[0])
        & (noisy[:, 1] >= mn[1])
        & (noisy[:, 1] < mx[1])
        & (noisy[:, 2] >= mn[2])
        & (noisy[:, 2] < mx[2])
    )
    if not np.any(inside):
        return cloud
    return np.vstack([cloud, noisy[inside]])


def clear_near_positions(
    cloud: np.ndarray, clear_pos: np.ndarray, radius: float = np.sqrt(2.0), use_3d: bool = True
) -> np.ndarray:
    """Remove points near listed collision-free start/goal positions.

    Each clear position contributes an exclusion sphere in 3D or circle in XY
    when ``use_3d`` is false.  A point must be outside every exclusion region
    to remain in the returned cloud.
    """
    if cloud.size == 0 or clear_pos.size == 0:
        return cloud
    keep = np.ones(len(cloud), dtype=bool)
    r2 = radius * radius
    # Accumulate the keep mask with AND so a point near any requested clear
    # position is removed.
    for cp in clear_pos:
        if use_3d:
            d2 = np.sum((cloud - cp[:3]) ** 2, axis=1)
        else:
            d2 = np.sum((cloud[:, :2] - cp[:2]) ** 2, axis=1)
        keep &= d2 >= r2
    return cloud[keep]


def load_clear_csv(path: str | Path) -> tuple[np.ndarray, bool]:
    """Load a two- or three-column clear-position CSV.

    Returns:
        A padded ``(N, 3)`` coordinate array and a flag indicating whether the
        input contained a real Z coordinate.  Two-dimensional positions receive
        ``z = 0`` solely to maintain the common point-array shape.
    """
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    use_3d = data.shape[1] >= 3
    if data.shape[1] == 2:
        padded = np.zeros((len(data), 3))
        padded[:, :2] = data
        return padded, False
    return data[:, :3], use_3d
