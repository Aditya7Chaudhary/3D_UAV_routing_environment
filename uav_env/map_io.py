"""Map I/O helpers: images, PCD, NumPy arrays (no ROS / no rosbag)."""

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
    The image is extruded along Z across the full map height.
    """
    path = Path(path)
    img = Image.open(path).convert("RGB")
    img_data = np.asarray(img, dtype=np.float64)
    dim_x = int(map_size[0] / resolution)
    dim_y = int(map_size[1] / resolution)
    dim_z = int(map_size[2] / resolution)
    ratio_x = img.width / max(dim_x, 1)
    ratio_y = img.height / max(dim_y, 1)

    color_avg = img_data.mean(axis=2)
    if negate:
        color_avg = 255.0 - color_avg
    occ = (255.0 - color_avg) / 255.0

    points = []
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
    """Load a PCD file via Open3D, returning (N,3) float array."""
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        return np.zeros((0, 3))
    return np.asarray(cloud.points, dtype=float)


def write_pcd(path: str | Path, points: np.ndarray) -> None:
    """Write (N,3) points to a PCD file."""
    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    o3d.io.write_point_cloud(str(path), cloud)


def save_npy(path: str | Path, points: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(points, dtype=float))


def gen_random_cloud(bound: float = 50.0, num: int = 10000, seed: int = 0) -> np.ndarray:
    """Sparse random obstacle points with two cleared corridors (mode 0)."""
    rng = np.random.default_rng(seed)
    pts = []
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
    if cloud.size == 0 or noise_prob <= 0:
        return cloud
    rng = np.random.default_rng(seed)
    mask = rng.random(len(cloud)) < noise_prob
    noisy = cloud[mask] + rng.normal(0.0, noise_std, size=(mask.sum(), 3))
    mn = map_origin
    mx = map_origin + map_size
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
    """Remove points near listed clear positions (collision-free start/goal regions)."""
    if cloud.size == 0 or clear_pos.size == 0:
        return cloud
    keep = np.ones(len(cloud), dtype=bool)
    r2 = radius * radius
    for cp in clear_pos:
        if use_3d:
            d2 = np.sum((cloud - cp[:3]) ** 2, axis=1)
        else:
            d2 = np.sum((cloud[:, :2] - cp[:2]) ** 2, axis=1)
        keep &= d2 >= r2
    return cloud[keep]


def load_clear_csv(path: str | Path) -> tuple[np.ndarray, bool]:
    """Load clear_pos CSV. Returns (Nx2 or Nx3 array, use_3d flag)."""
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    use_3d = data.shape[1] >= 3
    if data.shape[1] == 2:
        padded = np.zeros((len(data), 3))
        padded[:, :2] = data
        return padded, False
    return data[:, :3], use_3d
