"""
===============================================================================
3D UAV ROUTING ENVIRONMENT — POINT-CLOUD VISUALISATION
===============================================================================

This file replaces the RViz-oriented visualisation path from the original
project.  It offers a lightweight Matplotlib preview for files and automated
runs, plus an Open3D interactive window for manual inspection.  Both helpers
expect the repository-standard ``(N, 3)`` world-coordinate point cloud.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


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
