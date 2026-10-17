#!/usr/bin/env python3
"""Load source maps and optionally rasterise or inflate them into a voxel grid.

Supported modes are synthetic random points, occupancy images, and PCD files.
The script writes the loaded cloud and, when enabled, occupied voxel centres,
providing a ROS-free conversion path for planner inputs.

For every input, the source cloud can receive bounded synthetic noise before
being written. Grid inflation then expands occupied points into nearby voxels,
which is useful for accounting for UAV clearance in a voxel planner.

Usage (from repo root):
  python scripts/read_grid_map.py
  python scripts/read_grid_map.py --config config/read_grid_map.yaml
  python scripts/read_grid_map.py --mode 4 --file data/pcd/cloud_0917.pcd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.grid_map import BasicMapParams, GridMap, GridMapParams
from uav_env.map_io import (
    add_noise_to_cloud,
    gen_random_cloud,
    read_image_map,
    read_pcd,
    write_pcd,
)
from uav_env.visualize import plot_cloud_matplotlib, show_open3d


def load_config(path: Path) -> dict:
    """Load the YAML read/conversion configuration without unsafe YAML objects."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    """Resolve relative paths against the repository root.

    Absolute paths are left unchanged, allowing users to convert data stored
    outside this project without copying it into the repository first.
    """
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def load_one(mode: int, file_path: Path, cfg: dict) -> np.ndarray:
    """Load one cloud according to the configured input mode.

    Modes 2 and 3 intentionally stop with guidance because ROS bag support
    was removed from this standalone port.
    """
    mcfg = cfg["map"]
    map_size = np.array([mcfg["x_size"], mcfg["y_size"], mcfg["z_size"]], dtype=float)
    map_origin = np.array(
        [mcfg["x_origin"], mcfg["y_origin"], mcfg["z_origin"]], dtype=float
    )
    res = float(mcfg["resolution"])

    # Mode 0 has no input file: it builds the historical synthetic test cloud.
    if mode == 0:
        return gen_random_cloud(bound=max(map_size[:2]) / 2, num=10000, seed=cfg["params"].get("seed", 0))
    # Mode 1 maps image pixels into world voxel centres and extrudes them in Z.
    if mode == 1:
        return read_image_map(
            file_path,
            map_size=map_size,
            map_origin=map_origin,
            resolution=res,
            occ_th=float(cfg["img"]["occ_th"]),
            negate=bool(cfg["img"].get("negate", False)),
        )
    # Mode 4 already supplies world-coordinate obstacle points.
    if mode == 4:
        return read_pcd(file_path)
    if mode in (2, 3):
        raise SystemExit(
            "ROS bag modes (2/3) are not supported in this pure-Python port.\n"
            "Convert the bag to PCD first (e.g. with ROS once, or pcl_ros), then use --mode 4."
        )
    raise SystemExit(f"Unknown mode: {mode}")


def main():
    """Parse input options, convert maps, optionally grid them, and preview last.

    A folder is filtered to appropriate image/PCD extensions and processed in
    sorted order.  When several inputs are processed, their source filename is
    prefixed to each output to avoid overwriting prior conversions.
    """
    parser = argparse.ArgumentParser(description="Read / convert grid maps for UAV planning")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "read_grid_map.yaml")
    parser.add_argument("--mode", type=int, default=None, help="0=random, 1=image, 4=pcd")
    parser.add_argument("--file", type=Path, default=None, help="Input file path")
    parser.add_argument("--folder", type=Path, default=None, help="Process all files in folder")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--open3d", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = int(args.mode if args.mode is not None else cfg["map"]["mode"])
    mcfg = cfg["map"]
    pcfg = cfg.get("params", {})
    ocfg = cfg.get("output", {})

    files: list[Path] = []
    # Choose either a sorted batch of maps or one explicitly configured file.
    if args.folder or cfg.get("use_folder", False):
        folder = resolve_path(args.folder or cfg["folder_path"])
        exts = {".png", ".jpg", ".jpeg", ".pcd"} if mode in (1, 4) else {".png", ".pcd"}
        files = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
        if not files:
            raise SystemExit(f"No map files found in {folder}")
    else:
        files = [resolve_path(args.file or cfg["file_path"])]

    basic = BasicMapParams(
        map_origin=np.array(
            [mcfg["x_origin"], mcfg["y_origin"], mcfg["z_origin"]], dtype=float
        ),
        map_size=np.array([mcfg["x_size"], mcfg["y_size"], mcfg["z_size"]], dtype=float),
    )
    grid_mp = GridMapParams(basic_mp=basic, resolution=float(mcfg["resolution"]))
    grid = GridMap()
    grid.init_map(grid_mp)

    last_cloud = None
    last_grid = None
    for fp in files:
        print(f"Reading map: {fp}")
        cloud = load_one(mode, fp, cfg)
        # Noise is applied before serialization and optional voxel inflation,
        # so both outputs represent the same perturbed source cloud.
        cloud = add_noise_to_cloud(
            cloud,
            map_origin=basic.map_origin,
            map_size=basic.map_size,
            noise_std=float(pcfg.get("noise_std", 0.05)),
            noise_prob=float(pcfg.get("noise_prob", 0.1)) if pcfg.get("add_noise") else 0.0,
            seed=int(pcfg.get("seed", 0)),
        )
        print(f"  cloud points: {len(cloud)}")

        out_pcd = ROOT / ocfg.get("pcd_path", "outputs/read_map.pcd")
        if len(files) > 1:
            out_pcd = out_pcd.with_name(f"{fp.stem}_{out_pcd.name}")
        write_pcd(out_pcd, cloud)
        print(f"  saved: {out_pcd}")

        # Rebuild from free space for each file, then export canonical voxel
        # centres rather than the raw, potentially off-grid input points.
        if mcfg.get("publish_grid_centers", True):
            grid.clear_all_occ()
            grid.fill_map(cloud, inflated=float(mcfg.get("inflate_radius", 0.0)))
            grid_cloud = grid.publish_map()
            out_grid = ROOT / ocfg.get("grid_pcd_path", "outputs/read_map_grid.pcd")
            if len(files) > 1:
                out_grid = out_grid.with_name(f"{fp.stem}_{out_grid.name}")
            write_pcd(out_grid, grid_cloud)
            print(f"  grid centers: {len(grid_cloud)} -> {out_grid}")
            last_grid = grid_cloud

        last_cloud = cloud

    show = (not args.no_show) and bool(cfg.get("visualize", True))
    vis_pts = last_grid if last_grid is not None else last_cloud
    fig_path = ROOT / "outputs" / "read_map_preview.png"
    if args.open3d or cfg.get("use_open3d", False):
        show_open3d(vis_pts, title="Read grid map")
    else:
        plot_cloud_matplotlib(
            vis_pts, title="Read grid map", save_path=fig_path, show=show
        )


if __name__ == "__main__":
    main()
