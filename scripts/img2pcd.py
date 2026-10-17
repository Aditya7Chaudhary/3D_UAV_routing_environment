#!/usr/bin/env python3
"""Convert a 2D occupancy image into an extruded 3D PCD point cloud.

This small command-line wrapper exposes image dimensions, resolution, and
occupancy threshold without requiring a full ``read_grid_map`` configuration.
It is useful when converting a single image and inspecting the result quickly.

Usage:
  python scripts/img2pcd.py --input data/img/all/maze.png --output outputs/maze.pcd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.map_io import read_image_map, write_pcd
from uav_env.visualize import plot_cloud_matplotlib


def main():
    """Parse conversion arguments, create the cloud, save it, and optionally plot.

    Relative paths are rooted at the project directory so the command behaves
    consistently whether launched from the repository root or another folder.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--x-size", type=float, default=20.0)
    parser.add_argument("--y-size", type=float, default=10.0)
    parser.add_argument("--z-size", type=float, default=0.5)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--occ-th", type=float, default=0.7)
    parser.add_argument("--negate", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    # Preserve absolute caller paths; otherwise use the repository as the
    # stable base for both source and default output locations.
    inp = args.input if args.input.is_absolute() else ROOT / args.input
    out = args.output
    if out is None:
        out = ROOT / "outputs" / f"{inp.stem}.pcd"
    elif not out.is_absolute():
        out = ROOT / out

    # This standalone helper uses a zero origin; image conversion places voxel
    # centres positively within the supplied physical X/Y/Z dimensions.
    map_size = np.array([args.x_size, args.y_size, args.z_size], dtype=float)
    map_origin = np.zeros(3)
    cloud = read_image_map(
        inp,
        map_size=map_size,
        map_origin=map_origin,
        resolution=args.resolution,
        occ_th=args.occ_th,
        negate=args.negate,
    )
    write_pcd(out, cloud)
    print(f"Saved {out} with {len(cloud)} points")
    if args.show:
        plot_cloud_matplotlib(cloud, title=inp.name, show=True)


if __name__ == "__main__":
    main()
