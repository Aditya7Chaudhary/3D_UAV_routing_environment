#!/usr/bin/env python3
"""Convert a 2D occupancy image to a 3D PCD (standalone helper).

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

    inp = args.input if args.input.is_absolute() else ROOT / args.input
    out = args.output
    if out is None:
        out = ROOT / "outputs" / f"{inp.stem}.pcd"
    elif not out.is_absolute():
        out = ROOT / out

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
