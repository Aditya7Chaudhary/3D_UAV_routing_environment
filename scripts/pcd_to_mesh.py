#!/usr/bin/env python3
"""Voxelize a PCD and export a merged triangle mesh (OBJ).

Usage:
  python scripts/pcd_to_mesh.py --input outputs/maze.pcd --output outputs/maze.obj
  python scripts/pcd_to_mesh.py --input outputs/maze.pcd --gazebo   # also write Gazebo model files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.mesh_export import pcd_to_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument("--gazebo", action="store_true", help="Also write optional Gazebo model files")
    parser.add_argument("--model-name", type=str, default="map")
    args = parser.parse_args()

    inp = args.input if args.input.is_absolute() else ROOT / args.input
    out = args.output
    if out is None:
        out = ROOT / "outputs" / f"{inp.stem}.obj"
    elif not out.is_absolute():
        out = ROOT / out

    pcd_to_mesh(
        inp,
        out,
        voxel_size=args.voxel_size,
        write_gazebo=args.gazebo,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
