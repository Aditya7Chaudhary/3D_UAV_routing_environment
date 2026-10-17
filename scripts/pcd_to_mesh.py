#!/usr/bin/env python3
"""Voxelise an occupied PCD and export its merged box geometry as OBJ.

The wrapper forwards CLI paths and meshing options to ``uav_env.mesh_export``;
``--gazebo`` additionally writes descriptor files that load the OBJ as a
static Gazebo model. The voxel size controls both mesh detail and its output
complexity: smaller cells preserve more detail but create more boxes/faces.

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
    """Parse mesh-export arguments and delegate conversion to the library API.

    When no OBJ path is supplied, the input file's stem is reused beneath the
    repository ``outputs`` directory, keeping PCD and generated mesh names
    easy to associate.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument("--gazebo", action="store_true", help="Also write optional Gazebo model files")
    parser.add_argument("--model-name", type=str, default="map")
    args = parser.parse_args()

    # Resolve relative paths consistently with the project's other CLI tools.
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
