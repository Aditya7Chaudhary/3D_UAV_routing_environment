#!/usr/bin/env python3
"""Generate one or more structured 3D UAV obstacle maps from YAML settings.

The script turns configuration values into map and obstacle-generator
parameters, optionally clears prescribed start/goal areas, then saves PCD and
NPY point clouds. It is the standard command-line entry point for synthetic
dataset generation.

The configuration defines physical bounds, per-shape target densities,
environment presets, output numbering, optional dynamic velocities, and
visualisation defaults. Command line options override only the seed, sample
count, configuration location, and viewer choices; all other values remain
reproducible from the YAML file.

Usage (from repo root):
  python scripts/generate_structure_map.py
  python scripts/generate_structure_map.py --config config/structure_map.yaml
  python scripts/generate_structure_map.py --seed 42 --no-show
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

from uav_env.grid_map import BasicMapParams, GridMapParams
from uav_env.map_io import clear_near_positions, load_clear_csv, write_pcd
from uav_env.struct_map_gen import MapGenParams, StructMapGenerator
from uav_env.visualize import plot_cloud_matplotlib, plot_environment_matplotlib, show_open3d


def load_config(path: Path) -> dict:
    """Load the YAML map-generation configuration as a nested dictionary.

    ``safe_load`` accepts ordinary configuration values and deliberately does
    not construct arbitrary Python objects from YAML tags.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """Parse options, generate samples, optionally clear regions, save, and plot.

    Each sample uses ``seed + i`` so a dataset receives deterministic but
    distinct maps.  Only the last sample is visualised after the loop.
    """
    parser = argparse.ArgumentParser(description="Generate structured UAV obstacle maps")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "structure_map.yaml",
        help="YAML config path",
    )
    parser.add_argument("--seed", type=float, default=None, help="Override random seed")
    parser.add_argument("--samples", type=int, default=None, help="Number of maps to generate")
    parser.add_argument("--no-show", action="store_true", help="Do not open a plot window")
    parser.add_argument("--open3d", action="store_true", help="Visualize with Open3D")
    args = parser.parse_args()

    # Split optional configuration sections from required map/shape settings.
    cfg = load_config(args.config)
    mcfg = cfg["map"]
    pcfg = cfg["params"]
    ecfg = cfg.get("environment", {})
    dcfg = cfg.get("dataset", {})
    ccfg = cfg.get("clear", {})
    dyn = cfg.get("dyn", {})

    seed = float(args.seed if args.seed is not None else pcfg.get("seed", 1))
    samples = int(args.samples if args.samples is not None else dcfg.get("samples_num", 1))
    start_index = int(dcfg.get("start_index", 0))
    out_dir = ROOT / dcfg.get("path", "outputs/dataset")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Construct physical grid parameters first; the generator finalises the
    # derived bounds and voxel storage during ``init_params``.
    basic = BasicMapParams(
        map_origin=np.array(
            [mcfg["x_origin"], mcfg["y_origin"], mcfg["z_origin"]], dtype=float
        ),
        map_size=np.array([mcfg["x_size"], mcfg["y_size"], mcfg["z_size"]], dtype=float),
    )
    grid_mp = GridMapParams(basic_mp=basic, resolution=float(mcfg["resolution"]))
    # Transfer YAML ratios and size-distribution controls into typed settings.
    mgpa = MapGenParams(
        cylinder_ratio=float(pcfg["cylinder_ratio"]),
        circle_ratio=float(pcfg["circle_ratio"]),
        gate_ratio=float(pcfg["gate_ratio"]),
        ellip_ratio=float(pcfg["ellip_ratio"]),
        poly_ratio=float(pcfg["poly_ratio"]),
        w1=float(pcfg["w1"]),
        w2=float(pcfg["w2"]),
        w3=float(pcfg["w3"]),
        add_noise=bool(pcfg.get("add_noise", False)),
    )

    gen = StructMapGenerator()
    gen.vel_h = np.array(
        [dyn.get("v_x_h", 0.1), dyn.get("v_y_h", 0.1), dyn.get("v_z_h", 0.0)], dtype=float
    )
    gen.init_params(grid_mp)
    # ``MapGenParams`` are configuration, not defaults: assigning them here
    # makes all params.* values in the YAML affect structured and mixed maps.
    gen.mgpa = mgpa

    clear_pts = None
    use_3d = True
    # Clear positions reserve collision-free regions, commonly for starts/goals.
    if ccfg.get("enabled", False):
        clear_pts, use_3d = load_clear_csv(ROOT / ccfg["path"])

    last_cloud = None
    for i in range(samples):
        cur_seed = seed + i
        # Reset map state before every independent seed/sample.
        gen.clear()
        gen.generate_environment(ecfg, cur_seed, bool(dyn.get("dyn_mode", False)))
        if mcfg.get("simple_2d", False):
            cloud = gen.get_pc_2d()
        else:
            cloud = gen.get_pc()

        # Apply reservations after map construction so no generated point
        # survives inside a configured clear region.
        if clear_pts is not None:
            cloud = clear_near_positions(cloud, clear_pts, use_3d=use_3d)

        if dcfg.get("save_map", True):
            out_pcd = out_dir / f"pt{start_index + i}.pcd"
            write_pcd(out_pcd, cloud)
            np.save(out_dir / f"pt{start_index + i}.npy", cloud)
            print(f"Saved {out_pcd}  ({len(cloud)} points)")

        last_cloud = cloud

    if last_cloud is None:
        return

    fig_path = ROOT / "outputs" / "structure_map_preview.png"
    show = (not args.no_show) and bool(cfg.get("visualize", True))
    if args.open3d or cfg.get("use_open3d", False):
        show_open3d(last_cloud, title="Structure map")
    elif mcfg.get("simple_2d", False) or clear_pts is not None:
        # A cleared cloud may intentionally differ from the in-memory grid, so
        # retain the point renderer for this legacy export-only mode.
        plot_cloud_matplotlib(
            last_cloud,
            title="Structure map",
            save_path=fig_path,
            show=show,
        )
    else:
        plot_environment_matplotlib(
            gen,
            title=f"Structure map: {ecfg.get('type', 'structured')}",
            save_path=fig_path,
            show=show,
            environment_type=ecfg.get("type", "structured"),
        )


if __name__ == "__main__":
    main()
