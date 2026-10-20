#!/usr/bin/env python3
"""
===============================================================================
RANDOM 3D UAV ENVIRONMENT DATASET GENERATOR
===============================================================================

PURPOSE
-------
Generate a dataset of varied 3D UAV planning environments without requiring a
person to edit ``environment.type`` in YAML for every sample.  For each
iteration, this script selects an environment family at random, randomises its
family-specific parameters, generates its occupied point cloud, and saves all
artifacts needed to inspect or reuse the sample.

SAVED ARTIFACTS
---------------
<output-dir>/
    images/          One PNG preview for every generated environment.
    point_clouds/    Matching PCD and NPY coordinate files for every sample.
    metadata.jsonl   One JSON record per sample: type, seed, options, paths,
                     point count, and map configuration.
    dataset_summary.json
                     Run-wide generation settings and all sample records.

UNBALANCED SAMPLING
-------------------
The default weights intentionally favour structured and urban-block scenes,
then mountains, random land, corridors, and mixed terrain.  This produces a
varied but non-uniform dataset.  Pass ``--weights`` to change that distribution
without changing this script.

USAGE
-----
    python scripts/generate_environment_dataset.py --samples 100 --no-show
    python scripts/generate_environment_dataset.py --samples 500 --seed 42 \
        --output-dir outputs/uav_dataset
===============================================================================
"""

from __future__ import annotations

import argparse
import json
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
from uav_env.visualize import plot_cloud_matplotlib, plot_environment_matplotlib


# The order must match the six comma-separated values accepted by --weights.
ENVIRONMENT_TYPES = (
    "structured",
    "blocks",
    "mountains",
    "random_land",
    "corridor",
    "mixed",
)
DEFAULT_WEIGHTS = (0.35, 0.25, 0.15, 0.10, 0.10, 0.05)


def load_config(path: Path) -> dict:
    """Load base map dimensions and primitive-generator settings from YAML."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_weights(raw_weights: str) -> np.ndarray:
    """Validate and normalise six comma-separated unbalanced type weights.

    The values correspond, in order, to ``structured``, ``blocks``,
    ``mountains``, ``random_land``, ``corridor``, and ``mixed``.  Normalising
    lets users specify relative weights such as ``7,5,3,2,2,1``.
    """
    try:
        weights = np.asarray([float(value) for value in raw_weights.split(",")], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--weights must contain numbers separated by commas.") from exc
    if len(weights) != len(ENVIRONMENT_TYPES):
        expected = len(ENVIRONMENT_TYPES)
        raise argparse.ArgumentTypeError(f"--weights needs exactly {expected} values.")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise argparse.ArgumentTypeError("--weights must be non-negative and include a positive value.")
    return weights / weights.sum()


def build_map_params(config: dict) -> GridMapParams:
    """Convert the shared YAML ``map`` section into typed grid parameters."""
    map_cfg = config["map"]
    basic = BasicMapParams(
        map_origin=np.array(
            [map_cfg["x_origin"], map_cfg["y_origin"], map_cfg["z_origin"]],
            dtype=float,
        ),
        map_size=np.array(
            [map_cfg["x_size"], map_cfg["y_size"], map_cfg["z_size"]],
            dtype=float,
        ),
    )
    return GridMapParams(basic_mp=basic, resolution=float(map_cfg["resolution"]))


def build_shape_params(config: dict) -> MapGenParams:
    """Convert the YAML primitive-density settings into generator parameters."""
    params_cfg = config["params"]
    return MapGenParams(
        cylinder_ratio=float(params_cfg["cylinder_ratio"]),
        circle_ratio=float(params_cfg["circle_ratio"]),
        gate_ratio=float(params_cfg["gate_ratio"]),
        ellip_ratio=float(params_cfg["ellip_ratio"]),
        poly_ratio=float(params_cfg["poly_ratio"]),
        w1=float(params_cfg["w1"]),
        w2=float(params_cfg["w2"]),
        w3=float(params_cfg["w3"]),
        add_noise=bool(params_cfg.get("add_noise", False)),
    )


def sample_environment_options(environment_type: str, rng: np.random.Generator) -> dict:
    """Create varied, JSON-safe options for one selected environment family.

    A distinct sample seed controls the actual layout inside the environment
    generator.  This function additionally varies high-level terrain shape,
    building density, or corridor dimensions so same-type images do not simply
    differ by object placement.
    """
    options: dict[str, float | int | str] = {"type": environment_type}
    if environment_type in ("random_land", "mixed"):
        options.update(
            base_height=0.0,
            height_range=float(rng.uniform(0.4, 1.8)),
            smoothing_passes=int(rng.integers(2, 11)),
        )
    elif environment_type == "mountains":
        spread_min = float(rng.uniform(0.8, 2.0))
        options.update(
            base_height=0.0,
            mountain_count=int(rng.integers(3, 10)),
            peak_height=float(rng.uniform(0.8, 2.8)),
            spread_min=spread_min,
            spread_max=float(rng.uniform(max(spread_min, 2.0), 6.0)),
        )
    elif environment_type == "blocks":
        min_width = float(rng.uniform(0.6, 1.5))
        min_height = float(rng.uniform(0.5, 1.2))
        options.update(
            building_count=int(rng.integers(8, 41)),
            min_width=min_width,
            max_width=float(rng.uniform(max(min_width, 1.5), 4.5)),
            min_height=min_height,
            max_height=float(rng.uniform(max(min_height, 1.3), 2.9)),
        )
    elif environment_type == "corridor":
        options.update(
            width=float(rng.uniform(4.0, 10.0)),
            wall_height=float(rng.uniform(1.5, 2.9)),
            partition_spacing=float(rng.uniform(3.0, 8.0)),
            opening_width=float(rng.uniform(0.8, 2.5)),
        )
    return options


def main() -> None:
    """Generate the requested number of random environment samples and files."""
    parser = argparse.ArgumentParser(
        description="Generate an unbalanced dataset of varied 3D UAV environments."
    )
    parser.add_argument(
        "--samples",
        "-x",
        type=int,
        required=True,
        help="Number of environment images/point clouds to generate.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "structure_map.yaml",
        help="Base map and primitive settings YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "random_environment_dataset",
        help="Folder that will contain images, coordinates, and metadata.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Dataset-level reproducibility seed.")
    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=np.asarray(DEFAULT_WEIGHTS, dtype=float),
        help=(
            "Six comma-separated relative weights for structured,blocks,mountains,"
            "random_land,corridor,mixed. Default: 0.35,0.25,0.15,0.10,0.10,0.05"
        ),
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=80000,
        help="Maximum points drawn in each preview; coordinate files remain complete.",
    )
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be a positive integer.")

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    images_dir = output_dir / "images"
    clouds_dir = output_dir / "point_clouds"
    images_dir.mkdir(parents=True, exist_ok=True)
    clouds_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    grid_params = build_map_params(config)
    shape_params = build_shape_params(config)
    dataset_rng = np.random.default_rng(args.seed)
    generator = StructMapGenerator()
    generator.init_params(grid_params)
    generator.mgpa = shape_params

    # Optional clear zones are shared by every sample, e.g. for planner starts.
    clear_cfg = config.get("clear", {})
    clear_points = None
    use_3d = True
    if clear_cfg.get("enabled", False):
        clear_path = Path(clear_cfg["path"])
        clear_points, use_3d = load_clear_csv(
            clear_path if clear_path.is_absolute() else ROOT / clear_path
        )

    digits = max(4, len(str(args.samples)))
    records = []
    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        for index in range(args.samples):
            environment_type = str(dataset_rng.choice(ENVIRONMENT_TYPES, p=args.weights))
            options = sample_environment_options(environment_type, dataset_rng)
            sample_seed = int(dataset_rng.integers(0, np.iinfo(np.int64).max))

            # Reset prior shape/cloud state, then create one fully independent map.
            generator.clear()
            generator.generate_environment(options, sample_seed, if_dyn=False)
            cloud = generator.get_pc()
            if clear_points is not None:
                cloud = clear_near_positions(cloud, clear_points, use_3d=use_3d)

            stem = f"environment_{index:0{digits}d}_{environment_type}"
            pcd_path = clouds_dir / f"{stem}.pcd"
            npy_path = clouds_dir / f"{stem}.npy"
            image_path = images_dir / f"{stem}.png"
            write_pcd(pcd_path, cloud)
            np.save(npy_path, cloud)
            # Render the generator's existing occupied cells as exposed voxel
            # faces.  The PCD/NPY data and collision grid are untouched.
            if clear_points is None:
                plot_environment_matplotlib(
                    generator,
                    title=f"Environment {index}: {environment_type}",
                    save_path=image_path,
                    show=False,
                    environment_type=environment_type,
                )
            else:
                # Clear zones are an export-time legacy filter and therefore
                # need a cloud renderer rather than the unchanged grid.
                plot_cloud_matplotlib(
                    cloud,
                    title=f"Environment {index}: {environment_type}",
                    save_path=image_path,
                    show=False,
                    max_points=args.max_plot_points,
                )

            record = {
                "index": index,
                "type": environment_type,
                "seed": sample_seed,
                "options": options,
                "point_count": int(len(cloud)),
                "pcd": str(pcd_path.relative_to(output_dir)),
                "npy": str(npy_path.relative_to(output_dir)),
                "image": str(image_path.relative_to(output_dir)),
                "map": {
                    "origin": grid_params.basic_mp.map_origin.tolist(),
                    "size": grid_params.basic_mp.map_size.tolist(),
                    "resolution": grid_params.resolution,
                },
            }
            metadata_file.write(json.dumps(record) + "\n")
            records.append(record)
            print(
                f"[{index + 1}/{args.samples}] {environment_type}: "
                f"{len(cloud)} points -> {image_path.name}"
            )

    summary = {
        "samples": args.samples,
        "dataset_seed": args.seed,
        "environment_types": list(ENVIRONMENT_TYPES),
        "weights": {name: float(weight) for name, weight in zip(ENVIRONMENT_TYPES, args.weights)},
        "base_config": str(config_path),
        "records": records,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nCompleted {args.samples} samples in: {output_dir}")


if __name__ == "__main__":
    main()
