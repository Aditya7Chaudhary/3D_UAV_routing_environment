#!/usr/bin/env python3
"""Generate one low-cost comparison image for all supported environment types."""

from __future__ import annotations

import argparse
import copy
import gc
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.grid_map import BasicMapParams, GridMapParams
from uav_env.struct_map_gen import MapGenParams, StructMapGenerator
from uav_env.visualize import plot_environment_matplotlib


ENVIRONMENT_TYPES = ("structured", "blocks", "random_land", "mountains", "corridor", "mixed")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_map_params(config: dict, resolution: float) -> GridMapParams:
    map_cfg = config["map"]
    return GridMapParams(
        basic_mp=BasicMapParams(
            map_origin=np.array(
                [map_cfg["x_origin"], map_cfg["y_origin"], map_cfg["z_origin"]], dtype=float
            ),
            map_size=np.array(
                [map_cfg["x_size"], map_cfg["y_size"], map_cfg["z_size"]], dtype=float
            ),
        ),
        resolution=resolution,
    )


def build_shape_params(config: dict) -> MapGenParams:
    params = config["params"]
    return MapGenParams(
        occupancy_ratio=float(params.get("occupancy_ratio", 0.5)),
        cylinder_ratio=float(params["cylinder_ratio"]),
        circle_ratio=float(params["circle_ratio"]),
        gate_ratio=float(params["gate_ratio"]),
        ellip_ratio=float(params["ellip_ratio"]),
        poly_ratio=float(params["poly_ratio"]),
        w1=float(params["w1"]),
        w2=float(params["w2"]),
        w3=float(params["w3"]),
        add_noise=bool(params.get("add_noise", False)),
    )


def render_comparison(
    config: dict,
    map_params: GridMapParams,
    shape_params: MapGenParams,
    output_path: Path,
    seed: int,
    vertical_exaggeration: float,
) -> None:
    environment_options = dict(config.get("environment", {}))
    with tempfile.TemporaryDirectory(prefix="uav-environment-preview-") as temp_dir:
        temp_path = Path(temp_dir)
        thumbnails: list[tuple[str, np.ndarray]] = []

        for index, environment_type in enumerate(ENVIRONMENT_TYPES):
            started = time.perf_counter()
            print(f"[{index + 1}/{len(ENVIRONMENT_TYPES)}] Generating {environment_type}...")
            generator = StructMapGenerator()
            generator.init_params(copy.deepcopy(map_params))
            generator.mgpa = copy.deepcopy(shape_params)

            options = dict(environment_options)
            options["type"] = environment_type
            generator.generate_environment(options, seed + index, if_dyn=False)

            image_path = temp_path / f"{environment_type}.png"
            plot_environment_matplotlib(
                generator,
                title="",
                save_path=image_path,
                show=False,
                environment_type=environment_type,
                vertical_exaggeration=vertical_exaggeration,
            )
            with Image.open(image_path) as image:
                image.thumbnail((900, 675), Image.Resampling.LANCZOS)
                thumbnails.append((environment_type, np.asarray(image.convert("RGB")).copy()))

            del generator
            gc.collect()
            print(f"  Rendered in {time.perf_counter() - started:.1f}s")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for axis, (environment_type, image) in zip(axes.flat, thumbnails):
            axis.imshow(image)
            axis.set_title(environment_type.replace("_", " ").title())
            axis.axis("off")
        fig.suptitle(
            f"UAV environment previews ({map_params.basic_mp.map_size[0]:g} x "
            f"{map_params.basic_mp.map_size[1]:g} x {map_params.basic_mp.map_size[2]:g} m, "
            f"{map_params.resolution:g} m preview voxels)",
            fontsize=16,
        )
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate one comparison image for all six UAV environment types."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "structure_map.yaml",
        help="Map dimensions, environment options, and obstacle settings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "environment_types_preview.png",
        help="Path for the combined comparison image.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Preview-only voxel resolution in metres; 1.0 is recommended for large maps.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Base seed (defaults to params.seed).")
    parser.add_argument(
        "--vertical-exaggeration",
        type=float,
        default=1.0,
        help="Display-only Z multiplier.",
    )
    args = parser.parse_args()
    if args.resolution <= 0:
        parser.error("--resolution must be positive.")
    if args.vertical_exaggeration <= 0:
        parser.error("--vertical-exaggeration must be positive.")

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    config = load_config(config_path)
    resolution = float(args.resolution)
    seed = int(config["params"].get("seed", 1) if args.seed is None else args.seed)
    configured_resolution = float(config["map"]["resolution"])
    map_size = np.array(
        [config["map"]["x_size"], config["map"]["y_size"], config["map"]["z_size"]],
        dtype=float,
    )
    preview_voxels = int(np.prod(np.ceil(map_size / resolution).astype(int)))
    print(
        f"Preview resolution: {resolution:g} m (config resolution: "
        f"{configured_resolution:g} m); {preview_voxels:,} voxels per environment."
    )
    if resolution != configured_resolution:
        print("Using a coarser preview grid; config and production files are not modified.")

    started = time.perf_counter()
    render_comparison(
        config,
        build_map_params(config, resolution),
        build_shape_params(config),
        output_path,
        seed,
        args.vertical_exaggeration,
    )
    print(f"Saved comparison image: {output_path}")
    print(f"Total time: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
