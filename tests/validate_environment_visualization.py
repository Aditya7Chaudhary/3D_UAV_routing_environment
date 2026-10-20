#!/usr/bin/env python3
"""Generate, render, and validate representative UAV environment families.

This is intentionally an executable validation tool rather than a unit test:
it produces baseline point-cloud images and improved occupancy-surface images
side by side without touching a production dataset.  Every improved polygon
is derived from the generator's existing ``GridMap`` occupancy, so rendering
cannot alter planning or collision data.

Run from the repository root:

    python tests/validate_environment_visualization.py
    python tests/validate_environment_visualization.py --seed 42 --no-mixed
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.grid_map import BasicMapParams, GridMapParams
from uav_env.struct_map_gen import MapGenParams, StructMapGenerator
from uav_env.visualize import (
    plot_cloud_matplotlib,
    plot_environment_matplotlib,
    surface_bounds,
    voxel_surface_faces,
)


PRIMARY_TYPES = ("structured", "blocks", "random_land", "mountains", "corridor")


def load_config(path: Path) -> dict:
    """Read the existing YAML configuration without changing its format."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_map_params(config: dict) -> GridMapParams:
    """Create the repository's normal grid parameters from its YAML map block."""
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
        resolution=float(map_cfg["resolution"]),
    )


def build_shape_params(config: dict) -> MapGenParams:
    """Create normal structured-obstacle parameters from the existing YAML."""
    params = config["params"]
    return MapGenParams(
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


def validation_options(config: dict, environment_type: str) -> dict:
    """Use one representative, supported option set for an environment family."""
    env = dict(config.get("environment", {}))
    env["type"] = environment_type
    return env


def make_generator(grid_params: GridMapParams, shape_params: MapGenParams) -> StructMapGenerator:
    """Construct a fresh generator so repeatability checks are independent."""
    generator = StructMapGenerator()
    generator.init_params(copy.deepcopy(grid_params))
    generator.mgpa = copy.deepcopy(shape_params)
    return generator


def occupied_mask(generator: StructMapGenerator) -> np.ndarray:
    """Return the generator's authoritative occupied voxels as a Boolean grid."""
    grid = generator.grid_map
    return grid.occupancy.reshape(grid.mp.map_grid_size) > grid.mp.min_thrd


def voxel_extent(mask: np.ndarray, origin: np.ndarray, resolution: float):
    """Return the physical min/max corners of all occupied source voxels."""
    indices = np.argwhere(mask)
    if indices.size == 0:
        return None
    lower = np.asarray(origin, dtype=float) + indices.min(axis=0) * resolution
    upper = np.asarray(origin, dtype=float) + (indices.max(axis=0) + 1) * resolution
    return lower, upper


def terrain_statistics(mask: np.ndarray, origin: np.ndarray, resolution: float) -> dict:
    """Report column-top elevations inferred directly from occupied voxels."""
    indices = np.argwhere(mask)
    if indices.size == 0:
        return {"occupied_columns": 0}
    tops = np.full(mask.shape[:2], -1, dtype=int)
    np.maximum.at(tops, (indices[:, 0], indices[:, 1]), indices[:, 2])
    valid = tops >= 0
    elevations = origin[2] + (tops[valid] + 1) * resolution
    return {
        "occupied_columns": int(valid.sum()),
        "surface_z_min": float(elevations.min()),
        "surface_z_max": float(elevations.max()),
        "surface_z_mean": float(elevations.mean()),
    }


def primitive_voxel_mask(generator: StructMapGenerator, full_mask: np.ndarray) -> np.ndarray:
    """Return cells contributed by existing rasterised parametric shapes."""
    primitive = np.zeros_like(full_mask)
    grid = generator.grid_map
    for name in ("cylinder", "circle_gate", "rect_gate", "ellipsoid", "polyhedron"):
        for shape in getattr(generator.geo_map, name):
            if not shape.cloud:
                continue
            points = np.asarray(shape.cloud, dtype=float)
            indices = np.array([grid.pos_to_index(point) for point in points])
            valid = np.all((indices >= 0) & (indices < grid.mp.map_grid_size), axis=1)
            if np.any(valid):
                primitive[tuple(indices[valid].T)] = True
    return primitive & full_mask


def height_field_collision_compatibility(
    generator: StructMapGenerator, mask: np.ndarray
) -> tuple[bool, dict]:
    """Check that a stored visual height field agrees with voxel sampling.

    A continuous terrain surface can lie within one voxel of the collision
    boundary because occupancy is sampled at voxel centres.  This routine makes
    that limitation explicit instead of claiming bit-exact mesh correspondence.
    """
    field = generator.terrain_height_field
    if field is None:
        return True, {"available": False}
    xs, ys, heights = field
    grid = generator.grid_map
    res = grid.mp.resolution
    z0 = grid.mp.basic_mp.map_origin[2]
    if heights.shape != mask.shape[:2] or len(xs) != mask.shape[0] or len(ys) != mask.shape[1]:
        return False, {"available": True, "reason": "height-field dimensions differ from grid"}
    tops = np.full(mask.shape[:2], -1, dtype=int)
    indices = np.argwhere(mask)
    np.maximum.at(tops, (indices[:, 0], indices[:, 1]), indices[:, 2])
    occupied_columns = tops >= 0
    top_centres = z0 + (tops[occupied_columns] + 0.5) * res
    occupied_ok = bool(
        np.all(heights[occupied_columns] >= top_centres - 1e-9)
        and np.all(heights[occupied_columns] < top_centres + res + 1e-9)
    )
    first_centre = z0 + 0.5 * res
    empty_ok = bool(np.all(heights[~occupied_columns] < first_centre + 1e-9))
    offsets = np.abs(heights[occupied_columns] - (top_centres + 0.5 * res))
    return occupied_ok and empty_ok, {
        "available": True,
        "source": "generator_height_field",
        "max_distance_to_voxel_top_m": float(offsets.max()) if offsets.size else 0.0,
        "sampling_tolerance_m": float(res),
    }


def shape_statistics(generator: StructMapGenerator) -> dict:
    """Report actual registered parametric obstacles, without creating new ones."""
    result = {}
    for name in ("cylinder", "circle_gate", "rect_gate", "ellipsoid", "polyhedron"):
        shapes = getattr(generator.geo_map, name)
        result[name] = {
            "count": len(shapes),
            "voxel_points": int(sum(len(shape.cloud) for shape in shapes)),
        }
    return result


def reproducible_probe_points(generator: StructMapGenerator, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Choose occupied/free points established by the source occupancy grid."""
    grid = generator.grid_map
    occupied_indices = np.argwhere(mask)
    free_indices = np.argwhere(~mask)
    if occupied_indices.size == 0 or free_indices.size == 0:
        raise ValueError("Validation needs both occupied and free source voxels.")
    # Stable index positions avoid treating arbitrary world points as truth.
    def pick(indices: np.ndarray) -> np.ndarray:
        positions = np.linspace(0, len(indices) - 1, num=min(3, len(indices)), dtype=int)
        return np.array([grid.index_to_pos(indices[position]) for position in positions])

    return pick(occupied_indices), pick(free_indices)


def corridor_opening_probe(generator: StructMapGenerator, mask: np.ndarray) -> np.ndarray | None:
    """Find a free cell in an actual partition opening, if the source has one."""
    # A partition plane has both occupied and free interior cells at ground
    # level.  The selected free voxel is consequently established by the map,
    # not guessed from a drawing or from hard-coded coordinates.
    for x in range(mask.shape[0]):
        row = mask[x, :, 0]
        # Find a contiguous free run bounded by occupied cells on both sides.
        # This identifies a partition gap, not the broad free area outside the
        # two long corridor walls.
        free = ~row
        starts = np.flatnonzero(free & np.r_[True, ~free[:-1]])
        ends = np.flatnonzero(free & np.r_[~free[1:], True])
        for start, end in zip(starts, ends):
            if start > 0 and end < len(row) - 1 and row[start - 1] and row[end + 1]:
                y = (int(start) + int(end)) // 2
                return generator.grid_map.index_to_pos(np.array([x, y, 0]))
    return None


def validate_one(
    environment_type: str,
    seed: int,
    options: dict,
    grid_params: GridMapParams,
    shape_params: MapGenParams,
    output_dir: Path,
    vertical_exaggeration: float,
) -> dict:
    """Generate one map, render it twice, and run geometry/collision checks."""
    started = time.perf_counter()
    result: dict = {
        "type": environment_type,
        "seed": seed,
        "options": options,
        "checks": {},
        "warnings": [],
        "status": "fail",
    }
    try:
        generator = make_generator(grid_params, shape_params)
        generator.generate_environment(options, seed, if_dyn=False)
        cloud = generator.get_pc()
        mask = occupied_mask(generator)
        grid = generator.grid_map
        origin = grid.mp.basic_mp.map_origin
        maximum = grid.mp.basic_mp.max_range
        resolution = grid.mp.resolution

        # Geometry sanity uses source points and the source grid directly.
        finite = bool(np.isfinite(cloud).all())
        in_bounds = bool(
            np.all(cloud >= origin - 1e-9) and np.all(cloud < maximum + 1e-9)
        )
        non_empty = bool(len(cloud) > 0 and np.any(mask))
        point_cells_occupied = bool(all(grid.is_occ(point) == 1 for point in cloud))
        result["checks"].update(
            finite_coordinates=finite,
            non_empty=non_empty,
            source_points_in_bounds=in_bounds,
            source_points_marked_occupied=point_cells_occupied,
        )

        # Rendering consistency: the mesh is built from the same Boolean
        # occupancy array and must retain the exact exterior voxel extent.
        faces, _kinds = voxel_surface_faces(mask, origin, resolution)
        source_extent = voxel_extent(mask, origin, resolution)
        mesh_extent = surface_bounds(faces)
        extents_match = bool(
            source_extent is not None
            and mesh_extent is not None
            and np.allclose(source_extent[0], mesh_extent[0])
            and np.allclose(source_extent[1], mesh_extent[1])
        )
        result["checks"].update(
            occupancy_surface_source="exposed_faces_of_authoritative_occupancy_voxels",
            surface_has_faces=bool(len(faces) > 0),
            surface_faces_non_degenerate=bool(
                len(faces) > 0
                and np.all(
                    np.linalg.norm(
                        np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1
                    )
                    > 0.0
                )
            ),
            surface_extent_matches_occupied_voxels=extents_match,
        )
        terrain_mask = mask
        if environment_type == "mixed":
            # A primitive may sit above a terrain column.  Its top must not be
            # mistaken for the height-field terrain surface during validation.
            terrain_mask = mask & ~primitive_voxel_mask(generator, mask)
        terrain_compatible, terrain_visual = height_field_collision_compatibility(generator, terrain_mask)
        if terrain_visual["available"]:
            result["checks"]["terrain_height_field_matches_voxel_sampling"] = terrain_compatible
            result["terrain_visual_surface"] = terrain_visual
            result["warnings"].append(
                "Terrain rendering uses the generator's continuous height field; "
                "the collision boundary remains voxel-sampled within one resolution cell."
            )

        occupied_probes, free_probes = reproducible_probe_points(generator, mask)
        before = [grid.is_occ(point) for point in np.vstack((occupied_probes, free_probes))]
        opening_probe = corridor_opening_probe(generator, mask) if environment_type == "corridor" else None
        opening_is_free = (
            opening_probe is not None and grid.is_occ(opening_probe) == 0
            if environment_type == "corridor"
            else True
        )

        baseline_path = output_dir / "baseline" / f"{environment_type}.png"
        improved_path = output_dir / "improved" / f"{environment_type}.png"
        root_image_path = output_dir / f"{environment_type}.png"
        # This is deliberately the unchanged scatter renderer used for the
        # original dataset previews.
        plot_cloud_matplotlib(
            cloud,
            title=f"Baseline point cloud: {environment_type}, seed {seed}",
            save_path=baseline_path,
            show=False,
        )
        face_counts = plot_environment_matplotlib(
            generator,
            title=f"{environment_type.replace('_', ' ').title()} — seed {seed}",
            save_path=improved_path,
            show=False,
            environment_type=environment_type,
            vertical_exaggeration=vertical_exaggeration,
        )
        # Keep the required top-level images convenient while preserving the
        # baseline/improved comparison directories.
        # The top-level required image is byte-for-byte the improved render;
        # copying avoids a second expensive projection of the same geometry.
        shutil.copy2(improved_path, root_image_path)
        after = [grid.is_occ(point) for point in np.vstack((occupied_probes, free_probes))]
        collision_unchanged = before == after
        probes_correct = before[: len(occupied_probes)] == [1] * len(occupied_probes) and before[
            len(occupied_probes) :
        ] == [0] * len(free_probes)
        result["checks"].update(
            collision_probes_match_source=probes_correct,
            visualization_did_not_change_collision=collision_unchanged,
            corridor_opening_probe_free=opening_is_free,
        )
        if environment_type == "corridor" and opening_probe is None:
            result["warnings"].append("No flanked ground-level partition opening was found.")

        # Determinism is checked by independently rebuilding the map.
        repeated = make_generator(grid_params, shape_params)
        repeated.generate_environment(options, seed, if_dyn=False)
        result["checks"]["fixed_seed_reproducible"] = bool(
            np.array_equal(cloud, repeated.get_pc())
            and np.array_equal(mask, occupied_mask(repeated))
        )
        result.update(
            point_count=int(len(cloud)),
            occupied_voxels=int(mask.sum()),
            source_voxel_bounds={
                "min": source_extent[0].tolist() if source_extent else None,
                "max": source_extent[1].tolist() if source_extent else None,
            },
            mesh_voxel_bounds={
                "min": mesh_extent[0].tolist() if mesh_extent else None,
                "max": mesh_extent[1].tolist() if mesh_extent else None,
            },
            terrain=terrain_statistics(mask, origin, resolution),
            shape_sources=shape_statistics(generator),
            surface_face_counts={name: int(count) for name, count in face_counts.items()},
            images={
                "baseline": str(baseline_path.relative_to(output_dir)),
                "improved": str(improved_path.relative_to(output_dir)),
                "primary": str(root_image_path.relative_to(output_dir)),
            },
        )
        required = [value for value in result["checks"].values() if isinstance(value, bool)]
        result["status"] = "pass" if all(required) else "fail"
    except Exception as exc:  # Continue with the other environment families.
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result


def save_comparison_grid(records: list[dict], output_dir: Path) -> Path | None:
    """Write a labelled grid from the generated improved images."""
    successful = [record for record in records if record.get("status") == "pass"]
    if not successful:
        return None
    import matplotlib.pyplot as plt

    columns = 3
    rows = int(np.ceil(len(successful) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 5 * rows))
    axes = np.atleast_1d(axes).reshape((-1,))
    for axis, record in zip(axes, successful):
        image = plt.imread(output_dir / record["images"]["primary"])
        axis.imshow(image)
        axis.set_title(f"{record['type'].replace('_', ' ').title()}\nseed {record['seed']}")
        axis.axis("off")
    for axis in axes[len(successful) :]:
        axis.axis("off")
    fig.suptitle("UAV environment visualization validation", fontsize=16)
    fig.tight_layout()
    path = output_dir / "comparison_grid.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_text_report(report: dict, path: Path) -> None:
    """Write a compact human-readable companion to the JSON report."""
    lines = ["UAV environment visualization validation", "=" * 42]
    lines.append(f"Overall status: {report['overall_status'].upper()}")
    lines.append(f"Seed: {report['seed']}")
    lines.append(f"Output directory: {report['output_directory']}")
    lines.append("")
    for record in report["environments"]:
        lines.append(
            f"{record['type']}: {record['status'].upper()} "
            f"({record.get('point_count', 0)} points, {record.get('elapsed_seconds', 0):.3f}s)"
        )
        if "error" in record:
            lines.append(f"  error: {record['error']}")
        else:
            failed = [name for name, passed in record["checks"].items() if passed is False]
            lines.append("  checks: " + ("all passed" if not failed else "failed: " + ", ".join(failed)))
            for warning in record["warnings"]:
                lines.append(f"  warning: {warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate faithful UAV environment visualization.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "structure_map.yaml")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed; each environment uses seed + index.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "visualization_validation",
        help="Dedicated, non-production validation output directory.",
    )
    parser.add_argument("--no-mixed", action="store_true", help="Skip the mixed-environment integration sample.")
    parser.add_argument(
        "--vertical-exaggeration",
        type=float,
        default=1.0,
        help="Display-only Z multiplier; occupancy and reports stay in physical metres.",
    )
    args = parser.parse_args()
    if args.vertical_exaggeration <= 0:
        parser.error("--vertical-exaggeration must be positive.")

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    config = load_config(config_path)
    grid_params = build_map_params(config)
    shape_params = build_shape_params(config)
    environment_types = list(PRIMARY_TYPES)
    if not args.no_mixed:
        environment_types.append("mixed")

    records = []
    for index, environment_type in enumerate(environment_types):
        record = validate_one(
            environment_type=environment_type,
            seed=args.seed + index,
            options=validation_options(config, environment_type),
            grid_params=grid_params,
            shape_params=shape_params,
            output_dir=output_dir,
            vertical_exaggeration=args.vertical_exaggeration,
        )
        records.append(record)
        print(
            f"[{record['status'].upper()}] {environment_type} "
            f"({record.get('elapsed_seconds', 0):.3f}s)"
        )

    comparison = save_comparison_grid(records, output_dir)
    overall_pass = all(record["status"] == "pass" for record in records)
    report = {
        "overall_status": "pass" if overall_pass else "fail",
        "seed": args.seed,
        "config": str(config_path),
        "output_directory": str(output_dir),
        "vertical_exaggeration_display_only": args.vertical_exaggeration,
        "mixed_enabled": not args.no_mixed,
        "comparison_grid": str(comparison.relative_to(output_dir)) if comparison else None,
        "environments": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_text_report(report, output_dir / "validation_report.txt")
    print(f"Overall validation: {report['overall_status'].upper()} -> {output_dir}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
