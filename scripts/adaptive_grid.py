#!/usr/bin/env python3
"""Build and visualise adaptive octrees from saved UAV environment samples.

The stored point clouds represent occupied voxels at the resolution recorded
in ``metadata.jsonl``. Coarse cells are eight times that resolution and mixed
cells are bisected in each dimension three times, so the finest cells are
exactly one stored voxel wide (L/8). At each level, a cell is occupied at
80% or greater, empty at 0%, and mixed otherwise. Mixed cells at the finest
level remain non-traversable because the source data cannot resolve them any
further.

Usage from the repository root:
    python scripts/adaptive_grid.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENVIRONMENT_TYPES = ("structured", "blocks", "mountains", "random_land", "corridor", "mixed")
OCCUPIED_THRESHOLD = 0.8
SUBDIVISION_LEVELS = 3


@dataclass
class AdaptiveCell:
    """A terminal octree cell and the occupancy statistics used to classify it."""

    cell_id: int
    lower: np.ndarray
    upper: np.ndarray
    status: str
    occupancy_fraction: float
    depth: int
    index_lower: np.ndarray
    index_upper: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return (self.lower + self.upper) / 2.0


@dataclass
class AdaptiveGrid:
    """Adaptive terminal cells plus a native-resolution leaf lookup grid."""

    origin: np.ndarray
    size: np.ndarray
    resolution: float
    cells: list[AdaptiveCell]
    leaf_lookup: np.ndarray
    occupied_voxels: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.occupied_voxels.shape)

    def cell_at(self, position: np.ndarray) -> AdaptiveCell | None:
        """Return the terminal cell containing an in-bounds world coordinate."""
        point = np.asarray(position, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("A point must contain three finite coordinates.")
        if np.any(point < self.origin) or np.any(point >= self.origin + self.size):
            return None
        index = np.floor((point - self.origin) / self.resolution).astype(int)
        index = np.minimum(index, np.asarray(self.shape) - 1)
        cell_id = int(self.leaf_lookup[tuple(index)])
        return self.cells[cell_id] if cell_id >= 0 else None


def load_dataset_samples(dataset_dir: Path) -> tuple[list[dict], list[str]]:
    """Load the first metadata record and point cloud for each available type."""
    metadata_path = dataset_dir / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata was not found: {metadata_path}")

    selected: dict[str, dict] = {}
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line_number, line in enumerate(metadata_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {metadata_path}:{line_number}") from exc
            environment_type = str(record.get("type", ""))
            if environment_type in ENVIRONMENT_TYPES and environment_type not in selected:
                selected[environment_type] = record

    if not selected:
        raise ValueError(f"No supported environment samples were found in {metadata_path}")

    samples = []
    for environment_type in ENVIRONMENT_TYPES:
        record = selected.get(environment_type)
        if record is None:
            continue
        cloud_path = dataset_dir / record["npy"]
        if not cloud_path.is_file():
            raise FileNotFoundError(f"Saved point cloud is missing: {cloud_path}")
        points = np.load(cloud_path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError(f"Expected finite (N, 3) points in {cloud_path}")
        map_info = record.get("map", {})
        origin = np.asarray(map_info.get("origin"), dtype=float)
        size = np.asarray(map_info.get("size"), dtype=float)
        resolution = float(map_info.get("resolution", 0.0))
        if origin.shape != (3,) or size.shape != (3,) or np.any(size <= 0) or resolution <= 0:
            raise ValueError(f"Invalid map geometry in metadata record for {environment_type}")
        samples.append(
            {
                "record": record,
                "environment_type": environment_type,
                "points": points,
                "origin": origin,
                "size": size,
                "resolution": resolution,
            }
        )
    missing = [name for name in ENVIRONMENT_TYPES if name not in selected]
    return samples, missing


def occupancy_from_cloud(
    points: np.ndarray,
    origin: np.ndarray,
    size: np.ndarray,
    resolution: float,
) -> np.ndarray:
    """Rasterise saved obstacle points using the repository's floor indexing."""
    dims = np.ceil(size / resolution).astype(int)
    occupied = np.zeros(tuple(int(value) for value in dims), dtype=bool)
    if points.size == 0:
        return occupied
    indices = np.floor((points - origin) / resolution).astype(int)
    valid = np.all((indices >= 0) & (indices < dims), axis=1)
    if np.any(valid):
        occupied[tuple(indices[valid].T)] = True
    return occupied


def _cell_index_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    dims: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find base-voxel centres within a half-open cell's world-space bounds."""
    index_lower = np.ceil((lower - origin) / resolution - 0.5 - 1e-10).astype(int)
    index_upper = np.ceil((upper - origin) / resolution - 0.5 - 1e-10).astype(int)
    return np.maximum(index_lower, 0), np.minimum(index_upper, dims)


def build_adaptive_grid(
    points: np.ndarray,
    origin: np.ndarray,
    size: np.ndarray,
    resolution: float,
) -> AdaptiveGrid:
    """Build an octree, splitting mixed cells until their width is L/8."""
    occupied = occupancy_from_cloud(points, origin, size, resolution)
    dims = np.asarray(occupied.shape, dtype=int)
    root_width = resolution * (2**SUBDIVISION_LEVELS)
    leaf_lookup = np.full(occupied.shape, -1, dtype=np.int32)
    cells: list[AdaptiveCell] = []

    def visit(lower: np.ndarray, upper: np.ndarray, depth: int) -> None:
        clipped_lower = np.maximum(lower, origin)
        clipped_upper = np.minimum(upper, origin + size)
        if np.any(clipped_lower >= clipped_upper):
            return
        index_lower, index_upper = _cell_index_bounds(
            clipped_lower, clipped_upper, origin, resolution, dims
        )
        spans = index_upper - index_lower
        voxel_count = int(np.prod(spans))
        if voxel_count == 0:
            occupied_count = 0
        else:
            region = tuple(slice(index_lower[i], index_upper[i]) for i in range(3))
            occupied_count = int(np.count_nonzero(occupied[region]))
        fraction = occupied_count / voxel_count if voxel_count else 0.0

        if fraction == 0:
            status = "empty"
        elif fraction >= OCCUPIED_THRESHOLD:
            status = "occupied"
        elif depth < SUBDIVISION_LEVELS:
            midpoint = (lower + upper) / 2.0
            for x_side in (0, 1):
                for y_side in (0, 1):
                    for z_side in (0, 1):
                        side = np.array([x_side, y_side, z_side], dtype=bool)
                        child_lower = np.where(side, midpoint, lower)
                        child_upper = np.where(side, upper, midpoint)
                        visit(child_lower, child_upper, depth + 1)
            return
        else:
            # Conservatively treat unresolved mixed finest cells as blocked.
            status = "mixed"

        cell_id = len(cells)
        cell = AdaptiveCell(
            cell_id=cell_id,
            lower=clipped_lower.copy(),
            upper=clipped_upper.copy(),
            status=status,
            occupancy_fraction=fraction,
            depth=depth,
            index_lower=index_lower,
            index_upper=index_upper,
        )
        cells.append(cell)
        if voxel_count:
            region = tuple(slice(index_lower[i], index_upper[i]) for i in range(3))
            leaf_lookup[region] = cell_id

    root_counts = np.ceil(size / root_width).astype(int)
    for ix in range(int(root_counts[0])):
        for iy in range(int(root_counts[1])):
            for iz in range(int(root_counts[2])):
                lower = origin + np.array([ix, iy, iz], dtype=float) * root_width
                upper = lower + root_width
                visit(lower, upper, 0)

    if np.any(leaf_lookup < 0):
        missing = int(np.count_nonzero(leaf_lookup < 0))
        raise RuntimeError(f"Adaptive decomposition left {missing} base voxels unassigned.")
    return AdaptiveGrid(origin, size, resolution, cells, leaf_lookup, occupied)


def iter_grid_neighbors(grid: AdaptiveGrid) -> Iterator[tuple[int, int]]:
    """Yield unique free-cell pairs sharing a face in the base-voxel lookup."""
    free = np.fromiter(
        (cell.status == "empty" for cell in grid.cells),
        dtype=bool,
        count=len(grid.cells),
    )
    pairs = []
    for axis in range(3):
        lower_slice = [slice(None)] * 3
        upper_slice = [slice(None)] * 3
        lower_slice[axis] = slice(0, -1)
        upper_slice[axis] = slice(1, None)
        left = grid.leaf_lookup[tuple(lower_slice)]
        right = grid.leaf_lookup[tuple(upper_slice)]
        mask = (left >= 0) & (right >= 0) & (left != right)
        if np.any(mask):
            pair_batch = np.column_stack((left[mask], right[mask]))
            pair_batch.sort(axis=1)
            pairs.append(pair_batch)
    if not pairs:
        return
    unique_pairs = np.unique(np.concatenate(pairs, axis=0), axis=0)
    for left, right in unique_pairs:
        if free[left] and free[right]:
            yield int(left), int(right)


def grid_summary(grid: AdaptiveGrid) -> dict:
    """Return counts and finest-cell geometry for a saved decomposition."""
    counts = {state: 0 for state in ("empty", "occupied", "mixed")}
    for cell in grid.cells:
        counts[cell.status] += 1
    return {
        "cell_counts": counts,
        "total_terminal_cells": len(grid.cells),
        "finest_cell_side": grid.resolution,
        "coarse_cell_side": grid.resolution * (2**SUBDIVISION_LEVELS),
        "subdivision_levels": SUBDIVISION_LEVELS,
        "occupied_base_voxels": int(np.count_nonzero(grid.occupied_voxels)),
        "base_grid_shape": list(grid.shape),
    }


def save_adaptive_grid(grid: AdaptiveGrid, destination: Path) -> None:
    """Persist terminal cells and the base-voxel-to-cell lookup without pickle."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        origin=grid.origin,
        size=grid.size,
        resolution=np.asarray(grid.resolution),
        cell_lower=np.asarray([cell.lower for cell in grid.cells]),
        cell_upper=np.asarray([cell.upper for cell in grid.cells]),
        cell_status=np.asarray([cell.status for cell in grid.cells]),
        occupancy_fraction=np.asarray(
            [cell.occupancy_fraction for cell in grid.cells], dtype=float
        ),
        cell_depth=np.asarray([cell.depth for cell in grid.cells], dtype=np.int8),
        index_lower=np.asarray([cell.index_lower for cell in grid.cells], dtype=np.int32),
        index_upper=np.asarray([cell.index_upper for cell in grid.cells], dtype=np.int32),
        leaf_lookup=grid.leaf_lookup,
        occupied_voxels=grid.occupied_voxels,
    )


def load_adaptive_grid(source: Path) -> AdaptiveGrid:
    """Load a previously saved adaptive grid using only non-pickled arrays."""
    if not source.is_file():
        raise FileNotFoundError(f"Saved adaptive grid was not found: {source}")
    with np.load(source, allow_pickle=False) as data:
        required = {
            "origin",
            "size",
            "resolution",
            "cell_lower",
            "cell_upper",
            "cell_status",
            "occupancy_fraction",
            "cell_depth",
            "index_lower",
            "index_upper",
            "leaf_lookup",
            "occupied_voxels",
        }
        if not required.issubset(data.files):
            raise ValueError(f"Saved adaptive grid is incomplete: {source}")
        arrays = {name: data[name] for name in required}

    cell_count = len(arrays["cell_status"])
    if any(
        len(arrays[name]) != cell_count
        for name in (
            "cell_lower",
            "cell_upper",
            "occupancy_fraction",
            "cell_depth",
            "index_lower",
            "index_upper",
        )
    ):
        raise ValueError(f"Saved adaptive grid has inconsistent cell arrays: {source}")
    cells = [
        AdaptiveCell(
            cell_id=cell_id,
            lower=arrays["cell_lower"][cell_id],
            upper=arrays["cell_upper"][cell_id],
            status=str(arrays["cell_status"][cell_id]),
            occupancy_fraction=float(arrays["occupancy_fraction"][cell_id]),
            depth=int(arrays["cell_depth"][cell_id]),
            index_lower=arrays["index_lower"][cell_id],
            index_upper=arrays["index_upper"][cell_id],
        )
        for cell_id in range(cell_count)
    ]
    return AdaptiveGrid(
        origin=arrays["origin"],
        size=arrays["size"],
        resolution=float(arrays["resolution"]),
        cells=cells,
        leaf_lookup=arrays["leaf_lookup"],
        occupied_voxels=arrays["occupied_voxels"],
    )


def save_grid_preview(grid: AdaptiveGrid, environment_type: str, destination: Path) -> None:
    """Save a labelled XY cross-section through the adaptive 3D cells."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    z_plane = grid.origin[2] + grid.size[2] / 2.0
    colours = {"empty": "#e7f2e7", "occupied": "#e9a3a3", "mixed": "#f3d27a"}
    patches = []
    face_colours = []
    for cell in grid.cells:
        if cell.lower[2] <= z_plane < cell.upper[2]:
            patches.append(
                Rectangle(
                    (cell.lower[0], cell.lower[1]),
                    cell.upper[0] - cell.lower[0],
                    cell.upper[1] - cell.lower[1],
                )
            )
            face_colours.append(colours[cell.status])

    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    collection = PatchCollection(
        patches,
        facecolor=face_colours,
        edgecolor="#263238",
        linewidth=0.25,
    )
    axis.add_collection(collection)
    axis.set_xlim(grid.origin[0], grid.origin[0] + grid.size[0])
    axis.set_ylim(grid.origin[1], grid.origin[1] + grid.size[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_title(
        f"Adaptive grid: {environment_type} (XY slice at Z={z_plane:.2f} m)"
    )
    axis.grid(False)
    from matplotlib.patches import Patch

    axis.legend(
        handles=[Patch(facecolor=colours[key], edgecolor="#263238", label=key.title())
                 for key in ("empty", "occupied", "mixed")],
        loc="upper right",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "outputs" / "uav_dataset",
        help="Existing UAV dataset folder containing metadata.jsonl and point_clouds/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "adaptive_grid",
        help="Folder for per-type grid summaries and slice images.",
    )
    args = parser.parse_args()
    samples, missing = load_dataset_samples(args.dataset_dir)
    if missing:
        print("No saved sample for environment type(s): " + ", ".join(missing))

    for sample in samples:
        environment_type = sample["environment_type"]
        grid = build_adaptive_grid(
            sample["points"], sample["origin"], sample["size"], sample["resolution"]
        )
        sample_dir = args.output_dir / environment_type
        sample_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "environment_type": environment_type,
            "dataset_index": sample["record"]["index"],
            "source_npy": sample["record"]["npy"],
            **grid_summary(grid),
        }
        summary_path = sample_dir / "grid_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        grid_path = sample_dir / "adaptive_grid.npz"
        save_adaptive_grid(grid, grid_path)
        image_path = sample_dir / "adaptive_grid.png"
        save_grid_preview(grid, environment_type, image_path)
        print(
            f"{environment_type}: {summary['total_terminal_cells']} terminal cells "
            f"(empty={summary['cell_counts']['empty']}, "
            f"occupied={summary['cell_counts']['occupied']}, "
            f"mixed={summary['cell_counts']['mixed']}) -> {grid_path}"
        )


if __name__ == "__main__":
    main()
