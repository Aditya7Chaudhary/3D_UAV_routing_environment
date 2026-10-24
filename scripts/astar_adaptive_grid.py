#!/usr/bin/env python3
"""Run A* on adaptive grids for one saved sample of each available type.

Usage from the repository root:
    python scripts/astar_adaptive_grid.py
    python scripts/astar_adaptive_grid.py --start 1 1 1 --goal 28 28 5
"""

from __future__ import annotations

import argparse
import heapq
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_grid import (
    AdaptiveGrid,
    iter_grid_neighbors,
    load_adaptive_grid,
    load_dataset_samples,
)


def build_graph(grid: AdaptiveGrid) -> list[list[tuple[int, float]]]:
    """Build a weighted 6-connected graph over free terminal cells."""
    adjacency: list[list[tuple[int, float]]] = [[] for _ in grid.cells]
    for left, right in iter_grid_neighbors(grid):
        distance = float(np.linalg.norm(grid.cells[left].center - grid.cells[right].center))
        adjacency[left].append((right, distance))
        adjacency[right].append((left, distance))
    return adjacency


def astar(
    grid: AdaptiveGrid,
    adjacency: list[list[tuple[int, float]]],
    start: np.ndarray,
    goal: np.ndarray,
) -> list[int]:
    """Find the minimum-cost free-cell route between two world-space points."""
    start_cell = grid.cell_at(start)
    goal_cell = grid.cell_at(goal)
    if start_cell is None or goal_cell is None:
        raise ValueError("Start and goal must lie inside the map.")
    if start_cell.status != "empty":
        raise ValueError(f"Start lies in a {start_cell.status} cell.")
    if goal_cell.status != "empty":
        raise ValueError(f"Goal lies in a {goal_cell.status} cell.")

    start_id, goal_id = start_cell.cell_id, goal_cell.cell_id
    distances = {start_id: 0.0}
    predecessors: dict[int, int] = {}
    open_heap = [
        (float(np.linalg.norm(start_cell.center - goal_cell.center)), 0.0, start_id)
    ]

    while open_heap:
        _, cost_so_far, current = heapq.heappop(open_heap)
        if cost_so_far > distances.get(current, math.inf):
            continue
        if current == goal_id:
            route = [current]
            while current in predecessors:
                current = predecessors[current]
                route.append(current)
            route.reverse()
            return route

        for neighbour, edge_cost in adjacency[current]:
            next_cost = cost_so_far + edge_cost
            if next_cost < distances.get(neighbour, math.inf):
                distances[neighbour] = next_cost
                predecessors[neighbour] = current
                heuristic = float(
                    np.linalg.norm(grid.cells[neighbour].center - goal_cell.center)
                )
                heapq.heappush(
                    open_heap, (next_cost + heuristic, next_cost, neighbour)
                )
    raise ValueError("No path exists between the requested points.")


def connected_components(
    adjacency: list[list[tuple[int, float]]],
    grid: AdaptiveGrid,
) -> list[list[int]]:
    """Return graph components, excluding terminal obstacle cells."""
    components = []
    visited: set[int] = set()
    for node_id, neighbours in enumerate(adjacency):
        if not neighbours or node_id in visited:
            continue
        component = []
        queue = deque([node_id])
        visited.add(node_id)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbour, _ in adjacency[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        components.append(component)
    isolated_free = [
        node_id
        for node_id, cell in enumerate(grid.cells)
        if cell.status == "empty" and node_id not in visited
    ]
    components.extend([[node_id] for node_id in isolated_free])
    return components


def choose_distant_endpoints(
    grid: AdaptiveGrid,
    adjacency: list[list[tuple[int, float]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Choose distant free cell centres in the largest connected component."""
    components = connected_components(adjacency, grid)
    if not components:
        raise ValueError("The environment has no free cells for path planning.")
    component = max(components, key=len)
    if len(component) == 1:
        raise ValueError("The largest free-space component contains only one cell.")

    centers = np.asarray([grid.cells[node_id].center for node_id in component])
    first = int(np.argmax(np.linalg.norm(centers - centers.mean(axis=0), axis=1)))
    second = int(np.argmax(np.linalg.norm(centers - centers[first], axis=1)))
    return centers[first], centers[second]


def render_path(
    points: np.ndarray,
    grid: AdaptiveGrid,
    route: list[int],
    start: np.ndarray,
    goal: np.ndarray,
    title: str,
    destination: Path,
) -> None:
    """Save a 3D view of the stored environment with the A* path overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    route_points = np.asarray([grid.cells[node_id].center for node_id in route])
    figure = plt.figure(figsize=(11, 9), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    display_points = points
    if len(points) > 60000:
        sample_indices = np.linspace(0, len(points) - 1, 60000, dtype=int)
        display_points = points[sample_indices]
    axis.scatter(
        display_points[:, 0],
        display_points[:, 1],
        display_points[:, 2],
        s=1.0,
        c="#59636e",
        alpha=0.28,
        depthshade=False,
    )
    axis.plot(
        route_points[:, 0],
        route_points[:, 1],
        route_points[:, 2],
        color="#e31a1c",
        linewidth=2.6,
        label="A* route",
    )
    axis.scatter(*start, color="#16a34a", s=70, marker="o", label="A (start)")
    axis.scatter(*goal, color="#2563eb", s=80, marker="*", label="B (goal)")
    axis.set_xlim(grid.origin[0], grid.origin[0] + grid.size[0])
    axis.set_ylim(grid.origin[1], grid.origin[1] + grid.size[1])
    axis.set_zlim(grid.origin[2], grid.origin[2] + grid.size[2])
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title(title)
    axis.legend(loc="upper left")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def parse_point(raw_point: list[float] | None) -> np.ndarray | None:
    if raw_point is None:
        return None
    point = np.asarray(raw_point, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("Each start/goal point must contain three finite coordinates.")
    return point


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "outputs" / "uav_dataset",
        help="Existing UAV dataset folder; this script never generates environments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "adaptive_grid_paths",
        help="Folder for route images.",
    )
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=ROOT / "outputs" / "adaptive_grid",
        help="Folder containing networks created by scripts/adaptive_grid.py.",
    )
    parser.add_argument(
        "--start",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Optional world-space start point (must lie in free space).",
    )
    parser.add_argument(
        "--goal",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Optional world-space goal point (must lie in free space).",
    )
    args = parser.parse_args()
    if (args.start is None) != (args.goal is None):
        parser.error("--start and --goal must be supplied together.")

    samples, missing = load_dataset_samples(args.dataset_dir)
    if missing:
        print("No saved sample for environment type(s): " + ", ".join(missing))

    requested_start = parse_point(args.start)
    requested_goal = parse_point(args.goal)
    for sample in samples:
        environment_type = sample["environment_type"]
        grid_path = args.grid_dir / environment_type / "adaptive_grid.npz"
        grid = load_adaptive_grid(grid_path)
        if (
            not np.allclose(grid.origin, sample["origin"])
            or not np.allclose(grid.size, sample["size"])
            or not np.isclose(grid.resolution, sample["resolution"])
        ):
            raise ValueError(
                f"Saved grid geometry does not match dataset metadata for {environment_type}: "
                f"{grid_path}"
            )
        adjacency = build_graph(grid)
        if requested_start is None:
            start, goal = choose_distant_endpoints(grid, adjacency)
        else:
            start, goal = requested_start.copy(), requested_goal.copy()
        route = astar(grid, adjacency, start, goal)
        destination = args.output_dir / f"{environment_type}_astar_path.png"
        cost = sum(
            float(np.linalg.norm(grid.cells[left].center - grid.cells[right].center))
            for left, right in zip(route, route[1:])
        )
        render_path(
            sample["points"],
            grid,
            route,
            start,
            goal,
            f"A* path: {environment_type} "
            f"({len(route)} cells, graph cost {cost:.2f} m)",
            destination,
        )
        print(
            f"{environment_type}: {len(route)} cells, graph cost {cost:.2f} m "
            f"-> {destination}"
        )


if __name__ == "__main__":
    main()
