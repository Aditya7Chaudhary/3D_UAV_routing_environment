"""Voxelize a PCD and export a merged box mesh (OBJ). Optional Gazebo model files."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


def pcd_to_voxel_grid(pcd_file: str | Path, voxel_size: float = 0.2):
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(pcd_file))
    if pcd.is_empty():
        raise ValueError("Input PCD is empty!")
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    idxs = [v.grid_index for v in vg.get_voxels()]
    voxels = np.asarray(idxs, dtype=int)
    origin = np.array(vg.origin, dtype=float)
    return voxels, voxel_size, origin


def runs_in_row(row):
    runs = []
    in_run = False
    start = 0
    for x, val in enumerate(row):
        if val and not in_run:
            in_run = True
            start = x
        elif not val and in_run:
            runs.append((start, x - 1))
            in_run = False
    if in_run:
        runs.append((start, len(row) - 1))
    return runs


def greedy_xy_merge_rectangles(binary_grid):
    h, _w = binary_grid.shape
    active = {}
    rects = []
    for y in range(h):
        row_runs = runs_in_row(binary_grid[y])
        next_active = {}
        used_prev = set()
        for x1, x2 in row_runs:
            if (x1, x2) in active:
                y0, _ = active[(x1, x2)]
                next_active[(x1, x2)] = [y0, y]
                used_prev.add((x1, x2))
            else:
                next_active[(x1, x2)] = [y, y]
        for key, (y0, y1_cur) in active.items():
            if key not in used_prev:
                rects.append(((key[0], y0), (key[1], y1_cur)))
        active = next_active
    for key, (y0, y1_cur) in active.items():
        rects.append(((key[0], y0), (key[1], y1_cur)))
    return rects


def rectangles_per_z(voxels):
    by_z = defaultdict(list)
    for x, y, z in voxels:
        by_z[z].append((x, y))
    rects_z = {}
    for z, xy_list in by_z.items():
        xs = [x for x, _ in xy_list]
        ys = [y for _, y in xy_list]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        grid = np.zeros((h, w), dtype=bool)
        for x, y in xy_list:
            grid[y - y_min, x - x_min] = True
        rects = greedy_xy_merge_rectangles(grid)
        rects_abs = []
        for (x1, y1), (x2, y2) in rects:
            rects_abs.append(((x1 + x_min, y1 + y_min), (x2 + x_min, y2 + y_min)))
        rects_z[z] = rects_abs
    return rects_z


def boxes_to_mesh_from_rectangles(rects_z, voxel_size, origin):
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    for z, rects in rects_z.items():
        for (x1, y1), (x2, y2) in rects:
            nx = x2 - x1 + 1
            ny = y2 - y1 + 1
            size = np.array([nx, ny, 1.0], dtype=float) * voxel_size
            min_corner = origin + np.array([x1, y1, z], dtype=float) * voxel_size
            box = o3d.geometry.TriangleMesh.create_box(*size)
            box.translate(min_corner)
            mesh += box
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    return mesh


def save_optional_gazebo_model(mesh, model_name: str, output_dir: str | Path):
    """Write OBJ + optional Gazebo model descriptors (not required to run this repo)."""
    import open3d as o3d

    output_dir = Path(output_dir)
    model_path = output_dir / model_name
    mesh_dir = model_path / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_file = mesh_dir / "map.obj"
    o3d.io.write_triangle_mesh(str(mesh_file), mesh, write_vertex_normals=False)
    (model_path / "model.config").write_text(
        f"""<?xml version="1.0" ?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author><name>Auto</name></author>
  <description>Greedy XY merged voxel mesh</description>
</model>
""",
        encoding="utf-8",
    )
    (model_path / "model.sdf").write_text(
        f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry><mesh><uri>model://{model_name}/meshes/map.obj</uri></mesh></geometry>
      </collision>
      <visual name="visual">
        <geometry><mesh><uri>model://{model_name}/meshes/map.obj</uri></mesh></geometry>
      </visual>
    </link>
  </model>
</sdf>
""",
        encoding="utf-8",
    )
    return mesh_file


def pcd_to_mesh(
    pcd_file: str | Path,
    output_obj: str | Path,
    voxel_size: float = 0.2,
    write_gazebo: bool = False,
    model_name: str = "map",
):
    import open3d as o3d

    voxels, vs, origin = pcd_to_voxel_grid(pcd_file, voxel_size=voxel_size)
    if voxels.size == 0:
        raise ValueError("No occupied voxels after voxelization.")
    rects_z = rectangles_per_z(voxels)
    mesh = boxes_to_mesh_from_rectangles(rects_z, vs, origin)
    output_obj = Path(output_obj)
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_obj), mesh, write_vertex_normals=False)
    print(f"[INFO] OBJ saved: {output_obj}")
    if write_gazebo:
        save_optional_gazebo_model(mesh, model_name, output_obj.parent)
    return mesh
