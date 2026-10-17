"""
===============================================================================
3D UAV ROUTING ENVIRONMENT — POINT CLOUD TO MESH EXPORT
===============================================================================

PURPOSE
-------
This module turns an occupied point cloud into a triangle-mesh obstacle map.
It first voxelises the cloud, then merges adjacent occupied voxels into
rectangles within each horizontal X-Y layer.  One box is created for each
merged rectangle and the boxes together are exported as an OBJ file.

WHY MERGE VOXELS
----------------
Creating a separate cube for every occupied point produces an unnecessarily
large mesh.  Merging same-row runs and then matching runs in adjacent rows
preserves exact voxel occupancy while reducing the number of boxes and faces.

Gazebo support is optional.  When requested, the exported OBJ is accompanied
by minimal ``model.config`` and ``model.sdf`` descriptor files.
===============================================================================
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


def pcd_to_voxel_grid(pcd_file: str | Path, voxel_size: float = 0.2):
    """Load and voxelise a PCD using Open3D.

    Returns:
        ``(voxels, voxel_size, origin)`` where ``voxels`` is an integer
        ``(N, 3)`` grid-index array and ``origin`` is Open3D's world-space
        voxel-grid origin.  The origin is required to reconstruct box poses.
    """
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(pcd_file))
    if pcd.is_empty():
        raise ValueError("Input PCD is empty!")
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    # Open3D owns the conversion from floating-point points to voxel indices.
    idxs = [v.grid_index for v in vg.get_voxels()]
    voxels = np.asarray(idxs, dtype=int)
    origin = np.array(vg.origin, dtype=float)
    return voxels, voxel_size, origin


def runs_in_row(row):
    """Find every contiguous truthy run in one boolean grid row.

    Returned endpoints are inclusive, so a run ``(2, 4)`` covers cells 2, 3,
    and 4.  This convention is used consistently when converting rectangles
    back into physical box sizes.
    """
    runs = []
    in_run = False
    start = 0
    # Start a run when occupancy first becomes true; close it at the cell
    # immediately before occupancy becomes false.
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
    """Merge identical horizontal runs across adjacent rows into rectangles.

    A run can extend vertically only when the next row has the same inclusive
    X interval.  Runs that do not continue are emitted immediately; remaining
    active runs are emitted after the final row.
    """
    h, _w = binary_grid.shape
    active = {}
    rects = []
    # ``active`` maps a horizontal X run to the first and most recent Y row
    # over which that exact run has continued.
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
    """Group voxel indices by Z and merge each horizontal XY occupancy slice.

    The temporary boolean array covers only the slice's minimum bounding
    rectangle, then merged coordinates are shifted back into absolute voxel
    indices before they are returned.
    """
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
        # Mark every occupied X-Y coordinate relative to this slice's minimum.
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
    """Build one voxel-layer-high box per rectangle and combine them into a mesh.

    A rectangle spans X and Y cells only, so every generated box has exactly
    one voxel of Z height.  Mesh-cleaning removes overlapping duplicate
    topology created where independently emitted boxes touch.
    """
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    for z, rects in rects_z.items():
        for (x1, y1), (x2, y2) in rects:
            # Inclusive endpoints require +1 to obtain each box's voxel count.
            nx = x2 - x1 + 1
            ny = y2 - y1 + 1
            size = np.array([nx, ny, 1.0], dtype=float) * voxel_size
            min_corner = origin + np.array([x1, y1, z], dtype=float) * voxel_size
            box = o3d.geometry.TriangleMesh.create_box(*size)
            box.translate(min_corner)
            mesh += box
    # Clean the combined mesh before it is written to an interchange format.
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    return mesh


def save_optional_gazebo_model(mesh, model_name: str, output_dir: str | Path):
    """Write OBJ plus minimal Gazebo descriptors beneath ``output_dir/model_name``.

    The same mesh is used for collision and visual geometry, and the model is
    marked static because this export describes a fixed obstacle environment.
    Gazebo files are optional and are not needed by Python map generation.
    """
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
    """Voxelise a PCD, merge its voxel boxes, and write a triangle-mesh OBJ.

    Args:
        pcd_file: Input occupied point cloud.
        output_obj: Destination OBJ path.
        voxel_size: Grid-cell size used to define obstacle geometry.
        write_gazebo: Whether to create extra Gazebo model files.
        model_name: Gazebo model-directory and URI name.

    Returns:
        The in-memory Open3D mesh after export.
    """
    import open3d as o3d

    voxels, vs, origin = pcd_to_voxel_grid(pcd_file, voxel_size=voxel_size)
    if voxels.size == 0:
        raise ValueError("No occupied voxels after voxelization.")
    # Merge before creating boxes to make a compact OBJ for simulators/viewers.
    rects_z = rectangles_per_z(voxels)
    mesh = boxes_to_mesh_from_rectangles(rects_z, vs, origin)
    output_obj = Path(output_obj)
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_obj), mesh, write_vertex_normals=False)
    print(f"[INFO] OBJ saved: {output_obj}")
    if write_gazebo:
        save_optional_gazebo_model(mesh, model_name, output_obj.parent)
    return mesh
