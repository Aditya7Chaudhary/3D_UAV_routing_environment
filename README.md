# 3D_UAV_routing_environment

Pure-Python port of [Kumar Robotics `kr_param_map`](https://github.com/KumarRobotics/kr_param_map) for generating and loading **3D UAV routing / motion-planning environments**.

This repository **does not require ROS, ROS2, PCL, RViz, SDL, or Ubuntu**. It runs on Windows, macOS, and Linux with a normal Python 3 environment.

Original research paper (please cite if you use this for research):

```
@INPROCEEDINGS{10610207,
  author={Shao, Yifei Simon and Wu, Yuwei and Jarin-Lipschitz, Laura and Chaudhari, Pratik and Kumar, Vijay},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  title={Design and Evaluation of Motion Planners for Quadrotors in Environments with Varying Complexities},
  year={2024},
  pages={10033-10039},
  doi={10.1109/ICRA57147.2024.10610207}}
```

---

## File & folder guide (what each part does)

### Root

| Path | Role |
|------|------|
| `README.md` | This document: overview, file map, install, usage. |
| `LICENSE` | BSD 3-Clause license from the upstream project. |
| `requirements.txt` | Python dependencies (`numpy`, `scipy`, `Pillow`, `PyYAML`, `matplotlib`, `open3d`). |
| `.gitignore` | Ignores caches, virtualenvs, and generated `outputs/`. |

### Package: `uav_env/` (core library)

| Path | Role |
|------|------|
| `uav_env/__init__.py` | Package entry; exports the main public classes/helpers. |
| `uav_env/shapes.py` | Obstacle primitives: `Cylinder`, `Ellipsoid`, `Polyhedron`, `CircleGate`, `RectGate`, plus rotation helpers. Pure NumPy/SciPy (ConvexHull replaces QuickHull). |
| `uav_env/grid_map.py` | 3D occupancy grid: indexing, occupancy set/query, inflation fill, 2D projection of occupied columns. |
| `uav_env/geo_map.py` | Lightweight container that stores the generated geometric obstacle instances. |
| `uav_env/struct_map_gen.py` | **Structured map generator**: randomly places cylinders / circle gates / rect gates / ellipsoids / polytopes at target occupancy ratios; optional noise and dynamic motion. |
| `uav_env/map_io.py` | I/O helpers: image→3D cloud, PCD read/write, random cloud mode, noise injection, clear-zone filtering, CSV clear-position loading. |
| `uav_env/mesh_export.py` | Voxelize a PCD, greedy rectangle merge per Z-slice, export OBJ mesh (optional Gazebo model files). |
| `uav_env/visualize.py` | Visualization via Matplotlib (default) or Open3D (replaces RViz). |

### Scripts: `scripts/` (command-line entry points)

| Path | Role |
|------|------|
| `scripts/generate_structure_map.py` | CLI for the structured generator (replaces `structure_map` ROS node + launch). Writes PCD/NPY under `outputs/`. |
| `scripts/read_grid_map.py` | CLI to load images / PCD / random points, optionally inflate to grid centers (replaces `read_grid_map` ROS node). |
| `scripts/img2pcd.py` | Small standalone converter: one PNG → one PCD. |
| `scripts/pcd_to_mesh.py` | Standalone PCD → OBJ mesh exporter (Gazebo files optional). |

### Config: `config/`

| Path | Role |
|------|------|
| `config/structure_map.yaml` | Parameters formerly in `structure_map.launch` (map size, ratios, seed, dataset save path, dynamics). |
| `config/read_grid_map.yaml` | Parameters formerly in `read_grid_map.launch` (mode, file path, inflation, image thresholds). |

### Data: `data/`

| Path | Role |
|------|------|
| `data/img/all/` | Example 2D occupancy images (`maze.png`, `balls.png`, `empty.png`, …). |
| `data/img/maze/` | 100 maze PNGs for batch loading / dataset-style experiments. |
| `data/pcd/` | Example point-cloud map (`cloud_0917.pcd`). |
| `data/pc1/*/…png` + `.stl` | Preview images and mesh meshes of classic indoor maps (Levine, office, simple, skir). ROS `.bag` files are **not** shipped here — convert to PCD if you need them. |
| `data/clear_pos/` | CSV positions that should stay free of obstacles (`clear_2d.csv`, `clear_3d.csv`). |
| `data/semantics/` | Example semantic obstacle definitions (from the old `vicon_env` package). |
| `data/dataset/` | Sample generated PCD from the upstream dataset folder. |

### Docs & outputs

| Path | Role |
|------|------|
| `docs/` | Figures from the original repo (map annotation, example environments, inflation / resolution demos). |
| `outputs/` | Default write location for generated maps, previews, and meshes (created at runtime). |

---

## What changed vs the ROS package

| Original (ROS) | This repo (pure Python) |
|----------------|-------------------------|
| C++ + PCL + Eigen nodes | Python + NumPy / SciPy / Open3D |
| `roslaunch` / topics / RViz | YAML configs + CLI scripts + Matplotlib/Open3D |
| SDL_image for PNGs | Pillow |
| QuickHull C++ header | `scipy.spatial.ConvexHull` |
| ROS bag modes 2 & 3 | Removed — convert bags to `.pcd` and use mode `4` |
| Ubuntu / ROS Noetic or Humble required | Any OS with Python 3.9+ |

---

## Install

```bash
cd 3D_UAV_routing_environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## Quick start

### 1) Generate a structured 3D obstacle map

```bash
python scripts/generate_structure_map.py --config config/structure_map.yaml --no-show
```

Outputs:

- `outputs/dataset/pt0.pcd` / `pt0.npy`
- `outputs/structure_map_preview.png`

Tune obstacle density in `config/structure_map.yaml`:

```yaml
params:
  cylinder_ratio: 0.01
  circle_ratio: 0.005
  gate_ratio: 0.005
  ellip_ratio: 0.005
  poly_ratio: 0.005
```

Higher ratios → denser / harder environments for planners.

### 2) Load a maze image as a 3D map

```bash
python scripts/read_grid_map.py --config config/read_grid_map.yaml --no-show
```

Or load a PCD directly:

```bash
python scripts/read_grid_map.py --mode 4 --file data/pcd/cloud_0917.pcd --no-show
```

### 3) Image → PCD → mesh

```bash
python scripts/img2pcd.py --input data/img/all/maze.png --output outputs/maze.pcd
python scripts/pcd_to_mesh.py --input outputs/maze.pcd --output outputs/maze.obj
```

---

## Map modes (`read_grid_map`)

| Mode | Meaning |
|------|---------|
| `0` | Random sparse points with cleared corridors |
| `1` | 2D image extruded along Z |
| `4` | Read `.pcd` |

Modes `2` / `3` (ROS bags) from the original package are intentionally omitted.

---

## Using the library in your own planner

```python
from pathlib import Path
import numpy as np
from uav_env import (
    BasicMapParams, GridMapParams, MapGenParams, StructMapGenerator, write_pcd
)

basic = BasicMapParams(
    map_origin=np.array([-15.0, -15.0, 0.0]),
    map_size=np.array([30.0, 30.0, 3.0]),
)
grid_mp = GridMapParams(basic_mp=basic, resolution=0.2)
mgpa = MapGenParams(cylinder_ratio=0.01, circle_ratio=0.005)

gen = StructMapGenerator()
gen.init_params(grid_mp)
gen.random_uni_map_gen(mgpa, seed=1.0, if_dyn=False)
cloud = gen.get_pc()          # (N, 3) numpy array
write_pcd("my_map.pcd", cloud)
```

Collision checks against the occupancy grid:

```python
from uav_env import GridMap

grid = gen.grid_map
occupied = grid.is_occ(np.array([0.0, 0.0, 1.0]))  # 1=occ, 0=free, -1=OOB
```

---

## Tips

- Start with `resolution: 0.2` for faster iteration; use `0.1` to match the original denser maps.
- Keep total obstacle ratios modest on large maps — generation time scales with occupied voxels.
- Open3D visualization: add `--open3d` to either main script.
- Optional Gazebo model export still exists via `pcd_to_mesh.py --gazebo`, but **Gazebo is not required**.

---

## Maintenance / upstream

Upstream ROS packages: [KumarRobotics/kr_param_map](https://github.com/KumarRobotics/kr_param_map)

Original contact (upstream): Yuwei Wu (`yuweiwu@seas.upenn.edu`).
