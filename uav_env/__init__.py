"""Pure-Python 3D UAV routing / parametric map environment.

Ported from Kumar Robotics kr_param_map (ROS1/ROS2) to stand alone on Windows,
macOS, and Linux without ROS, PCL, or Ubuntu-only tooling.
"""

from .grid_map import BasicMapParams, GridMap, GridMapParams
from .struct_map_gen import MapGenParams, StructMapGenerator
from .map_io import read_image_map, read_pcd, write_pcd, save_npy

__all__ = [
    "BasicMapParams",
    "GridMap",
    "GridMapParams",
    "MapGenParams",
    "StructMapGenerator",
    "read_image_map",
    "read_pcd",
    "write_pcd",
    "save_npy",
]

__version__ = "1.0.0"
