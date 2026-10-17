"""
3D UAV routing environment public package interface.

This package is a pure-Python port of the Kumar Robotics parametric mapping
environment.  It runs on Windows, macOS, and Linux without ROS, PCL, or
Ubuntu-only dependencies.

The exports below are deliberately the small, high-level public API:

* ``BasicMapParams`` and ``GridMapParams`` describe physical voxel maps.
* ``GridMap`` stores and queries occupancy.
* ``MapGenParams`` and ``StructMapGenerator`` create random shape maps.
* The I/O helpers convert common map files to and from ``(N, 3)`` clouds.

Individual geometry classes, mesh conversion, and visualisation helpers remain
available through their own modules because they are more specialised tools.
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
