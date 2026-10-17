"""
===============================================================================
3D UAV ROUTING ENVIRONMENT — GEOMETRY MAP CONTAINER
===============================================================================

``GeoMap`` stores the original parametric shape objects after they have been
rasterised into occupancy points.  The separate lists preserve concrete shape
types, letting the structured generator iterate, reset, or move each family
without using fragile type labels.  It complements ``GridMap``, which only
knows about occupied voxel values rather than the source geometry.
===============================================================================
"""

from __future__ import annotations

from .shapes import CircleGate, Cylinder, Ellipsoid, Polyhedron, RectGate


class GeoMap:
    """Own one ordered collection for every supported obstacle family.

    Shapes are appended in generation order.  No geometry is copied: changes
    made to a stored object, such as dynamic centre updates, affect the exact
    instance the generator originally created.
    """
    def __init__(self):
        """Create empty collections for all supported obstacle families."""
        self.polyhedron: list[Polyhedron] = []
        self.cylinder: list[Cylinder] = []
        self.ellipsoid: list[Ellipsoid] = []
        self.circle_gate: list[CircleGate] = []
        self.rect_gate: list[RectGate] = []

    def clear_all(self):
        """Remove all currently registered geometry objects."""
        self.polyhedron.clear()
        self.cylinder.clear()
        self.ellipsoid.clear()
        self.circle_gate.clear()
        self.rect_gate.clear()

    def add(self, geo_rep):
        """Register one supported shape in the collection matching its type.

        Unsupported objects are rejected so map generation cannot silently
        omit geometry that was expected to be rasterised.  ``isinstance`` is
        used rather than class-name strings so subclasses are accepted too.
        """
        if isinstance(geo_rep, Polyhedron):
            self.polyhedron.append(geo_rep)
        elif isinstance(geo_rep, Cylinder):
            self.cylinder.append(geo_rep)
        elif isinstance(geo_rep, Ellipsoid):
            self.ellipsoid.append(geo_rep)
        elif isinstance(geo_rep, CircleGate):
            self.circle_gate.append(geo_rep)
        elif isinstance(geo_rep, RectGate):
            self.rect_gate.append(geo_rep)
        else:
            raise TypeError(f"Unsupported geometry type: {type(geo_rep)}")
