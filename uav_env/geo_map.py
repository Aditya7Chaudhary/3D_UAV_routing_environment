"""Container for geometric obstacle instances."""

from __future__ import annotations

from .shapes import CircleGate, Cylinder, Ellipsoid, Polyhedron, RectGate


class GeoMap:
    def __init__(self):
        self.polyhedron: list[Polyhedron] = []
        self.cylinder: list[Cylinder] = []
        self.ellipsoid: list[Ellipsoid] = []
        self.circle_gate: list[CircleGate] = []
        self.rect_gate: list[RectGate] = []

    def clear_all(self):
        self.polyhedron.clear()
        self.cylinder.clear()
        self.ellipsoid.clear()
        self.circle_gate.clear()
        self.rect_gate.clear()

    def add(self, geo_rep):
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
