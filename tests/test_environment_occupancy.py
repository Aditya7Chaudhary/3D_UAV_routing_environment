from __future__ import annotations

import unittest

import numpy as np

from uav_env.grid_map import BasicMapParams, GridMapParams
from uav_env.struct_map_gen import MapGenParams, StructMapGenerator


def make_generator(size: tuple[float, float, float], resolution: float) -> StructMapGenerator:
    generator = StructMapGenerator()
    generator.init_params(
        GridMapParams(
            basic_mp=BasicMapParams(
                map_origin=np.zeros(3),
                map_size=np.asarray(size, dtype=float),
            ),
            resolution=resolution,
        )
    )
    return generator


class EnvironmentOccupancyTests(unittest.TestCase):
    def make_occupancy_generator(self) -> StructMapGenerator:
        generator = make_generator((10.0, 10.0, 5.0), 0.5)
        generator.mgpa = MapGenParams(
            occupancy_ratio=0.2,
            cylinder_ratio=1.0,
            circle_ratio=0.0,
            gate_ratio=0.0,
            ellip_ratio=0.0,
            poly_ratio=0.0,
            w1=0.4,
            w2=0.6,
            w3=0.8,
        )
        return generator

    def test_all_environment_types_use_shared_occupancy_target(self) -> None:
        for environment_type in (
            "structured",
            "corridor",
            "random_land",
            "mountains",
            "blocks",
            "mixed",
            "random_terrain",
            "land",
            "mountain",
            "urban",
            "city",
        ):
            with self.subTest(environment_type=environment_type):
                generator = self.make_occupancy_generator()
                generator.generate_environment(
                    {"type": environment_type, "height_range": 1.0},
                    seed=17,
                )
                occupied = np.count_nonzero(
                    generator.grid_map.occupancy > generator.grid_map.mp.min_thrd
                )
                total = len(generator.grid_map.occupancy)
                self.assertEqual(occupied, round(total * generator.mgpa.occupancy_ratio))

    def test_random_land_places_primitives_to_shared_occupancy_target(self) -> None:
        generator = make_generator((10.0, 10.0, 5.0), 0.5)
        generator.mgpa = MapGenParams(
            occupancy_ratio=0.5,
            cylinder_ratio=1.0,
            circle_ratio=0.0,
            gate_ratio=0.0,
            ellip_ratio=0.0,
            poly_ratio=0.0,
            w1=0.4,
            w2=0.6,
            w3=0.8,
        )

        generator.generate_environment({"type": "random_land"}, seed=6)
        generator.generate_environment({"type": "random_land"}, seed=7)

        expected_voxels = int(np.prod(generator.grid_map.mp.map_grid_size) * 0.5)
        self.assertEqual(len(generator.get_pc()), expected_voxels)
        self.assertGreater(len(generator.geo_map.cylinder), 0)

    def test_structured_pillars_span_the_full_map_height(self) -> None:
        generator = make_generator((8.0, 8.0, 4.0), 1.0)
        generator.mgpa.occupancy_ratio = 0.5

        generator.generate_environment({"type": "structured"}, seed=11)

        occupied = (
            generator.grid_map.occupancy.reshape(generator.grid_map.mp.map_grid_size)
            > generator.grid_map.mp.min_thrd
        )
        column_counts = occupied.sum(axis=2)
        path = generator._free_path(
            generator._occupied_bool(),
            *generator._planning_endpoints(),
            generator._planning_clearance(),
        )
        path_mask = np.zeros_like(occupied)
        path_mask[tuple(np.asarray(path).T)] = True
        reserved_path = generator._dilate_mask(path_mask, generator._planning_clearance())
        partial_columns = (column_counts != 0) & (column_counts != 4)
        self.assertTrue(np.all(~partial_columns | reserved_path.any(axis=2)))
        self.assertEqual(int(occupied.sum()), 32 * 4)

    def test_mountains_and_mixed_apply_occupancy_to_local_obstacles(self) -> None:
        for environment_type in ("mountains", "mixed"):
            with self.subTest(environment_type=environment_type):
                generator = self.make_occupancy_generator()
                generator.generate_environment(
                    {
                        "type": environment_type,
                        "mountain_count": 6,
                        "peak_height": 2.0,
                        "spread_min": 0.5,
                        "spread_max": 2.5,
                    },
                    seed=17,
                )

                occupied = generator._occupied_bool()
                column_heights = occupied.sum(axis=2)
                floor_coverage = occupied[:, :, 0].mean()
                positive_heights = column_heights[column_heights > 0]
                obstacle_count = sum(
                    len(shapes)
                    for shapes in (
                        generator.geo_map.cylinder,
                        generator.geo_map.circle_gate,
                        generator.geo_map.rect_gate,
                        generator.geo_map.ellipsoid,
                        generator.geo_map.polyhedron,
                    )
                )

                self.assertEqual(int(occupied.sum()), round(occupied.size * 0.2))
                self.assertLess(floor_coverage, 0.8)
                self.assertGreater(len(np.unique(positive_heights)), 2)
                self.assertIsNone(generator.terrain_height_field)
                if environment_type == "mixed":
                    self.assertGreater(obstacle_count, 0)

    def test_every_environment_has_a_clearance_path_between_opposite_corners(self) -> None:
        for environment_type in (
            "structured",
            "corridor",
            "random_land",
            "mountains",
            "blocks",
            "mixed",
        ):
            with self.subTest(environment_type=environment_type):
                generator = self.make_occupancy_generator()
                generator.generate_environment(
                    {"type": environment_type, "height_range": 1.0},
                    seed=23,
                )
                start, goal = generator._planning_endpoints()
                path = generator._free_path(
                    generator._occupied_bool(),
                    start,
                    goal,
                    generator._planning_clearance(),
                )

                self.assertIsNotNone(path)
                self.assertEqual(path[0], start)
                self.assertEqual(path[-1], goal)
                self.assertEqual(generator.grid_map.is_occ(np.zeros(3)), 0)
                opposite = (
                    generator.mpa.basic_mp.max_range
                    - 0.01 * generator.mpa.resolution
                )
                self.assertEqual(generator.grid_map.is_occ(opposite), 0)

    def test_corridor_maze_spans_both_map_axes(self) -> None:
        generator = self.make_occupancy_generator()
        generator.generate_environment({"type": "corridor"}, seed=29)

        occupied = generator._occupied_bool().any(axis=2)
        occupied_x = np.count_nonzero(occupied.any(axis=1))
        occupied_y = np.count_nonzero(occupied.any(axis=0))

        self.assertGreater(occupied_x, int(0.7 * occupied.shape[0]))
        self.assertGreater(occupied_y, int(0.7 * occupied.shape[1]))

    def test_random_and_mixed_use_a_small_number_of_larger_primitives(self) -> None:
        for environment_type in ("random_land", "mixed"):
            with self.subTest(environment_type=environment_type):
                generator = self.make_occupancy_generator()
                generator.generate_environment(
                    {"type": environment_type, "height_range": 1.0},
                    seed=23,
                )
                obstacle_count = sum(
                    len(shapes)
                    for shapes in (
                        generator.geo_map.cylinder,
                        generator.geo_map.circle_gate,
                        generator.geo_map.rect_gate,
                        generator.geo_map.ellipsoid,
                        generator.geo_map.polyhedron,
                    )
                )

                self.assertGreater(obstacle_count, 0)
                self.assertLess(obstacle_count, 40)


if __name__ == "__main__":
    unittest.main()
