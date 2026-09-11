from pathlib import Path
import unittest

from navmem3d.capture.keyframes import select_pose_coverage, select_uniform
from navmem3d.capture.habitat import pinhole_intrinsics, quaternion_xyzw_to_matrix
from navmem3d.capture.inspect import inspect_sequence
from navmem3d.schemas import CaptureSequence, SchemaError, WorldManifest
from navmem3d.world.package import inspect_world
from navmem3d.world.gaussians import inspect_antimatter_splat
from navmem3d.world.interiorgs import register_interiorgs_scene
from navmem3d.semantic.query import query_entity_index
from navmem3d.semantic.filtering import build_searchable_entity_index
from navmem3d.navigation.goal_pose import build_observation_goal_index
from navmem3d.navigation.query_goal import query_navigation_goals
from navmem3d.navigation.patrol import build_interiorgs_patrol
from navmem3d.navigation.registration import estimate_bounds_registration
from navmem3d.semantic.geometry import derive_entity_geometry
from navmem3d.navigation.grid_runtime import (
    FREE,
    OCCUPIED,
    GridTransform,
    OccupancyGrid,
    astar_path,
    differential_drive_step,
)
import struct
import tempfile


ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sequence = CaptureSequence.load(ROOT / "examples/capture_sequence.json")

    def test_world_manifest_and_summary(self) -> None:
        path = ROOT / "examples/world_minimal/manifest.json"
        manifest = WorldManifest.load(path)
        self.assertEqual(manifest.world_id, "example_room_001")
        summary = inspect_world(path)
        self.assertEqual(summary["asset_format"], "ply")
        self.assertTrue(summary["asset_exists"])

    def test_uniform_is_deterministic_and_keeps_endpoints(self) -> None:
        ids = [frame.frame_id for frame in select_uniform(self.sequence.frames, 3)]
        self.assertEqual(ids, ["f000", "f002", "f004"])

    def test_pose_coverage_is_deterministic(self) -> None:
        first = [frame.frame_id for frame in select_pose_coverage(self.sequence.frames, 3)]
        second = [frame.frame_id for frame in select_pose_coverage(self.sequence.frames, 3)]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertIn("f000", first)
        self.assertEqual(len(set(first)), 3)

    def test_duplicate_frame_ids_are_rejected(self) -> None:
        data = self.sequence.to_dict()
        data["frames"][1]["frame_id"] = "f000"
        with self.assertRaises(SchemaError):
            CaptureSequence.from_dict(data)

    def test_habitat_camera_math(self) -> None:
        intrinsics = pinhole_intrinsics(640, 480, 90.0)
        self.assertAlmostEqual(intrinsics[0], 320.0)
        transform = quaternion_xyzw_to_matrix(0, 0, 0, 1, (1, 2, 3))
        self.assertEqual(transform[3:12:4], (1.0, 2.0, 3.0))

    def test_sequence_inspection_reports_fixture_files(self) -> None:
        summary = inspect_sequence(ROOT / "examples/capture_sequence.json")
        self.assertEqual(summary["frame_count"], 5)
        self.assertFalse(summary["rgb_complete"])

    def test_antimatter_splat_inspection(self) -> None:
        records = [
            struct.pack("<6f8B", 1, 2, 3, .1, .2, .3, 10, 20, 30, 255, 128, 128, 128, 128),
            struct.pack("<6f8B", -2, 4, 1, .4, .1, .2, 40, 50, 60, 127, 128, 128, 128, 128),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.splat"
            path.write_bytes(b"".join(records))
            summary = inspect_antimatter_splat(path)
        self.assertEqual(summary.gaussian_count, 2)
        self.assertEqual(summary.position_min, (-2.0, 2.0, 1.0))
        self.assertEqual(summary.position_max, (1.0, 4.0, 3.0))
        self.assertAlmostEqual(summary.position_p05[0], -1.85)
        self.assertAlmostEqual(summary.position_p95[0], 0.85)
        self.assertAlmostEqual(summary.mean_opacity, (255 + 127) / (2 * 255))

    def test_invalid_splat_size_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.splat"
            path.write_bytes(b"broken")
            with self.assertRaises(ValueError):
                inspect_antimatter_splat(path)

    def test_register_interiorgs_scene(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "0001_fixture"
            root.mkdir()
            (root / "3dgs_compressed.ply").write_bytes(b"ply\nend_header\n")
            (root / "labels.json").write_text(
                __import__("json").dumps([{"id": "chair_1"}]), encoding="utf-8"
            )
            (root / "occupancy.png").write_bytes(b"fixture")
            (root / "occupancy.json").write_text(
                __import__("json").dumps({"resolution": 0.01}), encoding="utf-8"
            )
            (root / "structure.json").write_text(
                __import__("json").dumps({"rooms": [{}], "ins": [{}, {}]}), encoding="utf-8"
            )
            output = Path(directory) / "worlds" / "manifest.json"
            manifest, summary = register_interiorgs_scene(root, output)
            loaded = WorldManifest.load(output)
        self.assertEqual(manifest.world_id, "interiorgs_0001_fixture")
        self.assertEqual(loaded.builder, "interiorgs_reference")
        self.assertEqual(summary["label_count"], 1)
        self.assertEqual(summary["room_count"], 1)
        self.assertEqual(summary["structure_instance_count"], 2)

    def test_grid_transform_round_trip(self) -> None:
        transform = GridTransform((0.05, 0, -2.0, 0, -0.05, 3.0, 0, 0, 1))
        world = transform.to_world(20, 10)
        pixel = transform.to_pixel(*world)
        self.assertAlmostEqual(pixel[0], 20)
        self.assertAlmostEqual(pixel[1], 10)

    def test_interiorgs_grid_transform_matches_official_formula(self) -> None:
        transform = GridTransform.from_interiorgs_metadata(
            {"scale": 0.05, "upper": [10.0, 2.0, 1.0], "lower": [-2.0, -7.0, 0.1]}
        )
        col, row = transform.to_pixel(8.0, -4.0)
        self.assertAlmostEqual(col, 40.0)
        self.assertAlmostEqual(row, 60.0)
        x, y = transform.to_world(col, row)
        self.assertAlmostEqual(x, 8.0)
        self.assertAlmostEqual(y, -4.0)

    def test_astar_respects_occupancy_and_robot_inflation(self) -> None:
        rows = tuple(
            tuple(OCCUPIED if col == 3 and row != 3 else FREE for col in range(7))
            for row in range(7)
        )
        grid = OccupancyGrid(rows, GridTransform((1, 0, 0, 0, 1, 0, 0, 0, 1)))
        path = astar_path(grid, (1, 3), (5, 3))
        self.assertIsNotNone(path)
        self.assertIn((3, 3), path)
        self.assertIsNone(
            astar_path(grid, (1, 3), (5, 3), blocked=grid.inflated_blocked(1))
        )

    def test_differential_drive_step(self) -> None:
        x, y, yaw = differential_drive_step((0, 0, 0), 1.0, 0.0, 0.5)
        self.assertAlmostEqual(x, 0.5)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(yaw, 0.0)

    def test_interiorgs_patrol_stays_on_navigable_pixels(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scene"
            root.mkdir()
            Image.new("L", (32, 24), FREE).save(root / "occupancy.png")
            (root / "occupancy.json").write_text(
                __import__("json").dumps(
                    {
                        "scale": 0.05,
                        "upper": [1.6, 1.2, 1.0],
                        "lower": [0.0, 0.0, 0.1],
                    }
                )
            )
            output = root / "patrol.json"
            patrol = build_interiorgs_patrol(
                root, output, frame_count=20, waypoint_count=4, robot_radius_m=0.1
            )
            grid = OccupancyGrid.from_interiorgs_scene(root)
            self.assertEqual(len(patrol["frames"]), 20)
            self.assertTrue(output.exists())
            self.assertTrue(
                all(grid.is_free(*frame["pixel"]) for frame in patrol["frames"])
            )

    def test_exact_entity_and_relation_query(self) -> None:
        index = {
            "world_id": "room",
            "entities": [
                {"entity_id": "sink_01", "labels": ["sink", "水槽"], "attributes": []},
                {
                    "entity_id": "printer_01",
                    "labels": ["printer", "打印机"],
                    "attributes": ["white", "白色"],
                    "relations": [{"predicate": "near", "object_id": "sink_01"}],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text(__import__("json").dumps(index), encoding="utf-8")
            result = query_entity_index(
                path,
                terms=["打印机"],
                attributes=["白色"],
                predicate="near",
                reference_terms=["水槽"],
            )
        self.assertEqual(result["match_count"], 1)
        self.assertEqual(result["matches"][0]["entity_id"], "printer_01")

    def test_searchable_filter_keeps_compact_and_multiview_entities(self) -> None:
        fixture = {
            "scene_robust_bounds": [[0, 0, 0], [10, 10, 10]],
            "entities": [
                {
                    "entity_id": "compact",
                    "view_count": 1,
                    "derived_geometry": {"extent": [1, 1, 1]},
                    "relations": [{"predicate": "near", "object_id": "wall"}],
                },
                {
                    "entity_id": "supported",
                    "view_count": 2,
                    "derived_geometry": {"extent": [4, 1, 1]},
                    "relations": [],
                },
                {
                    "entity_id": "wall",
                    "view_count": 3,
                    "derived_geometry": {"extent": [9, 8, 1]},
                    "relations": [],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "geometry.json"
            output = Path(directory) / "searchable.json"
            source.write_text(__import__("json").dumps(fixture), encoding="utf-8")
            result = build_searchable_entity_index(source, output)
        self.assertEqual(result["searchable_entity_count"], 2)
        self.assertEqual(result["structural_region_count"], 1)
        self.assertEqual(result["entities"][0]["relations"], [])

    def test_observation_goal_uses_supporting_view(self) -> None:
        entities = {
            "entities": [
                {
                    "entity_id": "sofa",
                    "labels": ["sofa"],
                    "membership_uri": "sofa.npy",
                    "observations": [{"view_index": 0}, {"view_index": 0}],
                    "derived_geometry": {"center": [0, 0, 2]},
                }
            ]
        }
        views = {"views": [{"view_id": "v0", "camera_position": [0, 0, 0]}]}
        with tempfile.TemporaryDirectory() as directory:
            entity_path = Path(directory) / "entities.json"
            view_path = Path(directory) / "views.json"
            output_path = Path(directory) / "goals.json"
            entity_path.write_text(__import__("json").dumps(entities), encoding="utf-8")
            view_path.write_text(__import__("json").dumps(views), encoding="utf-8")
            result = build_observation_goal_index(entity_path, view_path, output_path)
        goal = result["entities"][0]["goal_candidates"]
        self.assertEqual(len(goal), 1)
        self.assertEqual(goal[0]["forward"], [0.0, 0.0, 1.0])

    def test_geometry_accepts_standard_3dgs_ply(self) -> None:
        import numpy as np
        from plyfile import PlyData, PlyElement

        vertex = np.array(
            [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (2.0, 4.0, 6.0)],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            PlyData([PlyElement.describe(vertex, "vertex")]).write(root / "world.ply")
            members = root / "memberships"
            members.mkdir()
            np.save(members / "target.npy", np.array([0, 1], dtype=np.uint32))
            fused = {
                "entities": [{"entity_id": "target", "membership_uri": "memberships/target.npy"}]
            }
            (root / "fused.json").write_text(__import__("json").dumps(fused), encoding="utf-8")
            result = derive_entity_geometry(root / "fused.json", root / "world.ply", root / "geometry.json")
        self.assertEqual(result["entities"][0]["derived_geometry"]["center"], [0.5, 1.0, 1.5])

    def test_bounds_registration_encodes_rdf_to_interiorgs_axes(self) -> None:
        import numpy as np
        from PIL import Image
        from plyfile import PlyData, PlyElement

        vertex = np.array(
            [(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            PlyData([PlyElement.describe(vertex, "vertex")]).write(root / "b.ply")
            scene = root / "scene"
            scene.mkdir()
            Image.new("L", (2, 2), FREE).save(scene / "occupancy.png")
            (scene / "occupancy.json").write_text(
                __import__("json").dumps({"scale": 1.0, "lower": [-2, -3, 0.1], "upper": [2, 3, 1]}),
                encoding="utf-8",
            )
            result = estimate_bounds_registration(root / "b.ply", scene, output_path=root / "transform.json")
        matrix = np.asarray(result["transform_b_to_a"]).reshape(4, 4)
        # RDF's Y-down maps to InteriorGS Z-up, and Z-forward maps to A's back axis.
        self.assertGreater(matrix[1, 2], 0.0)
        self.assertLess(matrix[2, 1], 0.0)

    def test_navigation_query_joins_entity_and_goal_indices(self) -> None:
        entities = {
            "entities": [
                {
                    "entity_id": "sofa",
                    "labels": ["sofa", "沙发"],
                    "attributes": [],
                    "membership_uri": "sofa.npy",
                    "relations": [],
                }
            ]
        }
        goals = {
            "requires_map_registration": True,
            "reachability_filter_applied": False,
            "entities": [
                {"entity_id": "sofa", "goal_candidates": [{"candidate_id": "g0"}]}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            entity_path = Path(directory) / "entities.json"
            goal_path = Path(directory) / "goals.json"
            entity_path.write_text(__import__("json").dumps(entities), encoding="utf-8")
            goal_path.write_text(__import__("json").dumps(goals), encoding="utf-8")
            result = query_navigation_goals(
                entity_path, goal_path, terms=["沙发"], max_goals_per_entity=1
            )
        self.assertEqual(result["match_count"], 1)
        self.assertEqual(result["targets"][0]["goal_candidates"][0]["candidate_id"], "g0")


if __name__ == "__main__":
    unittest.main()
