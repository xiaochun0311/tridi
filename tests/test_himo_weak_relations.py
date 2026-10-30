"""Synthetic relation fixtures check geometry and audit boundaries, not model quality."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

_spec = importlib.util.spec_from_file_location("himo_weak_relations", Path(__file__).parents[1] / "tridi/data/himo_weak_relations.py")
relations = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(relations)


def centres(main_path, other_path=None):
    main_path = np.asarray(main_path)
    result = np.zeros((2, len(main_path), 3))
    result[0, :, 0] = main_path
    result[1, :, 0] = 2. if other_path is None else other_path
    return result


class HIMOWeakRelationsTest(unittest.TestCase):
    def test_centroid_column_rotation_and_translation(self):
        rotation = np.tile(np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]), (2, 1, 1))
        objects = {name: dict(rot=rotation, transl=np.tile([3., 4., 5.], (2, 1))) for name in ("A", "B")}
        transformed = relations.world_centres(objects, ["A", "B"], [[1., 2., 0.], [0., 0., 0.]])
        np.testing.assert_allclose(transformed[0], [[1., 5., 5.], [1., 5., 5.]])
        np.testing.assert_allclose(transformed[1], [[3., 4., 5.], [3., 4., 5.]])

    @unittest.skipUnless(importlib.util.find_spec("trimesh"), "trimesh is installed on the target server")
    def test_true_triangle_area_weighted_surface_centroid(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "mesh.obj"
            path.write_text("v 0 0 0\nv 2 0 0\nv 0 1 0\nv 10 0 0\nv 14 0 0\nv 10 2 0\nf 1 2 3\nf 4 5 6\n")
            properties = relations._mesh_properties(path)
            expected = (np.array([2 / 3, 1 / 3, 0]) + 4 * np.array([34 / 3, 2 / 3, 0])) / 5
            np.testing.assert_allclose(properties["surface_centroid"], expected)
            self.assertEqual(properties["area_m2"], 5.)

    def test_approach_away_and_other_target(self):
        approach = centres(np.linspace(0., .4, 21))
        action, evidence = relations.classify_relation(approach, 0, 0, 20)
        self.assertEqual(action, "approach_other_object")
        self.assertEqual(evidence["target_slot"], 1)
        action, evidence = relations.classify_relation(approach[:, ::-1], 0, 0, 20)
        self.assertEqual(action, "move_away_from_other_object")
        action, evidence = relations.classify_relation(approach[::-1], 1, 0, 20)
        self.assertEqual(action, "approach_other_object")
        self.assertEqual(evidence["target_slot"], 0)

    def test_counterfactual_motion_and_distance_path_rejections(self):
        values = centres(np.zeros(21), np.linspace(2., 1., 21))
        values[0, :, 1] = np.linspace(0., .04, 21)
        action, evidence = relations.classify_relation(values, 0, 0, 20)
        self.assertIsNone(action)
        self.assertGreater(evidence["main_centre_motion_m"], .03)
        self.assertLess(evidence["main_only_distance_change_m"], .001)
        path = np.interp(np.arange(61), [0, 20, 40, 60], [0., .8, -.6, .2])
        action, evidence = relations.classify_relation(centres(path), 0, 0, 60)
        self.assertIsNone(action)
        self.assertLess(evidence["smoothed_distance_directness"], .55)
        self.assertIsNone(evidence["target_slot"])

    def test_generate_preserves_states_splits_v1_evidence_and_sources(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("object_pose", "smplx", "text", "object_mesh"):
                (root / name).mkdir()
            sequence = "S03T001"
            positions = np.zeros((41, 3))
            positions[:, 0] = np.r_[np.linspace(0, .4, 21), np.linspace(.4, 0., 21)[1:]]
            objects = {name: dict(rot=np.tile(np.eye(3), (41, 1, 1)),
                transl=positions if name == "A" else np.tile([2., 0., 0.], (41, 1))) for name in ("A", "B")}
            object_file = root / "object_pose" / f"{sequence}.npy"
            source_file = root / "smplx" / f"{sequence}.npz"
            text_file = root / "text" / f"{sequence}.txt"
            np.save(object_file, objects)
            np.savez(source_file, body_pose=np.zeros((41, 21, 3)))
            text_file.write_text("Caption is provenance only.")
            row = dict(sequence=sequence, subject="S03", split="train", status="weak_temporal_v1",
                review_status="machine_checked", object_slots={"0": "A", "1": "B"}, total_source_frames=41,
                source_frame_index_base=0, keyframes=dict(S0=0, S1=20, S2=40),
                local_object_pose_sha256=relations._sha(object_file), source_member_verification={"sha256": relations._sha(source_file)},
                source_caption=text_file.read_text(), source_text_sha256=relations._sha(text_file),
                actions=[dict(action_id="move_object", object_slot=0, target_slot=None, start=start, end=end)
                    for start, end in ((1, 20), (21, 40))],
                weak_labeling=dict(algorithm="weak_temporal_v1", thresholds={"minimum": 6}, evidence={"quality": .9}))
            original = deepcopy(row)
            path = root / "v1.json"
            path.write_text(json.dumps(dict(status="weak_temporal_v1", candidates=[row])))
            source_sha = relations._sha(path)
            properties = dict(sha256="1" * 64, surface_centroid=[1., 1., 0.], area_m2=1., vertices=3, faces=1)
            with patch.object(relations, "_mesh_properties", return_value=properties):
                result = relations.generate(root, path, root / "v2.json")
            changed = result["candidates"][0]
            self.assertEqual(changed["status"], "weak_temporal_v2")
            self.assertEqual([a["action_id"] for a in changed["actions"]], ["approach_other_object", "move_away_from_other_object"])
            for key in ("keyframes", "subject", "split", "total_source_frames", "source_member_verification", "source_caption"):
                self.assertEqual(changed[key], original[key])
            self.assertEqual([(a["start"], a["end"]) for a in changed["actions"]], [(1, 20), (21, 40)])
            self.assertEqual(changed["weak_labeling"]["evidence"]["base_weak_labeling"], original["weak_labeling"])
            self.assertEqual(changed["weak_labeling"]["evidence"]["base_actions"], original["actions"])
            self.assertEqual(relations._sha(path), source_sha)
            self.assertEqual(result["coverage"]["train"]["target_counts"], {"0": 0, "1": 2, "None": 0})
            self.assertTrue(result["closed_action_coverage"])
            with self.assertRaisesRegex(ValueError, "new relation annotation"):
                relations.generate(root, path, path)
            bad = json.loads(path.read_text())
            bad["candidates"][0]["weak_labeling"]["algorithm"] = "weak_temporal_v2"
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "machine-checked v1"):
                relations.generate(root, path, root / "bad.json")


if __name__ == "__main__":
    unittest.main()
