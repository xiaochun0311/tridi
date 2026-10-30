"""Synthetic weak-label fixtures verify rules and leakage, not HIMO generation quality."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

# Load the numpy-only generator without importing the existing torch-based data package.
_spec = importlib.util.spec_from_file_location("himo_weak", Path(__file__).parents[1] / "tridi/data/himo_weak.py")
weak = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(weak)


def trajectories(kind="move", ambiguous=False):
    total = 90
    progress = np.zeros(total)
    progress[10:30] = np.linspace(0, 1, 20)
    progress[30:50] = 1
    progress[50:70] = np.linspace(1, 2, 20)
    progress[70:] = 2
    position = np.zeros((total, 3))
    if kind in ("move", "mixed"):
        position[:, 0] = progress * .2
    angles = progress * np.pi / 3 if kind in ("rotate", "mixed") else np.zeros(total)
    rotation = np.tile(np.eye(3), (total, 1, 1))
    rotation[:, 0, 0] = rotation[:, 1, 1] = np.cos(angles)
    rotation[:, 1, 0] = np.sin(angles)
    rotation[:, 0, 1] = -np.sin(angles)
    return {"A": dict(rot=rotation, transl=position),
        "B": dict(rot=rotation.copy() if ambiguous else np.tile(np.eye(3), (total, 1, 1)),
                  transl=position.copy() if ambiguous else np.zeros((total, 3)))}


class HIMOWeakTest(unittest.TestCase):
    def test_observable_action_coverage_and_bounds(self):
        for kind, expected in (("move", "move_object"), ("rotate", "rotate_object"), ("mixed", "move_and_rotate_object")):
            row, reason = weak.find_stages(trajectories(kind))
            self.assertIsNone(reason)
            self.assertEqual([action["action_id"] for action in row["actions"]], [expected, expected])
            frames = [row["keyframes"][key] for key in ("S0", "S1", "S2")]
            self.assertTrue(0 <= frames[0] < frames[1] < frames[2] < row["total_source_frames"])
            self.assertTrue(all(frames[i] < action["start"] <= action["end"] == frames[i + 1]
                for i, action in enumerate(row["actions"])))
            for index, stage in enumerate(row["weak_labeling"]["evidence"]["stages"]):
                endpoint = stage["raw_condition_to_target"]
                self.assertEqual(endpoint["action_id"], expected)
                self.assertEqual((endpoint["start_frame"], endpoint["end_frame"]), tuple(frames[index:index + 2]))

    def test_rejects_accumulated_drift_that_changes_actual_target_action(self):
        values = trajectories()
        # Slow rotation is individually inactive, but accumulates into the second target.
        for item in values.values():
            for key, data in item.items():
                item[key] = np.concatenate([data[:40], np.repeat(data[39:40], 20, axis=0), data[40:]])
        angles = np.zeros(110)
        angles[34:64] = np.linspace(0, np.pi / 10, 30)
        angles[64:] = np.pi / 10
        rotation = values["A"]["rot"]
        rotation[:, 0, 0] = rotation[:, 1, 1] = np.cos(angles)
        rotation[:, 1, 0] = np.sin(angles)
        rotation[:, 0, 1] = -np.sin(angles)
        self.assertEqual(weak.find_stages(values)[1], "no_clear_adjacent_pair")

    def test_rejects_ambiguous_objects_and_tracking_jumps(self):
        self.assertEqual(weak.find_stages(trajectories(ambiguous=True))[1], "no_clear_adjacent_pair")
        values = trajectories()
        values["A"]["transl"][20, 0] += 1
        self.assertEqual(weak.find_stages(values)[1], "tracking_jump")
        values = trajectories()
        values["C"] = deepcopy(values["B"])
        self.assertEqual(weak.find_stages(values)[1], "not_two_objects")

    def test_deterministic_subject_splits_and_provenance(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("object_pose", "smplx", "text"):
                (root / name).mkdir()
            for subject in ("S01", "S02", "S03", "S08", "S09", "S10", "S16", "S20"):
                for index, kind in enumerate(("move", "rotate", "mixed"), 1):
                    sequence = f"{subject}T{index:03d}"
                    np.save(root / "object_pose" / f"{sequence}.npy", trajectories(kind))
                    np.savez(root / "smplx" / f"{sequence}.npz", body_pose=np.zeros((90, 21, 3)))
                    (root / "text" / f"{sequence}.txt").write_text("A non-temporal source caption.")
            first = weak.generate(root, root / "first.json", max_per_split=(4, 4, 4))
            second = weak.generate(root, root / "second.json", max_per_split=(4, 4, 4))
            self.assertEqual(first, second)
            self.assertEqual(first["selected_counts"], dict(train=4, val=4, test=4))
            subjects = {split: set(values) for split, values in first["selected_subjects"].items()}
            self.assertFalse(subjects["train"] & subjects["val"] or subjects["train"] & subjects["test"] or subjects["val"] & subjects["test"])
            self.assertEqual(set(first["selected_action_counts"]["train"]),
                {"move_object", "rotate_object", "move_and_rotate_object"})
            self.assertEqual(len({row["sequence"] for row in first["candidates"]}), 12)
            self.assertEqual(first["selected_order_variation"]["all"]["different_action_ids_fraction"], 0.)
            for row in first["candidates"]:
                self.assertEqual(row["status"], "weak_temporal_v1")
                self.assertEqual(row["review_status"], "machine_checked")
                self.assertEqual(len(row["source_member_verification"]["sha256"]), 64)
                self.assertEqual(len(row["source_text_sha256"]), 64)
                self.assertTrue(row["weak_labeling"]["evidence"])
            with self.assertRaisesRegex(ValueError, "new weak annotation"):
                weak.generate(root, root / "first.json")


if __name__ == "__main__":
    unittest.main()
