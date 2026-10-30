"""Synthetic unit fixtures check data boundaries; they are not research data."""
from copy import deepcopy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from tridi.data.himo_ordered import inspect, load_triplet, prepare
from tridi.model.ordered_scene.protocol import FIXED, index_sha256


class GeometryFixture:
    def descriptors(self, names):
        return torch.zeros(len(names), 2, 12)

    def contacts(self, shape, human, object_pose, names):
        return torch.zeros(len(names), 2, 2, 690)

    def protocol(self):
        ids = list(range(690))
        return dict(FIXED, model_sha256="1" * 64, contact_vertex_indices=ids,
                    contact_indices_sha256=index_sha256(ids))


class HIMODataTest(unittest.TestCase):
    def test_inventory_weak_opt_in_does_not_claim_human_acceptance(self):
        row = dict(self.row, split="train", status="weak_temporal_v2", review_status="machine_checked",
            weak_labeling=dict(algorithm="weak_temporal_v2"))
        labels = {"candidates": [row]}
        self.assertFalse(inspect(self.root, labels)["annotations"]["labels_ready_for_prepare"])
        report = inspect(self.root, labels, allow_weak=True)["annotations"]
        self.assertTrue(report["labels_ready_for_prepare"])
        self.assertEqual(report["accepted"], 0)
        self.assertEqual(report["machine_checked"], 1)

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("smplx", "object_pose", "text"):
            (self.root / name).mkdir()
        self.row = self.make_source("S01T001")

    def make_source(self, sequence):
        params = dict(body_pose=np.zeros((4, 21, 3), dtype=np.float32),
            lhand_pose=np.zeros((4, 15, 3), dtype=np.float32),
            rhand_pose=np.zeros((4, 15, 3), dtype=np.float32),
            global_orient=np.zeros((4, 3), dtype=np.float32),
            transl=np.arange(12, dtype=np.float32).reshape(4, 3) / 10,
            betas=np.zeros(10, dtype=np.float32), expression=np.zeros((4, 10), dtype=np.float32))
        params["global_orient"][1, 2] = np.pi / 2
        np.savez(self.root / "smplx" / f"{sequence}.npz", **params)
        objects = {name: dict(rot=np.tile(np.eye(3), (4, 1, 1)),
                    transl=np.column_stack([np.arange(4) + offset, np.zeros((4, 2))]))
                   for name, offset in [("A", 1), ("B", 9)]}
        np.save(self.root / "object_pose" / f"{sequence}.npy", objects)
        return dict(sequence=sequence, subject=sequence.split("T")[0],
            status="pending_human_review", review_status="unreviewed_by_human",
            object_slots={"0": "B", "1": "A"}, keyframes=dict(S0=0, S1=1, S2=3),
            total_source_frames=4, actions=[
                dict(action_id="move", start=1, end=1, object_slot=0, target_slot=None),
                dict(action_id="place", start=2, end=3, object_slot=0, target_slot=1)])

    def test_native_layout_explicit_slots_and_no_future_condition(self):
        loaded = load_triplet(self.root, self.row)
        self.assertEqual(loaded["object_names"], ["B", "A"])
        self.assertEqual(set(loaded["conditions"]), {"initial_object_pose"})
        self.assertEqual(tuple(loaded["targets"]["human"].shape), (2, 315))
        self.assertEqual(loaded["conditions"]["initial_object_pose"][0, 6].item(), 9)
        self.assertEqual(loaded["targets"]["object_pose"][0, 0, 6].item(), 10)
        torch.testing.assert_close(loaded["targets"]["human"][0, :6],
            torch.tensor([0., -1., 0., 1., 0., 0.]), atol=2e-6, rtol=0.)
        path = self.root / "object_pose/S01T001.npy"
        objects = np.load(path, allow_pickle=True).item()
        objects["B"]["transl"][1:, 0] += 30
        np.save(path, objects)
        changed = load_triplet(self.root, self.row)
        torch.testing.assert_close(loaded["conditions"]["initial_object_pose"], changed["conditions"]["initial_object_pose"])
        self.assertFalse(torch.equal(loaded["targets"]["object_pose"], changed["targets"]["object_pose"]))

    def test_unreviewed_and_cross_sequence_splits_rejected_before_geometry(self):
        with self.assertRaisesRegex(ValueError, "human acceptance"):
            prepare(self.root, {"candidates": [self.row]}, None, self.root / "out.pth")
        row = deepcopy(self.row)
        row.update(status="verified", review_status="accepted_by_human", split="train")
        other = dict(row, split="val")
        with self.assertRaisesRegex(ValueError, "cross data splits"):
            prepare(self.root, {"candidates": [row, other]}, None, self.root / "out.pth")

    def test_prepare_records_review_and_unseen_action_coverage(self):
        first = dict(self.row, status="verified", review_status="accepted_by_human", split="train")
        second = self.make_source("S02T002")
        second.update(status="verified", review_status="accepted_by_human", split="val")
        second["actions"][0]["action_id"] = "new_action"
        result = prepare(self.root, {"candidates": [first, second]}, GeometryFixture(), self.root / "prepared.pth")
        self.assertEqual(set(result), {"conditions", "targets", "metadata", "vocabulary", "representation"})
        self.assertEqual(result["metadata"][1]["unseen_action_ids_relative_to_train"], ["new_action"])
        self.assertEqual(result["metadata"][0]["action_annotation"], "verified")
        self.assertEqual(result["metadata"][0]["review_status"], "accepted_by_human")
        self.assertTrue((self.root / "prepared.pth").is_file())

    def test_source_integrity_and_face_policy(self):
        row = deepcopy(self.row)
        row["source_member_verification"] = {"sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "SHA256 changed"):
            load_triplet(self.root, row)
        path = self.root / "smplx/S01T001.npz"
        with np.load(path) as source:
            values = {key: source[key] for key in source.files}
        values["expression"][0, 0] = 1.
        np.savez(path, **values)
        with self.assertRaisesRegex(ValueError, "zero-face"):
            load_triplet(self.root, self.row)

    def test_inspect_does_not_promote_candidate_labels(self):
        import h5py
        folder = self.root / "processed_2o"
        folder.mkdir()
        with h5py.File(folder / "train.h5", "w") as data:
            group = data.create_group("0")
            group["transl"] = np.zeros((4, 3))
            group["object_state/A"] = np.zeros((4, 9))
            group["object_state/B"] = np.zeros((4, 9))
        report = inspect(self.root, {"candidates": [self.row]})
        self.assertEqual(report["processed"]["processed_2o/train"]["records"], 1)
        self.assertFalse(report["annotations"]["labels_ready_for_prepare"])
        self.assertEqual(self.row["status"], "pending_human_review")

    def test_weak_requires_opt_in_and_preserves_machine_status(self):
        row = deepcopy(self.row)
        row.update(status="weak_temporal_v1", review_status="machine_checked", split="train",
            weak_labeling=dict(algorithm="weak_temporal_v1", thresholds={"minimum_frames": 6},
                evidence={"stages": [{"net_translation_m": .1}, {"net_translation_m": .2}]}),
            source_caption="Recorded caption", source_text_sha256=hashlib.sha256(b"Recorded caption").hexdigest())
        (self.root / "text/S01T001.txt").write_text("Recorded caption")
        row["actions"][0]["action_id"] = "move_object"
        row["actions"][1]["action_id"] = "rotate_object"
        row["actions"][1]["target_slot"] = None
        with self.assertRaisesRegex(ValueError, "human acceptance"):
            prepare(self.root, {"candidates": [row]}, None, self.root / "strict.pth")
        result = prepare(self.root, {"candidates": [row]}, GeometryFixture(), self.root / "weak.pth", allow_weak=True)
        metadata = result["metadata"][0]
        self.assertEqual(metadata["status"], "weak_temporal_v1")
        self.assertEqual(metadata["action_annotation"], "weak_temporal_v1")
        self.assertEqual(metadata["review_status"], "machine_checked")
        self.assertEqual(metadata["weak_labeling"], row["weak_labeling"])
        self.assertEqual(metadata["source_caption"], "Recorded caption")
        for changed in (dict(row, status="pending_human_review"), dict(row, weak_labeling={}),
                        dict(row, review_status="accepted_by_human")):
            with self.assertRaisesRegex(ValueError, "human acceptance"):
                prepare(self.root, {"candidates": [changed]}, None, self.root / "forged.pth", allow_weak=True)
        other = deepcopy(row)
        other.update(sequence="S01T002", split="val")
        with self.assertRaisesRegex(ValueError, "subject-level isolation"):
            prepare(self.root, {"candidates": [row, other]}, None, self.root / "leaked.pth", allow_weak=True)

    def test_v2_versions_and_relation_target_contract(self):
        row = deepcopy(self.row)
        row.update(status="weak_temporal_v2", review_status="machine_checked", split="train",
            weak_labeling=dict(algorithm="weak_temporal_v2", thresholds={"distance_m": .05}, evidence={"stages": [{"relation": True}]}))
        row["actions"][0].update(action_id="approach_other_object", object_slot=0, target_slot=1)
        row["actions"][1].update(action_id="rotate_object", object_slot=0, target_slot=None)
        with self.assertRaisesRegex(ValueError, "human acceptance"):
            prepare(self.root, {"candidates": [row]}, None, self.root / "strict_v2.pth")
        result = prepare(self.root, {"candidates": [row]}, GeometryFixture(), self.root / "v2.pth", allow_weak=True)
        self.assertEqual(result["conditions"]["action_targets"].tolist(), [[1, 2]])
        self.assertEqual(result["metadata"][0]["weak_labeling"], row["weak_labeling"])
        mixed = deepcopy(row)
        mixed["weak_labeling"]["algorithm"] = "weak_temporal_v1"
        with self.assertRaisesRegex(ValueError, "human acceptance"):
            prepare(self.root, {"candidates": [mixed]}, None, self.root / "mixed.pth", allow_weak=True)
        v1 = deepcopy(row)
        v1.update(status="weak_temporal_v1")
        v1["weak_labeling"]["algorithm"] = "weak_temporal_v1"
        with self.assertRaisesRegex(ValueError, "incompatible"):
            prepare(self.root, {"candidates": [v1]}, None, self.root / "v1_rel.pth", allow_weak=True)
        for index, target in ((0, None), (0, 0), (1, 1)):
            invalid = deepcopy(row)
            invalid["actions"][index]["target_slot"] = target
            with self.assertRaisesRegex(ValueError, "targets"):
                prepare(self.root, {"candidates": [invalid]}, None, self.root / "target.pth", allow_weak=True)


if __name__ == "__main__":
    unittest.main()
