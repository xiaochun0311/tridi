import unittest

import torch

from tridi.model.ordered_scene.evaluate import rotation_error_degrees, scene_metrics, select_test_indices, swap_actions


class HeldOutMetrics(unittest.TestCase):
    def test_rigid_rotation_units_and_contact_macro(self):
        identity = torch.tensor([1., 0, 0, 0, 1, 0])
        truth = dict(object_pose=torch.zeros(1, 2, 2, 9), human=torch.zeros(1, 2, 315),
                     contact=torch.zeros(1, 2, 2, 690))
        truth["object_pose"][..., :6] = identity
        truth["contact"][..., 0] = 1
        prediction = {key: value.clone() for key, value in truth.items()}
        prediction["object_pose"][..., :6] = torch.tensor([0., -1, 0, 1, 0, 0])
        prediction["object_pose"][..., 6] = .1
        prediction["human"][..., 312] = .2
        prediction["contact"] = truth["contact"] * 2 - 1
        joints = torch.zeros(1, 2, 52, 3)
        shifted = joints.clone()
        shifted[..., 2] = .003
        result = scene_metrics(prediction, truth, shifted, joints)
        for key, expected in dict(object_translation_mm=100, object_rotation_degrees=90,
                human_root_translation_mm=200, native52_world_mpjpe_mm=3,
                contact_macro_precision=1, contact_macro_recall=1, contact_macro_f1=1).items():
            self.assertAlmostEqual(float(result[key]), expected, places=4)
        self.assertEqual(float(rotation_error_degrees(identity, identity)), 0)
        prediction["contact"].fill_(-1)
        self.assertEqual(float(scene_metrics(prediction, truth, joints, joints)["contact_macro_f1"]), 0)
        with self.assertRaises(ValueError):
            rotation_error_degrees(torch.zeros(6), identity)

    def test_test_selection_requires_unseen_people_and_seen_actions(self):
        data = dict(metadata=[dict(split=split, subject=subject) for split, subject in
                    (("train", "S01"), ("val", "S02"), ("test", "S03"), ("test", "S04"))],
                    conditions=dict(action_ids=torch.tensor([[0, 1], [0, 1], [1, 0], [0, 1]])))
        self.assertEqual(select_test_indices(data, 2, 42), select_test_indices(data, 2, 42))
        self.assertEqual(set(select_test_indices(data, 2, 42)), {2, 3})
        data["metadata"][2]["subject"] = "S01"
        with self.assertRaises(ValueError):
            select_test_indices(data, 2, 42)
        data["metadata"][2]["subject"] = "S03"
        data["conditions"]["action_ids"][2, 0] = 2
        with self.assertRaises(ValueError):
            select_test_indices(data, 2, 42)
        with self.assertRaises(ValueError):
            select_test_indices(data, 0, 42)

    def test_order_swap_preserves_initial_geometry_and_binds_action_references(self):
        conditions = dict(initial_object_pose=torch.randn(1, 2, 9), descriptors=torch.randn(1, 2, 12),
                          action_ids=torch.tensor([[0, 1]]), action_objects=torch.tensor([[1, 0]]),
                          action_targets=torch.tensor([[2, 1]]))
        changed = swap_actions(conditions)
        for key, value in conditions.items():
            self.assertTrue(torch.equal(changed[key], value.flip(1) if key.startswith("action_") else value))


if __name__ == "__main__":
    unittest.main()
