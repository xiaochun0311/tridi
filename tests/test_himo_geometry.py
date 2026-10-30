"""Native geometry regressions; real-asset checks skip when HIMO is unavailable."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import trimesh

from tridi.model.ordered_scene.geometry import NativeSceneGeometry
from tridi.model.ordered_scene.protocol import assert_same_representation
from tridi.utils.geometry import batch_rodrigues, matrix_to_rotation_6d


MODELS = Path(os.environ.get("HIMO_TEST_MODELS", "/home/yanghc/tridi/data/smplx_models"))
DATA = Path(os.environ.get("HIMO_TEST_ROOT", "/home/yanghc/tridi/data/raw/himo/unpacked-20261029/data"))
REFERENCE = Path(os.environ.get("HIMO_TEST_REFERENCE",
    "/home/yanghc/tridi/outputs/himo-small-stage-20261029/visual/S01T005/decoded_full.npz"))


class RigidTransformTest(unittest.TestCase):
    def test_threshold_query_matches_global_triangle_distance(self):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=.2)
        points = np.random.default_rng(42).normal(0, .2, (150, 3))
        points = np.vstack([points, np.asarray(mesh.vertices)[:5], [[10., 10., 10.]]])
        distance = trimesh.proximity.closest_point(mesh, points)[1]
        result = NativeSceneGeometry._surface_near(mesh, points, .05)
        np.testing.assert_array_equal(result, distance < .05)

    def test_row6d_rotates_a_canonical_mesh_without_modification(self):
        geometry = NativeSceneGeometry.__new__(NativeSceneGeometry)
        geometry.device = torch.device("cpu")
        mesh = trimesh.Trimesh([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], [[0, 1, 2]], process=False)
        geometry._meshes = {"triangle": mesh}
        pose = torch.tensor([0., -1., 0., 1., 0., 0., 1., 2., 3.])
        result = geometry.object_vertices("triangle", pose).numpy()
        np.testing.assert_allclose(result, [[1, 2, 3], [1, 3, 3], [0, 2, 3]], atol=1e-7)
        np.testing.assert_array_equal(mesh.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        with self.assertRaises(ValueError):
            geometry.object_vertices("triangle", torch.zeros(9))


@unittest.skipUnless((MODELS / "smplx/SMPLX_NEUTRAL.npz").is_file() and (DATA / "smplx/S01T005.npz").is_file(),
                     "Native HIMO assets are not installed")
class NativeGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.geometry = NativeSceneGeometry(MODELS, DATA / "object_mesh")
        with np.load(DATA / "smplx/S01T005.npz", allow_pickle=False) as source:
            frames = np.asarray([0, 45])
            angles = np.concatenate([source["global_orient"][:, None], source["body_pose"],
                                     source["lhand_pose"], source["rhand_pose"]], axis=1)[frames]
            rotation = batch_rodrigues(torch.from_numpy(angles.reshape(-1, 3))).reshape(2, 52, 3, 3)
            cls.human = torch.cat([matrix_to_rotation_6d(rotation).reshape(2, 312),
                                  torch.from_numpy(source["transl"][frames])], dim=1)[None].float()
            cls.shape = torch.from_numpy(source["betas"])[None].float()

    def test_fixed_contact_asset_and_full_finger_layout(self):
        geometry = self.geometry
        self.assertEqual(len(np.unique(geometry.indices)), 690)
        self.assertEqual(int(np.isin(geometry.indices, geometry.left_hand_indices).sum()), 70)
        self.assertEqual(int(np.isin(geometry.indices, geometry.right_hand_indices).sum()), 70)
        vertices, joints = geometry.decode(self.shape, self.human)
        self.assertEqual(tuple(joints.shape), (1, 2, 52, 3))
        # Independent earlier raw-sequence decoder is a regression anchor for full hands and coordinates.
        if REFERENCE.is_file():
            with np.load(REFERENCE, allow_pickle=False) as reference:
                expected = reference["vertices"][[0, 45]]
                expected_joints = reference["joints"][[0, 45]][:, list(range(22)) + list(range(25, 55))]
            np.testing.assert_allclose(vertices.detach().numpy()[0], expected, atol=2e-6)
            np.testing.assert_allclose(joints.detach().numpy()[0], expected_joints, atol=2e-6)
        changed = self.human.clone()
        angle = torch.tensor([[0., 0., 1.]])
        changed[0, :, 37 * 6:38 * 6] = matrix_to_rotation_6d(batch_rodrigues(angle))[0]
        changed_vertices, changed_joints = geometry.decode(self.shape, changed)
        self.assertGreater(float((changed_vertices - vertices).abs().max()), .001)
        np.testing.assert_allclose(changed_joints.detach().numpy()[:, :, 22:37],
                                   joints.detach().numpy()[:, :, 22:37], atol=2e-6)
        self.assertGreater(float((changed_joints[:, :, 37:] - joints[:, :, 37:]).abs().max()), .001)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "indices.npy"
            np.save(path, geometry.indices, allow_pickle=False)
            reloaded = NativeSceneGeometry(MODELS, DATA / "object_mesh", indices_path=path,
                                           representation=geometry.protocol())
            assert_same_representation(geometry.protocol(), reloaded.protocol())
            incompatible = dict(geometry.protocol(), model_sha256="0" * 64)
            with self.assertRaises(ValueError):
                NativeSceneGeometry(MODELS, DATA / "object_mesh", representation=incompatible)

    def test_surface_contact_slot_order_and_export_roundtrip(self):
        geometry = self.geometry
        vertices, _ = geometry.decode(self.shape, self.human)
        pose = torch.zeros(1, 2, 2, 9)
        pose[..., :6] = torch.tensor([1., 0., 0., 0., 1., 0.])
        # Place one recorded canonical vertex exactly on the first native sampled point.
        anchor = torch.as_tensor(np.asarray(geometry._mesh("Lemon").vertices)[0], dtype=torch.float32)
        pose[0, :, 0, 6:] = vertices[0, :, geometry.indices[0]].detach() - anchor
        pose[0, :, 1, 6:] = 100
        contacts = geometry.contacts(self.shape, self.human, pose, [["Lemon", "Trashcan"]])
        self.assertEqual(tuple(contacts.shape), (1, 2, 2, 690))
        self.assertTrue(bool((contacts[0, :, 0, 0] == 1).all()))
        self.assertTrue(bool((contacts[:, :, 1] == 0).all()))
        swapped = geometry.contacts(self.shape, self.human, pose.flip(2), [["Trashcan", "Lemon"]])
        torch.testing.assert_close(swapped, contacts.flip(2))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "meshes"
            report = geometry.export(dict(shape=self.shape, human=self.human, object_pose=pose, contact=contacts),
                                     [["Lemon", "Trashcan"]], output)
            self.assertFalse(report["geometry_corrections"])
            self.assertEqual(np.load(report["joints"], allow_pickle=False).shape, (1, 2, 52, 3))
            for frame, row in enumerate(report["scenes"]):
                body = trimesh.load(row["human"], process=False, maintain_order=True)
                np.testing.assert_allclose(body.vertices, vertices[0, frame].detach().numpy(), atol=2e-6)
                for obj in row["objects"]:
                    restored = trimesh.load(obj["path"], process=False, maintain_order=True)
                    expected = geometry.object_vertices(obj["name"], pose[0, frame, obj["slot"]]).numpy()
                    np.testing.assert_allclose(restored.vertices, expected, atol=2e-5)
            saved = json.loads((output / "meshes.json").read_text())
            assert_same_representation(saved["representation"], geometry.protocol())
            with self.assertRaises(FileExistsError):
                geometry.export(dict(shape=self.shape, human=self.human, object_pose=pose),
                                [["Lemon", "Trashcan"]], output)


if __name__ == "__main__":
    unittest.main()
