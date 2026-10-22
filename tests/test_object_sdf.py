"""Run with: python -m unittest discover -s tests -p test_object_sdf.py -v."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
import trimesh

from tridi.core.trainer import Trainer
from tridi.model.tridi import TriDiModel
from tridi.model.wrappers.object_sdf import ObjectSDF
from tridi.utils.geometry import matrix_to_rotation_6d


IDENTITY = matrix_to_rotation_6d(torch.eye(3)).unsqueeze(0)


class ObjectSDFTests(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        self.sdf = ObjectSDF({7: self.mesh}, resolution=24, padding=0.25)

    def query(self, points, sdf=None):
        return (self.sdf if sdf is None else sdf).query(
            points, torch.tensor([7]), IDENTITY, torch.zeros(1, 3)
        )

    def test_box_signed_distances(self):
        points = torch.tensor([[[0.25, 0.11, 0.07], [0.5, 0.11, 0.07], [0.7, 0.11, 0.07]]])
        distances = self.query(points)
        torch.testing.assert_close(distances, torch.tensor([[-0.25, 0.0, 0.2]]), atol=2e-6, rtol=0)

    def test_query_gradient_matches_face_normal(self):
        points = torch.tensor([[[0.7, 0.11, 0.07]]], requires_grad=True)
        gradient, = torch.autograd.grad(self.query(points).sum(), points)
        torch.testing.assert_close(gradient, torch.tensor([[[1.0, 0.0, 0.0]]]), atol=2e-5, rtol=0)

    def test_pose_scale_batch_and_repeated_noncontiguous_classes(self):
        meshes = {7: trimesh.creation.box(extents=(1.0, 2.0, 0.5)),
                  42: trimesh.creation.box(extents=(2.0, 0.8, 1.5))}
        sdf = ObjectSDF(meshes, resolution=24, padding=0.25)
        classes = torch.tensor([7, 42, 7])
        rotations = torch.tensor(np.stack([
            trimesh.transformations.rotation_matrix(angle, axis)[:3, :3]
            for angle, axis in [(np.pi / 2, [0, 0, 1]), (np.pi / 4, [0, 1, 0]),
                                (np.pi / 6, [1, 0, 0])]
        ]), dtype=torch.float32)
        rotation6 = matrix_to_rotation_6d(rotations).requires_grad_(True)
        translation = torch.tensor([[1.0, -2.0, 0.3], [-0.5, 0.7, 2.0], [3.0, 1.0, -0.4]],
                                   requires_grad=True)
        scale = torch.tensor([2.0, 0.5, 1.5])
        half_x = torch.tensor([0.5, 1.0, 0.5])
        canonical = torch.zeros(3, 2, 3)
        canonical[:, 0, 0] = half_x * 0.75
        canonical[:, 1, 0] = half_x + 0.2
        canonical[:, :, 1] = 0.05
        canonical[:, :, 2] = 0.02
        points = ((torch.bmm(canonical, rotations.transpose(1, 2))
                   + translation.detach()[:, None, :]) * scale[:, None, None]).requires_grad_(True)
        distances = sdf.query(points, classes, rotation6, translation, scale)
        expected = torch.stack([-half_x * 0.25, torch.full((3,), 0.2)], dim=1) * scale[:, None]
        torch.testing.assert_close(distances, expected, atol=5e-6, rtol=0)
        for gradient in torch.autograd.grad(distances.sum(), (points, rotation6, translation)):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_outside_grid_has_positive_distance_and_gradient(self):
        points = torch.tensor([[[2.0, 0.11, 0.07]]], requires_grad=True)
        distances = self.query(points)
        gradient, = torch.autograd.grad(distances.sum(), points)
        torch.testing.assert_close(distances, torch.tensor([[1.5]]), atol=2e-6, rtol=0)
        torch.testing.assert_close(gradient, torch.tensor([[[1.0, 0.0, 0.0]]]), atol=2e-5, rtol=0)

    def test_cache_reuse_and_geometry_resolution_padding_invalidation(self):
        points = torch.tensor([[[0.7, 0.11, 0.07]]])
        with tempfile.TemporaryDirectory() as folder:
            cached = ObjectSDF({7: self.mesh}, resolution=16, padding=0.25, cache_dir=folder)
            expected = self.query(points, cached)
            self.assertEqual(len(list(Path(folder).glob('*.npz'))), 1)
            reloaded = ObjectSDF({7: self.mesh}, resolution=16, padding=0.25, cache_dir=folder)
            with patch('igl.signed_distance', side_effect=AssertionError('Cache was not reused')):
                torch.testing.assert_close(self.query(points, reloaded), expected, atol=0, rtol=0)
            moved_mesh = self.mesh.copy()
            moved_mesh.apply_translation([0.1, 0, 0])
            for mesh, resolution, padding in [(moved_mesh, 16, 0.25), (self.mesh, 18, 0.25),
                                              (self.mesh, 16, 0.3)]:
                self.query(points, ObjectSDF({7: mesh}, resolution=resolution, padding=padding,
                                            cache_dir=folder))
            self.assertEqual(len(list(Path(folder).glob('*.npz'))), 4)

    def test_open_mesh_rejected_without_convex_hull_fallback(self):
        opened = self.mesh.copy()
        opened.update_faces(np.arange(len(opened.faces) - 1))
        sdf = ObjectSDF({7: opened}, resolution=16)
        with self.assertRaisesRegex(ValueError, 'closed'):
            self.query(torch.zeros(1, 1, 3), sdf)

    def test_open_mesh_requires_explicit_approximate_sign_opt_in(self):
        opened = self.mesh.copy()
        opened.update_faces(np.arange(len(opened.faces) - 1))
        sdf = ObjectSDF({7: opened}, resolution=16, allow_open=True)
        distances = self.query(torch.tensor([[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]]), sdf)
        self.assertTrue(torch.isfinite(distances).all())
        self.assertLess(distances[0, 0].item(), 0)
        self.assertGreater(distances[0, 1].item(), 0)

    def test_thin_mesh_retains_negative_interior(self):
        thin = trimesh.creation.box(extents=(2.0, 1.0, 0.001))
        sdf = ObjectSDF({7: thin}, resolution=24)
        distances = self.query(torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0008]]]), sdf)
        self.assertTrue(torch.isfinite(distances).all())
        self.assertLess(distances[0, 0].item(), 0)
        self.assertGreater(distances[0, 1].item(), 0)

    def test_invalid_resolution_and_padding_rejected(self):
        for kwargs in ({'resolution': 7}, {'padding': 0}, {'padding': -0.1}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                ObjectSDF({7: self.mesh}, **kwargs)

    def test_invalid_point_shape_and_class_batch_rejected(self):
        for points in (torch.zeros(2, 3), torch.zeros(1, 2, 2)):
            with self.subTest(shape=points.shape), self.assertRaisesRegex(ValueError, 'shape'):
                self.query(points)
        with self.assertRaisesRegex(ValueError, 'batch'):
            self.sdf.query(torch.zeros(1, 2, 3), torch.tensor([7, 7]), IDENTITY, torch.zeros(1, 3))

    def test_invalid_scale_rejected(self):
        for scale in (0, -1, float('nan'), float('inf')):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, 'scale'):
                self.sdf.query(torch.zeros(1, 1, 3), torch.tensor([7]), IDENTITY,
                               torch.zeros(1, 3), scale)


class GuidanceTests(unittest.TestCase):
    """Exercise the real sampling-guidance method with tiny synthetic body geometry."""

    def setUp(self):
        self.sdf = ObjectSDF({7: trimesh.creation.box()}, resolution=24, padding=0.25)

        def meshes(output, classes, sbj_gender=None):
            offsets = output.sbj_c.new_tensor([[0.0, 0.0, 0.0], [0.0, 0.02, 0.01]])
            vertices = output.sbj_c[:, None, :] + output.sbj_shape[:, None, :3] * 0.1 + offsets
            return vertices, output.obj_c[:, None, :]

        self.model = SimpleNamespace(
            device=torch.device('cpu'), data_sbj_channels=325, data_obj_channels=9,
            data_contacts_channels=128, object_sdf_resolution=24,
            sdf_penetration_weight=1.0, sdf_contact_weight=1.0,
            mesh_model=SimpleNamespace(get_meshes_wkpts_th=meshes, query_object_sdf=self.sdf.query),
            contact_model=SimpleNamespace(decode_contacts_th=lambda gt, z, sample: (z[:, :2], z[:, :2] > 0.5)),
            split_output=TriDiModel.split_output,
        )
        self.batch = SimpleNamespace(obj_class=torch.tensor([7]), sbj_contact_indexes=torch.tensor([[0, 1]]))

    def prediction(self, x, contact=False):
        pred = torch.zeros(1, 462)
        pred[:, 10:16] = IDENTITY
        pred[:, 16:322] = IDENTITY.repeat(1, 51)
        pred[:, 322:325] = torch.tensor([x, 0.1, 0.07])
        pred[:, 325:331] = IDENTITY
        pred[:, 334:] = float(contact)
        return pred

    def guidance(self, mode, pred):
        return TriDiModel.get_prediction_from_cg(
            self.model, mode, pred, pred[:, :325], pred[:, 325:334], pred[:, 334:], self.batch, 10
        )

    def test_penetration_guidance_pushes_body_out_and_preserves_shape(self):
        pred = self.prediction(0.4)
        original = pred.clone()
        gradient = self.guidance('100', pred)
        self.assertEqual(gradient.shape, (1, 325))
        self.assertGreater(gradient[0, 322].item(), 0)
        torch.testing.assert_close(gradient[:, :10], torch.zeros(1, 10), atol=0, rtol=0)
        torch.testing.assert_close(pred, original, atol=0, rtol=0)
        updated = pred.clone()
        updated[:, :325] += 0.01 * gradient
        before = self.sdf.query(pred[:, None, 322:325], self.batch.obj_class, pred[:, 325:331], pred[:, 331:334])
        after = self.sdf.query(updated[:, None, 322:325], self.batch.obj_class,
                               updated[:, 325:331], updated[:, 331:334])
        self.assertLess(torch.relu(-after).sum().item(), torch.relu(-before).sum().item())

    def test_object_only_guidance_keeps_body_conditioned(self):
        gradient = self.guidance('010', self.prediction(0.4))
        self.assertEqual(gradient.shape, (1, 9))
        self.assertLess(gradient[0, 6].item(), 0)

    def test_contact_guidance_closes_external_gap(self):
        gradient = self.guidance('100', self.prediction(0.7, contact=True))
        self.assertLess(gradient[0, 322].item(), 0)

    def test_contact_only_generation_has_no_geometry_correction(self):
        gradient = self.guidance('001', self.prediction(0.4))
        torch.testing.assert_close(gradient, torch.zeros(1, 128), atol=0, rtol=0)


class TrainingLossTests(unittest.TestCase):
    """Run the real training losses against a box without constructing SMPL or CUDA models."""

    def setUp(self):
        sdf = ObjectSDF({7: trimesh.creation.box()}, resolution=24, padding=0.25)
        self.query = Mock(side_effect=sdf.query)
        self.trainer = Trainer.__new__(Trainer)
        self.trainer.cfg = SimpleNamespace(train=SimpleNamespace(
            loss_t_stamp_threshold=250, losses={'obj_sdf_penetration': 1.0}
        ))
        self.trainer.mesh_model = SimpleNamespace(query_object_sdf=self.query)

    def fixture(self, scales=(1.0,)):
        count = len(scales)
        scale = torch.tensor(scales)
        truth = torch.tensor([[[0.5, 0.1, 0.07], [0.3, 0.15, 0.12]]]).repeat(count, 1, 1)
        predicted = torch.tensor([[[0.4, 0.1, 0.07], [0.8, 0.15, 0.12]]]).repeat(count, 1, 1)
        batch = SimpleNamespace(
            obj_class=torch.full((count,), 7), obj_R=IDENTITY.repeat(count, 1),
            obj_c=torch.zeros(count, 3), sbj_vertices=truth.requires_grad_(True),
            sbj_contact_indexes=torch.tensor([[0, 1]]).repeat(count, 1), scale=scale,
        )
        output = SimpleNamespace(
            sbj_vertices=(predicted * scale[:, None, None]).requires_grad_(True),
            obj_R=IDENTITY.repeat(count, 1), obj_c=torch.zeros(count, 3, requires_grad=True),
            timesteps_sbj=torch.zeros(count, dtype=torch.long),
            timesteps_obj=torch.zeros(count, dtype=torch.long),
        )
        return batch, output

    def loss(self, batch, output, *keys):
        self.trainer.cfg.train.losses = {key: 1.0 for key in keys}
        return self.trainer.compute_loss(batch, output, {})

    def test_both_subject_and_object_timesteps_gate_sdf_losses(self):
        batch, output = self.fixture((1.0, 1.0, 1.0))
        output.timesteps_sbj = torch.tensor([0, 0, 251])
        output.timesteps_obj = torch.tensor([0, 251, 0])
        loss, log = self.loss(batch, output, 'obj_sdf_penetration', 'obj_sdf_contact')
        self.assertGreater(log['loss/obj_sdf_penetration'], 0)
        self.assertGreater(log['loss/obj_sdf_contact'], 0)
        loss.backward()
        self.assertGreater(output.sbj_vertices.grad[0].abs().sum().item(), 0)
        torch.testing.assert_close(output.sbj_vertices.grad[1:], torch.zeros(2, 2, 3), atol=0, rtol=0)

    def test_empty_timestep_masks_produce_finite_differentiable_zero(self):
        for key in ('obj_sdf_penetration', 'obj_sdf_contact'):
            with self.subTest(loss=key):
                batch, output = self.fixture((1.0, 1.0))
                output.timesteps_sbj = torch.tensor([251, 0])
                output.timesteps_obj = torch.tensor([0, 251])
                self.query.reset_mock()
                loss, log = self.loss(batch, output, key)
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(log['loss/total'], 0)
                loss.backward()
                torch.testing.assert_close(output.sbj_vertices.grad, torch.zeros(2, 2, 3), atol=0, rtol=0)
                self.query.assert_not_called()

    def test_no_ground_truth_surface_contacts_produces_zero_gradient(self):
        batch, output = self.fixture()
        batch.sbj_vertices = torch.tensor([[[0.3, 0.1, 0.07], [0.2, 0.15, 0.12]]], requires_grad=True)
        loss, log = self.loss(batch, output, 'obj_sdf_contact')
        self.assertEqual(log['loss/obj_sdf_contact'], 0)
        loss.backward()
        torch.testing.assert_close(output.sbj_vertices.grad, torch.zeros(1, 2, 3), atol=0, rtol=0)
        self.assertIsNone(batch.sbj_vertices.grad)

    def test_contact_targets_use_unscaled_truth_and_predictions_use_batch_scale(self):
        batch, output = self.fixture((2.0, 0.5))
        # The first GT point is within 5 mm; the second is outside that threshold.
        batch.sbj_vertices = torch.tensor([[[0.503, 0.1, 0.07], [0.508, 0.15, 0.12]]]).repeat(2, 1, 1)
        batch.sbj_vertices.requires_grad_(True)
        canonical = torch.tensor([[[0.7, 0.1, 0.07], [0.9, 0.15, 0.12]]]).repeat(2, 1, 1)
        output.sbj_vertices = (canonical * batch.scale[:, None, None]).requires_grad_(True)
        loss, log = self.loss(batch, output, 'obj_sdf_contact')
        self.assertAlmostEqual(log['loss/obj_sdf_contact'], (0.4 ** 2 + 0.1 ** 2) / 2, places=5)
        self.assertEqual(self.query.call_count, 2)
        gt_call, predicted_call = self.query.call_args_list
        torch.testing.assert_close(gt_call.args[0], batch.sbj_vertices, atol=0, rtol=0)
        self.assertNotIn('scale', gt_call.kwargs)
        torch.testing.assert_close(predicted_call.kwargs['scale'], batch.scale, atol=0, rtol=0)
        loss.backward()
        self.assertTrue((output.sbj_vertices.grad[:, 0, 0] > 0).all())
        self.assertTrue((output.obj_c.grad[:, 0] < 0).all())
        torch.testing.assert_close(output.sbj_vertices.grad[:, 1], torch.zeros(2, 3), atol=0, rtol=0)
        self.assertIsNone(batch.sbj_vertices.grad)

    def test_penetration_gradients_separate_body_and_object(self):
        batch, output = self.fixture()
        loss, log = self.loss(batch, output, 'obj_sdf_penetration')
        self.assertAlmostEqual(log['loss/obj_sdf_penetration'], 0.005, places=6)
        loss.backward()
        self.assertLess(output.sbj_vertices.grad[0, 0, 0].item(), 0)
        self.assertGreater(output.obj_c.grad[0, 0].item(), 0)
        torch.testing.assert_close(output.sbj_vertices.grad[:, 1], torch.zeros(1, 3), atol=0, rtol=0)
        output.sbj_vertices = (output.sbj_vertices - 0.1 * output.sbj_vertices.grad).detach().requires_grad_(True)
        output.obj_c = (output.obj_c - 0.1 * output.obj_c.grad).detach().requires_grad_(True)
        improved, _ = self.loss(batch, output, 'obj_sdf_penetration')
        self.assertLess(improved.item(), loss.item())

    def test_configured_sdf_losses_require_enabled_object_sdf(self):
        cfg = SimpleNamespace(model=SimpleNamespace(object_sdf_resolution=0),
                              train=SimpleNamespace(losses={'obj_sdf_contact': 1.0}))
        with self.assertRaisesRegex(ValueError, 'object_sdf_resolution'):
            Trainer(cfg, SimpleNamespace())


if __name__ == '__main__':
    unittest.main()
