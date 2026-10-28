"""Focused checks of bounded in-diffusion hand refinement, without SMPL assets."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
import trimesh

from tridi.model.sdf_refinement import refine_sdf_prediction
from tridi.core.sampler import Sampler
from tridi.model.tridi import TriDiModel
from tridi.model.wrappers.object_sdf import ObjectSDF


IDENTITY = torch.tensor([[1., 0., 0., 0., 1., 0.]])


class SDFRefinementTests(unittest.TestCase):
    def setUp(self):
        self.sdf = ObjectSDF({7: trimesh.creation.box()}, resolution=24, padding=.25)
        self.unoptimized_parameter = torch.nn.Parameter(torch.tensor(.02))
        self.couple_body = False

        def meshes(output, classes, sbj_gender=None):
            # Local full-joint 16 and 17 parameters occupy full-vector 106,112.
            x0, x1 = output.sbj_pose[:, 90], output.sbj_pose[:, 96]
            body_x = 3 * x0 - 1.2 if self.couple_body else torch.ones_like(x0) * .9
            x = torch.stack([x0, x1, body_x], dim=1)
            y = torch.ones_like(x) * (.09 + self.unoptimized_parameter)
            z = torch.ones_like(x) * .07
            vertices = torch.stack([x, y, z], dim=-1)
            return vertices, output.obj_c[:, None]

        self.decoder = Mock(side_effect=lambda gt, z, sample:
                            (torch.zeros(len(z), 2), torch.zeros(len(z), 2, dtype=torch.bool)))
        self.model = SimpleNamespace(
            data_sbj_channels=325, data_obj_channels=9, data_contacts_channels=128,
            cg_scale=.1, sdf_refine_steps=5, sdf_refine_max_delta=.1,
            sdf_contact_weight=1., sdf_penetration_weight=5.,
            split_output=TriDiModel.split_output,
            mesh_model=SimpleNamespace(
                get_meshes_wkpts_th=meshes,
                get_hand_mask=lambda gender, count: torch.tensor([[True, True, False]]).expand(count, -1),
                query_object_sdf=self.sdf.query),
            contact_model=SimpleNamespace(decode_contacts_th=self.decoder))

    def fixture(self, count=1, target=True):
        reference = torch.zeros(count, 462)
        reference[:, 10:16] = IDENTITY
        reference[:, 16:322] = IDENTITY.repeat(1, 51)
        reference[:, 106], reference[:, 112] = .7, .8
        reference[:, 322:325] = torch.tensor([.01, .02, .03])
        reference[:, 325:331] = IDENTITY
        batch = SimpleNamespace(
            obj_class=torch.full((count,), 7), sbj_gender=torch.ones(count, dtype=torch.bool),
            sbj_contact_indexes=torch.tensor([[0, 1]]).expand(count, -1),
            sbj_contact_target=torch.tensor([[target, False]]).expand(count, -1))
        return reference, batch

    def guidance(self, mode, reference, batch):
        with torch.no_grad():
            return refine_sdf_prediction(self.model, mode, reference, batch)

    def updated(self, reference, guidance, mode='100'):
        result = reference.clone()
        offset, packed_offset = 0, 0
        for enabled, width in zip(mode, (325, 9, 128)):
            if enabled == '1':
                result[:, offset:offset + width] += self.model.cg_scale * guidance[:, packed_offset:packed_offset + width]
                packed_offset += width
            offset += width
        return result

    def test_raw_contact_closes_gap_when_decoder_has_no_hand_targets(self):
        reference, batch = self.fixture()
        original = reference.clone()
        guidance = self.guidance('100', reference, batch)
        changed = self.updated(reference, guidance)
        self.assertEqual(guidance.shape, (1, 325))
        self.assertLess(changed[0, 106].item(), reference[0, 106].item())
        self.assertGreaterEqual(changed[0, 106].item(), .5)
        self.decoder.assert_not_called()
        torch.testing.assert_close(reference, original, atol=0, rtol=0)
        self.assertIsNone(self.unoptimized_parameter.grad)

    def test_explicit_empty_target_does_not_invent_contacts(self):
        reference, batch = self.fixture(target=False)
        guidance = self.guidance('100', reference, batch)
        torch.testing.assert_close(guidance, torch.zeros_like(guidance), atol=0, rtol=0)
        self.decoder.assert_not_called()

    def test_generated_contact_mode_ignores_raw_target(self):
        reference, batch = self.fixture(target=True)
        guidance = self.guidance('111', reference, batch)
        self.assertEqual(guidance.shape, (1, 462))
        torch.testing.assert_close(guidance, torch.zeros_like(guidance), atol=0, rtol=0)
        self.decoder.assert_called_once()

    def test_missing_raw_target_uses_decoder(self):
        reference, batch = self.fixture()
        batch.sbj_contact_target = None
        self.guidance('100', reference, batch)
        self.decoder.assert_called_once()

    def test_penetration_can_improve_without_contact_targets(self):
        reference, batch = self.fixture(target=False)
        reference[:, 106] = .4
        changed = self.updated(reference, self.guidance('100', reference, batch))
        self.assertGreater(changed[0, 106].item(), reference[0, 106].item())
        self.assertLess(changed[0, 106].item(), .51)
        self.assertIsNone(self.unoptimized_parameter.grad)

    def test_only_arm_hand_parameters_change_with_bounded_delta(self):
        reference, batch = self.fixture()
        self.model.cg_scale = 2.
        self.model.sdf_refine_steps = 10
        self.model.sdf_refine_max_delta = .03
        changed = self.updated(reference, self.guidance('111', reference, batch), mode='111')
        # Generated contacts have an empty decoder target in this fixture.
        torch.testing.assert_close(changed, reference, atol=0, rtol=0)
        changed = self.updated(reference, self.guidance('100', reference, batch))
        allowed = torch.zeros(462, dtype=torch.bool)
        allowed[106:322] = True
        torch.testing.assert_close(changed[:, ~allowed], reference[:, ~allowed], atol=0, rtol=0)
        self.assertGreater((changed - reference).abs().sum().item(), 0)
        self.assertLessEqual((changed - reference).abs().max().item(), .030001)

    def test_row_specific_sampled_indexes_select_correct_hand_vertex(self):
        reference, batch = self.fixture(count=2)
        batch.sbj_contact_indexes = torch.tensor([[0, 1], [1, 0]])
        changed = self.updated(reference, self.guidance('100', reference, batch))
        self.assertLess(changed[0, 106].item(), reference[0, 106].item())
        self.assertEqual(changed[0, 112].item(), reference[0, 112].item())
        self.assertLess(changed[1, 112].item(), reference[1, 112].item())
        self.assertEqual(changed[1, 106].item(), reference[1, 106].item())

    def test_nonhand_penetration_cannot_be_introduced_to_close_hand_gap(self):
        reference, batch = self.fixture()
        self.couple_body = True
        self.model.cg_scale = .2
        self.model.sdf_refine_steps = 20
        self.model.sdf_refine_max_delta = .4
        changed = self.updated(reference, self.guidance('100', reference, batch))
        body_x = 3 * changed[0, 106].item() - 1.2
        self.assertGreaterEqual(body_x, .49899)
        self.assertLess(changed[0, 106].item(), reference[0, 106].item())

    def test_more_shallow_collisions_cannot_hide_increased_total_penetration(self):
        # One 99mm penetration becomes two 79mm penetrations. The conditional
        # mean and maximum improve, while fixed-denominator penetration worsens.
        def meshes(output, classes, sbj_gender=None):
            x0 = output.sbj_pose[:, 90]
            x = torch.stack([x0, .5 - 4 * (x0 - .4), torch.ones_like(x0) * .9], dim=1)
            return torch.stack([x, torch.ones_like(x) * .11,
                                torch.ones_like(x) * .07], dim=-1), output.obj_c[:, None]

        self.model.mesh_model.get_meshes_wkpts_th = meshes
        self.model.cg_scale = .02 / (.198 * self.model.sdf_penetration_weight)
        self.model.sdf_refine_steps = 1
        for region_mask in ([True, True, False], [False, False, True]):
            with self.subTest(hand_mask=region_mask):
                self.model.mesh_model.get_hand_mask = lambda gender, count: torch.tensor([region_mask]).expand(count, -1)
                reference, batch = self.fixture(target=False)
                reference[:, 106] = .4
                candidate = reference.clone()
                candidate[:, 106] += .02

                def depths(value):
                    output = self.model.split_output(value)
                    vertices, _ = meshes(output, batch.obj_class)
                    signed = self.sdf.query(vertices, batch.obj_class, output.obj_R, output.obj_c)
                    return torch.relu(-signed - .001)[0, :2]

                before, after = depths(reference), depths(candidate)
                self.assertLess(after.square().sum().item() / int((after > 0).sum()),
                                before.square().sum().item() / int((before > 0).sum()))
                self.assertLess(after.max().item(), before.max().item())
                self.assertGreater(after.square().mean().item(), before.square().mean().item())
                guidance = self.guidance('100', reference, batch)
                torch.testing.assert_close(guidance, torch.zeros_like(guidance), atol=0, rtol=0)

    def test_zero_scale_or_conditioned_human_returns_packed_zero(self):
        reference, batch = self.fixture()
        self.model.cg_scale = 0.
        guidance = self.guidance('100', reference, batch)
        torch.testing.assert_close(guidance, torch.zeros(1, 325), atol=0, rtol=0)
        self.model.cg_scale = .1
        for mode, width in [('010', 9), ('001', 128)]:
            guidance = self.guidance(mode, reference, batch)
            torch.testing.assert_close(guidance, torch.zeros(1, width), atol=0, rtol=0)


class SamplerSettingsTests(unittest.TestCase):
    def test_sampling_step_forwards_requested_scheduler_and_step_count(self):
        sampler = Sampler.__new__(Sampler)
        sampler.cfg = SimpleNamespace(sample=SimpleNamespace(
            mode='111', c='ddim', num_inference_steps=73))
        sampler.model = Mock(return_value=torch.zeros(1, 462))
        sampler.model.split_output.return_value = 'decoded-result'
        batch = SimpleNamespace()
        result, captions = sampler.sample_step(batch)
        sampler.model.assert_called_once_with(
            batch, 'sample', sample_type='111', scheduler='ddim', num_inference_steps=73)
        sampler.model.eval.assert_called_once_with()
        self.assertEqual(result, 'decoded-result')
        self.assertEqual(captions, [])


if __name__ == '__main__':
    unittest.main()
