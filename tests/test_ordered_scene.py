"""Contract and wiring tests; random tensors here are not research training data."""
import os
import unittest
from copy import deepcopy
from unittest.mock import patch

import torch
from torch import nn

from tridi.model.ordered_scene.model import OrderedSceneDenoiser
from tridi.model.ordered_scene.pipeline import (
    STATE, STATE_SHAPES, diffusion_loss, reconstruction_loss, sample, sample_tri_timesteps,
    validate_conditions, validate_training_data,
)
from tridi.model.ordered_scene.cli import CHECKPOINT_SCHEMA, validate_checkpoint, validate_inference_payload
from tridi.model.ordered_scene.protocol import FIXED, index_sha256


def conditions():
    return dict(initial_object_pose=torch.randn(2, 2, 9), descriptors=torch.randn(2, 2, 12),
        action_ids=torch.tensor([[0, 1], [1, 2]]), action_objects=torch.tensor([[0, 1], [1, 0]]),
        action_targets=torch.tensor([[2, 0], [0, 2]]))


def targets():
    values = {key: torch.randn(2, *shape) for key, shape in STATE_SHAPES.items()}
    values["contact"] = torch.randint(0, 2, (2, 2, 2, 690)).float()
    return values


def representation():
    ids = list(range(690))
    return dict(FIXED, model_sha256="0" * 64, contact_vertex_indices=ids,
                contact_indices_sha256=index_sha256(ids))


class ConstantHeads(nn.Module):
    """Check the sampler actually uses object outputs, not copied input poses."""
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.num_actions, self.descriptor_dim = 3, 12

    def forward(self, **batch):
        return {key: torch.full_like(batch[key], .125 if key == "object_pose" else .25) for key in STATE}


class RecordingHeads(ConstantHeads):
    def forward(self, **batch):
        self.inputs = {key: value.detach().clone() for key, value in batch.items()}
        return {key: torch.zeros_like(batch[key]) + self.anchor for key in STATE}


class OrderedSceneContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(731)
        cls.model = OrderedSceneDenoiser(3, checkpoint=os.environ["TRIDI_INIT_CHECKPOINT"])

    def test_initial_only_boundary(self):
        cond = conditions()
        self.assertEqual(validate_conditions(cond, 3), 2)
        with self.assertRaises(ValueError):
            validate_conditions(dict(cond, object_pose=torch.randn(2, 2, 2, 9)), 3)
        cond["initial_object_pose"] = torch.randn(2, 2, 2, 9)
        with self.assertRaises(ValueError):
            validate_conditions(cond, 3)

    def test_recorded_temporal_contract(self):
        fixture = dict(conditions=conditions(), targets=targets(), vocabulary=["pick", "push", "place"],
            representation=representation(),
            metadata=[dict(subject=f"fixture_{i}", sequence_id=f"unit_{i}", split=split,
                initial_frame=0, target_frames=[10, 20], two_task_objects=True,
                provenance="recorded", action_annotation="verified", review_status="accepted_by_human")
                for i, split in enumerate(["train", "val"])])
        self.assertEqual(validate_training_data(fixture), 2)
        fixture["metadata"][0]["review_status"] = "unreviewed_by_human"
        with self.assertRaises(ValueError):
            validate_training_data(fixture)
        fixture["metadata"][0]["review_status"] = "accepted_by_human"
        fixture["metadata"][0]["provenance"] = "rule_rollout"
        with self.assertRaises(ValueError):
            validate_training_data(fixture)
        fixture["metadata"][0]["provenance"] = "recorded"
        fixture["metadata"][0]["target_frames"] = [0, 20]
        with self.assertRaises(ValueError):
            validate_training_data(fixture)

    def test_three_clocks_control_only_their_input_modality(self):
        model, cond = RecordingHeads(), conditions()
        truth = {key: torch.zeros(2, *shape) for key, shape in STATE_SHAPES.items()}
        noise = {key: torch.ones_like(value) for key, value in truth.items()}
        diffusion_loss(model, truth, cond, torch.zeros(2, 3, dtype=torch.long), noise=noise)
        baseline = model.inputs
        affected = [{"shape", "human"}, {"object_pose"}, {"contact"}]
        for branch in range(3):
            clocks = torch.zeros(2, 3, dtype=torch.long)
            clocks[:, branch] = 900
            diffusion_loss(model, truth, cond, clocks, noise=noise)
            for key in STATE:
                if key in affected[branch]:
                    self.assertFalse(torch.equal(baseline[key], model.inputs[key]))
                else:
                    self.assertTrue(torch.equal(baseline[key], model.inputs[key]))
        clocks = torch.tensor([[0, 400, 900], [250, 0, 500]])
        with torch.no_grad(), patch.object(self.model.core, "prepare_time", wraps=self.model.core.prepare_time) as timing:
            self.model.eval()(**targets(), **cond, t=clocks)
        for branch in range(3):
            self.assertTrue(torch.equal(timing.call_args.args[branch], clocks[:, branch]))

    def test_tridi_sparse_clock_patterns_and_loss_families(self):
        clock = sample_tri_timesteps(2048, torch.Generator().manual_seed(47))
        patterns = {tuple(row) for row in (clock > 0).int().tolist()}
        self.assertEqual(patterns, {(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1), (0, 0, 1), (0, 1, 0), (1, 0, 0)})
        self.assertTrue(((clock >= 0) & (clock < 1000)).all())
        self.assertTrue(torch.equal(clock, sample_tri_timesteps(2048, torch.Generator().manual_seed(47))))
        truth = {key: torch.zeros(2, *shape) for key, shape in STATE_SHAPES.items()}
        truth["contact"].fill_(-1)
        first = {key: value.clone() for key, value in truth.items()}
        first["human"].fill_(1)
        second = dict(first, human=first["human"] * 2)
        self.assertTrue(torch.allclose(reconstruction_loss(second, truth), 2 * reconstruction_loss(first, truth)))
        first = dict(truth, contact=truth["contact"] + 1)
        second = dict(truth, contact=truth["contact"] + 2)
        self.assertTrue(torch.allclose(reconstruction_loss(second, truth), 4 * reconstruction_loss(first, truth)))

    def test_checkpoint_geometry_and_initial_object_bindings(self):
        saved = dict(schema=CHECKPOINT_SCHEMA, data_validated=True, training_steps=1,
            representation=representation(), vocabulary=["pick", "push", "place"], num_actions=3, descriptor_dim=12)
        payload = dict(conditions=conditions(), vocabulary=saved["vocabulary"], representation=representation(),
                       object_names=[["Cup", "Pot"], ["Glasses", "Case"]])
        self.assertEqual(validate_inference_payload(saved, payload), 2)
        with self.assertRaises(ValueError):
            validate_checkpoint(dict(saved, schema="ordered-scene-v1"))
        with self.assertRaises(ValueError):
            validate_checkpoint(dict(saved, training_steps=0))
        wrong = deepcopy(payload)
        wrong["representation"]["model_sha256"] = "1" * 64
        with self.assertRaises(ValueError):
            validate_inference_payload(saved, wrong)
        wrong["representation"] = dict(representation(), body_model="smplh")
        with self.assertRaises(ValueError):
            validate_inference_payload(saved, wrong)
        with self.assertRaises(ValueError):
            validate_inference_payload(saved, dict(payload, targets=targets()))
        with self.assertRaises(ValueError):
            validate_inference_payload(saved, dict(payload, object_names=[["Cup", "Pot"]]))

    def test_gradients_include_objects_and_actions(self):
        model = self.model.train()
        model.zero_grad(set_to_none=True)
        loss = diffusion_loss(model, targets(), conditions(), torch.tensor([250, 700]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for layer in [model.action_embedding, model.core.decoder_O_orient, model.core.decoder_O_transl]:
            self.assertGreater(sum(p.grad.abs().sum().item() for p in layer.parameters()), 0.)
        for parameter in model.parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(model.initialization["strictly_loaded_tensors"], 170)

    def test_object_references_and_order_wiring(self):
        model = self.model.eval()
        cond, state = conditions(), targets()
        state["contact"] = 2 * state["contact"] - 1
        t = torch.tensor([500, 500])
        with torch.no_grad():
            first = model(**state, **cond, t=t)
            permuted = dict(cond, initial_object_pose=cond["initial_object_pose"].flip(1),
                descriptors=cond["descriptors"].flip(1), action_objects=1-cond["action_objects"],
                action_targets=torch.where(cond["action_targets"] == 2, 2, 1-cond["action_targets"]))
            altered_state = dict(state, object_pose=state["object_pose"].flip(2), contact=state["contact"].flip(2))
            swapped = model(**altered_state, **permuted, t=t)
            for key in STATE:
                restored = swapped[key].flip(2) if key in {"contact", "object_pose"} else swapped[key]
                self.assertLess((first[key]-restored).abs().max().item(), 2e-5)
            reverse = dict(cond, **{key: cond[key].flip(1) for key in ["action_ids", "action_objects", "action_targets"]})
            response = model(**state, **reverse, t=t)
            self.assertGreater((first["object_pose"]-response["object_pose"]).abs().max().item(), 1e-6)

    def test_sampler_updates_future_objects_from_noise(self):
        cond = conditions()
        generated = sample(ConstantHeads(), cond, seed=19, steps=2)
        self.assertEqual(set(generated), set(STATE))
        self.assertTrue(torch.allclose(generated["object_pose"], torch.full((2, 2, 2, 9), .125), atol=1e-6))
        self.assertFalse(torch.equal(generated["object_pose"][:, 0], cond["initial_object_pose"]))
        real = sample(self.model, cond, seed=19, steps=2)
        again = sample(self.model, cond, seed=19, steps=2)
        for key in STATE:
            self.assertTrue(torch.isfinite(real[key]).all())
            self.assertTrue(torch.equal(real[key], again[key]))


if __name__ == "__main__":
    unittest.main()
