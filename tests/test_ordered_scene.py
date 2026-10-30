"""Contract and wiring tests; random tensors here are not research training data."""
import os
import unittest
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from tridi.model.ordered_scene.model import OrderedSceneDenoiser
from tridi.model.ordered_scene import cli
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
            supervision="human_verified",
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


def training_fixture():
    cond, truth = conditions(), targets()
    cond["action_ids"] = torch.tensor([[0, 1], [1, 0]])
    return dict(conditions={key: torch.cat([value, value]) for key, value in cond.items()},
        targets={key: torch.cat([value, value]) for key, value in truth.items()},
        vocabulary=["move", "rotate", "unused"], representation=representation(),
        metadata=[dict(subject=f"fixture_{i}", sequence_id=f"unit_{i}", split=split,
            initial_frame=0, target_frames=[10, 20], two_task_objects=True, provenance="recorded",
            action_annotation="verified", review_status="accepted_by_human")
            for i, split in enumerate(["train", "train", "val", "val"])])


class TinyTrainingHeads(nn.Module):
    """Exercise real losses, optimizer moments and dropout RNG without the large transformer."""
    def __init__(self, num_actions, descriptor_dim=12, checkpoint=None):
        super().__init__()
        self.anchor = nn.Parameter(torch.randn(4))
        self.dropout = nn.Dropout(.25)
        self.num_actions, self.descriptor_dim = num_actions, descriptor_dim

    def forward(self, **batch):
        return {key: self.anchor[i] + .01 * batch[key].tanh() + .01 * self.dropout(torch.ones_like(batch[key]))
                for i, key in enumerate(STATE)}

    def manifest(self):
        return dict(test_fixture=True)


class OrderedTrainingContract(unittest.TestCase):
    def test_inference_rejects_target_relations_absent_from_training(self):
        saved = dict(schema=CHECKPOINT_SCHEMA, data_validated=True, training_steps=1,
            supervision="weak_temporal_v1", representation=representation(),
            vocabulary=["move", "rotate", "unused"], num_actions=3, descriptor_dim=12,
            condition_coverage=dict(action_targets_seen=[2]))
        payload = dict(conditions=conditions(), vocabulary=saved["vocabulary"], representation=representation(),
            object_names=[["Cup", "Pot"], ["Glasses", "Case"]])
        payload["conditions"]["action_targets"].fill_(2)
        self.assertEqual(validate_inference_payload(saved, payload), 2)
        for slot in [0, 1]:
            payload["conditions"]["action_targets"][0, 0] = slot
            with self.assertRaisesRegex(ValueError, "not supervised"):
                validate_inference_payload(saved, payload)
        for seen in [[], [True], [3], ["2"], [2, 2]]:
            with self.assertRaises(ValueError):
                validate_checkpoint(dict(saved, condition_coverage=dict(action_targets_seen=seen)))

    def test_weak_annotations_require_opt_in_and_machine_evidence(self):
        data = training_fixture()
        for row in data["metadata"]:
            row.update(action_annotation="weak_temporal_v1", review_status="machine_checked",
                status="weak_temporal_v1", weak_labeling=dict(algorithm="weak_temporal_v1",
                thresholds={"translation_m": .03}, evidence={"observed_stage_count": 2}))
        with self.assertRaises(ValueError):
            validate_training_data(data)
        self.assertEqual(validate_training_data(data, allow_weak=True), 4)
        missing = deepcopy(data)
        del missing["metadata"][0]["weak_labeling"]["evidence"]
        with self.assertRaises(ValueError):
            validate_training_data(missing, allow_weak=True)
        data["metadata"][0]["status"] = "pending"
        with self.assertRaises(ValueError):
            validate_training_data(data, allow_weak=True)

    def test_closed_set_validation_rejects_unseen_actions(self):
        data = training_fixture()
        data["conditions"]["action_ids"][2, 0] = 2
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data.pth"
            torch.save(data, path)
            args = SimpleNamespace(data=path, output=root / "fit", steps=2, batch_size=2, lr=1e-3)
            with self.assertRaisesRegex(ValueError, "Closed-set"):
                cli.train(args)
            self.assertFalse(args.output.exists())

    def test_resume_preserves_optimizer_clocks_dropout_rng_and_validation(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data.pth"
            data = training_fixture()
            for row in data["metadata"]:
                row.update(action_annotation="weak_temporal_v1", review_status="machine_checked",
                    weak_labeling=dict(algorithm="weak_temporal_v1", thresholds={"translation_m": .03},
                    evidence={"observed_stage_count": 2}))
            torch.save(data, path)
            args = SimpleNamespace(data=path, output=root / "full", steps=4, batch_size=2, lr=1e-3,
                initialization_checkpoint=None, seed=71, device="cpu", allow_weak=True,
                eval_every=2, eval_steps=2, max_val_samples=1, log_every=1)
            with patch.object(cli, "OrderedSceneDenoiser", TinyTrainingHeads), redirect_stdout(StringIO()):
                with patch.object(cli, "sample", wraps=cli.sample) as sampler:
                    cli.train(args)
                    self.assertTrue(all(len(call.args[1]["action_ids"]) == 1 and call.kwargs["steps"] == 2
                                        for call in sampler.call_args_list))
                full = torch.load(args.output / "last.pth", weights_only=False)
                args.output = root / "resumed"
                original_loss, calls = cli.diffusion_loss, 0

                def interrupt_after_two(*values, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        raise RuntimeError("simulated interruption")
                    return original_loss(*values, **kwargs)

                with patch.object(cli, "diffusion_loss", side_effect=interrupt_after_two):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        cli.train(args)
                args.resume = args.output / "last.pth"
                interrupted = torch.load(args.resume, weights_only=False)
                self.assertEqual(interrupted["training_steps"], 2)
                cli.train(args)
                resumed = torch.load(args.resume, weights_only=False)
                self.assertEqual(resumed["training_steps"], 4)
                self.assertEqual(resumed["history"], full["history"])
                self.assertEqual(resumed["supervision"], "weak_temporal_v1")
                self.assertEqual(resumed["optimizer"]["param_groups"][0]["betas"], (.95, .999))
                self.assertEqual(resumed["optimizer"]["param_groups"][0]["weight_decay"], 1e-6)
                for key in full["model"]:
                    self.assertTrue(torch.equal(full["model"][key], resumed["model"][key]))
                for name in ["generator_state", "torch_rng_state"]:
                    self.assertTrue(torch.equal(full[name], resumed[name]))
                for index, values in full["optimizer"]["state"].items():
                    for name, value in values.items():
                        self.assertTrue(torch.equal(value, resumed["optimizer"]["state"][index][name]))
                args.steps, args.lr = 6, 2e-3
                with self.assertRaisesRegex(ValueError, "identical data"):
                    cli.train(args)
                args.lr = 1e-3
                data["conditions"]["descriptors"][0, 0, 0] += .1
                torch.save(data, path)
                with self.assertRaisesRegex(ValueError, "identical data"):
                    cli.train(args)


if __name__ == "__main__":
    unittest.main()
