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
from tridi.model.ordered_scene import pipeline
from tridi.model.ordered_scene.pipeline import (
    STATE, STATE_SHAPES, diffusion_loss, reconstruction_loss, sample, sample_tri_timesteps,
    validate_conditions, validate_training_data,
)
from tridi.model.ordered_scene.cli import (
    CHECKPOINT_SCHEMA, LEGACY_CHECKPOINT_SCHEMA, checkpoint_model_profile,
    validate_checkpoint, validate_inference_payload,
)
from tridi.model.ordered_scene.protocol import FIXED, index_sha256
from tridi.model.ordered_scene.regularization import contact_local_points
from tridi.utils.geometry import rotation_6d_to_matrix


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
            supervision="human_verified", model_profile="legacy",
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
    def __init__(self, num_actions, descriptor_dim=12, checkpoint=None, model_profile="legacy"):
        super().__init__()
        self.anchor = nn.Parameter(torch.randn(4))
        self.dropout = nn.Dropout(.25)
        self.num_actions, self.descriptor_dim = num_actions, descriptor_dim
        self.model_profile = model_profile

    def forward(self, **batch):
        return {key: self.anchor[i] + .01 * batch[key].tanh() + .01 * self.dropout(torch.ones_like(batch[key]))
                for i, key in enumerate(STATE)}

    def manifest(self):
        return dict(test_fixture=True)


class TinyContactGeometry:
    indices = list(range(690))
    _rotations = staticmethod(rotation_6d_to_matrix)

    def decode(self, shape, human):
        points = human[..., 312:].unsqueeze(2) + shape[:, None, None, :3]
        points = points.expand(-1, -1, 690, -1)
        return points, points


class OrderedContactLossContract(unittest.TestCase):
    def test_disabled_contact_loss_preserves_original_loss_and_rng(self):
        model = TinyTrainingHeads(3)
        truth, cond, clocks = targets(), conditions(), torch.tensor([[0, 200, 999], [700, 0, 0]])
        signed = dict(truth, contact=truth["contact"] * 2 - 1)
        outputs = []
        handle = model.register_forward_hook(lambda _model, _inputs, output: outputs.append(output))
        try:
            torch.manual_seed(519)
            original = diffusion_loss(model, truth, cond, clocks)
            rng = torch.get_rng_state().clone()
            expected = reconstruction_loss(outputs[-1], signed)
            torch.testing.assert_close(original, expected, atol=0, rtol=0)
            torch.manual_seed(519)
            with patch.object(pipeline, "relative_contact_loss", side_effect=AssertionError("geometry used")):
                disabled = diffusion_loss(model, truth, cond, clocks, geometry=object(),
                    contact_weight=0., target_contact_points=object())
            torch.testing.assert_close(disabled, original, atol=0, rtol=0)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        finally:
            handle.remove()

    def test_contact_gate_uses_human_and_object_clocks_and_preserves_cache_rows(self):
        model, truth, cond = RecordingHeads(), targets(), conditions()
        truth["contact"].fill_(1)
        geometry = TinyContactGeometry()
        points = torch.arange(2 * 2 * 2 * 690 * 3).reshape(2, 2, 2, 690, 3).float()
        clocks = torch.tensor([[250, 250, 999], [0, 251, 0]])
        noise = {key: torch.zeros_like(value) for key, value in truth.items()}
        base = diffusion_loss(model, truth, cond, clocks, noise=noise)
        with patch.object(pipeline, "relative_contact_loss", return_value=torch.tensor(3.)) as contact:
            regularized = diffusion_loss(model, truth, cond, clocks, noise=noise, geometry=geometry,
                contact_weight=.2, target_contact_points=points)
        torch.testing.assert_close(regularized, base + .6, atol=0, rtol=0)
        self.assertEqual(len(contact.call_args.args[1]["human"]), 1)
        torch.testing.assert_close(contact.call_args.kwargs["target_points"], points[:1])
        self.assertTrue((contact.call_args.args[2]["contact"] > 0).all())
        with patch.object(pipeline, "relative_contact_loss", side_effect=AssertionError("inactive geometry used")):
            inactive = clocks.clone()
            inactive[0, 0] = 251
            diffusion_loss(model, truth, cond, inactive, noise=noise, geometry=geometry, contact_weight=.2)
            truth["contact"].zero_()
            diffusion_loss(model, truth, cond, clocks, noise=noise, geometry=geometry, contact_weight=.2)
        for weight in [-1., float("inf"), float("nan"), True]:
            with self.assertRaisesRegex(ValueError, "contact_weight"):
                diffusion_loss(model, truth, cond, clocks, contact_weight=weight)
        for maximum in [-1, 1000, 250., True]:
            with self.assertRaisesRegex(ValueError, "contact_max_timestep"):
                diffusion_loss(model, truth, cond, clocks, contact_max_timestep=maximum)
        with self.assertRaisesRegex(ValueError, "geometry"):
            diffusion_loss(model, truth, cond, clocks, contact_weight=.1)

    def test_relative_contact_loss_has_real_shape_human_and_object_gradients(self):
        truth, cond, geometry = targets(), conditions(), TinyContactGeometry()
        truth["contact"].fill_(1)
        model = RecordingHeads()
        model.states = nn.ParameterDict({key: nn.Parameter(value.clone()) for key, value in truth.items()})
        with torch.no_grad():
            model.states["shape"][:, :3] += .03
            model.states["human"][..., 312:] += .02
            model.states["object_pose"][..., 6:] -= .01
        model.forward = lambda **_batch: dict(model.states.items())
        noise = {key: torch.zeros_like(value) for key, value in truth.items()}
        clocks = torch.tensor([0, 250])
        base = diffusion_loss(model, truth, cond, clocks, noise=noise)
        with torch.no_grad():
            cached = contact_local_points(geometry, truth)
        regularized = diffusion_loss(model, truth, cond, clocks, noise=noise, geometry=geometry,
            contact_weight=.1, target_contact_points=cached)
        self.assertGreater(float((regularized - base).detach()), 0.)
        (regularized - base).backward()
        for key in ("shape", "human", "object_pose"):
            grad = model.states[key].grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0., key)
        self.assertEqual(float(model.states["contact"].grad.abs().sum()), 0.)


class OrderedTrainingContract(unittest.TestCase):
    def test_relation_weak_version_is_explicit_in_data_and_checkpoint(self):
        data = training_fixture()
        for row in data["metadata"]:
            row.update(action_annotation="weak_temporal_v2", review_status="machine_checked",
                weak_labeling=dict(algorithm="weak_temporal_v2", thresholds={"distance_m": .05},
                evidence={"relation": "approach_other_object"}))
        with self.assertRaises(ValueError):
            validate_training_data(data)
        self.assertEqual(validate_training_data(data, allow_weak=True), 4)
        inconsistent = deepcopy(data)
        inconsistent["metadata"][0]["weak_labeling"]["algorithm"] = "weak_temporal_v1"
        with self.assertRaises(ValueError):
            validate_training_data(inconsistent, allow_weak=True)
        saved = dict(schema=cli.CHECKPOINT_SCHEMA, model_profile="root_affine_v1", data_validated=True,
            training_steps=1, supervision="weak_temporal_v2", vocabulary=data["vocabulary"],
            num_actions=len(data["vocabulary"]), descriptor_dim=12, representation=data["representation"])
        self.assertIs(validate_checkpoint(saved), saved)

    def test_contact_cache_uses_only_training_rows_before_model_seed(self):
        data = training_fixture()
        for row, split in zip(data["metadata"], ["train", "val", "train", "val"]):
            row["split"] = split
        data["targets"]["shape"][:, 0] = torch.tensor([10., 20., 30., 40.])
        decoded, instances, batches = [], [], []

        class CacheGeometry(TinyContactGeometry):
            def __init__(self, *args, **kwargs):
                instances.append(kwargs)
                torch.rand(17)  # Geometry construction must precede the model seed.

            def decode(self, shape, human):
                self_test.assertFalse(torch.is_grad_enabled())
                decoded.extend(shape[:, 0].tolist())
                return super().decode(shape, human)

        self_test = self
        original = diffusion_loss

        def checked_loss(model, truth, cond, clocks, **kwargs):
            self.assertEqual(kwargs["contact_weight"], .2)
            expected = contact_local_points(TinyContactGeometry(), truth)
            torch.testing.assert_close(kwargs["target_contact_points"], expected, atol=0, rtol=0)
            batches.append(truth["shape"][:, 0].tolist())
            return original(model, truth, cond, clocks, scales=kwargs["scales"])

        with TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(data, root / "data.pth")
            args = SimpleNamespace(data=root / "data.pth", output=root / "baseline", steps=2,
                batch_size=2, lr=1e-3, initialization_checkpoint=None, seed=71, device="cpu",
                eval_every=2, eval_steps=2, max_val_samples=1, log_every=1)
            fake_module = SimpleNamespace(NativeSceneGeometry=CacheGeometry)
            with patch.object(cli, "OrderedSceneDenoiser", TinyTrainingHeads), redirect_stdout(StringIO()):
                cli.train(args)
                baseline = torch.load(args.output / "last.pth", weights_only=False)
                args.output, args.contact_weight = root / "contact", .2
                args.models_root, args.meshes_root = root / "models", root / "meshes"
                with patch.dict("sys.modules", {"tridi.model.ordered_scene.geometry": fake_module}), \
                        patch.object(cli, "diffusion_loss", side_effect=checked_loss):
                    cli.train(args)
                trained = torch.load(args.output / "last.pth", weights_only=False)
                best = torch.load(args.output / "best.pth", weights_only=False)
            self.assertEqual(decoded, [10., 30.])
            self.assertEqual(len(instances), 1)
            self.assertEqual(instances[0]["representation"], data["representation"])
            self.assertEqual(len(batches), 2)
            self.assertEqual(trained["train_settings"]["contact_weight"], .2)
            self.assertEqual(best["contact_regularization"]["max_timestep"], 250)
            for key in baseline["model"]:
                self.assertTrue(torch.equal(baseline["model"][key], trained["model"][key]))
            self.assertTrue(torch.equal(baseline["torch_rng_state"], trained["torch_rng_state"]))

    def test_checkpoint_profile_is_explicit_and_legacy_cannot_claim_affine(self):
        saved = dict(schema=CHECKPOINT_SCHEMA, data_validated=True, training_steps=1,
            supervision="human_verified", model_profile="root_affine_v1", representation=representation(),
            vocabulary=["move"], num_actions=1, descriptor_dim=12)
        self.assertEqual(checkpoint_model_profile(validate_checkpoint(saved)), "root_affine_v1")
        missing = dict(saved)
        del missing["model_profile"]
        with self.assertRaisesRegex(ValueError, "model_profile"):
            validate_checkpoint(missing)
        old = dict(missing, schema=LEGACY_CHECKPOINT_SCHEMA)
        self.assertEqual(checkpoint_model_profile(validate_checkpoint(old)), "legacy")
        with self.assertRaisesRegex(ValueError, "Legacy v2"):
            validate_checkpoint(dict(old, model_profile="root_affine_v1"))
        with self.assertRaises(ValueError):
            validate_checkpoint(dict(saved, model_profile="unknown"))

    def test_inference_rejects_target_relations_absent_from_training(self):
        saved = dict(schema=CHECKPOINT_SCHEMA, data_validated=True, training_steps=1,
            supervision="weak_temporal_v1", model_profile="legacy", representation=representation(),
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
                self.assertEqual(resumed["model_profile"], "legacy")
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
                args.model_profile = "root_affine_v1"
                with self.assertRaisesRegex(ValueError, "identical data"):
                    cli.train(args)
                args.model_profile = "legacy"
                args.contact_max_timestep = 249
                with self.assertRaisesRegex(ValueError, "identical data"):
                    cli.train(args)
                args.contact_max_timestep = 250
                args.contact_weight, args.models_root, args.meshes_root = .1, root / "models", root / "meshes"
                with self.assertRaisesRegex(ValueError, "identical data"):
                    cli.train(args)
                args.contact_weight = 0.
                data["conditions"]["descriptors"][0, 0, 0] += .1
                torch.save(data, path)
                with self.assertRaisesRegex(ValueError, "identical data"):
                    cli.train(args)


class OrderedModelProfileContract(unittest.TestCase):
    def test_affine_profile_preserves_weights_rng_and_only_changes_root_decoder(self):
        torch.set_num_threads(2)
        torch.manual_seed(811)
        legacy = OrderedSceneDenoiser(3, checkpoint=None, model_profile="legacy").eval()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(811)
        affine = OrderedSceneDenoiser(3, checkpoint=None, model_profile="root_affine_v1").eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(legacy.state_dict()), set(affine.state_dict()))
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, affine.state_dict()[name]), name)
        self.assertIsInstance(legacy.core.decoder_S_transl.projection[1], nn.SiLU)
        self.assertIsInstance(affine.core.decoder_S_transl.projection[1], nn.Identity)
        self.assertEqual(affine.manifest()["model_profile"], "root_affine_v1")
        cond, state, t = conditions(), targets(), torch.tensor([500, 500])
        with torch.no_grad():
            first = legacy(**state, **cond, t=t)
            second = affine(**state, **cond, t=t)
        for name in ("shape", "object_pose", "contact"):
            torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)
        torch.testing.assert_close(first["human"][..., :312], second["human"][..., :312], atol=0, rtol=0)
        self.assertGreater(float((first["human"][..., 312:] - second["human"][..., 312:]).abs().max()), 1e-4)
        # The affine profile can represent intermediate values below SiLU's global minimum.
        required = torch.full((2, 3), -2.)
        desired = affine.core.decoder_S_transl.projection[2](required)
        restored = affine.core.decoder_S_transl.projection[2](affine.core.decoder_S_transl.projection[1](required))
        torch.testing.assert_close(restored, desired, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
