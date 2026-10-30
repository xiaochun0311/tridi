from types import SimpleNamespace
import unittest

import torch
from torch import nn

from tridi.model.denoising.transformer_uni_3 import Projection
from tridi.model.ordered_scene.cli import validate_checkpoint
from tridi.model.ordered_scene.pipeline import CONDITIONS, STATE, STATE_SHAPES
from tridi.model.ordered_scene.smallfit import (SCHEMA, activation_statistics, denoise_once,
    gradient_statistics, normalized_body_errors, observe_translation_heads, select_train_indices)


def fixture():
    generator = torch.Generator().manual_seed(4)
    count = 9
    conditions = dict(initial_object_pose=torch.randn(count, 2, 9, generator=generator),
        descriptors=torch.randn(count, 2, 12, generator=generator),
        action_ids=torch.tensor([[i % 3, (i + 1) % 3] for i in range(count)]),
        action_objects=torch.tensor([[0, 1]] * count), action_targets=torch.full((count, 2), 2))
    targets = {key: torch.randn(count, *shape, generator=generator) for key, shape in STATE_SHAPES.items()}
    targets["contact"] = (targets["contact"] > 0).float()
    metadata = [dict(split="train" if i < 7 else "val" if i == 7 else "test",
        subject="S01", sequence_id=f"S01T{i:03}", object_names=[f"a{i}", f"b{i}"]) for i in range(count)]
    return dict(conditions=conditions, targets=targets, metadata=metadata)


class TinyHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Module()
        for name in ("decoder_S_transl", "decoder_O_transl", "decoder_S_shape", "decoder_S_orient", "decoder_S_pose"):
            setattr(self.core, name, Projection(3, 3))

    def forward(self, **values):
        self.last_condition_keys = {key for key in values if key not in STATE and key != "t"}
        out = {key: values[key].clone() for key in STATE}
        out["human"][..., 312:] = self.core.decoder_S_transl(values["human"][..., 312:])
        out["object_pose"][..., 6:] = self.core.decoder_O_transl(values["object_pose"][..., 6:])
        return out


class SmallfitDiagnosticTest(unittest.TestCase):
    def test_selection_is_repeatable_train_only_and_rejects_conflicting_conditions(self):
        data = fixture()
        # Rows 0/6 have one identical condition but contradicting recorded futures.
        for value in data["conditions"].values():
            value[6] = value[0]
        selected = select_train_indices(data, 4, 42)
        self.assertEqual(selected, select_train_indices(data, 4, 42))
        self.assertTrue(set(selected) <= {1, 2, 3, 4, 5})
        self.assertEqual(len(set(selected)), 4)
        actions = set(data["conditions"]["action_ids"][selected].flatten().tolist())
        self.assertEqual(actions, {0, 1, 2})
        self.assertEqual(data["metadata"][7]["split"], "val")
        self.assertEqual(data["metadata"][8]["split"], "test")
        with self.assertRaises(ValueError):
            select_train_indices(data, 6, 42)
        with self.assertRaises(ValueError):
            validate_checkpoint(dict(schema=SCHEMA, diagnostic_only=True))

    def test_stationary_silu_and_identity_probes_observe_without_changing_outputs(self):
        pre = torch.full((4, 2, 3), -1.278464542761)
        stats = activation_statistics(pre, torch.zeros_like(pre))
        self.assertLess(stats["active_derivative_rms"], 1e-6)
        self.assertEqual(stats["active_derivative_abs_below_001_fraction"], 1)
        identity = activation_statistics(pre, pre, "Identity")
        self.assertEqual(identity["active_derivative_rms"], 1)
        self.assertLess(identity["hypothetical_silu_derivative_rms"], 1e-6)
        model = TinyHeads()
        tensor = torch.randn(4, 2, 3)
        expected = model.core.decoder_S_transl(tensor)
        with observe_translation_heads(model) as observations:
            actual = model.core.decoder_S_transl(tensor)
            model.core.decoder_O_transl(tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.tensor(observations["human_root"]["output_batch_variance"]),
            actual.var(0, unbiased=False))
        self.assertFalse(model.core.decoder_S_transl._forward_hooks)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, .25)
        gradients = gradient_statistics(model)
        for value in gradients.values():
            self.assertEqual(value["parameter_count"], value["gradient_elements"])
            self.assertAlmostEqual(value["grad_rms"], .25)

    def test_reconstruction_noise_is_fixed_and_does_not_consume_training_rng(self):
        data = fixture()
        targets = {key: value[:2] for key, value in data["targets"].items()}
        conditions = {key: value[:2] for key, value in data["conditions"].items()}
        model = TinyHeads().train()
        torch.manual_seed(99)
        before = torch.get_rng_state().clone()
        first, _ = denoise_once(model, targets, conditions, 0, 123)
        again, _ = denoise_once(model, targets, conditions, 0, 123)
        high, _ = denoise_once(model, targets, conditions, 500, 123)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(model.training)
        self.assertEqual(model.last_condition_keys, set(CONDITIONS))
        for key in STATE:
            torch.testing.assert_close(first[key], again[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(first["human"], high["human"]))
        self.assertLess(float((first["contact"] - (2 * targets["contact"] - 1)).abs().mean()), .01)

    def test_body_breakdown_uses_actual_normalization_without_inventing_gradient_weights(self):
        target = dict(shape=torch.zeros(2, 10), human=torch.zeros(2, 2, 315))
        prediction = {key: value.clone() for key, value in target.items()}
        prediction["human"][..., 312:] = .2
        scales = dict(shape=torch.ones(10), human=torch.full((2, 315), .1))
        result = normalized_body_errors(prediction, target, scales)
        self.assertAlmostEqual(result["standardized_mae"]["root"], 2)
        self.assertAlmostEqual(result["baseline_body_loss_contributions"]["root"], 2 * 6 / 640)
        self.assertEqual(result["standardized_mae"]["joints"], 0)


if __name__ == "__main__":
    unittest.main()
