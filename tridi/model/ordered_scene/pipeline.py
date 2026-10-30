"""Joint scene diffusion from initial objects and two ordered action conditions.

Training contacts are binary; denoising and sampling use signed contact scores.
Metadata declarations require a separate source audit, not just this schema check.
"""
from numbers import Integral

import torch
from diffusers import DDIMScheduler

from .protocol import validate_representation


STATE = ("shape", "human", "object_pose", "contact")
CONDITIONS = ("initial_object_pose", "descriptors", "action_ids", "action_objects", "action_targets")
STATE_SHAPES = {"shape": (10,), "human": (2, 315), "object_pose": (2, 2, 9), "contact": (2, 2, 690)}
INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def scheduler():
    return DDIMScheduler(num_train_timesteps=1000, beta_start=1e-5, beta_end=.008,
                         beta_schedule="linear", prediction_type="sample", clip_sample=False)


def sample_tri_timesteps(batch, generator=None):
    """Use TriDi's seven sparse patterns and one shared nonzero level per row.

    A zero clock is the scheduler's near-clean t=0, not an exactly clean input.
    Sampling all modalities jointly uses the same clock for all three branches.
    """
    if not isinstance(batch, Integral) or isinstance(batch, bool) or batch < 1:
        raise ValueError("Timestep batch must be positive")
    patterns = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1],
                             [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    selected = torch.randint(0, len(patterns), (batch,), generator=generator)
    level = torch.randint(1, 1000, (batch, 1), generator=generator)
    return patterns[selected] * level


def _keys(values, expected, label):
    if not isinstance(values, dict) or set(values) != set(expected):
        raise ValueError(f"{label} requires exactly {tuple(expected)}")


def _tensor(value, shape, label, integer=False):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{label} must be a tensor of shape {shape}")
    valid_dtype = value.dtype in INTEGER_DTYPES if integer else value.is_floating_point()
    if not valid_dtype or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} has an invalid dtype or nonfinite values")


def validate_conditions(conditions, num_actions, descriptor_dim=12):
    """Reject future poses/body/contact values at the inference boundary."""
    _keys(conditions, CONDITIONS, "conditions")
    initial = conditions["initial_object_pose"]
    if not isinstance(initial, torch.Tensor) or initial.ndim != 3 or len(initial) == 0:
        raise ValueError("initial_object_pose must contain a nonempty batch of two objects")
    if not isinstance(num_actions, Integral) or isinstance(num_actions, bool) or num_actions < 1:
        raise ValueError("num_actions must be a positive vocabulary size")
    batch = len(initial)
    _tensor(initial, (batch, 2, 9), "initial_object_pose")
    _tensor(conditions["descriptors"], (batch, 2, descriptor_dim), "descriptors")
    for key, limit in [("action_ids", num_actions), ("action_objects", 2), ("action_targets", 3)]:
        value = conditions[key]
        _tensor(value, (batch, 2), key, integer=True)
        if bool(((value < 0) | (value >= limit)).any()):
            raise ValueError(f"{key} indices must be in [0, {limit}); target 2 means no target")
    return batch


def _validate_state(state, batch, label):
    _keys(state, STATE, label)
    for key, shape in STATE_SHAPES.items():
        _tensor(state[key], (batch, *shape), f"{label}.{key}")


def validate_training_data(data, descriptor_dim=12, allow_weak=False):
    """Validate the recorded S0 -> S1 -> S2 supervision contract."""
    _keys(data, ("conditions", "targets", "metadata", "vocabulary", "representation"), "training data")
    validate_representation(data["representation"])
    vocabulary = data["vocabulary"]
    if not isinstance(vocabulary, list) or not vocabulary or any(not isinstance(v, str) or not v.strip() for v in vocabulary):
        raise ValueError("vocabulary must list nonempty action names")
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("Action vocabulary entries must be unique")
    batch = validate_conditions(data["conditions"], len(vocabulary), descriptor_dim)
    _validate_state(data["targets"], batch, "targets")
    contact = data["targets"]["contact"]
    if not bool(((contact == 0) | (contact == 1)).all()):
        raise ValueError("Recorded contact targets must be binary 0/1")
    metadata = data["metadata"]
    if not isinstance(metadata, list) or len(metadata) != batch:
        raise ValueError("metadata must contain one source record per scene")
    groups = {}
    for row in metadata:
        if not isinstance(row, dict):
            raise ValueError("Each metadata entry must be a source record")
        if row.get("split") not in {"train", "val", "test"}:
            raise ValueError("Each source record requires train/val/test split")
        for key in ("subject", "sequence_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Each source record requires {key}")
        human = row.get("action_annotation") == "verified" and row.get("review_status") == "accepted_by_human"
        weak = (allow_weak and row.get("action_annotation") == "weak_temporal_v1"
                and row.get("review_status") == "machine_checked")
        if (row.get("status") == "pending" or row.get("two_task_objects") is not True
                or row.get("provenance") != "recorded" or not (human or weak)):
            raise ValueError("Requires recorded two-task-object sequences with human-verified labels or explicitly allowed weak_temporal_v1 labels")
        if weak:
            evidence = row.get("weak_labeling")
            if (not isinstance(evidence, dict) or evidence.get("algorithm") != "weak_temporal_v1"
                    or not isinstance(evidence.get("thresholds"), dict) or not evidence["thresholds"]
                    or not isinstance(evidence.get("evidence"), dict) or not evidence["evidence"]):
                raise ValueError("Weak labels require their algorithm, thresholds and nonempty evidence")
        initial, frames = row.get("initial_frame"), row.get("target_frames")
        if not isinstance(frames, (list, tuple)) or len(frames) != 2:
            raise ValueError("target_frames must identify S1 and S2")
        times = [initial, *frames]
        if any(not isinstance(t, Integral) or isinstance(t, bool) for t in times) or not 0 <= times[0] < times[1] < times[2]:
            raise ValueError("Require source frame indices 0 <= S0 < S1 < S2")
        group = (row["subject"], row["sequence_id"])
        if group in groups and groups[group] != row["split"]:
            raise ValueError("A recorded sequence cannot cross data splits")
        groups[group] = row["split"]
    return batch


def _model_conditions(model, conditions):
    batch = validate_conditions(conditions, model.num_actions, model.descriptor_dim)
    parameter = next(model.parameters())
    values = {key: value.to(device=parameter.device, dtype=(torch.long if key.startswith("action_") else parameter.dtype))
              for key, value in conditions.items()}
    return batch, values, parameter


@torch.no_grad()
def sample(model, conditions, seed, steps=40):
    """Generate all four state tensors from noise; accepts no target state."""
    if not isinstance(steps, Integral) or isinstance(steps, bool) or not 1 <= steps <= 1000:
        raise ValueError("DDIM steps must be between 1 and 1000")
    batch, cond, parameter = _model_conditions(model, conditions)
    schedule = scheduler()
    schedule.set_timesteps(steps, device=parameter.device)
    generator = torch.Generator(device=parameter.device).manual_seed(seed)
    state = {key: torch.randn((batch, *shape), generator=generator, device=parameter.device, dtype=parameter.dtype)
             for key, shape in STATE_SHAPES.items()}
    was_training = model.training
    model.eval()
    try:
        for timestep in schedule.timesteps:
            predicted = model(**state, **cond, t=timestep.expand(batch))
            _validate_state(predicted, batch, "prediction")
            state = {key: schedule.step(predicted[key], timestep, state[key], eta=0.).prev_sample for key in STATE}
    finally:
        model.train(was_training)
    _validate_state(state, batch, "sample")
    return state


def reconstruction_loss(prediction, target, scales=None):
    """TriDi loss families; contact states here are signed scores, not binary."""
    if scales is not None and (not isinstance(scales, dict) or not set(scales) <= set(STATE)):
        raise ValueError("Loss scales may only describe the four generated state tensors")
    errors = {}
    for key in STATE:
        scale = torch.as_tensor((scales or {}).get(key, 1.), device=target[key].device, dtype=target[key].dtype)
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("Loss scales must be finite and positive")
        errors[key] = (prediction[key] - target[key]) / scale
    # Shape is shared across frames; weight it by its channels within the human modality.
    body_size = target["shape"].numel() + target["human"].numel()
    body = (errors["shape"].abs().sum() + errors["human"].abs().sum()) / body_size
    objects = errors["object_pose"].abs().mean()
    positive = target["contact"] > 0
    count = positive.sum(-1, keepdim=True)
    ratio = ((690 - count) / count.clamp_min(1)).clamp(1, 50)
    weight = torch.where(positive, ratio, torch.ones_like(ratio))
    contacts = (errors["contact"].square() * weight).sum() / weight.sum()
    return body + objects + contacts


def diffusion_loss(model, targets, conditions, t, noise=None, scales=None):
    """Denoise native human, future objects and contacts with separate clocks."""
    batch, cond, parameter = _model_conditions(model, conditions)
    _validate_state(targets, batch, "targets")
    if not isinstance(t, torch.Tensor) or tuple(t.shape) not in {(batch,), (batch, 3)}:
        raise ValueError("t must have shape [B] or [B,3] for human/object/contact clocks")
    _tensor(t, tuple(t.shape), "t", integer=True)
    if bool(((t < 0) | (t >= 1000)).any()):
        raise ValueError("Diffusion timesteps must be in [0, 1000)")
    target = {key: value.to(device=parameter.device, dtype=parameter.dtype) for key, value in targets.items()}
    if not bool(((target["contact"] == 0) | (target["contact"] == 1)).all()):
        raise ValueError("Contact targets must be binary 0/1")
    target["contact"] = 2 * target["contact"] - 1
    if noise is not None:
        _validate_state(noise, batch, "noise")
        noise = {key: value.to(device=parameter.device, dtype=parameter.dtype) for key, value in noise.items()}
    else:
        noise = {key: torch.randn_like(value) for key, value in target.items()}
    schedule = scheduler()
    t = t.to(device=parameter.device, dtype=torch.long)
    times = t[:, None].expand(-1, 3) if t.ndim == 1 else t
    clock = {"shape": times[:, 0], "human": times[:, 0], "object_pose": times[:, 1], "contact": times[:, 2]}
    noisy = {key: schedule.add_noise(value, noise[key], clock[key]) for key, value in target.items()}
    prediction = model(**noisy, **cond, t=t)
    _validate_state(prediction, batch, "prediction")
    return reconstruction_loss(prediction, target, scales)
