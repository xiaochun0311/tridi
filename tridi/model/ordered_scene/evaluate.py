"""Held-out generation checks; recorded future states never enter the sampler."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from .cli import validate_checkpoint, validate_inference_payload
from .geometry import NativeSceneGeometry
from .model import OrderedSceneDenoiser
from .pipeline import STATE, sample, validate_training_data
from .protocol import assert_same_representation


def rotation_error_degrees(predicted, recorded):
    first = NativeSceneGeometry._rotations(predicted)
    second = NativeSceneGeometry._rotations(recorded)
    relative = first @ second.transpose(-1, -2)
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))


def scene_metrics(predicted, recorded, predicted_joints, recorded_joints):
    """Return one scalar per scene; contact macro averages the four maps."""
    mean = lambda value: value.flatten(1).mean(1)
    found, truth = predicted["contact"] > 0, recorded["contact"] > 0
    true_positive = (found & truth).sum(-1).float()
    false_positive = (found & ~truth).sum(-1).float()
    false_negative = (~found & truth).sum(-1).float()
    return dict(
        object_translation_mm=1000 * mean(torch.linalg.vector_norm(
            predicted["object_pose"][..., 6:] - recorded["object_pose"][..., 6:], dim=-1)),
        object_rotation_degrees=mean(rotation_error_degrees(
            predicted["object_pose"][..., :6], recorded["object_pose"][..., :6])),
        human_root_translation_mm=1000 * mean(torch.linalg.vector_norm(
            predicted["human"][..., 312:] - recorded["human"][..., 312:], dim=-1)),
        native52_world_mpjpe_mm=1000 * mean(torch.linalg.vector_norm(
            predicted_joints - recorded_joints, dim=-1)),
        contact_macro_precision=mean(true_positive / (true_positive + false_positive).clamp_min(1)),
        contact_macro_recall=mean(true_positive / (true_positive + false_negative).clamp_min(1)),
        contact_macro_f1=mean(2 * true_positive / (2 * true_positive + false_positive + false_negative).clamp_min(1)),
    )


def select_test_indices(data, maximum, seed):
    if maximum < 1:
        raise ValueError("--max-test must be positive")
    groups = {split: [i for i, row in enumerate(data["metadata"]) if row["split"] == split]
              for split in ("train", "val", "test")}
    if not groups["train"] or not groups["test"]:
        raise ValueError("Evaluation requires explicit train and test splits")
    subjects = {split: {data["metadata"][i]["subject"] for i in ids} for split, ids in groups.items()}
    if subjects["test"] & (subjects["train"] | subjects["val"]):
        raise ValueError("Held-out-person evaluation requires test subjects absent from train and val")
    trained_actions = set(data["conditions"]["action_ids"][groups["train"]].flatten().tolist())
    test_actions = set(data["conditions"]["action_ids"][groups["test"]].flatten().tolist())
    if test_actions - trained_actions:
        raise ValueError("Test action IDs must have training examples for this closed-vocabulary evaluation")
    permutation = torch.randperm(len(groups["test"]), generator=torch.Generator().manual_seed(seed)).tolist()
    return [groups["test"][i] for i in permutation[:maximum]]


def swap_actions(conditions):
    return {key: value.flip(1) if key.startswith("action_") else value.clone()
            for key, value in conditions.items()}


def _mean(values):
    return {key: float(value.mean()) for key, value in values.items()}


def _generate(model, conditions, seed, steps, batch_size):
    output = {key: [] for key in STATE}
    for start in range(0, len(conditions["initial_object_pose"]), batch_size):
        batch = {key: value[start:start + batch_size] for key, value in conditions.items()}
        result = sample(model, batch, seed + start, steps)
        for key in STATE:
            output[key].append(result[key].cpu())
    return {key: torch.cat(parts) for key, parts in output.items()}


def _plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, key, title in zip(axes.flat,
            ("object_translation_mm", "object_rotation_degrees", "native52_world_mpjpe_mm", "contact_macro_f1"),
            ("Object translation (mm, lower is better)", "Object rotation (deg, lower is better)",
             "Human 52-joint world MPJPE (mm, lower is better)", "Contact macro F1 (higher is better)")):
        labels = ["Initialized", "Trained"]
        values = [report["initialized"]["mean"][key], report["trained"]["mean"][key]]
        if key in report["static_object_baseline"]:
            labels.append("Static objects")
            values.append(report["static_object_baseline"][key])
        bars = axis.bar(labels, values, color=["#a6adb7", "#357abd", "#dc9b50"][:len(values)])
        axis.bar_label(bars, fmt="%.3g", padding=3)
        axis.set_title(title, fontsize=10)
        axis.set_ylim(0, max(max(values) * 1.2, .01))
    figure.suptitle(f"Held-out people, {report['test_scenes_evaluated']} scenes; {report['supervision']}")
    figure.savefig(output, dpi=160)
    plt.close(figure)


@torch.no_grad()
def evaluate(args):
    if args.output.exists():
        raise ValueError("Use a new evaluation output directory")
    if args.batch_size < 1 or args.export_count < 0:
        raise ValueError("Batch size must be positive and export count nonnegative")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    validate_training_data(data, allow_weak=args.allow_weak)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(saved)
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != saved.get("data_sha256"):
        raise ValueError("Evaluation data must match the checkpoint's declared training dataset")
    assert_same_representation(saved["representation"], data["representation"])
    initialization = saved.get("initialization", {})
    if (type(saved.get("training_seed")) is not int or initialization.get("sha256") !=
            hashlib.sha256(args.initialization_checkpoint.read_bytes()).hexdigest()):
        raise ValueError("Initialization comparison requires the recorded training seed and matching TriDi initialization SHA")
    ids = select_test_indices(data, args.max_test, args.seed)
    conditions = {key: value[ids] for key, value in data["conditions"].items()}
    names = [data["metadata"][i]["object_names"] for i in ids]
    payload = dict(conditions=conditions, vocabulary=data["vocabulary"], representation=data["representation"], object_names=names)
    validate_inference_payload(saved, payload)
    geometry = NativeSceneGeometry(args.models_root, args.meshes_root, device="cpu", representation=data["representation"])
    if not torch.allclose(geometry.descriptors(names), conditions["descriptors"], atol=1e-6, rtol=1e-5):
        raise ValueError("Recorded object names do not match initial geometry descriptors")
    recorded = {key: value[ids].cpu() for key, value in data["targets"].items()}
    recorded_joints = geometry.decode(recorded["shape"], recorded["human"])[1]
    model = OrderedSceneDenoiser(saved["num_actions"], saved["descriptor_dim"], checkpoint=None).to(args.device)
    model.load_state_dict(saved["model"], strict=True)
    trained = _generate(model, conditions, args.seed, args.steps, args.batch_size)
    swapped_conditions = swap_actions(conditions)
    swapped = _generate(model, swapped_conditions, args.seed, args.steps, args.batch_size)
    del model
    torch.manual_seed(saved["training_seed"])
    model = OrderedSceneDenoiser(saved["num_actions"], saved["descriptor_dim"], checkpoint=args.initialization_checkpoint).to(args.device)
    initialized = _generate(model, conditions, args.seed, args.steps, args.batch_size)
    metrics = {name: scene_metrics(prediction, recorded, geometry.decode(prediction["shape"], prediction["human"])[1], recorded_joints)
               for name, prediction in (("initialized", initialized), ("trained", trained), ("swapped_actions", swapped))}
    stationary = conditions["initial_object_pose"][:, None].expand(-1, 2, -1, -1)
    static = dict(object_translation_mm=float(1000 * torch.linalg.vector_norm(stationary[..., 6:] - recorded["object_pose"][..., 6:], dim=-1).mean()),
                  object_rotation_degrees=float(rotation_error_degrees(stationary[..., :6], recorded["object_pose"][..., :6]).mean()))
    changed = torch.stack([(conditions[key] != swapped_conditions[key]).any(1)
                           for key in ("action_ids", "action_objects", "action_targets")]).any(0)
    alteration = 1000 * torch.linalg.vector_norm(trained["object_pose"][..., 6:] - swapped["object_pose"][..., 6:], dim=-1).mean((1, 2))
    report = dict(supervision=saved.get("supervision", "unspecified"), training_steps=saved["training_steps"],
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), data_sha256=saved["data_sha256"],
        initialization=initialization, initialization_seed=saved["training_seed"], sampling_seed=args.seed,
        sampling_steps=args.steps, sampling_batch_size=args.batch_size, test_scenes_evaluated=len(ids),
        total_test_scenes=sum(row["split"] == "test" for row in data["metadata"]),
        test_subjects=sorted({data["metadata"][i]["subject"] for i in ids}), held_out_people=True,
        selected_indices=ids, action_counts=dict(Counter(data["vocabulary"][i] for i in conditions["action_ids"].flatten().tolist())),
        static_object_baseline=static, contact_empty_map_policy="Precision, recall and F1 are zero when their denominator is zero; macro averages S1/S2 x two objects",
        action_order_diagnostic=dict(changed_condition_scenes=int(changed.sum()),
            object_translation_change_mm=float(alteration.mean()),
            object_translation_change_on_changed_conditions_mm=float(alteration[changed].mean()) if bool(changed.any()) else None,
            interpretation="Same-noise sensitivity check only; the recorded reference belongs to the original order, so swapped-reference error does not establish order correctness."),
        limitations=["One sampled output is compared with one recorded future; plausible alternative motions can have high reference error.",
                     "Phase labels are automatic coarse motion labels when supervision is weak_temporal_v1; these tests do not establish natural-language action understanding.",
                     "Contact scores are compared with unsigned 5cm proximity labels, not grasp quality or penetration-free physics."])
    for name, values in metrics.items():
        report[name] = dict(mean=_mean(values), per_scene=[dict(sequence_id=data["metadata"][index]["sequence_id"],
            subject=data["metadata"][index]["subject"], **{key: float(value[j]) for key, value in values.items()})
            for j, index in enumerate(ids)])
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    torch.save(dict(conditions=payload, recorded_targets=recorded, trained=trained, initialized=initialized,
                    swapped_actions=swapped, note="Targets are stored for evaluation only and were never passed to sample()"), args.output / "samples.pth")
    count = min(args.export_count, len(ids))
    if count:
        for name, values in (("trained", trained), ("recorded", recorded)):
            geometry.export({key: value[:count] for key, value in values.items()}, names[:count], args.output / (name + "_meshes"))
    _plot(report, args.output / "heldout_comparison.png")
    print(json.dumps(dict(output=str(args.output), test_scenes=len(ids), trained=report["trained"]["mean"])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "checkpoint", "initialization-checkpoint", "models-root", "meshes-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-test", default=32, type=int)
    parser.add_argument("--steps", default=40, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--export-count", default=4, type=int)
    parser.add_argument("--allow-weak", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        evaluate(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
