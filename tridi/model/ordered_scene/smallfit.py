"""Training-set memorization diagnostic; its checkpoints are not production weights."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .cli import save_checkpoint, subset
from .evaluate import rotation_error_degrees, scene_metrics
from .geometry import NativeSceneGeometry
from .model import OrderedSceneDenoiser
from .pipeline import CONDITIONS, STATE, diffusion_loss, sample, sample_tri_timesteps, scheduler, validate_training_data
from .regularization import contact_local_points


SCHEMA = "himo-smallfit-diagnostic-v1"


def _digest(values, keys, index, quantize=False):
    digest = hashlib.sha256()
    for key in keys:
        value = values[key][index].detach().cpu().contiguous()
        if quantize and value.is_floating_point():
            value = (value.double() * 10000).round().long()
        digest.update(key.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def select_train_indices(data, count, seed):
    """Reject conflicting near-identical conditions; never relabel val/test as train."""
    if type(count) is not int or count < 1:
        raise ValueError("Diagnostic scene count must be positive")
    train = [i for i, row in enumerate(data["metadata"]) if row["split"] == "train"]
    groups = {}
    for index in train:
        signature = _digest(data["conditions"], CONDITIONS, index, quantize=True)
        groups.setdefault(signature, []).append(index)
    eligible = [indices[0] for indices in groups.values()
                if len({_digest(data["targets"], STATE, i) for i in indices}) == 1]
    if len(eligible) < count:
        raise ValueError("Not enough unique nonconflicting training conditions for the requested diagnostic")
    def coverage(index):
        return ({("action", int(value)) for value in data["conditions"]["action_ids"][index]}
                | {("slot", int(value)) for value in data["conditions"]["action_objects"][index]})
    required = set().union(*(coverage(i) for i in train))
    generator = torch.Generator().manual_seed(seed)
    pool = [eligible[i] for i in torch.randperm(len(eligible), generator=generator).tolist()]
    chosen, seen_pairs = [], set()
    while len(chosen) < count:
        index = max(pool, key=lambda i: (len(coverage(i) & required),
                    tuple(data["metadata"][i]["object_names"]) not in seen_pairs))
        chosen.append(index)
        required -= coverage(index)
        seen_pairs.add(tuple(data["metadata"][index]["object_names"]))
        pool.remove(index)
    if required:
        raise ValueError("Selected diagnostic set does not cover the training action and object-slot classes")
    return chosen


def activation_statistics(preactivation, output, activation="SiLU"):
    pre, out = preactivation.detach().float(), output.detach().float()
    sigmoid = pre.sigmoid()
    silu_derivative = sigmoid * (1 + pre * (1 - sigmoid))
    if activation not in {"SiLU", "Identity"}:
        raise ValueError("Translation diagnostic only supports the actual SiLU or Identity activation")
    derivative = silu_derivative if activation == "SiLU" else torch.ones_like(pre)
    flatten = lambda value: value.reshape(-1, value.shape[-1])
    return dict(activation=activation, pre_activation_min=float(pre.min()), pre_activation_max=float(pre.max()),
        pre_activation_mean=float(pre.mean()), pre_activation_channel_std=flatten(pre).std(0, unbiased=False).cpu().tolist(),
        active_derivative_abs_mean=float(derivative.abs().mean()),
        active_derivative_rms=float(derivative.square().mean().sqrt()),
        active_derivative_abs_below_001_fraction=float((derivative.abs() < .01).float().mean()),
        hypothetical_silu_derivative_abs_mean=float(silu_derivative.abs().mean()),
        hypothetical_silu_derivative_rms=float(silu_derivative.square().mean().sqrt()),
        output_min=float(out.min()), output_max=float(out.max()),
        output_channel_std=flatten(out).std(0, unbiased=False).cpu().tolist(),
        output_batch_variance=out.var(0, unbiased=False).cpu().tolist())


@contextmanager
def observe_translation_heads(model):
    """Hooks record values only; they never replace tensors or gradients."""
    pre, raw, result, handles = {}, {}, {}, []
    for name, module in (("human_root", model.core.decoder_S_transl),
                         ("object_translation", model.core.decoder_O_transl)):
        def before(_module, _inputs, output, name=name):
            pre[name] = output.detach()
        def after(_module, _inputs, output, name=name):
            raw[name] = (pre[name], output.detach(), type(_module.projection[1]).__name__)
        handles.extend([module.projection[0].register_forward_hook(before), module.register_forward_hook(after)])
    try:
        yield result
    finally:
        for handle in handles:
            handle.remove()
        result.update({name: activation_statistics(*values) for name, values in raw.items()})


def gradient_statistics(model):
    output = {}
    for label, module in (("human_root", model.core.decoder_S_transl),
            ("human_global_orientation", model.core.decoder_S_orient),
            ("human_joints", model.core.decoder_S_pose), ("shape", model.core.decoder_S_shape),
            ("object_translation", model.core.decoder_O_transl)):
        gradients = [p.grad.detach().float().flatten() for p in module.parameters() if p.grad is not None]
        if not gradients:
            output[label] = dict(parameter_count=sum(p.numel() for p in module.parameters()), gradient_elements=0)
            continue
        values = torch.cat(gradients)
        output[label] = dict(parameter_count=sum(p.numel() for p in module.parameters()),
            gradient_elements=values.numel(), grad_l1_sum=float(values.abs().sum()),
            grad_l1_mean=float(values.abs().mean()), grad_l2=float(values.norm()),
            grad_rms=float(values.square().mean().sqrt()))
    return output


def normalized_body_errors(prediction, target, scales):
    human = (prediction["human"] - target["human"]) / scales["human"]
    shape = (prediction["shape"] - target["shape"]) / scales["shape"]
    parts = dict(root=human[..., 312:].abs(), global_orientation=human[..., :6].abs(),
                 joints=human[..., 6:312].abs(), shape=shape.abs())
    denominator = human.numel() + shape.numel()
    return dict(standardized_mae={key: float(value.mean()) for key, value in parts.items()},
        baseline_body_loss_contributions={key: float(value.sum() / denominator) for key, value in parts.items()})


@torch.no_grad()
def denoise_once(model, targets, conditions, timestep, seed):
    """Target-noised reconstruction probe, explicitly separate from generation."""
    parameter = next(model.parameters())
    state = {key: value.to(parameter.device) for key, value in targets.items()}
    state["contact"] = state["contact"] * 2 - 1
    cond = {key: value.to(parameter.device) for key, value in conditions.items()}
    clocks = torch.full((len(state["shape"]),), timestep, device=parameter.device, dtype=torch.long)
    generator = torch.Generator(device=parameter.device).manual_seed(seed)
    schedule = scheduler()
    noisy = {key: schedule.add_noise(value, torch.randn(value.shape, device=value.device,
            dtype=value.dtype, generator=generator), clocks) for key, value in state.items()}
    was_training = model.training
    model.eval()
    try:
        with observe_translation_heads(model) as activations:
            predicted = model(**noisy, **cond, t=clocks)
    finally:
        model.train(was_training)
    return {key: value.cpu() for key, value in predicted.items()}, activations


def macro_f1(found, truth):
    true_positive = (found & truth).sum(-1).float()
    denominator = found.sum(-1) + truth.sum(-1)
    return float((2 * true_positive / denominator.clamp_min(1)).mean())


@torch.no_grad()
def assess(model, targets, conditions, geometry, true_joints, scales, seed, sampling_steps):
    predictions, activations = {}, {}
    for name, clock in (("low_noise_t0", 0), ("high_noise_t500", 500)):
        predictions[name], activations[name] = denoise_once(model, targets, conditions, clock, seed + 70001)
    with observe_translation_heads(model) as final_activations:
        predictions["ddim_generation"] = {key: value.cpu() for key, value in sample(
            model, conditions, seed + 90001, steps=sampling_steps).items()}
    activations["ddim_generation_final_step"] = final_activations
    scores = {}
    for name, predicted in predictions.items():
        joints = geometry.decode(predicted["shape"], predicted["human"])[1]
        values = scene_metrics(predicted, targets, joints, true_joints)
        scores[name] = {key: float(value.mean()) for key, value in values.items()}
        scores[name]["body_loss_breakdown"] = normalized_body_errors(predicted, targets, scales)
    return scores, activations, predictions


def run(args):
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1 or not 1 <= args.sampling_steps <= 1000:
        raise ValueError("Steps, batch size and evaluation interval must be positive; DDIM steps must be 1..1000")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Learning rate must be positive")
    if args.export_count < 0 or args.model_profile not in {"legacy", "root_affine_v1"}:
        raise ValueError("Export count must be nonnegative; use a declared model profile")
    if (args.loss_profile not in {"baseline", "relative_contact_v1"} or
            not math.isfinite(args.contact_weight) or args.contact_weight < 0 or
            (args.loss_profile == "baseline" and args.contact_weight != 0) or
            (args.loss_profile == "relative_contact_v1" and args.contact_weight <= 0)):
        raise ValueError("Baseline requires contact weight 0; relative_contact_v1 requires a finite positive contact weight")
    if type(args.contact_max_timestep) is not int or not 0 <= args.contact_max_timestep < 1000:
        raise ValueError("Contact maximum timestep must be an integer in [0, 1000)")
    if args.output.exists() and args.resume is None:
        raise ValueError("Use a new diagnostic output directory")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    validate_training_data(data, allow_weak=args.allow_weak)
    indices = select_train_indices(data, args.scenes, args.seed)
    all_train = [i for i, row in enumerate(data["metadata"]) if row["split"] == "train"]
    targets, conditions = subset(data["targets"], indices, "cpu"), subset(data["conditions"], indices, "cpu")
    scales = {key: data["targets"][key][all_train].std(0, unbiased=False).clamp_min(.1)
              for key in STATE if key != "contact"}
    names = [data["metadata"][i]["object_names"] for i in indices]
    geometry = NativeSceneGeometry(args.models_root, args.meshes_root, device="cpu", representation=data["representation"])
    true_joints = geometry.decode(targets["shape"], targets["human"])[1]
    static = conditions["initial_object_pose"][:, None].expand(-1, 2, -1, -1)
    static_score = dict(object_translation_mm=float(1000 * torch.linalg.vector_norm(
        static[..., 6:] - targets["object_pose"][..., 6:], dim=-1).mean()),
        object_rotation_degrees=float(rotation_error_degrees(static[..., :6], targets["object_pose"][..., :6]).mean()))
    settings = dict(data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(), indices=indices,
        seed=args.seed, batch_size=args.batch_size, device=args.device, lr=args.lr,
        eval_every=args.eval_every, sampling_steps=args.sampling_steps, loss_profile=args.loss_profile,
        model_profile=args.model_profile, contact_weight=args.contact_weight,
        contact_max_timestep=args.contact_max_timestep,
        contact_gating="max(human_timestep, object_timestep) <= contact_max_timestep",
        loss_scale_source="all original train records; no val/test", optimizer="AdamW", weight_decay=1e-6, betas=[.95, .999])
    saved = None
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (saved.get("schema") != SCHEMA or saved.get("settings") != settings or
                Path(args.resume).resolve().parent != args.output.resolve() or saved["step"] >= args.steps):
            raise ValueError("Resume requires the same diagnostic settings, output directory and a larger total step budget")
    training_geometry, contact_points = None, None
    if args.contact_weight > 0:
        training_geometry = NativeSceneGeometry(args.models_root, args.meshes_root,
            device=args.device, representation=data["representation"])
        with torch.no_grad():
            contact_points = contact_local_points(training_geometry, targets).detach()
    torch.manual_seed(args.seed)
    model_args = dict(checkpoint=None if saved else args.init_checkpoint)
    if args.model_profile != "legacy":
        model_args["model_profile"] = args.model_profile
    model = OrderedSceneDenoiser(len(data["vocabulary"]), **model_args).to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-6, betas=(.95, .999))
    generator = torch.Generator().manual_seed(args.seed)
    history, start_step = [], 0
    if saved is not None:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if torch.device(args.device).type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng_state"], device=args.device)
        history, start_step = saved["history"], saved["step"]
    args.output.mkdir(parents=True, exist_ok=args.resume is not None)
    protocol = dict(schema=SCHEMA, diagnostic_only=True, independent_evaluation=False,
        note="Training and assessment use the same selected original-train records. This is a memorization test, not a validation or generalization result.",
        settings=settings, selected_sources=[data["metadata"][i] for i in indices],
        conditions_fingerprint_policy="Initial poses/descriptors quantized to 1e-4; integer action conditions exact. Conflicting future states at one fingerprint are excluded.",
        representation=data["representation"], initialization=saved["initialization"] if saved else model.manifest(),
        static_object_baseline=static_score, contact_empty_map_policy="Empty/empty maps contribute zero F1, matching held-out evaluation")
    def write_json(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    write_json("protocol.json", protocol)
    if start_step == 0:
        scores, activations, _ = assess(model, targets, conditions, geometry, true_joints, scales, args.seed, args.sampling_steps)
        history.append(dict(step=0, metrics=scores, translation_head_activations=activations))
        write_json("history.json", history)
        print(json.dumps(history[-1]), flush=True)
    started = time.monotonic()
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        ids = torch.randint(len(indices), (args.batch_size,), generator=generator).tolist()
        clocks = sample_tri_timesteps(len(ids), generator=generator)
        checked = step == 1 or step % args.eval_every == 0 or step == args.steps
        optimizer.zero_grad(set_to_none=True)
        loss_args = dict(scales=scales, geometry=training_geometry, contact_weight=args.contact_weight,
            target_contact_points=contact_points[ids] if contact_points is not None else None,
            contact_max_timestep=args.contact_max_timestep)
        if checked:
            with observe_translation_heads(model) as training_activations:
                loss = diffusion_loss(model, subset(targets, ids, args.device), subset(conditions, ids, args.device), clocks, **loss_args)
        else:
            loss = diffusion_loss(model, subset(targets, ids, args.device), subset(conditions, ids, args.device), clocks, **loss_args)
        loss.backward()
        gradients = gradient_statistics(model) if checked else None
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise ValueError("Nonfinite diagnostic loss or gradient")
        optimizer.step()
        if checked:
            scores, activations, predictions = assess(model, targets, conditions, geometry, true_joints, scales, args.seed, args.sampling_steps)
            history.append(dict(step=step, training_loss=float(loss.detach()), gradient_norm_before_clip=float(norm),
                decoder_gradients_before_clip=gradients, training_translation_head_activations=training_activations,
                metrics=scores, translation_head_activations=activations))
            write_json("history.json", history)
            save_checkpoint(args.output / "last.pth", dict(schema=SCHEMA, diagnostic_only=True,
                model=model.state_dict(), optimizer=optimizer.state_dict(), generator_state=generator.get_state(),
                torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state(args.device)
                    if torch.device(args.device).type == "cuda" else None,
                settings=settings, step=step, history=history, representation=data["representation"],
                initialization=protocol["initialization"], vocabulary=data["vocabulary"]))
            print(json.dumps(dict(step=step, training_loss=float(loss.detach()), metrics=scores,
                elapsed_seconds=time.monotonic()-started)), flush=True)
        elif step % 50 == 0:
            print(json.dumps(dict(step=step, training_loss=float(loss.detach()), elapsed_seconds=time.monotonic()-started)), flush=True)
    final = predictions["ddim_generation"]
    observed = torch.cat([geometry.contacts(final["shape"][i:i+4], final["human"][i:i+4],
        final["object_pose"][i:i+4], names[i:i+4]).cpu() for i in range(0, len(indices), 4)]) > 0
    head, truth = final["contact"] > 0, targets["contact"] > 0
    contact = dict(head_vs_generated_geometry_macro_f1=macro_f1(head, observed),
        generated_geometry_vs_recorded_labels_macro_f1=macro_f1(observed, truth),
        generated_geometry_positive_fraction=float(observed.float().mean()),
        predicted_head_positive_fraction=float(head.float().mean()), recorded_positive_fraction=float(truth.float().mean()))
    report = dict(schema=SCHEMA, diagnostic_only=True, independent_evaluation=False,
        step=args.steps, settings=settings, initial=history[0], final=history[-1],
        static_object_baseline=static_score, actual_geometric_contacts=contact,
        ddim_beats_static_object_position=history[-1]["metrics"]["ddim_generation"]["object_translation_mm"] < static_score["object_translation_mm"],
        generated_geometry_corrected=False)
    write_json("report.json", report)
    torch.save(dict(diagnostic_only=True, conditions=conditions, recorded_targets=targets,
        predictions=predictions, geometry_contacts=observed, indices=indices), args.output / "samples.pth")
    if args.export_count:
        count = min(args.export_count, len(indices))
        for label, state in (("recorded", targets), ("generated", final)):
            geometry.export({key: value[:count] for key, value in state.items()}, names[:count], args.output / (label + "_meshes"))
    print(json.dumps(dict(output=str(args.output), report=report)), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "init-checkpoint", "models-root", "meshes-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--scenes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--export-count", type=int, default=2)
    parser.add_argument("--loss-profile", choices=["baseline", "relative_contact_v1"], default="baseline")
    parser.add_argument("--contact-weight", type=float, default=0.)
    parser.add_argument("--contact-max-timestep", type=int, default=250)
    parser.add_argument("--model-profile", choices=["legacy", "root_affine_v1"], default="legacy")
    parser.add_argument("--allow-weak", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        run(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
