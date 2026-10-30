"""Train or sample the ordered scene model; no rule-based pose refinement."""
import argparse
import hashlib
import json
from pathlib import Path

import math
import time

import torch

from .model import MODEL_PROFILES, OrderedSceneDenoiser
from .pipeline import (
    STATE, diffusion_loss, reconstruction_loss, sample, sample_tri_timesteps,
    validate_conditions, validate_training_data,
)
from .protocol import assert_same_representation, validate_representation


CHECKPOINT_SCHEMA = "tridi-himo-ordered-v3"
LEGACY_CHECKPOINT_SCHEMA = "tridi-himo-ordered-v2"


def subset(values, ids, device):
    return {key: value[ids].to(device) for key, value in values.items()}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def checkpoint_model_profile(saved):
    if saved.get("schema") == LEGACY_CHECKPOINT_SCHEMA:
        if saved.get("model_profile", "legacy") != "legacy":
            raise ValueError("Legacy v2 checkpoints can only use the legacy model profile")
        return "legacy"
    if saved.get("schema") != CHECKPOINT_SCHEMA or saved.get("model_profile") not in MODEL_PROFILES:
        raise ValueError("Native v3 checkpoints require an explicit supported model_profile")
    return saved["model_profile"]


def validate_checkpoint(saved):
    if (not isinstance(saved, dict) or saved.get("schema") not in {CHECKPOINT_SCHEMA, LEGACY_CHECKPOINT_SCHEMA}
            or saved.get("data_validated") is not True
            or saved.get("supervision") not in {"human_verified", "weak_temporal_v1"}
            or type(saved.get("training_steps")) is not int or saved["training_steps"] < 1):
        raise ValueError("Requires a trained native HIMO checkpoint; legacy, diagnostic and untrained weights are incompatible")
    checkpoint_model_profile(saved)
    validate_representation(saved.get("representation"))
    vocabulary = saved.get("vocabulary")
    if (not isinstance(vocabulary, list) or not vocabulary
            or any(not isinstance(v, str) or not v.strip() for v in vocabulary)
            or len(set(vocabulary)) != len(vocabulary)
            or saved.get("num_actions") != len(vocabulary) or saved.get("descriptor_dim") != 12):
        raise ValueError("Invalid checkpoint action vocabulary or geometry descriptor layout")
    if "condition_coverage" in saved:
        coverage = saved["condition_coverage"]
        seen = coverage.get("action_targets_seen") if isinstance(coverage, dict) else None
        if (not isinstance(seen, list) or not seen or any(type(value) is not int or not 0 <= value <= 2 for value in seen)
                or seen != sorted(set(seen))):
            raise ValueError("Checkpoint action_targets_seen must list sorted unique supervised target references in [0, 2]")
    return saved


def validate_inference_payload(saved, payload):
    """Return batch size after checking initial-only inputs and asset identities."""
    validate_checkpoint(saved)
    if not isinstance(payload, dict) or set(payload) != {"conditions", "vocabulary", "representation", "object_names"}:
        raise ValueError("Inference requires only conditions, vocabulary, representation and initial object_names")
    if payload["vocabulary"] != saved["vocabulary"]:
        raise ValueError("Inference action vocabulary must match the checkpoint")
    assert_same_representation(saved["representation"], payload["representation"])
    batch = validate_conditions(payload["conditions"], saved["num_actions"], saved["descriptor_dim"])
    if "condition_coverage" in saved:
        requested = set(payload["conditions"]["action_targets"].flatten().tolist())
        if not requested <= set(saved["condition_coverage"]["action_targets_seen"]):
            raise ValueError("Inference target references were not supervised in this checkpoint's training data")
    names = payload["object_names"]
    if (not isinstance(names, list) or len(names) != batch or any(
            not isinstance(row, (list, tuple)) or len(row) != 2
            or any(not isinstance(name, str) or not name.strip() for name in row) for row in names)):
        raise ValueError("object_names must bind two nonempty initial mesh names per scene")
    return batch


def train(args):
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    validate_training_data(data, allow_weak=getattr(args, "allow_weak", False))
    supervision = "weak_temporal_v1" if any(row["action_annotation"] == "weak_temporal_v1" for row in data["metadata"]) else "human_verified"
    indices = {split: [i for i, row in enumerate(data["metadata"]) if row["split"] == split]
               for split in ["train", "val", "test"]}
    if not indices["train"] or not indices["val"] or args.steps < 1:
        raise ValueError("Training requires separate train/val sequences and a positive step budget")
    if args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Batch size and learning rate must be positive")
    train_actions = set(data["conditions"]["action_ids"][indices["train"]].flatten().tolist())
    val_actions = set(data["conditions"]["action_ids"][indices["val"]].flatten().tolist())
    if not val_actions <= train_actions:
        raise ValueError("Closed-set validation requires every validation action to occur in training")
    coverage = dict(action_targets_seen=sorted(set(data["conditions"]["action_targets"][indices["train"]].flatten().tolist())))
    model_profile = getattr(args, "model_profile", "legacy")
    if model_profile not in MODEL_PROFILES:
        raise ValueError("Unknown model_profile")
    settings = dict(batch_size=args.batch_size, lr=args.lr, seed=args.seed, device=str(args.device),
        model_profile=model_profile,
        training_protocol="tri_sparse_x0_l1_balanced_mse_v1",
        weight_decay=getattr(args, "weight_decay", 1e-6), betas=list(getattr(args, "betas", (.95, .999))),
        eval_every=getattr(args, "eval_every", 200), eval_steps=getattr(args, "eval_steps", 20),
        max_val_samples=getattr(args, "max_val_samples", 32), log_every=getattr(args, "log_every", 50))
    for key in ("eval_every", "eval_steps", "max_val_samples", "log_every"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (settings["eval_steps"] > 1000 or not math.isfinite(settings["weight_decay"]) or settings["weight_decay"] < 0
            or len(settings["betas"]) != 2 or any(not math.isfinite(v) or not 0 <= v < 1 for v in settings["betas"])):
        raise ValueError("Invalid DDIM steps or AdamW hyperparameters")
    resume = getattr(args, "resume", None)
    if args.output.exists() and resume is None:
        raise ValueError("Use a new output directory to preserve earlier results")
    data_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
    saved = None
    if resume is not None:
        resume = Path(resume)
        if resume.name != "last.pth" or resume.resolve().parent != args.output.resolve():
            raise ValueError("Resume requires last.pth in the same training output directory")
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        validate_checkpoint(saved)
        saved_settings = dict(saved.get("train_settings", {}))
        saved_settings.setdefault("model_profile", checkpoint_model_profile(saved))
        if (saved.get("checkpoint_kind") != "last" or saved.get("data_sha256") != data_sha
                or saved.get("supervision") != supervision or saved_settings != settings
                or checkpoint_model_profile(saved) != model_profile
                or saved.get("condition_coverage") != coverage
                or saved.get("vocabulary") != data["vocabulary"]):
            raise ValueError("Resume requires identical data, supervision, vocabulary and training/evaluation hyperparameters")
        assert_same_representation(saved["representation"], data["representation"])
        if saved["training_steps"] >= args.steps:
            raise ValueError("Total steps must exceed the completed resume step")
    torch.manual_seed(args.seed)
    model = OrderedSceneDenoiser(len(data["vocabulary"]), checkpoint=None if saved else args.initialization_checkpoint,
        model_profile=model_profile).to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]))
    scales = {key: data["targets"][key][indices["train"]].std(0, unbiased=False).clamp_min(.1)
              for key in STATE if key != "contact"}
    args.output.mkdir(parents=True, exist_ok=resume is not None)
    protocol = dict(input_sha256=data_sha,
        seed=args.seed, steps=args.steps, split_sizes={key: len(ids) for key, ids in indices.items()},
        model=model.manifest(), vocabulary=data["vocabulary"],
        representation=data["representation"], checkpoint_schema=CHECKPOINT_SCHEMA,
        model_profile=model_profile,
        supervision=supervision, train_settings=settings,
        condition_coverage=coverage,
        conditioning="Only initial object poses/geometry and ordered action references",
        targets="Recorded S1/S2 body, both object poses, and contacts; no IK in sampling",
        data_note="Schema declarations require a separately audited recorded data source")
    generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 104729)
    order = torch.randperm(len(indices["val"]), generator=val_generator).tolist()
    val_subset = [indices["val"][i] for i in order[:settings["max_val_samples"]]]
    protocol["validation_indices"] = val_subset
    best, history, start_step = float("inf"), [], 0
    cuda = torch.device(args.device).type == "cuda"
    if saved is not None:
        required = {"optimizer", "generator_state", "torch_rng_state", "cuda_rng_state", "best_score", "history", "protocol"}
        if not required <= set(saved):
            raise ValueError("Resume checkpoint is missing optimizer, RNG or history state")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if cuda:
            if saved["cuda_rng_state"] is None:
                raise ValueError("CUDA resume requires the original CUDA RNG state")
            torch.cuda.set_rng_state(saved["cuda_rng_state"], device=args.device)
        best, history, start_step = saved["best_score"], saved["history"], saved["training_steps"]
        protocol["model"] = saved["protocol"]["model"]
        if saved["protocol"]["validation_indices"] != val_subset:
            raise ValueError("Resume validation subset does not match")
    save_json(args.output / "protocol.json", protocol)
    save_json(args.output / "history.json", history)
    if cuda:
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.monotonic()
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        chosen = torch.randint(len(indices["train"]), (args.batch_size,), generator=generator)
        ids = [indices["train"][i] for i in chosen.tolist()]
        t = sample_tri_timesteps(len(ids), generator=generator)
        loss = diffusion_loss(model, subset(data["targets"], ids, args.device),
            subset(data["conditions"], ids, args.device), t, scales=scales)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise ValueError("Nonfinite training loss or gradient")
        optimizer.step()
        evaluated = step % settings["eval_every"] == 0 or step == args.steps
        if evaluated:
            total, count = 0., 0
            for start in range(0, len(val_subset), args.batch_size):
                val_ids = val_subset[start:start + args.batch_size]
                predicted = sample(model, subset(data["conditions"], val_ids, args.device), args.seed + 1 + start,
                    steps=settings["eval_steps"])
                target = subset(data["targets"], val_ids, args.device)
                target["contact"] = target["contact"] * 2 - 1
                val_scales = {key: value.to(args.device) for key, value in scales.items()}
                score = reconstruction_loss(predicted, target, val_scales).item()
                total += score * len(val_ids)
                count += len(val_ids)
            score = total / count
            if not math.isfinite(score):
                raise ValueError("Nonfinite validation loss")
            history.append(dict(step=step, training_loss=float(loss.detach()), validation_score=score))
            if score < best:
                best = score
                save_checkpoint(args.output / "best.pth", dict(schema=CHECKPOINT_SCHEMA, model=model.state_dict(),
                    model_profile=model_profile,
                    num_actions=model.num_actions, descriptor_dim=model.descriptor_dim,
                    vocabulary=data["vocabulary"], training_steps=step, data_validated=True,
                    representation=data["representation"], supervision=supervision,
                    condition_coverage=coverage,
                    training_seed=args.seed, initialization=protocol["model"],
                    validation_score=score, data_sha256=data_sha))
            save_json(args.output / "history.json", history)
            print(json.dumps(history[-1]), flush=True)
        if step % settings["log_every"] == 0 or evaluated:
            elapsed = time.monotonic() - started
            progress = dict(step=step, total_steps=args.steps, training_loss=float(loss.detach()),
                optimizer_steps_per_second=(step - start_step) / elapsed,
                supervision=supervision, validation_samples=len(val_subset))
            if cuda:
                progress["peak_allocated_mb"] = torch.cuda.max_memory_allocated(args.device) / 2**20
            print(json.dumps(progress), flush=True)
            save_checkpoint(args.output / "last.pth", dict(schema=CHECKPOINT_SCHEMA, checkpoint_kind="last",
                model_profile=model_profile,
                model=model.state_dict(), optimizer=optimizer.state_dict(),
                num_actions=model.num_actions, descriptor_dim=model.descriptor_dim,
                vocabulary=data["vocabulary"], training_steps=step, data_validated=True,
                representation=data["representation"], supervision=supervision, data_sha256=data_sha,
                condition_coverage=coverage,
                training_seed=args.seed, initialization=protocol["model"],
                train_settings=settings, protocol=protocol, best_score=best, history=history,
                generator_state=generator.get_state(), torch_rng_state=torch.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state(args.device) if cuda else None))


def save_checkpoint(path, values):
    temporary = path.with_suffix(".tmp")
    torch.save(values, temporary)
    temporary.replace(path)


def generate(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(saved)
    payload = torch.load(args.conditions, map_location="cpu", weights_only=False)
    validate_inference_payload(saved, payload)
    if args.output.exists():
        raise ValueError("Use a new generation output path")
    model_profile = checkpoint_model_profile(saved)
    model = OrderedSceneDenoiser(saved["num_actions"], saved["descriptor_dim"], checkpoint=None,
        model_profile=model_profile).to(args.device)
    model.load_state_dict(saved["model"], strict=True)
    prediction = sample(model, payload["conditions"], args.seed, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema=CHECKPOINT_SCHEMA, prediction={k: v.cpu() for k, v in prediction.items()},
        model_profile=model_profile,
        supervision=saved["supervision"],
        representation=saved["representation"], vocabulary=saved["vocabulary"], object_names=payload["object_names"],
        conditions=payload, seed=args.seed, checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        generation="Joint neural diffusion; no supplied future object poses or IK"), args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate-data")
    check.add_argument("--data", required=True, type=Path)
    check.add_argument("--allow-weak", action="store_true")
    fit = commands.add_parser("train")
    fit.add_argument("--data", required=True, type=Path)
    fit.add_argument("--output", required=True, type=Path)
    fit.add_argument("--steps", required=True, type=int)
    fit.add_argument("--initialization-checkpoint", default="assets/gb_main.pth", type=Path)
    fit.add_argument("--batch-size", default=16, type=int)
    fit.add_argument("--lr", default=1e-4, type=float)
    fit.add_argument("--seed", default=42, type=int)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--allow-weak", action="store_true")
    fit.add_argument("--resume", type=Path)
    fit.add_argument("--eval-every", default=200, type=int)
    fit.add_argument("--eval-steps", default=20, type=int)
    fit.add_argument("--max-val-samples", default=32, type=int)
    fit.add_argument("--log-every", default=50, type=int)
    fit.add_argument("--weight-decay", default=1e-6, type=float)
    fit.add_argument("--betas", default=(.95, .999), nargs=2, type=float)
    fit.add_argument("--model-profile", choices=MODEL_PROFILES, default="legacy")
    gen = commands.add_parser("sample")
    gen.add_argument("--conditions", required=True, type=Path)
    gen.add_argument("--checkpoint", required=True, type=Path)
    gen.add_argument("--output", required=True, type=Path)
    gen.add_argument("--seed", default=20261027, type=int)
    gen.add_argument("--steps", default=40, type=int)
    gen.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        if args.command == "validate-data":
            data = torch.load(args.data, map_location="cpu", weights_only=False)
            print(json.dumps(dict(valid=True, samples=validate_training_data(data, allow_weak=args.allow_weak))))
        elif args.command == "train":
            train(args)
        else:
            generate(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
