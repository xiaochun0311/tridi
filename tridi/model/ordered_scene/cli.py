"""Train or sample the ordered scene model; no rule-based pose refinement."""
import argparse
import hashlib
import json
from pathlib import Path

import math

import torch

from .model import OrderedSceneDenoiser
from .pipeline import (
    STATE, diffusion_loss, reconstruction_loss, sample, sample_tri_timesteps,
    validate_conditions, validate_training_data,
)
from .protocol import assert_same_representation, validate_representation


CHECKPOINT_SCHEMA = "tridi-himo-ordered-v2"


def subset(values, ids, device):
    return {key: value[ids].to(device) for key, value in values.items()}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def validate_checkpoint(saved):
    if (not isinstance(saved, dict) or saved.get("schema") != CHECKPOINT_SCHEMA
            or saved.get("data_validated") is not True
            or type(saved.get("training_steps")) is not int or saved["training_steps"] < 1):
        raise ValueError("Requires a trained native HIMO checkpoint; legacy, diagnostic and untrained weights are incompatible")
    validate_representation(saved.get("representation"))
    vocabulary = saved.get("vocabulary")
    if (not isinstance(vocabulary, list) or not vocabulary
            or any(not isinstance(v, str) or not v.strip() for v in vocabulary)
            or len(set(vocabulary)) != len(vocabulary)
            or saved.get("num_actions") != len(vocabulary) or saved.get("descriptor_dim") != 12):
        raise ValueError("Invalid checkpoint action vocabulary or geometry descriptor layout")
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
    names = payload["object_names"]
    if (not isinstance(names, list) or len(names) != batch or any(
            not isinstance(row, (list, tuple)) or len(row) != 2
            or any(not isinstance(name, str) or not name.strip() for name in row) for row in names)):
        raise ValueError("object_names must bind two nonempty initial mesh names per scene")
    return batch


def train(args):
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    validate_training_data(data)
    indices = {split: [i for i, row in enumerate(data["metadata"]) if row["split"] == split]
               for split in ["train", "val", "test"]}
    if not indices["train"] or not indices["val"] or args.steps < 1:
        raise ValueError("Training requires separate train/val sequences and a positive step budget")
    if args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Batch size and learning rate must be positive")
    if args.output.exists():
        raise ValueError("Use a new output directory to preserve earlier results")
    torch.manual_seed(args.seed)
    model = OrderedSceneDenoiser(len(data["vocabulary"]), checkpoint=args.initialization_checkpoint).to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scales = {key: data["targets"][key][indices["train"]].std(0, unbiased=False).clamp_min(.1)
              for key in STATE if key != "contact"}
    args.output.mkdir(parents=True)
    protocol = dict(input_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        seed=args.seed, steps=args.steps, split_sizes={key: len(ids) for key, ids in indices.items()},
        model=model.manifest(), vocabulary=data["vocabulary"],
        representation=data["representation"], checkpoint_schema=CHECKPOINT_SCHEMA,
        conditioning="Only initial object poses/geometry and ordered action references",
        targets="Recorded S1/S2 body, both object poses, and contacts; no IK in sampling",
        data_note="Schema declarations require a separately audited recorded data source")
    save_json(args.output / "protocol.json", protocol)
    generator = torch.Generator().manual_seed(args.seed)
    best, history = float("inf"), []
    for step in range(1, args.steps + 1):
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
        if step % 200 == 0 or step == args.steps:
            total, count = 0., 0
            for start in range(0, len(indices["val"]), args.batch_size):
                val_ids = indices["val"][start:start + args.batch_size]
                predicted = sample(model, subset(data["conditions"], val_ids, args.device), args.seed + 1 + start)
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
                torch.save(dict(schema=CHECKPOINT_SCHEMA, model=model.state_dict(),
                    num_actions=model.num_actions, descriptor_dim=model.descriptor_dim,
                    vocabulary=data["vocabulary"], training_steps=step, data_validated=True,
                    representation=data["representation"],
                    validation_score=score, data_sha256=protocol["input_sha256"]), args.output / "best.pth")
            save_json(args.output / "history.json", history)
            print(json.dumps(history[-1]), flush=True)


def generate(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(saved)
    payload = torch.load(args.conditions, map_location="cpu", weights_only=False)
    validate_inference_payload(saved, payload)
    if args.output.exists():
        raise ValueError("Use a new generation output path")
    model = OrderedSceneDenoiser(saved["num_actions"], saved["descriptor_dim"], checkpoint=None).to(args.device)
    model.load_state_dict(saved["model"], strict=True)
    prediction = sample(model, payload["conditions"], args.seed, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema=CHECKPOINT_SCHEMA, prediction={k: v.cpu() for k, v in prediction.items()},
        representation=saved["representation"], vocabulary=saved["vocabulary"], object_names=payload["object_names"],
        conditions=payload, seed=args.seed, checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        generation="Joint neural diffusion; no supplied future object poses or IK"), args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate-data")
    check.add_argument("--data", required=True, type=Path)
    fit = commands.add_parser("train")
    fit.add_argument("--data", required=True, type=Path)
    fit.add_argument("--output", required=True, type=Path)
    fit.add_argument("--steps", required=True, type=int)
    fit.add_argument("--initialization-checkpoint", default="assets/gb_main.pth", type=Path)
    fit.add_argument("--batch-size", default=16, type=int)
    fit.add_argument("--lr", default=1e-4, type=float)
    fit.add_argument("--seed", default=42, type=int)
    fit.add_argument("--device", default="cuda")
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
            print(json.dumps(dict(valid=True, samples=validate_training_data(data))))
        elif args.command == "train":
            train(args)
        else:
            generate(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
