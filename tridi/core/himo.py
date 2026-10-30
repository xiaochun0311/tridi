"""HIMO's native two-object/two-frame workflow through TriDi's main entry."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from tridi.data import himo_ordered as data_io
from tridi.model.ordered_scene import cli
from tridi.model.ordered_scene.geometry import NativeSceneGeometry
from tridi.model.ordered_scene.model import OrderedSceneDenoiser
from tridi.model.ordered_scene.pipeline import diffusion_loss, sample, validate_training_data
from tridi.model.ordered_scene.protocol import assert_same_representation


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def geometry(cfg, representation=None):
    return NativeSceneGeometry(cfg.env.smpl_folder, Path(cfg.himo.root) / "object_mesh",
        device="cpu" if cfg.run.cpu else "cuda", indices_path=cfg.himo.contact_indices,
        representation=representation)


def diagnose(cfg):
    """Use real pending candidates only for tensor/gradient/export checks."""
    annotations = json.loads(Path(cfg.himo.annotations).read_text())
    row = annotations["candidates"][cfg.himo.condition_index]
    raw = data_io.load_triplet(Path(cfg.himo.root), row)
    geom = geometry(cfg)
    device = "cpu" if cfg.run.cpu else "cuda"
    names = [raw["object_names"]]
    target = {key: value.unsqueeze(0).to(device) for key, value in raw["targets"].items()}
    target["contact"] = geom.contacts(target["shape"], target["human"], target["object_pose"], names).to(device)
    vocabulary = sorted({action["action_id"] for action in row["actions"]})
    conditions = dict(initial_object_pose=raw["conditions"]["initial_object_pose"].unsqueeze(0).to(device),
        descriptors=geom.descriptors(names).to(device),
        action_ids=torch.tensor([[vocabulary.index(action["action_id"]) for action in row["actions"]]], device=device),
        action_objects=torch.tensor([[action["object_slot"] for action in row["actions"]]], device=device),
        action_targets=torch.tensor([[2 if action["target_slot"] is None else action["target_slot"] for action in row["actions"]]], device=device))
    model = OrderedSceneDenoiser(len(vocabulary), checkpoint=cfg.himo.initialization_checkpoint).to(device)
    loss = diffusion_loss(model, target, conditions, torch.tensor([[200, 600, 900]], device=device))
    loss.backward()
    gradients = {name: float(sum((p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None), 0.))
        for name, module in dict(human=model.core.decoder_S_pose, objects=model.core.decoder_O_transl,
                                 contacts=model.contact_head, actions=model.action_embedding).items()}
    if not torch.isfinite(loss) or not all(value > 0 for value in gradients.values()):
        raise ValueError("HIMO diagnostic failed: nonfinite loss or disconnected branch")
    prediction = sample(model, conditions, cfg.run.seed, steps=2)
    out = Path(cfg.run.path) / "diagnostic"
    out.mkdir(exist_ok=False)
    geom.export({key: value.detach().cpu() for key, value in target.items()}, names, out / "recorded_meshes")
    geom.export({key: value.detach().cpu() for key, value in prediction.items()}, names, out / "untrained_shape_check")
    report = dict(source=row["sequence"], annotation_status=row["status"],
        loss=float(loss.detach()), branch_gradient_sums=gradients, optimizer_steps=0,
        pretrained=model.manifest(), representation=geom.protocol(),
        target_shapes={key: list(value.shape) for key, value in target.items()},
        sample_shapes={key: list(value.shape) for key, value in prediction.items()},
        semantic_training=False, trained_checkpoint_written=False,
        note="Pending labels are used only to exercise conditioning tensors; not verified semantic supervision. Diagnostic samples have no quality claim.")
    save_json(out / "report.json", report)
    print(json.dumps(dict(output=str(out), loss=report["loss"], semantic_training=False)), flush=True)


def run_himo(cfg):
    torch.set_num_threads(2)
    out = Path(cfg.run.path)
    job = cfg.run.job
    prepared = Path(cfg.himo.prepared_data)
    device = "cpu" if cfg.run.cpu else "cuda"
    if cfg.himo.supervision not in {"human_verified", "weak_temporal_v1"}:
        raise ValueError("himo.supervision must be human_verified or weak_temporal_v1")
    allow_weak = cfg.himo.supervision == "weak_temporal_v1"
    if job == "inspect":
        report = data_io.inspect(Path(cfg.himo.root), cfg.himo.annotations)
        save_json(out / "himo_inspection.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    elif job == "diagnose":
        diagnose(cfg)
    elif job == "prepare":
        data_io.prepare(Path(cfg.himo.root), Path(cfg.himo.annotations), geometry(cfg), prepared, allow_weak=allow_weak)
        print(json.dumps(dict(prepared_data=str(prepared))), flush=True)
    elif job == "train":
        if not prepared.is_file():
            data_io.prepare(Path(cfg.himo.root), Path(cfg.himo.annotations), geometry(cfg), prepared, allow_weak=allow_weak)
        values = torch.load(prepared, map_location="cpu", weights_only=False)
        validate_training_data(values, allow_weak=allow_weak)
        assert_same_representation(values["representation"], geometry(cfg, values["representation"]).protocol())
        cli.train(SimpleNamespace(data=prepared, output=out / "fit", steps=cfg.train.max_steps,
            initialization_checkpoint=Path(cfg.himo.initialization_checkpoint),
            batch_size=cfg.dataloader.batch_size, lr=cfg.optimizer.lr, seed=cfg.run.seed, device=device,
            allow_weak=allow_weak, resume=None if cfg.resume.checkpoint is None else Path(cfg.resume.checkpoint),
            eval_every=cfg.himo.eval_every, eval_steps=cfg.himo.eval_steps,
            max_val_samples=cfg.himo.max_val_samples, log_every=cfg.himo.log_every,
            weight_decay=cfg.optimizer.weight_decay, betas=tuple(cfg.optimizer.kwargs.get("betas", (.95, .999)))))
    elif job == "conditions":
        values = torch.load(prepared, map_location="cpu", weights_only=False)
        validate_training_data(values, allow_weak=allow_weak)
        index = cfg.himo.condition_index
        if not 0 <= index < len(values["metadata"]):
            raise ValueError("himo.condition_index is outside prepared data")
        payload = dict(conditions={key: value[index:index + 1] for key, value in values["conditions"].items()},
            vocabulary=values["vocabulary"], representation=values["representation"],
            object_names=[values["metadata"][index]["object_names"]])
        path = out / "initial_conditions.pth"
        if path.exists():
            raise ValueError("Preserve existing inference conditions; use a new run name")
        torch.save(payload, path)
        print(json.dumps(dict(conditions=str(path), future_targets_included=False)), flush=True)
    elif job == "sample":
        if cfg.sample.target not in {"meshes", "tensors"}:
            raise ValueError("HIMO sample.target must be meshes or tensors")
        if cfg.resume.checkpoint is None or cfg.himo.conditions is None:
            raise ValueError("HIMO sampling requires resume.checkpoint and himo.conditions")
        saved = torch.load(cfg.resume.checkpoint, map_location="cpu", weights_only=False)
        payload = torch.load(cfg.himo.conditions, map_location="cpu", weights_only=False)
        cli.validate_inference_payload(saved, payload)
        geom = geometry(cfg, saved["representation"])
        descriptors = geom.descriptors(payload["object_names"]).cpu()
        if not torch.allclose(descriptors, payload["conditions"]["descriptors"].cpu(), atol=1e-6, rtol=1e-5):
            raise ValueError("Initial geometry names do not match the supplied object descriptors")
        prediction_path = out / "generation.pth"
        cli.generate(SimpleNamespace(conditions=Path(cfg.himo.conditions), checkpoint=Path(cfg.resume.checkpoint),
            output=prediction_path, seed=cfg.run.seed, steps=cfg.sample.num_inference_steps, device=device))
        result = torch.load(prediction_path, map_location="cpu", weights_only=False)
        if cfg.sample.target == "meshes":
            geom.export(result["prediction"], result["object_names"], out / "meshes")
    else:
        raise ValueError("HIMO supports inspect, diagnose, prepare, train, conditions and sample")
