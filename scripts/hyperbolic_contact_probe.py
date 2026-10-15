"""Train isolated contact autoencoders; does not modify TriDi or its checkpoints."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

from tridi.model.conditioning.contact_ae_clip import ContactDec, ContactEnc


def expmap0(v):
    norm = v.square().sum(-1, keepdim=True).clamp_min(1e-16).sqrt()
    x = v * (norm.tanh() / norm)
    radius = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return x * (.995 / radius).clamp_max(1)


def squared_distance(x, y, geometry):
    xx, yy = x.square().sum(-1, keepdim=True), y.square().sum(-1)[None]
    delta = (xx + yy - 2 * x @ y.T).clamp_min(0)
    if geometry == "euclidean":
        # Match the Poincare metric's local factor 4 at the origin.
        return 4 * delta
    ratio = delta / ((1 - xx) * (1 - yy)).clamp_min(1e-8)
    return 4 * torch.asinh(ratio.clamp_min(1e-12).sqrt()).square()


def self_check():
    v = torch.tensor([[.1, -.2], [0., 0.]], dtype=torch.float64, requires_grad=True)
    x = expmap0(v)
    origin = torch.zeros(1, 2, dtype=v.dtype)
    d2 = squared_distance(x, origin, "hyperbolic")[:, 0]
    assert torch.allclose(d2, 4 * v.square().sum(-1), atol=1e-10)
    own = squared_distance(x, x, "hyperbolic")
    assert torch.allclose(own, own.T, atol=1e-10)
    assert own.diagonal().max() < 1e-9
    d2.sum().backward()
    assert torch.isfinite(v.grad).all()
    near_boundary = expmap0(torch.tensor([[100., 0.]]))
    assert near_boundary.norm() < 1
    assert torch.isfinite(squared_distance(near_boundary, -near_boundary, "hyperbolic")).all()
    tiny = torch.tensor([[1e-5, 2e-5]], dtype=torch.float64)
    assert torch.allclose(squared_distance(expmap0(tiny), origin, "hyperbolic"),
                          squared_distance(tiny, origin, "euclidean"), rtol=1e-6, atol=1e-12)
    print("GEOMETRY_CHECKS_PASSED", flush=True)


class ContactAE(nn.Module):
    def __init__(self, dim, geometry, prior):
        super().__init__()
        self.geometry = geometry
        self.encoder = ContactEnc(690, 256, dim)
        if geometry == "mlp" or geometry.endswith("_aux"):
            self.decoder = ContactDec(690, 256, dim)
        else:
            self.prototypes = nn.Parameter(torch.randn(690, dim) * .01)
            self.bias = nn.Parameter(torch.logit(prior.clamp(.001, .999)))

    def forward(self, mask):
        z = self.encoder(mask)
        if self.geometry == "mlp" or self.geometry.endswith("_aux"):
            return self.decoder(z)
        p = self.prototypes
        if self.geometry == "hyperbolic":
            z, p = expmap0(z), expmap0(p)
        return self.bias - squared_distance(z, p, self.geometry)


def latent_regularizer(model, mask):
    z = .1 * model.encoder(mask)
    geometry = model.geometry.removesuffix("_aux")
    if geometry == "hyperbolic":
        z = expmap0(z)
    other = mask.roll(1, 0)
    intersection = (mask * other).sum(-1)
    union = (mask + other - mask * other).sum(-1)
    target = torch.where(union > 0, 1 - intersection / union.clamp_min(1), 0.)
    d2 = squared_distance(z, z.roll(1, 0), geometry).diagonal()
    return F.mse_loss(d2, target.square())


def load_contacts(root, output):
    masks, metadata, counts = {}, {}, {}
    idx = np.load("assets/smpl_template_decimated_idxs.npy")
    keypoints = {p.stem: np.load(p)["cartesian"] for p in (root / "object_keypoints").glob("*.npz")}
    plans = {"train": ("train", 10, [f"s{i}" for i in range(1, 7)], 10),
             "val": ("train", 10, ["s7", "s8"], 10),
             "test": ("test", 1, ["s9", "s10"], 1)}
    verified_distance = False
    for split, (file_split, fps, subjects, stride) in plans.items():
        rows, infos = [], []
        with h5py.File(root / f"dataset_{file_split}_{fps}fps.hdf5") as f:
            for subject in subjects:
                if subject not in f:
                    continue
                for name in sorted(f[subject]):
                    seq = f[subject][name]
                    obj = seq.attrs["obj"]
                    assert obj in keypoints, obj
                    for i in range(0, int(seq.attrs["T"]), stride):
                        assert abs(float(seq["prep_s"][i]) - 1) < 1e-6, "Non-unit scale requires matching TriDi scaling."
                        points = keypoints[obj] @ seq["obj_R"][i].reshape(3, 3).T + seq["obj_c"][i]
                        vertices = seq["sbj_v"][i][idx]
                        distance = cKDTree(points).query(vertices, workers=1)[0]
                        if not verified_distance:
                            reference = torch.cdist(torch.tensor(vertices)[None].float(),
                                                    torch.tensor(points)[None].float()).min(-1).values[0].numpy()
                            assert np.allclose(distance, reference, atol=2e-5)
                            assert np.array_equal(distance < .05, reference < .05)
                            verified_distance = True
                        rows.append(distance < .05)
                        infos.append({"subject": subject, "sequence": name, "frame": i, "object": obj})
        masks[split] = torch.tensor(np.stack(rows), dtype=torch.float32)
        metadata[split] = infos
        counts[split] = {"frames": len(rows), "subjects": sorted({r["subject"] for r in infos}),
                         "objects": sorted({r["object"] for r in infos}),
                         "sequences": len({(r["subject"], r["sequence"]) for r in infos}),
                         "positive_fraction": float(masks[split].mean()),
                         "empty_frames": int((masks[split].sum(-1) == 0).sum())}
        print("DATA", split, counts[split], flush=True)
    assert not set(counts["train"]["subjects"]) & set(counts["val"]["subjects"])
    assert not set(counts["train"]["subjects"]) & set(counts["test"]["subjects"])
    assert not set(counts["val"]["subjects"]) & set(counts["test"]["subjects"])
    np.savez_compressed(output / "contact_masks.npz", **{k: v.numpy() for k, v in masks.items()})
    (output / "frames.json").write_text(json.dumps(metadata, indent=2))
    return masks, metadata, counts


def metrics(probability, target, threshold):
    pred, truth = probability >= threshold, target.bool()
    tp = (pred & truth).sum().item()
    fp = (pred & ~truth).sum().item()
    fn = (~pred & truth).sum().item()
    per_tp = (pred & truth).sum(-1).float()
    per_den = pred.sum(-1) + truth.sum(-1)
    per_f1 = torch.where(per_den > 0, 2 * per_tp / per_den.clamp_min(1), 1.)
    nonempty = truth.any(-1)
    return {"micro_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "micro_iou": tp / max(tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "mean_frame_f1": float(per_f1.mean()),
            "mean_nonempty_frame_f1": float(per_f1[nonempty].mean()) if nonempty.any() else None,
            "mean_predicted_vertices": float(pred.float().sum(-1).mean()),
            "mean_true_vertices": float(truth.float().sum(-1).mean()),
            "bce": float(F.binary_cross_entropy(probability.clamp(1e-6, 1 - 1e-6), target))}


@torch.no_grad()
def predict(model, masks):
    return torch.cat([model(x).sigmoid() for x in masks.split(128)])


def fit(geometry, dim, seed, masks, output, steps):
    torch.manual_seed(seed)
    model = ContactAE(dim, geometry, masks["train"].mean(0))
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    batch_rng = torch.Generator().manual_seed(seed + 1000)
    weight = ((1 - masks["train"].mean()) / masks["train"].mean()).clamp_max(20)
    best, history = None, []
    start = time.time()
    max_grad = 0.
    for step in range(1, steps + 1):
        batch = masks["train"][torch.randint(len(masks["train"]), (128,), generator=batch_rng)]
        optimizer.zero_grad()
        reconstruction = F.binary_cross_entropy_with_logits(model(batch), batch, pos_weight=weight)
        regularizer = latent_regularizer(model, batch) if geometry.endswith("_aux") else reconstruction * 0
        loss = reconstruction + .03 * regularizer
        assert torch.isfinite(loss), (geometry, step)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        assert torch.isfinite(grad), (geometry, step)
        max_grad = max(max_grad, float(grad))
        optimizer.step()
        if step % 100 == 0 or step == steps:
            probability = predict(model, masks["val"])
            candidates = [(metrics(probability, masks["val"], t)["micro_f1"], float(t))
                          for t in np.linspace(.05, .95, 19)]
            score, threshold = max(candidates)
            history.append({"step": step, "train_loss": float(loss.detach()),
                            "reconstruction_loss": float(reconstruction.detach()),
                            "latent_regularizer": float(regularizer.detach()),
                            "val_micro_f1": score, "threshold": threshold})
            if best is None or score > best["score"]:
                best = {"score": score, "threshold": threshold, "step": step,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            print("FIT", geometry, seed, history[-1], flush=True)
    model.load_state_dict(best["state"])
    probability = predict(model, masks["test"])
    row = {"geometry": geometry, "dim": dim, "seed": seed,
           "parameters": sum(p.numel() for p in model.parameters()), "steps": steps,
           "best_step": best["step"], "threshold": best["threshold"],
           "val_micro_f1": best["score"], "test": metrics(probability, masks["test"], best["threshold"]),
           "seconds": time.time() - start, "max_gradient_norm_before_clip": max_grad, "history": history}
    if geometry in ["hyperbolic", "hyperbolic_aux"]:
        with torch.no_grad():
            scale = .1 if geometry.endswith("_aux") else 1.
            radius = expmap0(scale * model.encoder(masks["test"])).norm(dim=-1)
            row["test_latent_max_radius"] = float(radius.max())
            row["test_latent_fraction_radius_above_099"] = float((radius > .99).float().mean())
    name = f"{geometry}_d{dim}_seed{seed}"
    torch.save({"model": best["state"], "geometry": geometry, "dim": dim,
                "threshold": best["threshold"], "metadata": row}, output / f"{name}.pth")
    np.save(output / f"{name}_test_probabilities.npy", probability.numpy())
    print("RESULT", json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output", default="outputs/hyperbolic-contact-20261014")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--geometries", nargs="+", choices=["euclidean", "hyperbolic", "mlp", "euclidean_aux", "hyperbolic_aux"],
                        default=["euclidean", "hyperbolic", "mlp"])
    parser.add_argument("--cache", default=None, help="Reuse masks and frame metadata from a completed probe.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    self_check()
    for geometry in ["euclidean_aux", "hyperbolic_aux"]:
        torch.manual_seed(123)
        model = ContactAE(4, geometry, torch.full((690,), .01))
        batch = (torch.rand(8, 690) < .03).float()
        check = F.binary_cross_entropy_with_logits(model(batch), batch) + .03 * latent_regularizer(model, batch)
        check.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print("AUXILIARY_GRADIENT_CHECKS_PASSED", flush=True)
    if args.self_check:
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "results.json").exists(), "Use a new output directory to preserve earlier runs."
    protected = [Path(".gitignore")] + list(Path("tridi").rglob("*.py")) + list(Path("config").glob("*.*"))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    report = {"completed": False, "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "device": "cpu", "curvature": -1,
              "scope": "Contact reconstruction, not full diffusion or generation quality; no explicit hierarchy supervision.",
              "protocol": "GRAB train s1-s6, validation s7-s8, test s9-s10; 1fps sampling; threshold and checkpoint selected only on validation.",
              "matched_comparison": "Compare euclidean/hyperbolic prototypes or euclidean_aux/hyperbolic_aux: within each pair identical architecture, dimension, parameter count, initialization, batches, Adam, steps, BCE weight; Euclidean metric scaled by 4 to match local hyperbolic metric.",
              "auxiliary_method": "MLP encoder/decoder in tangent coordinates; auxiliary loss fits squared geometric distances to squared contact-mask Jaccard distances. Weight 0.03, latent scale 0.1; hyperbolic points expmap0(0.1*z), curvature -1. This is metric regularization, not intrinsic diffusion or an exp/log-only transformation.",
              "mlp_reference": "Same latent dimension, existing ContactEnc/ContactDec architecture; different parameter count, separate architecture reference.",
              "limitation": "Correlated frames, short training, GRAB only; no claim of hierarchical structure or overall generation improvement.",
              "rows": []}
    path = output / "results.json"
    path.write_text(json.dumps(report, indent=2))
    if args.cache:
        cache = Path(args.cache)
        previous = json.loads((cache / "results.json").read_text())
        assert previous["completed"]
        masks = {k: torch.tensor(v, dtype=torch.float32) for k, v in np.load(cache / "contact_masks.npz").items()}
        metadata = json.loads((cache / "frames.json").read_text())
        counts = previous["data"]
        report["cache_source"] = str(cache)
        report["cache_masks_sha256"] = hashlib.sha256((cache / "contact_masks.npz").read_bytes()).hexdigest()
    else:
        masks, metadata, counts = load_contacts(Path("data/preprocessed/grab_smplh"), output)
    report["data"] = counts
    for seed in args.seeds:
        for geometry in args.geometries:
            row = fit(geometry, args.dim, seed, masks, output, args.steps)
            probability = torch.tensor(np.load(output / f"{geometry}_d{args.dim}_seed{seed}_test_probabilities.npy"))
            row["by_object"] = {obj: metrics(probability[index], masks["test"][index], row["threshold"])
                                for obj in counts["test"]["objects"]
                                for index in [[i for i, r in enumerate(metadata["test"]) if r["object"] == obj]]}
            report["rows"].append(row)
            path.write_text(json.dumps(report, indent=2))
    report["summary"] = {geometry: {"mean_test_micro_f1": float(np.mean([r["test"]["micro_f1"] for r in report["rows"] if r["geometry"] == geometry])),
                                     "std_test_micro_f1": float(np.std([r["test"]["micro_f1"] for r in report["rows"] if r["geometry"] == geometry])),
                                     "mean_test_micro_iou": float(np.mean([r["test"]["micro_iou"] for r in report["rows"] if r["geometry"] == geometry]))}
                         for geometry in args.geometries}
    report["protected_source_unchanged"] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in before.items())
    assert report["protected_source_unchanged"]
    report["completed"] = True
    path.write_text(json.dumps(report, indent=2))
    print("HYPERBOLIC_CONTACT_PROBE_COMPLETE", json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
