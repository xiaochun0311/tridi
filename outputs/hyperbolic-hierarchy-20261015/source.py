"""SMPL-H tree-distance reconstruction probe; no pose-generation claims."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch


def distances(z, geometry):
    # Lorentz hyperboloid coordinates (curvature -1); double precision.
    if geometry == "euclidean":
        return torch.cdist(z, z)
    t = (1 + z.square().sum(-1)).sqrt()
    cosh = t[:, None] * t[None, :] - z @ z.T
    return torch.acosh(cosh.clamp_min(1 + 1e-12))


def hashes():
    paths = sorted(Path("tridi").rglob("*.py")) + sorted(Path("config").glob("*"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def main():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    output = Path("outputs/hyperbolic-hierarchy-20261015")
    output.mkdir(parents=True, exist_ok=True)
    before = hashes()
    source = Path("data/smplx_models/smplh/SMPLH_MALE.pkl")
    with source.open("rb") as f:
        table = np.asarray(pickle.load(f, encoding="latin1")["kintree_table"])
    assert np.array_equal(table[1], np.arange(52))
    parents = table[0].astype(np.int64)
    parents[0] = -1
    n = len(parents)
    target = np.full((n, n), np.inf)
    np.fill_diagonal(target, 0)
    for i, p in enumerate(parents[1:], 1):
        assert 0 <= p < i
        target[i, p] = target[p, i] = 1
    for k in range(n):
        target = np.minimum(target, target[:, k, None] + target[None, k, :])
    assert np.isfinite(target).all() and np.max(target) > 1
    # Known radial distance in the Lorentz model, plus finite-gradient check.
    check = torch.tensor([[0., 0.], [np.sinh(2.), 0.]], requires_grad=True)
    check_d = distances(check, "hyperbolic")
    assert abs(check_d[0, 1].item() - 2) < 1e-10
    check_d[0, 1].backward()
    assert torch.isfinite(check.grad).all()
    print("TREE_AND_GEOMETRY_CHECKS_PASSED", flush=True)
    a, b = np.triu_indices(n, 1)
    edges = np.flatnonzero(target[a, b] == 1)
    other = np.flatnonzero(target[a, b] > 1)
    np.random.default_rng(20261015).shuffle(other)
    cut1, cut2 = int(.6 * len(other)), int(.8 * len(other))
    splits = {"train": np.concatenate([edges, other[:cut1]]),
              "val": other[cut1:cut2], "test": other[cut2:]}
    truth = torch.tensor(target[a, b])
    rows = []
    for dim in [2, 4, 8]:
        for seed in [42, 43, 44]:
            for geometry in ["euclidean", "hyperbolic"]:
                # Validation selects learning rate and checkpoint equally for both geometries.
                best = None
                for lr in [.01, .05]:
                    torch.manual_seed(seed)
                    z = torch.nn.Parameter(.1 * torch.randn(n, dim))
                    optimizer = torch.optim.Adam([z], lr=lr)
                    for step in range(1, 2001):
                        optimizer.zero_grad()
                        pred = distances(z, geometry)[a, b]
                        loss = ((pred[splits["train"]] - truth[splits["train"]]) ** 2).mean()
                        assert torch.isfinite(loss)
                        loss.backward()
                        assert torch.isfinite(z.grad).all()
                        optimizer.step()
                        if step % 100 == 0:
                            with torch.no_grad():
                                pred = distances(z, geometry)[a, b]
                                val = ((pred[splits["val"]] - truth[splits["val"]]) ** 2).mean().item()
                                if best is None or val < best[0]:
                                    best = (val, z.detach().clone(), lr, step)
                val, z, lr, step = best
                pred = distances(z, geometry)[a, b].numpy()
                row = dict(dim=dim, seed=seed, geometry=geometry, lr=lr, step=step)
                for split, idx in splits.items():
                    error = pred[idx] - truth.numpy()[idx]
                    row[split + "_rmse"] = float(np.sqrt(np.mean(error ** 2)))
                    row[split + "_relative_error"] = float(np.mean(abs(error) / truth.numpy()[idx]))
                np.save(output / f"{geometry}_{dim}_{seed}.npy", z.numpy())
                rows.append(row)
                print(json.dumps(row), flush=True)
    result = dict(completed=True, source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  parents=parents.tolist(), splits={k: v.tolist() for k, v in splits.items()},
                  pair_indices=[a.tolist(), b.tolist()], results=rows,
                  protected_source_unchanged=(before == hashes()),
                  protocol="52 fixed joints; unit-edge tree distances; 2 learning rates x 2000 steps; validation selection; held-out pairs, not unseen joints or poses")
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print("COMPLETED", flush=True)


if __name__ == "__main__":
    main()
