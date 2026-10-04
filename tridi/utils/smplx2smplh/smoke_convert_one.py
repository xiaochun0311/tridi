#!/usr/bin/env python3
"""Smoke test: run SMPL-X -> SMPL+H conversion on ONE GRAB sequence."""
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import grab as grabmod

grab_path = Path("/home/yanghc/tridi/data/raw/grab")
smplx_path = Path("/home/yanghc/tridi/data/smplx_models")

cfg = SimpleNamespace(
    only_contact=True,
    grab_path=grab_path,
    batch_size=128,
    num_workers=0,
    downsample=True,
    save_meshes=False,
    model_path=str(smplx_path),
    transfer_data_path="./transfer_data",
)

sequence = grab_path / "grab" / "s1" / "banana_pass_1.npz"
seq_data = grabmod.parse_npz(sequence)
seq_info = {
    "gender": seq_data["gender"],
    "sbj_id": str(seq_data["sbj_id"]),
    "action": "_".join(sequence.stem.split("_")[1:]),
    "obj_name": str(seq_data["obj_name"]),
}
seq_info["output_seq_path"] = (
    grab_path / "grab_smplh" / seq_info["sbj_id"] / f"{seq_info['obj_name']}_{seq_info['action']}"
)
seq_info["output_seq_path"].mkdir(exist_ok=True, parents=True)

t0 = time.time()
frame_mask, faces, verts = grabmod.create_smplx_meshes(seq_data, cfg)
seq_info["frame_mask"] = frame_mask
print(f"[smoke] frames selected (30fps, contact only): {int(frame_mask.sum())}", flush=True)

mesh_dataset = grabmod.MeshInMemory(faces, verts, seq_info)
t1 = time.time()
grabmod.run_smplx2smplh_conversion(mesh_dataset, seq_info, cfg)
t2 = time.time()
print(f"[smoke] SMPL-X meshes: {t1-t0:.1f}s, fitting: {t2-t1:.1f}s, total: {t2-t0:.1f}s", flush=True)

# verify output pkl exists
out = grab_path / "grab_smplh" / seq_info["sbj_id"] / f"{seq_info['obj_name']}_{seq_info['action']}" / "sequence_data.pkl"
print(f"[smoke] output exists: {out.is_file()}  size: {out.stat().st_size if out.is_file() else 0}", flush=True)
