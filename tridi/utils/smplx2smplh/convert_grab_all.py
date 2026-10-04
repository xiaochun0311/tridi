#!/usr/bin/env python3
"""Convert GRAB SMPL-X -> SMPL+H for selected subjects (resumable, parallelizable).

Run from inside tridi/utils/smplx2smplh (transfer_data/ and local packages are
resolved relative to cwd). Use CUDA_VISIBLE_DEVICES to pin a GPU per process.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import tqdm

import grab as grabmod

OBJECTS = [
    "banana", "binoculars", "camera", "coffeemug", "cup", "doorknob", "eyeglasses",
    "flute", "flashlight", "fryingpan", "gamecontroller", "hammer", "headphones",
    "knife", "lightbulb", "mouse", "mug", "phone", "teapot", "toothbrush", "wineglass",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--grab-path", type=Path, required=True)
    ap.add_argument("-s", "--smplx-path", type=Path, required=True)
    ap.add_argument("--subjects", nargs="+", default=None, help="e.g. s1 s2 s3")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    cfg = SimpleNamespace(
        only_contact=True,
        grab_path=args.grab_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        downsample=True,
        save_meshes=False,
        model_path=str(args.smplx_path),
        transfer_data_path="./transfer_data",
        objects=OBJECTS,
    )

    all_seqs = sorted(cfg.grab_path.glob("grab/*/*.npz"))
    seqs = [s for s in all_seqs if s.stem.split("_")[0] in cfg.objects]
    if args.subjects:
        seqs = [s for s in seqs if s.parent.name in args.subjects]
    seqs = sorted(seqs)

    done = 0
    for sequence in tqdm.tqdm(seqs, total=len(seqs), ncols=80):
        seq_data = grabmod.parse_npz(sequence)
        seq_info = {
            "gender": seq_data["gender"],
            "sbj_id": str(seq_data["sbj_id"]),
            "action": "_".join(sequence.stem.split("_")[1:]),
            "obj_name": str(seq_data["obj_name"]),
        }
        seq_info["output_seq_path"] = (
            cfg.grab_path / "grab_smplh" / seq_info["sbj_id"] / f"{seq_info['obj_name']}_{seq_info['action']}"
        )
        seq_info["output_seq_path"].mkdir(exist_ok=True, parents=True)
        if (seq_info["output_seq_path"] / "sequence_data.pkl").is_file():
            done += 1
            continue

        frame_mask, faces, verts = grabmod.create_smplx_meshes(seq_data, cfg)
        if int(frame_mask.sum()) < 1:
            continue
        seq_info["frame_mask"] = frame_mask

        mesh_dataset = grabmod.MeshInMemory(faces, verts, seq_info)
        grabmod.run_smplx2smplh_conversion(mesh_dataset, seq_info, cfg)

    print(f"[convert_grab_all] finished total={len(seqs)} already_done={done}", flush=True)


if __name__ == "__main__":
    main()
