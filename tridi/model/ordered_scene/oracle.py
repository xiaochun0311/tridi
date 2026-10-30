"""Zero-training source -> native decode -> export/reload checks for HIMO."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import trimesh
from smplx.lbs import batch_rodrigues

from .geometry import NativeSceneGeometry
from .pipeline import validate_training_data


TOLERANCE_M = 1e-5


def maximum_error(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Oracle arrays require matching finite shapes")
    return float(np.abs(first - second).max(initial=0.))


def contact_agreement(found, expected, distances=None):
    """Empty maps agree exactly; they are not assigned an arbitrary F1 score."""
    found, expected = np.asarray(found, dtype=bool), np.asarray(expected, dtype=bool)
    if found.shape != expected.shape:
        raise ValueError("Contact maps require the same shape")
    mismatch = found != expected
    count = int(mismatch.sum())
    boundary = 0
    margins = []
    if count and distances is not None:
        distances = np.asarray(distances)
        if distances.shape != found.shape or not np.isfinite(distances[mismatch]).all():
            raise ValueError("Oracle mismatch distances require finite matching entries")
        margins = np.abs(distances[mismatch] - .05).tolist()
        boundary = int(np.count_nonzero(np.asarray(margins) <= TOLERANCE_M))
    return dict(mismatch_points=count, threshold_boundary_points=boundary,
                unexplained_points=count - boundary, strict_match=(count == 0),
                mismatch_distance_from_threshold_m=margins)


def _source_reference(geometry, states, condition, metadata):
    """Decode axis-angle fields directly, independent of the packed row6D layout."""
    paths, hashes = metadata["source_files"], metadata["source_sha256"]
    for key in ("smplx", "object_pose"):
        if hashlib.sha256(Path(paths[key]).read_bytes()).hexdigest() != hashes[key]:
            raise ValueError(f"Recorded {key} SHA changed for {metadata['sequence_id']}")
    frames = metadata["target_frames"]
    with np.load(paths["smplx"], allow_pickle=False) as saved:
        params = {key: saved[key] for key in saved.files}
    def rotation(key, joints):
        values = torch.as_tensor(params[key][frames].copy(), dtype=torch.float32)
        return batch_rodrigues(values.reshape(-1, 3)).reshape(2, joints, 3, 3)
    identity = torch.eye(3).expand(2, 1, 3, 3)
    beta = torch.as_tensor(params["betas"].copy(), dtype=torch.float32).reshape(1, 10)
    result = geometry.model(betas=beta.expand(2, -1),
        transl=torch.as_tensor(params["transl"][frames].copy(), dtype=torch.float32),
        global_orient=rotation("global_orient", 1), body_pose=rotation("body_pose", 21),
        left_hand_pose=rotation("lhand_pose", 15), right_hand_pose=rotation("rhand_pose", 15),
        jaw_pose=identity, leye_pose=identity, reye_pose=identity, expression=torch.zeros(2, 10))
    joints = torch.cat([result.joints[:, :22], result.joints[:, 25:55]], dim=1)
    objects = np.load(paths["object_pose"], allow_pickle=True).item()
    errors = dict(source_shape_max_abs=maximum_error(states["shape"].numpy()[0], beta.numpy()[0]),
        source_human_translation_max_abs_m=maximum_error(states["human"].numpy()[0, :, 312:], params["transl"][frames]),
        source_object_rotation_max_abs=0., source_object_translation_max_abs_m=0.,
        source_object_vertices_max_abs_m=0., source_initial_object_max_abs=0.)
    for slot, name in enumerate(metadata["object_names"]):
        item = objects[name]
        pose = states["object_pose"][0, :, slot]
        source_rotation = item["rot"][frames]
        errors["source_object_rotation_max_abs"] = max(errors["source_object_rotation_max_abs"],
            maximum_error(geometry._rotations(pose[:, :6]).numpy(), source_rotation))
        errors["source_object_translation_max_abs_m"] = max(errors["source_object_translation_max_abs_m"],
            maximum_error(pose[:, 6:].numpy(), item["transl"][frames]))
        initial_frame = metadata["initial_frame"]
        initial = np.r_[item["rot"][initial_frame, :2].reshape(6), item["transl"][initial_frame]]
        errors["source_initial_object_max_abs"] = max(errors["source_initial_object_max_abs"],
            maximum_error(condition[0, slot].numpy(), initial))
        canonical = np.asarray(geometry._mesh(name).vertices)
        for time, frame in enumerate(frames):
            world = canonical @ item["rot"][frame].T + item["transl"][frame]
            errors["source_object_vertices_max_abs_m"] = max(errors["source_object_vertices_max_abs_m"],
                maximum_error(geometry.object_vertices(name, pose[time]).numpy(), world))
    return result.vertices.numpy(), joints.numpy(), errors


def _check_scene(geometry, states, condition, metadata, directory):
    names = [metadata["object_names"]]
    vertices, joints = geometry.decode(states["shape"], states["human"])
    source_vertices, source_joints, errors = _source_reference(geometry, states, condition, metadata)
    errors.update(source_human_vertices_max_abs_m=maximum_error(vertices.numpy()[0], source_vertices),
                  source_native52_joints_max_abs_m=maximum_error(joints.numpy()[0], source_joints))
    exported = geometry.export(states, names, directory)
    errors["reloaded_native52_joints_max_abs_m"] = maximum_error(
        np.load(exported["joints"], allow_pickle=False), joints.numpy())
    errors.update(reloaded_human_vertices_max_abs_m=0., reloaded_object_vertices_max_abs_m=0.)
    native = geometry.contacts(states["shape"], states["human"], states["object_pose"], names).numpy()[0] > 0
    cached = states["contact"].numpy()[0] > 0
    maps = []
    threshold = geometry.protocol()["contact_threshold_m"]
    for time, row in enumerate(exported["scenes"]):
        body = trimesh.load(row["human"], process=False, maintain_order=True)
        errors["reloaded_human_vertices_max_abs_m"] = max(errors["reloaded_human_vertices_max_abs_m"],
            maximum_error(body.vertices, vertices.numpy()[0, time]))
        points = np.asarray(body.vertices)[geometry.indices]
        for obj in row["objects"]:
            slot, name = obj["slot"], obj["name"]
            mesh = trimesh.load(obj["path"], process=False, maintain_order=True)
            expected_vertices = geometry.object_vertices(name, states["object_pose"][0, time, slot]).numpy()
            errors["reloaded_object_vertices_max_abs_m"] = max(errors["reloaded_object_vertices_max_abs_m"],
                maximum_error(mesh.vertices, expected_vertices))
            found = geometry._surface_near(mesh, points, threshold)
            mismatch = (native[time, slot] != cached[time, slot]) | (found != cached[time, slot])
            distances = np.full(690, np.inf)
            if mismatch.any():
                distances[mismatch] = trimesh.proximity.closest_point(mesh, points[mismatch])[1]
            maps.append(dict(state=f"S{time + 1}", slot=slot, object=name,
                cached_positive_points=int(cached[time, slot].sum()),
                native_vs_cache=contact_agreement(native[time, slot], cached[time, slot], distances),
                reloaded_vs_cache=contact_agreement(found, cached[time, slot], distances)))
    unexplained = sum(item[comparison]["unexplained_points"] for item in maps
                      for comparison in ("native_vs_cache", "reloaded_vs_cache"))
    strict = all(item[comparison]["strict_match"] for item in maps
                 for comparison in ("native_vs_cache", "reloaded_vs_cache"))
    return dict(sequence_id=metadata["sequence_id"], subject=metadata["subject"], split=metadata["split"],
        errors=errors, contact_maps=maps, strict_contact_match=strict,
        passed=all(value <= TOLERANCE_M for value in errors.values()) and unexplained == 0)


@torch.no_grad()
def oracle(args):
    if args.output.exists() or args.max_scenes < 1:
        raise ValueError("Use a new output directory and positive --max-scenes")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    count = validate_training_data(data, allow_weak=args.allow_weak)
    indices = torch.randperm(count, generator=torch.Generator().manual_seed(args.seed))[:args.max_scenes].tolist()
    geometry = NativeSceneGeometry(args.models_root, args.meshes_root, device="cpu", representation=data["representation"])
    args.output.mkdir(parents=True)
    records = []
    for position, index in enumerate(indices):
        states = {key: value[index:index + 1].cpu() for key, value in data["targets"].items()}
        condition = data["conditions"]["initial_object_pose"][index:index + 1].cpu()
        metadata = data["metadata"][index]
        if position < 2:
            record = _check_scene(geometry, states, condition, metadata, args.output / f"scene{position:03d}")
        else:
            with TemporaryDirectory(prefix="roundtrip-", dir=args.output) as temporary:
                record = _check_scene(geometry, states, condition, metadata, Path(temporary) / "meshes")
        record["dataset_index"] = index
        records.append(record)
        print(json.dumps(dict(scene=position + 1, total=len(indices), sequence=metadata["sequence_id"], passed=record["passed"])), flush=True)
    report = dict(passed=all(row["passed"] for row in records),
        strict_contact_match=all(row["strict_contact_match"] for row in records), scenes=len(records), seed=args.seed,
        tolerance_m=TOLERANCE_M, data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        maximum_errors={key: max(row["errors"][key] for row in records) for key in records[0]["errors"]},
        optimizer_steps=0, generated_states_corrected=False,
        contact_policy="Pointwise agreement; empty maps match. Any changed point within 1e-5m of the 5cm threshold is reported explicitly and tolerated.",
        note="GT axis-angle sources, packed row6D targets and exported/reloaded meshes must agree. No learned model is evaluated.", records=records)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "models-root", "meshes-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--max-scenes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-weak", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not oracle(args)["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
