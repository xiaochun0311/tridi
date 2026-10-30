"""Read native HIMO states with explicit human or opt-in weak supervision."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from smplx.lbs import batch_rodrigues


def _annotations(value):
    value = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    rows = value.get("candidates") if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Annotations must contain a nonempty candidates list")
    return rows


def inspect(root, annotations=None):
    """Inventory records, not inferred action boundaries or source-sequence IDs."""
    import h5py

    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Missing HIMO root: {root}")
    report = dict(root=str(root), resources={name: (root / name).is_dir() for name in
        ("smplx", "joints", "object_pose", "object_mesh", "text", "segments", "processed_2o", "processed_3o")}, processed={})
    for kind in ("processed_2o", "processed_3o"):
        for split in ("train", "val", "test"):
            path = root / kind / f"{split}.h5"
            if path.is_file():
                with h5py.File(path, "r") as data:
                    groups = list(data.values())
                    report["processed"][f"{kind}/{split}"] = dict(records=len(groups),
                        frames=sum(len(group["transl"]) for group in groups),
                        object_counts=dict(Counter(str(len(group["object_state"])) for group in groups)))
    report["note"] = "HDF5 groups may be cropped records; their numeric IDs are not original sequence IDs."
    if annotations is not None:
        rows = _annotations(annotations)
        accepted = sum(row.get("status") == "verified" and row.get("review_status") == "accepted_by_human" for row in rows)
        valid_splits = sum(row.get("split") in {"train", "val", "test"} for row in rows)
        report["annotations"] = dict(records=len(rows), accepted=accepted, valid_splits=valid_splits,
            labels_ready_for_prepare=(accepted == valid_splits == len(rows)),
            statuses=dict(Counter(str(row.get("status")) for row in rows)))
    return report


def _array(value, shape, label):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"Invalid {label}: expected finite numeric {shape}")
    return value


def _sha(path, expected=None):
    if not path.is_file():
        raise ValueError(f"Missing recorded source: {path}")
    result = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and result != expected:
        raise ValueError(f"Source SHA256 changed: {path}")
    return result


def load_triplet(root, row):
    """Read real states even for diagnostics; this function never verifies labels."""
    root = Path(root).resolve()
    sequence = row.get("sequence", "")
    if not isinstance(sequence, str) or not re.fullmatch(r"S\d{2}T\d{3}", sequence):
        raise ValueError("Requires an original HIMO SXXTXXX sequence ID")
    subject = sequence.split("T")[0]
    if row.get("subject", subject) != subject or row.get("source_frame_index_base", 0) != 0:
        raise ValueError("Subject or original zero-based frame convention does not match")
    slots, keyframes = row.get("object_slots"), row.get("keyframes")
    if not isinstance(slots, dict) or set(slots) != {"0", "1"}:
        raise ValueError("Explicit, distinct object_slots 0 and 1 are required")
    names = [slots["0"], slots["1"]]
    if any(not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."} for name in names) or len(set(names)) != 2:
        raise ValueError("Invalid object name")
    if not isinstance(keyframes, dict) or set(keyframes) != {"S0", "S1", "S2"}:
        raise ValueError("Explicit S0/S1/S2 keyframes are required")
    frames = [keyframes[key] for key in ("S0", "S1", "S2")]
    if any(type(frame) is not int for frame in frames) or not 0 <= frames[0] < frames[1] < frames[2]:
        raise ValueError("Require original frame indices 0 <= S0 < S1 < S2")
    actions = row.get("actions")
    if not isinstance(actions, list) or len(actions) != 2:
        raise ValueError("Exactly two annotated actions are required")
    for index, action in enumerate(actions):
        if (not isinstance(action, dict) or not isinstance(action.get("action_id"), str) or not action["action_id"].strip()
                or type(action.get("start")) is not int or type(action.get("end")) is not int
                or not frames[index] < action["start"] <= action["end"] == frames[index + 1]
                or type(action.get("object_slot")) is not int or action["object_slot"] not in (0, 1)
                or (action.get("target_slot") is not None and
                    (type(action["target_slot"]) is not int or action["target_slot"] not in (0, 1)))):
            raise ValueError("Action bounds, identifiers or object references do not match the selected states")
    source = root / "smplx" / f"{sequence}.npz"
    object_path = root / "object_pose" / f"{sequence}.npy"
    hashes = dict(smplx=_sha(source, row.get("source_member_verification", {}).get("sha256")),
                  object_pose=_sha(object_path, row.get("local_object_pose_sha256")))
    with np.load(source, allow_pickle=False) as saved:
        params = {key: saved[key] for key in saved.files}
    total = len(params["body_pose"])
    if frames[-1] >= total or row.get("total_source_frames", total) != total:
        raise ValueError("Annotated frames do not match the complete original sequence")
    poses = [_array(params[key], shape, key) for key, shape in [
        ("global_orient", (total, 3)), ("body_pose", (total, 21, 3)),
        ("lhand_pose", (total, 15, 3)), ("rhand_pose", (total, 15, 3))]]
    translation = _array(params["transl"], (total, 3), "transl")
    beta = np.asarray(params["betas"])
    if beta.shape == (1, 10):
        beta = beta[0]
    beta = _array(beta, (10,), "shared betas")
    for key in ("expression", "jaw_pose", "leye_pose", "reye_pose"):
        if key in params and (not np.isfinite(params[key]).all() or np.any(np.abs(params[key]) > 1e-8)):
            raise ValueError(f"Nonzero {key} is incompatible with the fixed zero-face protocol")
    axis = np.concatenate([poses[0][:, None], *poses[1:]], axis=1)[frames[1:]]
    rotation = batch_rodrigues(torch.as_tensor(axis.copy(), dtype=torch.float32).reshape(-1, 3)).reshape(2, 52, 3, 3)
    human = torch.cat([rotation[..., :2, :].reshape(2, 312),
                       torch.as_tensor(translation[frames[1:]].copy(), dtype=torch.float32)], dim=-1)
    recorded_objects = np.load(object_path, allow_pickle=True).item()
    if not isinstance(recorded_objects, dict) or set(recorded_objects) != set(names):
        raise ValueError("Two-object supervision cannot silently drop a third task object")
    object_states = []
    for name in names:
        item = recorded_objects[name]
        rotation = _array(item["rot"], (total, 3, 3), f"{name}.rot")
        position = _array(item["transl"], (total, 3), f"{name}.transl")
        if not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(rotation), 1., atol=1e-5):
            raise ValueError(f"{name} matrices must be proper rotations")
        object_states.append(np.concatenate([rotation[frames, :2, :].reshape(3, 6), position[frames]], axis=-1))
    object_states = torch.as_tensor(np.stack(object_states, axis=1).copy(), dtype=torch.float32)
    metadata = dict(subject=subject, sequence_id=sequence, initial_frame=frames[0], target_frames=frames[1:],
        total_source_frames=total, object_names=names, provenance="recorded", two_task_objects=True,
        action_annotation=row.get("status", "unreviewed"), review_status=row.get("review_status", "unreviewed"),
        split=row.get("split"), source_files=dict(smplx=str(source), object_pose=str(object_path)),
        source_sha256=hashes, actions=deepcopy(actions), limitations=deepcopy(row.get("limitations", [])),
        physical_scene_complete=row.get("physical_scene_complete"))
    if "source_text_sha256" in row:
        text_path = root / "text" / f"{sequence}.txt"
        metadata["source_sha256"]["text"] = _sha(text_path, row["source_text_sha256"])
        metadata["source_files"]["text"] = str(text_path)
    for key in ("status", "weak_labeling", "source_caption", "source_text_sha256"):
        if key in row:
            metadata[key] = deepcopy(row[key])
    return dict(conditions=dict(initial_object_pose=object_states[0]),
                targets=dict(shape=torch.as_tensor(beta.copy(), dtype=torch.float32), human=human,
                             object_pose=object_states[1:]), metadata=metadata, object_names=names)


def prepare(root, annotations, geometry, output, allow_weak=False):
    """Export accepted labels, or exact machine-checked labels with explicit opt-in."""
    rows = _annotations(annotations)
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new prepared-data path")
    groups, subjects = {}, {}
    weak_dataset = any(row.get("status") == "weak_temporal_v1" for row in rows)
    for row in rows:
        accepted = row.get("status") == "verified" and row.get("review_status") == "accepted_by_human"
        weak = row.get("status") == "weak_temporal_v1" and row.get("review_status") == "machine_checked"
        evidence = row.get("weak_labeling", {})
        valid_weak = (isinstance(evidence, dict) and evidence.get("algorithm") == "weak_temporal_v1"
            and isinstance(evidence.get("thresholds"), dict) and bool(evidence["thresholds"])
            and isinstance(evidence.get("evidence"), dict) and bool(evidence["evidence"]))
        if not accepted and not (allow_weak and weak and valid_weak):
            raise ValueError("Pending/rejected stage labels cannot be prepared for training; human acceptance is required")
        if weak and any(action.get("action_id") not in {"move_object", "rotate_object", "move_and_rotate_object"}
                        for action in row.get("actions", [])):
            raise ValueError("Weak kinematic labels only support observable object movement/rotation actions")
        split = row.get("split")
        if split not in {"train", "val", "test"}:
            raise ValueError("Each accepted annotation requires an explicit train/val/test split")
        key = row.get("sequence")
        if key in groups and groups[key] != split:
            raise ValueError("An original recorded sequence cannot cross data splits")
        groups[key] = split
        subject = str(key).split("T")[0]
        if weak_dataset and subject in subjects and subjects[subject] != split:
            raise ValueError("Weak supervision requires subject-level isolation across data splits")
        subjects[subject] = split
    triplets = [load_triplet(root, row) for row in rows]
    names = [item["object_names"] for item in triplets]
    vocabulary = sorted({action["action_id"] for row in rows for action in row["actions"]})
    index = {action: i for i, action in enumerate(vocabulary)}
    targets = {key: torch.stack([item["targets"][key] for item in triplets]) for key in ("shape", "human", "object_pose")}
    conditions = dict(initial_object_pose=torch.stack([item["conditions"]["initial_object_pose"] for item in triplets]),
        descriptors=geometry.descriptors(names).detach().cpu(),
        action_ids=torch.tensor([[index[action["action_id"]] for action in row["actions"]] for row in rows]),
        action_objects=torch.tensor([[action["object_slot"] for action in row["actions"]] for row in rows]),
        action_targets=torch.tensor([[2 if action.get("target_slot") is None else action["target_slot"] for action in row["actions"]] for row in rows]))
    targets["contact"] = torch.cat([geometry.contacts(targets["shape"][start:start + 4],
        targets["human"][start:start + 4], targets["object_pose"][start:start + 4], names[start:start + 4]).detach().cpu()
        for start in range(0, len(triplets), 4)])
    train_actions = {action["action_id"] for row in rows if row["split"] == "train" for action in row["actions"]}
    metadata = [item["metadata"] for item in triplets]
    for item in metadata:
        item["unseen_action_ids_relative_to_train"] = sorted({action["action_id"] for action in item["actions"]} - train_actions)
    result = dict(conditions=conditions, targets=targets, metadata=metadata, vocabulary=vocabulary,
                  representation=geometry.protocol())
    from tridi.model.ordered_scene.pipeline import validate_training_data
    validate_training_data(result, allow_weak=allow_weak)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    return result
