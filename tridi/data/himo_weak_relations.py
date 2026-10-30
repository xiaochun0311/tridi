"""Add observable centre-distance relations to fixed, machine-checked HIMO v1 stages."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ALGORITHM = "weak_temporal_v2"
BASE_ALGORITHM = "weak_temporal_v1"
KINEMATIC_ACTIONS = {"move_object", "rotate_object", "move_and_rotate_object"}
THRESHOLDS = dict(smooth_frames=5, min_distance_change_m=.05, min_main_centre_motion_m=.03,
    min_main_only_distance_change_m=.03, min_distance_directness=.55)
LIMITATIONS = ["Relations are machine weak labels for surface-centroid approach/retreat, not placement, containment, contact or task completion.",
    "Relation actions may include rotation; they do not prescribe a rotation angle.",
    "Source text is provenance only; v1 states, action bounds and subject splits are unchanged."]


def _sha(path, expected=None):
    result = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and result != expected:
        raise ValueError(f"Source SHA256 changed: {path}")
    return result


def _mesh_properties(path):
    import trimesh

    mesh = trimesh.load(str(path), process=False, force="mesh")
    areas = np.asarray(mesh.area_faces)
    if not len(areas) or not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError(f"Missing finite triangle surface: {path}")
    centroid = (np.asarray(mesh.triangles_center) * areas[:, None]).sum(0) / areas.sum()
    if not np.isfinite(centroid).all():
        raise ValueError(f"Nonfinite surface centroid: {path}")
    return dict(sha256=_sha(path), surface_centroid=centroid.tolist(),
        area_m2=float(areas.sum()), vertices=len(mesh.vertices), faces=len(mesh.faces))


def world_centres(objects, names, centroids):
    """Transform canonical surface centroids, never substitute raw pose origins."""
    centres = []
    total = len(objects[names[0]]["transl"])
    for name, centroid in zip(names, centroids):
        rotation = np.asarray(objects[name]["rot"], dtype=float)
        position = np.asarray(objects[name]["transl"], dtype=float)
        if (rotation.shape != (total, 3, 3) or position.shape != (total, 3)
                or not np.isfinite(rotation).all() or not np.isfinite(position).all()
                or not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-5)
                or not np.allclose(np.linalg.det(rotation), 1., atol=1e-5)):
            raise ValueError("Invalid recorded object poses")
        centres.append(np.einsum("tij,j->ti", rotation, centroid) + position)
    return np.stack(centres)


def classify_relation(centres, main, start, end):
    """Return a conservative relation and auditable evidence for actual target endpoints."""
    other = 1 - main
    distances = np.linalg.norm(centres[0, start:end + 1] - centres[1, start:end + 1], axis=1)
    change = float(distances[-1] - distances[0])
    main_motion = float(np.linalg.norm(centres[main, end] - centres[main, start]))
    other_motion = float(np.linalg.norm(centres[other, end] - centres[other, start]))
    main_only = float(np.linalg.norm(centres[main, end] - centres[other, start]) - distances[0])
    other_only = float(np.linalg.norm(centres[other, end] - centres[main, start]) - distances[0])
    width = THRESHOLDS["smooth_frames"]
    padded = np.pad(distances, (width // 2, width // 2), mode="edge")
    smoothed = sum(padded[i:i + len(distances)] for i in range(width)) / width
    path = float(np.abs(np.diff(smoothed)).sum())
    directness = float(abs(smoothed[-1] - smoothed[0]) / max(path, 1e-12))
    eligible = (abs(change) >= THRESHOLDS["min_distance_change_m"]
        and main_motion >= THRESHOLDS["min_main_centre_motion_m"]
        and np.sign(change) * main_only >= THRESHOLDS["min_main_only_distance_change_m"]
        and directness >= THRESHOLDS["min_distance_directness"])
    relation = ("approach_other_object" if change < 0 else "move_away_from_other_object") if eligible else None
    return relation, dict(start_frame=start, end_frame=end, main_slot=main, other_slot=other,
        centre_distance_start_m=float(distances[0]), centre_distance_end_m=float(distances[-1]),
        centre_distance_change_m=change, main_centre_motion_m=main_motion, other_centre_motion_m=other_motion,
        main_only_distance_change_m=main_only, other_only_distance_change_m=other_only,
        smoothed_distance_path_m=path, smoothed_distance_directness=directness,
        relation=relation, target_slot=other if relation else None)


def generate(root, input_path, output):
    """Create a new v2 file; retain all rows and all source states from its v1 input."""
    root, input_path, output = Path(root).resolve(), Path(input_path).resolve(), Path(output)
    if output.exists() or output.resolve() == input_path:
        raise ValueError("Use a new relation annotation output path; never overwrite the input")
    base = json.loads(input_path.read_text(encoding="utf-8"))
    rows = base.get("candidates")
    if not isinstance(rows, list) or not rows or base.get("status") != BASE_ALGORITHM:
        raise ValueError("Requires a nonempty weak_temporal_v1 annotation file")
    label_sha = _sha(input_path)
    mesh_cache, subjects, sequences, converted = {}, {}, set(), []
    for original in rows:
        row = deepcopy(original)
        sequence = row.get("sequence", "")
        weak = row.get("weak_labeling", {})
        if (not re.fullmatch(r"S\d{2}T\d{3}", sequence) or row.get("subject") != sequence.split("T")[0]
                or row.get("status") != BASE_ALGORITHM or row.get("review_status") != "machine_checked"
                or not isinstance(weak, dict) or weak.get("algorithm") != BASE_ALGORITHM
                or not isinstance(weak.get("thresholds"), dict) or not weak["thresholds"]
                or not isinstance(weak.get("evidence"), dict) or not weak["evidence"]):
            raise ValueError("Input rows must be explicit machine-checked v1 labels with evidence")
        split, subject = row.get("split"), row["subject"]
        if split not in {"train", "val", "test"} or sequence in sequences or (subject in subjects and subjects[subject] != split):
            raise ValueError("Requires unique sequences and isolated subject-level splits")
        sequences.add(sequence)
        subjects[subject] = split
        slots, keyframes = row.get("object_slots", {}), row.get("keyframes", {})
        if set(slots) != {"0", "1"} or set(keyframes) != {"S0", "S1", "S2"} or row.get("source_frame_index_base", 0) != 0:
            raise ValueError("Requires explicit object slots and original zero-based S0/S1/S2")
        names = [slots[str(i)] for i in (0, 1)]
        if (len(set(names)) != 2 or any(not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."} for name in names)):
            raise ValueError("Invalid explicit object names")
        paths = dict(object_pose=root / "object_pose" / f"{sequence}.npy",
            smplx=root / "smplx" / f"{sequence}.npz", text=root / "text" / f"{sequence}.txt")
        expected = dict(object_pose=row.get("local_object_pose_sha256"),
            smplx=row.get("source_member_verification", {}).get("sha256"), text=row.get("source_text_sha256"))
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in expected.values()):
            raise ValueError("v1 source SHA256 values are required")
        for key, path in paths.items():
            _sha(path, expected[key])
        objects = np.load(paths["object_pose"], allow_pickle=True).item()
        if not isinstance(objects, dict) or set(objects) != set(names):
            raise ValueError("Cannot change or omit source task objects")
        for name in names:
            if name not in mesh_cache:
                mesh_cache[name] = _mesh_properties(root / "object_mesh" / f"{name}.obj")
        centres = world_centres(objects, names, [mesh_cache[name]["surface_centroid"] for name in names])
        frames = [keyframes[key] for key in ("S0", "S1", "S2")]
        if (any(type(frame) is not int for frame in frames) or not 0 <= frames[0] < frames[1] < frames[2] < centres.shape[1]
                or row.get("total_source_frames") != centres.shape[1]):
            raise ValueError("Selected original states do not match source frames")
        actions = row.get("actions")
        if not isinstance(actions, list) or len(actions) != 2:
            raise ValueError("Requires the original two v1 actions")
        evidence = []
        for index, action in enumerate(actions):
            if (action.get("action_id") not in KINEMATIC_ACTIONS or action.get("target_slot") is not None
                    or type(action.get("object_slot")) is not int or action["object_slot"] not in (0, 1)
                    or type(action.get("start")) is not int or type(action.get("end")) is not int
                    or not frames[index] < action["start"] <= action["end"] == frames[index + 1]):
                raise ValueError("Cannot expand invalid or non-kinematic v1 action bounds/references")
            relation, stage = classify_relation(centres, action["object_slot"], frames[index], frames[index + 1])
            stage["original_action"] = deepcopy(action)
            if relation:
                action.update(action_id=relation, target_slot=1 - action["object_slot"])
            evidence.append(stage)
        row.update(status=ALGORITHM, review_status="machine_checked", weak_labeling=dict(algorithm=ALGORITHM,
            base_algorithm=BASE_ALGORITHM, thresholds=dict(THRESHOLDS), evidence=dict(
                base_weak_labeling=deepcopy(original["weak_labeling"]), base_actions=deepcopy(original["actions"]),
                source_label_sha256=label_sha, meshes={name: deepcopy(mesh_cache[name]) for name in names}, stages=evidence)))
        row["limitations"] = deepcopy(original.get("limitations", [])) + list(LIMITATIONS)
        converted.append(row)
    coverage = {}
    train_actions = {action["action_id"] for row in converted if row["split"] == "train" for action in row["actions"]}
    for split in ("train", "val", "test"):
        actions = [action for row in converted if row["split"] == split for action in row["actions"]]
        targets = Counter("None" if action["target_slot"] is None else str(action["target_slot"]) for action in actions)
        counts = Counter(action["action_id"] for action in actions)
        pairs = defaultdict(Counter)
        for action in actions:
            pairs[action["action_id"]]["None" if action["target_slot"] is None else str(action["target_slot"])] += 1
        coverage[split] = dict(action_counts=dict(counts), target_counts={key: targets[key] for key in ("0", "1", "None")},
            action_target_counts={key: dict(value) for key, value in pairs.items()},
            unseen_actions_relative_to_train=sorted(set(counts) - train_actions))
    report = dict(schema="himo-weak-temporal-relations-v2", status=ALGORITHM, review_status="machine_checked",
        base_algorithm=BASE_ALGORITHM, source_labels=str(input_path), source_label_sha256=label_sha,
        thresholds=dict(THRESHOLDS), base_report={key: value for key, value in base.items() if key != "candidates"},
        selected_counts=dict(Counter(row["split"] for row in converted)), coverage=coverage,
        closed_action_coverage=all(not value["unseen_actions_relative_to_train"] for value in coverage.values()),
        target_counts=dict(Counter("None" if action["target_slot"] is None else str(action["target_slot"])
            for row in converted for action in row["actions"])),
        relation_action_counts=dict(Counter(stage["relation"] for row in converted
            for stage in row["weak_labeling"]["evidence"]["stages"] if stage["relation"] is not None)),
        relation_stages=sum(stage["relation"] is not None for row in converted for stage in row["weak_labeling"]["evidence"]["stages"]),
        limitations=list(LIMITATIONS), candidates=converted)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = generate(args.root, args.input, args.output)
    print(json.dumps({key: value for key, value in report.items() if key not in {"candidates", "base_report"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
