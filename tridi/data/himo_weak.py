"""Reproducible kinematic weak stages; these are not human-reviewed action semantics."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ALGORITHM = "weak_temporal_v1"
VAL_SUBJECTS = {"S01", "S08", "S16"}
TEST_SUBJECTS = {"S02", "S09", "S20"}
THRESHOLDS = dict(smooth_frames=5, activity_translation_m_per_frame=.003,
    activity_rotation_deg_per_frame=1., dominance_ratio=2., min_phase_frames=6,
    bridge_idle_frames=1, max_gap_frames=60, min_net_translation_m=.03,
    min_net_rotation_deg=12., min_translation_directness=.55,
    min_rotation_directness=.5, max_step_translation_m=.15, max_step_rotation_deg=45.,
    significant_backtrack_m=.01, max_translation_reversals=1, projection_noise_m=.0005,
    translation_excursion_path_m=.06, min_excursion_translation_directness=.35,
    rotation_excursion_path_deg=24., min_excursion_rotation_directness=.4)
LIMITATIONS = [
    "Machine-checked kinematic phases, not human-verified semantic action boundaries or task completion.",
    "Only dominant object translation/rotation is labeled; no drinking, insertion, wearing or grasp claim.",
    "Recorded object translation origins need not be geometric centres; frame thresholds do not assume a frame rate.",
    "Source text is retained only for provenance and is not used to infer stage labels.",
    "S0 is an observed intermediate state; untracked room/support geometry may be absent."]


def _smooth(values):
    width = THRESHOLDS["smooth_frames"]
    padded = np.pad(values, ((width // 2, width // 2),) + ((0, 0),) * (values.ndim - 1), mode="edge")
    return sum(padded[i:i + len(values)] for i in range(width)) / width


def _angle(rotation):
    return np.degrees(np.arccos(np.clip((np.trace(rotation, axis1=-2, axis2=-1) - 1.) / 2., -1., 1.)))


def _runs(labels):
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [(int(labels[start]), int(start), int(end - 1)) for start, end in zip(edges[:-1], edges[1:])]


def find_stages(objects):
    """Return at most one pair, using only the complete recorded object trajectories."""
    names = sorted(objects)
    if len(names) != 2:
        return None, "not_two_objects"
    positions, rotations, raw_positions, raw_rotations = [], [], [], []
    total = len(objects[names[0]]["transl"])
    if total < 2 * THRESHOLDS["min_phase_frames"] + 3:
        return None, "short_sequence"
    for name in names:
        position = np.asarray(objects[name]["transl"], dtype=float)
        rotation = np.asarray(objects[name]["rot"], dtype=float)
        if (position.shape != (total, 3) or rotation.shape != (total, 3, 3)
                or not np.isfinite(position).all() or not np.isfinite(rotation).all()
                or not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-5)
                or not np.allclose(np.linalg.det(rotation), 1., atol=1e-5)):
            return None, "invalid_recorded_pose"
        if (np.linalg.norm(np.diff(position, axis=0), axis=-1).max() > THRESHOLDS["max_step_translation_m"]
                or _angle(rotation[1:] @ rotation[:-1].transpose(0, 2, 1)).max() > THRESHOLDS["max_step_rotation_deg"]):
            return None, "tracking_jump"
        raw_positions.append(position)
        raw_rotations.append(rotation)
        positions.append(_smooth(position))
        u, _, vh = np.linalg.svd(_smooth(rotation))
        correction = np.tile(np.eye(3), (total, 1, 1))
        correction[:, 2, 2] = np.linalg.det(u @ vh)
        rotations.append(u @ correction @ vh)
    positions, rotations = np.stack(positions), np.stack(rotations)
    raw_positions, raw_rotations = np.stack(raw_positions), np.stack(raw_rotations)
    delta = np.diff(positions, axis=1, prepend=positions[:, :1])
    translation_step = np.linalg.norm(delta, axis=-1)
    rotation_step = np.concatenate([np.zeros((2, 1)), _angle(rotations[:, 1:] @ rotations[:, :-1].transpose(0, 1, 3, 2))], axis=1)
    activity = np.maximum(translation_step / THRESHOLDS["activity_translation_m_per_frame"],
                          rotation_step / THRESHOLDS["activity_rotation_deg_per_frame"])
    dominant = activity.argmax(axis=0)
    best, other = activity.max(axis=0), activity.min(axis=0)
    labels = np.where(best < 1., -1, np.where(best >= THRESHOLDS["dominance_ratio"] * other, dominant, -2))
    for label, start, end in _runs(labels):
        if label == -1 and end - start + 1 <= THRESHOLDS["bridge_idle_frames"] and start > 0 and end + 1 < total:
            if labels[start - 1] >= 0 and labels[start - 1] == labels[end + 1]:
                labels[start:end + 1] = labels[start - 1]
    active = [(label, start, end) for label, start, end in _runs(labels) if label >= 0]
    choices = []
    for first, second in zip(active, active[1:]):
        if (min(first[2] - first[1] + 1, second[2] - second[1] + 1) < THRESHOLDS["min_phase_frames"]
                or first[1] == 0 or second[2] >= total - 1
                or second[1] - first[2] - 1 > THRESHOLDS["max_gap_frames"]
                or np.any(labels[first[2] + 1:second[1]] == -2)):
            continue
        stages = []
        for slot, start, end in (first, second):
            origin = start - 1
            displacement = positions[slot, end] - positions[slot, origin]
            distance = float(np.linalg.norm(displacement))
            angle = float(_angle(rotations[slot, end] @ rotations[slot, origin].T))
            path = float(translation_step[slot, start:end + 1].sum())
            angular_path = float(rotation_step[slot, start:end + 1].sum())
            directness = distance / max(path, 1e-12)
            rotational_directness = angle / max(angular_path, 1e-12)
            moves = distance >= THRESHOLDS["min_net_translation_m"]
            rotates = angle >= THRESHOLDS["min_net_rotation_deg"]
            projection = delta[slot, start:end + 1] @ (displacement / max(distance, 1e-12))
            signed = np.sign(projection[np.abs(projection) >= THRESHOLDS["projection_noise_m"]])
            reversals = int(np.count_nonzero(signed[1:] != signed[:-1]))
            backtrack = float(-projection[projection < 0].sum())
            if (not (moves or rotates) or (moves and directness < THRESHOLDS["min_translation_directness"])
                    or (rotates and rotational_directness < THRESHOLDS["min_rotation_directness"])
                    or (path >= THRESHOLDS["translation_excursion_path_m"] and directness < THRESHOLDS["min_excursion_translation_directness"])
                    or (angular_path >= THRESHOLDS["rotation_excursion_path_deg"] and rotational_directness < THRESHOLDS["min_excursion_rotation_directness"])
                    or (reversals > THRESHOLDS["max_translation_reversals"] and backtrack > THRESHOLDS["significant_backtrack_m"])):
                break
            action = "move_and_rotate_object" if moves and rotates else "move_object" if moves else "rotate_object"
            stages.append(dict(action_id=action, object_slot=slot, start=start, end=end, target_slot=None,
                evidence=dict(active_start=start, active_end=end, net_translation_m=distance,
                    net_rotation_deg=angle, translation_path_m=path, rotation_path_deg=angular_path,
                    translation_directness=directness, rotation_directness=rotational_directness,
                    translation_reversals=reversals, backtrack_m=backtrack,
                    dominant_activity_share=float(activity[slot, start:end + 1].sum() / activity[:, start:end + 1].sum()))))
        if len(stages) != 2:
            continue
        frames = [first[1] - 1, first[2], second[2]]
        for index, stage in enumerate(stages):
            start, end = frames[index:index + 2]
            distances = np.linalg.norm(raw_positions[:, end] - raw_positions[:, start], axis=-1)
            angles = _angle(raw_rotations[:, end] @ raw_rotations[:, start].transpose(0, 2, 1))
            slot = stage["object_slot"]
            moves = distances[slot] >= THRESHOLDS["min_net_translation_m"]
            rotates = angles[slot] >= THRESHOLDS["min_net_rotation_deg"]
            raw_action = ("move_and_rotate_object" if moves and rotates else "move_object" if moves
                          else "rotate_object" if rotates else None)
            if raw_action != stage["action_id"]:
                break
            stage["evidence"]["raw_condition_to_target"] = dict(start_frame=start, end_frame=end,
                action_id=raw_action, object_slot=slot, net_translation_m=float(distances[slot]),
                net_rotation_deg=float(angles[slot]), all_object_translation_m=distances.tolist(),
                all_object_rotation_deg=angles.tolist())
        if any("raw_condition_to_target" not in stage["evidence"] for stage in stages):
            continue
        # No repeated substantial translation reversals across the whole chosen window.
        repeated = False
        for slot in (0, 1):
            increments = delta[slot, first[1]:second[2] + 1]
            _, _, vh = np.linalg.svd(increments, full_matrices=False)
            signed_motion = increments @ vh[0]
            signs = np.sign(signed_motion[np.abs(signed_motion) >= THRESHOLDS["projection_noise_m"]])
            repeated |= (np.count_nonzero(signs[1:] != signs[:-1]) > THRESHOLDS["max_translation_reversals"]
                and min(signed_motion[signed_motion > 0].sum(), -signed_motion[signed_motion < 0].sum()) > THRESHOLDS["significant_backtrack_m"])
        if repeated:
            continue
        quality = sum(stage["evidence"]["dominant_activity_share"] for stage in stages)
        choices.append((quality, first[1] - 1, stages))
    if not choices:
        return None, "no_clear_adjacent_pair"
    _, initial, stages = max(choices, key=lambda choice: (choice[0], -choice[1]))
    evidence = dict(stages=[stage.pop("evidence") for stage in stages],
        selection="highest summed dominant activity share; earliest pair breaks ties",
        repeated_translation_reversal_check="passed", selected_kinematic_endpoints=True)
    return dict(object_slots={str(i): name for i, name in enumerate(names)},
        keyframes=dict(S0=initial, S1=stages[0]["end"], S2=stages[1]["end"]),
        actions=stages, total_source_frames=total,
        weak_labeling=dict(algorithm=ALGORITHM, thresholds=dict(THRESHOLDS), evidence=evidence)), None


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample(rows, maximum, rng):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(tuple(row["object_slots"].values()), tuple(action["action_id"] for action in row["actions"]))].append(row)
    keys = sorted(buckets)
    rng.shuffle(keys)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected, used = [], set()
    # Reserve observable action classes when available before balancing small object/action buckets.
    for action in sorted({item["action_id"] for row in rows for item in row["actions"]}):
        for key in keys:
            match = next((row for row in buckets[key] if row["sequence"] not in used
                and any(item["action_id"] == action for item in row["actions"])), None)
            if match is not None and len(selected) < maximum:
                selected.append(match)
                used.add(match["sequence"])
                break
    while len(selected) < min(maximum, len(rows)):
        for key in keys:
            while buckets[key] and buckets[key][-1]["sequence"] in used:
                buckets[key].pop()
            if buckets[key] and len(selected) < maximum:
                row = buckets[key].pop()
                selected.append(row)
                used.add(row["sequence"])
    return sorted(selected, key=lambda row: row["sequence"])


def generate(root, output, seed=42, max_per_split=(512, 64, 64)):
    """Write a new weak-only annotation file; never edit human-review candidates."""
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise ValueError("Use a new weak annotation output path")
    if not (root / "object_pose").is_dir() or len(max_per_split) != 3 or any(n < 1 for n in max_per_split):
        raise ValueError("Requires recorded object_pose sources and positive train/val/test caps")
    candidates, rejected, source_subjects = [], Counter(), Counter()
    for path in sorted((root / "object_pose").glob("*.npy")):
        if not re.fullmatch(r"S\d{2}T\d{3}", path.stem):
            continue
        objects = np.load(path, allow_pickle=True).item()
        if not isinstance(objects, dict):
            rejected["invalid_object_source"] += 1
            continue
        if len(objects) == 2:
            source_subjects[path.stem.split("T")[0]] += 1
        row, reason = find_stages(objects)
        if row is None:
            rejected[reason] += 1
            continue
        sequence, subject = path.stem, path.stem.split("T")[0]
        source, text = root / "smplx" / f"{sequence}.npz", root / "text" / f"{sequence}.txt"
        if not source.is_file() or not text.is_file():
            rejected["missing_smplx_or_text"] += 1
            continue
        with np.load(source, allow_pickle=False) as data:
            if len(data["body_pose"]) != row["total_source_frames"]:
                rejected["source_frame_count_mismatch"] += 1
                continue
            if any(key in data and (not np.isfinite(data[key]).all() or np.any(np.abs(data[key]) > 1e-8))
                   for key in ("expression", "jaw_pose", "leye_pose", "reye_pose")):
                rejected["nonzero_face"] += 1
                continue
        split = "val" if subject in VAL_SUBJECTS else "test" if subject in TEST_SUBJECTS else "train"
        row.update(sequence=sequence, subject=subject, split=split, source_frame_index_base=0,
            status=ALGORITHM, review_status="machine_checked", source_member_verification={"sha256": _sha(source)},
            local_object_pose_sha256=_sha(path), source_caption=text.read_text(encoding="utf-8"),
            source_text_sha256=_sha(text), limitations=list(LIMITATIONS), physical_scene_complete=False)
        candidates.append(row)
    rng = np.random.default_rng(seed)
    chosen = []
    for split, maximum in zip(("train", "val", "test"), max_per_split):
        chosen.extend(_sample([row for row in candidates if row["split"] == split], maximum, rng))
    train_actions = {action["action_id"] for row in chosen if row["split"] == "train" for action in row["actions"]}
    kept = [row for row in chosen if row["split"] == "train" or all(action["action_id"] in train_actions for action in row["actions"])]
    rejected["heldout_action_absent_from_train"] = len(chosen) - len(kept)
    variation = {}
    for split in ("all", "train", "val", "test"):
        subset = [row for row in kept if split == "all" or row["split"] == split]
        different_objects = sum(row["actions"][0]["object_slot"] != row["actions"][1]["object_slot"] for row in subset)
        different_actions = sum(row["actions"][0]["action_id"] != row["actions"][1]["action_id"] for row in subset)
        variation[split] = dict(records=len(subset), different_object_slots=different_objects,
            different_action_ids=different_actions, different_object_slots_fraction=different_objects / max(len(subset), 1),
            different_action_ids_fraction=different_actions / max(len(subset), 1))
    report = dict(schema="himo-weak-temporal-v1", status=ALGORITHM, review_status="machine_checked", seed=seed,
        thresholds=dict(THRESHOLDS), limits=dict(zip(("train", "val", "test"), max_per_split)),
        source_two_object_subjects=dict(sorted(source_subjects.items())),
        requested_heldout_subjects=dict(val=sorted(VAL_SUBJECTS), test=sorted(TEST_SUBJECTS)),
        missing_requested_subjects=sorted((VAL_SUBJECTS | TEST_SUBJECTS) - source_subjects.keys()),
        quality_candidates=dict(Counter(row["split"] for row in candidates)),
        selected_counts=dict(Counter(row["split"] for row in kept)), rejected=dict(rejected),
        selected_subjects={split: sorted({row["subject"] for row in kept if row["split"] == split}) for split in ("train", "val", "test")},
        selected_action_counts={split: dict(Counter(action["action_id"] for row in kept if row["split"] == split for action in row["actions"])) for split in ("train", "val", "test")},
        selected_order_variation=variation,
        limitations=list(LIMITATIONS), candidates=kept)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=512)
    parser.add_argument("--max-val", type=int, default=64)
    parser.add_argument("--max-test", type=int, default=64)
    args = parser.parse_args()
    result = generate(args.root, args.output, args.seed, (args.max_train, args.max_val, args.max_test))
    print(json.dumps({key: value for key, value in result.items() if key != "candidates"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
