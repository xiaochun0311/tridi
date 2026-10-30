"""Prevent same-shaped SMPL-H and SMPL-X parameters from being mixed."""
import hashlib
import re
import struct


WEAK_SUPERVISIONS = frozenset({"weak_temporal_v1", "weak_temporal_v2"})


FIXED = dict(schema="himo-native-smplx-v1", body_model="smplx", gender="neutral",
             num_betas=10, num_vertices=10475, units="m", use_pca=False,
             pose_layout="global_body_left_right_row6d_translation", face_policy="zero",
             contact_threshold_m=.05, contact_metric="unsigned_point_to_triangle",
             contact_sampling="neutral_template_fps550_body70_left70_right70",
             descriptor="canonical_mesh_extents_mean_absmean_std12")


def index_sha256(indices):
    return hashlib.sha256(struct.pack("<690q", *indices)).hexdigest()


def validate_representation(value):
    keys = set(FIXED) | {"model_sha256", "contact_vertex_indices", "contact_indices_sha256"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Requires the complete native HIMO SMPL-X representation protocol")
    if any(value[key] != expected for key, expected in FIXED.items()):
        raise ValueError("HIMO requires native neutral SMPL-X, full hands, row6d and metre units")
    ids = value["contact_vertex_indices"]
    if not isinstance(ids, list) or len(ids) != 690 or any(
            type(i) is not int or not 0 <= i < 10475 for i in ids) or len(set(ids)) != 690:
        raise ValueError("Requires 690 unique native SMPL-X contact vertex indices")
    for key in ("model_sha256", "contact_indices_sha256"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[0-9a-f]{64}", value[key]):
            raise ValueError(f"Invalid {key}")
    if value["contact_indices_sha256"] != index_sha256(ids):
        raise ValueError("Contact vertex indices do not match their SHA256")
    return value


def assert_same_representation(first, second):
    validate_representation(first)
    validate_representation(second)
    differences = [key for key in first if first[key] != second[key]]
    if differences:
        raise ValueError("Incompatible HIMO representation: " + ", ".join(differences))
