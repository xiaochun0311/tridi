"""Native HIMO geometry: full-hand SMPL-X, rigid objects and surface contacts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import smplx
import torch
import trimesh

from tridi.utils.geometry import rotation_6d_to_matrix
from .protocol import FIXED, assert_same_representation, index_sha256, validate_representation


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _farthest(vertices, candidates, count):
    """Deterministic sampling within a fixed anatomical region."""
    points = np.asarray(vertices, dtype=np.float64)[candidates]
    distances = np.full(len(points), np.inf)
    chosen = np.empty(count, dtype=np.int64)
    next_id = int(np.square(points - points.mean(0)).sum(1).argmax())
    for index in range(count):
        chosen[index] = next_id
        distances = np.minimum(distances, np.square(points - points[next_id]).sum(1))
        distances[chosen[:index + 1]] = -1
        next_id = int(distances.argmax())
    return np.asarray(candidates, dtype=np.int64)[chosen]


class NativeSceneGeometry:
    def __init__(self, models_root, meshes_root, device="cpu", indices_path=None, representation=None):
        self.models_root, self.meshes_root = Path(models_root), Path(meshes_root)
        self.device = torch.device(device)
        self.model_path = self.models_root / "smplx/SMPLX_NEUTRAL.npz"
        self.model = smplx.build_layer(str(self.models_root), model_type="smplx", gender="neutral",
                                     num_betas=10, use_pca=False).to(self.device).eval().requires_grad_(False)
        if self.model.v_template.shape != (10475, 3):
            raise ValueError("HIMO requires the native 10475-vertex SMPL-X model")
        weights = self.model.lbs_weights.detach().cpu().numpy()
        self.left_hand_indices = np.flatnonzero(weights[:, [20] + list(range(25, 40))].sum(1) > .5)
        self.right_hand_indices = np.flatnonzero(weights[:, [21] + list(range(40, 55))].sum(1) > .5)
        body = np.setdiff1d(np.arange(10475), np.union1d(self.left_hand_indices, self.right_hand_indices))
        template = self.model.v_template.detach().cpu().numpy()
        expected_indices = np.concatenate([_farthest(template, body, 550),
            _farthest(template, self.left_hand_indices, 70), _farthest(template, self.right_hand_indices, 70)])
        if representation is not None:
            validate_representation(representation)
            self.indices = np.asarray(representation["contact_vertex_indices"], dtype=np.int64)
        elif indices_path is not None and Path(indices_path).exists():
            self.indices = np.load(indices_path, allow_pickle=False)
        else:
            self.indices = expected_indices
        if (self.indices.shape != (690,) or self.indices.dtype.kind not in "iu" or
                len(np.unique(self.indices)) != 690 or self.indices.min() < 0 or self.indices.max() >= 10475):
            raise ValueError("Requires 690 unique native SMPL-X vertex indices")
        self.indices = self.indices.astype(np.int64)
        if not np.array_equal(self.indices, expected_indices):
            raise ValueError("Contact indices differ from the fixed native SMPL-X FPS protocol")
        if (np.isin(self.indices, self.left_hand_indices).sum() != 70 or
                np.isin(self.indices, self.right_hand_indices).sum() != 70):
            raise ValueError("Native contact sampling requires 70 vertices on each hand")
        self.model_sha256 = _sha256(self.model_path)
        if representation is not None:
            assert_same_representation(self.protocol(), representation)
        if indices_path is not None:
            path = Path(indices_path)
            if path.exists():
                if not np.array_equal(np.load(path, allow_pickle=False), self.indices):
                    raise ValueError("Existing contact sampling differs; preserve the original asset")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    np.save(stream, self.indices, allow_pickle=False)
        self._meshes = {}

    def protocol(self):
        ids = self.indices.tolist()
        return dict(FIXED, model_sha256=self.model_sha256, contact_vertex_indices=ids,
                    contact_indices_sha256=index_sha256(ids))

    def _tensor(self, value, shape, label):
        value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tuple(value.shape) != tuple(shape) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid {label}; expected finite {tuple(shape)}")
        return value

    @staticmethod
    def _rotations(sixd):
        first, second = sixd[..., :3], sixd[..., 3:]
        if bool((torch.linalg.vector_norm(first, dim=-1) < 1e-8).any()) or bool(
                (torch.linalg.vector_norm(torch.cross(first, second, dim=-1), dim=-1) < 1e-8).any()):
            raise ValueError("Degenerate 6D rotation cannot define a rigid transform")
        return rotation_6d_to_matrix(sixd)

    def decode(self, shape, human):
        batch = len(shape)
        shape = self._tensor(shape, (batch, 10), "shape")
        human = self._tensor(human, (batch, 2, 315), "human")
        rotation = self._rotations(human[..., :312].reshape(batch * 2, 52, 6))
        count = batch * 2
        identity = torch.eye(3, device=self.device).expand(count, 1, 3, 3)
        result = self.model(betas=shape[:, None].expand(-1, 2, -1).reshape(count, 10),
            transl=human[..., 312:].reshape(count, 3), global_orient=rotation[:, :1],
            body_pose=rotation[:, 1:22], left_hand_pose=rotation[:, 22:37], right_hand_pose=rotation[:, 37:52],
            jaw_pose=identity, leye_pose=identity, reye_pose=identity,
            expression=torch.zeros(count, 10, device=self.device))
        # HIMO's stored first52 includes face joints; it is not this full-hand layout.
        joints = torch.cat([result.joints[:, :22], result.joints[:, 25:55]], dim=1)
        return result.vertices.reshape(batch, 2, 10475, 3), joints.reshape(batch, 2, 52, 3)

    def _mesh(self, name):
        if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Object names must identify canonical HIMO meshes")
        if name not in self._meshes:
            mesh = trimesh.load(self.meshes_root / (name + ".obj"), process=False)
            if (not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0 or
                    not np.isfinite(mesh.vertices).all()):
                raise ValueError(f"Invalid canonical mesh: {name}")
            self._meshes[name] = mesh
        return self._meshes[name]

    @staticmethod
    def _names(object_names, batch):
        if len(object_names) != batch or any(isinstance(row, str) or len(row) != 2 for row in object_names):
            raise ValueError("Object names must preserve two explicit slots per scene")
        return object_names

    def descriptors(self, object_names):
        self._names(object_names, len(object_names))
        rows = []
        for pair in object_names:
            row = []
            for name in pair:
                vertices = np.asarray(self._mesh(name).vertices)
                row.append(np.concatenate([np.ptp(vertices, axis=0), vertices.mean(0),
                                           np.abs(vertices).mean(0), vertices.std(0)]))
            rows.append(row)
        return torch.as_tensor(np.asarray(rows), dtype=torch.float32, device=self.device)

    def object_vertices(self, name, pose):
        pose = self._tensor(pose, (9,), "object pose")
        vertices = torch.as_tensor(np.asarray(self._mesh(name).vertices), dtype=torch.float32, device=self.device)
        return vertices @ self._rotations(pose[:6]).T + pose[6:]

    def _contact_inputs(self, shape, human, object_pose, object_names):
        batch = len(shape)
        names = self._names(object_names, batch)
        pose = self._tensor(object_pose, (batch, 2, 2, 9), "object pose")
        rotations = self._rotations(pose[..., :6]).cpu().numpy()
        translation = pose[..., 6:].cpu().numpy()
        vertices = self.decode(shape, human)[0][:, :, self.indices].cpu().numpy()
        return names, rotations, translation, vertices

    @torch.no_grad()
    def contact_distances(self, shape, human, object_pose, object_names):
        names, rotations, translation, vertices = self._contact_inputs(shape, human, object_pose, object_names)
        batch = len(names)
        distances = np.empty((batch, 2, 2, 690), dtype=np.float32)
        for b, pair in enumerate(names):
            for time in range(2):
                for slot, name in enumerate(pair):
                    canonical = (vertices[b, time] - translation[b, time, slot]) @ rotations[b, time, slot]
                    mesh = self._mesh(name)
                    # Detailed thin meshes can allocate many candidate triangles per query.
                    distances[b, time, slot] = np.concatenate([
                        trimesh.proximity.closest_point(mesh, canonical[start:start + 128])[1]
                        for start in range(0, 690, 128)])
        if not np.isfinite(distances).all():
            raise ValueError("Nonfinite point-to-triangle contact distances")
        return torch.as_tensor(distances, dtype=torch.float32, device=self.device)

    @staticmethod
    def _surface_near(mesh, points, threshold):
        # Threshold labels need only nearby triangles, not a global nearest query.
        triangles, tree = mesh.triangles, mesh.triangles_tree
        result = np.zeros(len(points), dtype=bool)
        for index, point in enumerate(np.asarray(points, dtype=np.float64)):
            candidates = np.fromiter(tree.intersection(np.r_[point - threshold, point + threshold]), dtype=np.int64)
            for start in range(0, len(candidates), 4096):
                ids = candidates[start:start + 4096]
                nearest = trimesh.triangles.closest_point(triangles[ids], np.broadcast_to(point, (len(ids), 3)))
                if not np.isfinite(nearest).all():
                    raise ValueError("Nonfinite point-to-triangle contact query")
                if np.any(np.square(nearest - point).sum(1) < threshold ** 2):
                    result[index] = True
                    break
        return result

    @torch.no_grad()
    def contacts(self, shape, human, object_pose, object_names):
        names, rotations, translation, vertices = self._contact_inputs(shape, human, object_pose, object_names)
        result = np.empty((len(names), 2, 2, 690), dtype=np.float32)
        for b, pair in enumerate(names):
            for time in range(2):
                for slot, name in enumerate(pair):
                    points = (vertices[b, time] - translation[b, time, slot]) @ rotations[b, time, slot]
                    result[b, time, slot] = self._surface_near(self._mesh(name), points, FIXED["contact_threshold_m"])
        return torch.as_tensor(result, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def export(self, prediction, object_names, outputdir):
        batch = len(prediction["shape"])
        names = self._names(object_names, batch)
        pose = self._tensor(prediction["object_pose"], (batch, 2, 2, 9), "object pose")
        vertices, joints = self.decode(prediction["shape"], prediction["human"])
        root = Path(outputdir).resolve()
        root.mkdir(parents=True, exist_ok=False)
        paths = []
        for b, pair in enumerate(names):
            for time in range(2):
                prefix = f"scene{b:03d}_S{time + 1}"
                body_path = root / (prefix + "_human.ply")
                trimesh.Trimesh(vertices[b, time].cpu().numpy(), self.model.faces, process=False).export(body_path)
                row = dict(scene=b, state=f"S{time + 1}", human=str(body_path), objects=[])
                for slot, name in enumerate(pair):
                    mesh = self._mesh(name)
                    world = self.object_vertices(name, pose[b, time, slot]).cpu().numpy()
                    path = root / f"{prefix}_object{slot}_{name}.ply"
                    trimesh.Trimesh(world, mesh.faces, process=False).export(path)
                    row["objects"].append(dict(slot=slot, name=name, path=str(path)))
                paths.append(row)
        joints_path = root / "native52_joints.npy"
        np.save(joints_path, joints.cpu().numpy(), allow_pickle=False)
        result = dict(representation=self.protocol(), scenes=paths, joints=str(joints_path),
                      geometry_corrections=False, face_policy="zero", contact_metric="unsigned proximity, not penetration")
        (root / "meshes.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
