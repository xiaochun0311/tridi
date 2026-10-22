"""Canonical mesh-backed object SDFs, negative inside and positive outside."""
import hashlib
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import igl
import numpy as np
import torch
import torch.nn.functional as F

from tridi.utils.geometry import rotation_6d_to_matrix


class ObjectSDF:
    def __init__(self, meshes, resolution=64, padding=0.1, cache_dir=None, device='cpu', allow_open=False):
        if resolution < 8 or padding <= 0:
            raise ValueError('Object SDF requires resolution >= 8 and positive padding.')
        self.meshes = meshes
        self.resolution = resolution
        self.padding = padding
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.device = device
        self.allow_open = allow_open
        self.fields = {}

    def _field(self, class_id):
        if class_id in self.fields:
            return self.fields[class_id]
        mesh = self.meshes[class_id].copy()
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals(multibody=True)
        closed = mesh.is_watertight and mesh.is_volume
        if not closed and not self.allow_open:
            raise ValueError(
                f'Object {class_id}: SDF requires a closed, consistently oriented volume; '
                'repair the source mesh explicitly (no convex-hull fallback).'
            )
        if not closed:
            logging.getLogger(__name__).warning(
                'Object %s is not a closed volume: using approximate winding-number signs; '
                'distances near holes are not reliable collision ground truth.', class_id
            )
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if not np.isfinite(vertices).all():
            raise ValueError(f'Object {class_id}: non-finite mesh vertices.')
        extent = np.ptp(vertices, axis=0)
        if extent.min() <= 0:
            raise ValueError(f'Object {class_id}: degenerate mesh bounds.')
        # Keep resolution along thin axes (e.g. knife blades) instead of cubic bounds.
        lower = vertices.min(0) - self.padding * extent
        upper = vertices.max(0) + self.padding * extent

        key = hashlib.sha256(vertices.tobytes() + faces.tobytes()
                             + repr((1, self.resolution, self.padding, self.allow_open)).encode()).hexdigest()
        cache = self.cache_dir / f'{key}.npz' if self.cache_dir is not None else None
        if cache is not None and cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                values = data['sdf']
        else:
            x, y, z = [np.linspace(lo, hi, self.resolution) for lo, hi in zip(lower, upper)]
            zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
            points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
            values = np.concatenate([
                igl.signed_distance(part, vertices, faces,
                                    igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL if closed
                                    else igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER)[0]
                for part in np.array_split(points, max(1, (len(points) + 32767) // 32768))
            ]).astype(np.float32).reshape((self.resolution,) * 3)
            if not np.isfinite(values).all():
                raise ValueError(f'Object {class_id}: non-finite SDF grid.')
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(dir=cache.parent, suffix='.npz', delete=False) as fp:
                    temporary = Path(fp.name)
                    np.savez_compressed(fp, sdf=values)
                temporary.replace(cache)
        if values.shape != (self.resolution,) * 3 or not np.isfinite(values).all():
            raise ValueError(f'Object {class_id}: invalid SDF cache {cache}.')
        field = (
            torch.as_tensor(values, dtype=torch.float32, device=self.device)[None, None],
            torch.as_tensor(lower, dtype=torch.float32, device=self.device),
            torch.as_tensor(upper, dtype=torch.float32, device=self.device),
        )
        self.fields[class_id] = field
        return field

    def query(self, points, obj_class, obj_R, obj_c, scale=1.0):
        """Query [B,N,3] points after the same world scaling as MeshModel.

        MeshModel poses row vectors as (v @ R.T + obj_c) * scale.
        SDF values use points' units; gradients reach points, rotation and translation.
        """
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError('SDF points must have shape [B,N,3].')
        scale = torch.as_tensor(1.0 if scale is None else scale,
                                dtype=points.dtype, device=points.device).reshape(-1, 1, 1)
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('SDF scale must be finite and positive.')
        rotation = rotation_6d_to_matrix(obj_R.reshape(-1, 6))
        canonical = torch.bmm(points / scale - obj_c.unsqueeze(1), rotation)
        classes = obj_class.to(device=points.device).reshape(-1)
        if len(classes) != len(points):
            raise ValueError('SDF object classes must match the point batch.')
        distances = points.new_zeros(points.shape[:2])
        for class_id in classes.unique().tolist():
            indexes = torch.where(classes == class_id)[0]
            grid, lower, upper = [v.to(points) for v in self._field(class_id)]
            query = canonical[indexes]
            clamped = torch.maximum(lower, torch.minimum(query, upper))
            normalized = 2 * (clamped - lower) / (upper - lower) - 1
            values = F.grid_sample(
                grid.expand(len(indexes), -1, -1, -1, -1),
                normalized.reshape(len(indexes), 1, 1, -1, 3),
                mode='bilinear', padding_mode='border', align_corners=True,
            ).reshape(len(indexes), -1)
            # Outside the padded grid use a positive distance extension, not zero padding.
            values = values + torch.linalg.vector_norm(query - clamped, dim=-1)
            distances = distances.index_copy(0, indexes, values)
        return distances * scale.reshape(-1, 1)

    def to(self, device):
        self.device = device
        self.fields = {key: tuple(v.to(device) for v in field)
                       for key, field in self.fields.items()}
        return self
