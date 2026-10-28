"""Load canonical object points using the checkpoint's bundled references."""
import logging
import pickle
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


def _validate(entry, object_name, source):
    if not isinstance(entry, dict) or 'cartesian' not in entry:
        raise ValueError(f'{source}: object {object_name} needs cartesian keypoints.')
    points = np.asarray(entry['cartesian'])
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.issubdtype(points.dtype, np.number)
            or not np.isrealobj(points) or not np.isfinite(points).all()):
        raise ValueError(f'{source}: object {object_name} keypoints must be finite nonempty Nx3 coordinates.')
    return dict(entry, cartesian=points)


def load_object_keypoints(name, root, classid2obj, assets_folder=None):
    """Prefer official points without rewriting shared preprocessed datasets.

    A custom dataset with no bundled reference falls back to its NPZ files.
    A present reference must contain every requested object: never silently mix
    training-reference and regenerated points within the same dataset.
    """
    root = Path(root)
    reference = Path(assets_folder or './assets') / 'object_keypoints' / f'{name}.pkl'
    bundled = None
    if reference.is_file():
        with reference.open('rb') as fp:
            bundled = pickle.load(fp)
        if not isinstance(bundled, dict):
            raise ValueError(f'{reference}: expected an object-name dictionary.')
    result = {}
    for class_id, object_name in classid2obj.items():
        preprocessed = root / 'object_keypoints' / f'{object_name}.npz'
        if bundled is None:
            with np.load(preprocessed, allow_pickle=False) as data:
                entry = dict(data)
            result[class_id] = _validate(entry, object_name, preprocessed)
            continue
        if object_name not in bundled:
            raise KeyError(f'{reference}: missing requested object {object_name}.')
        entry = _validate(bundled[object_name], object_name, reference)
        if preprocessed.is_file():
            try:
                with np.load(preprocessed, allow_pickle=False) as data:
                    points = data['cartesian']
                matches = (points.shape == entry['cartesian'].shape
                           and np.allclose(points, entry['cartesian'], atol=1e-6, rtol=0))
            except (OSError, ValueError, KeyError):
                matches = False
            if not matches:
                logger.warning(
                    '%s/%s: preprocessed object keypoints differ from the bundled model reference; '
                    'using %s without modifying %s.', name, object_name, reference, preprocessed
                )
        result[class_id] = entry
    return result
