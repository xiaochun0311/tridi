"""Canonical keypoint loading must not silently use cropped regenerated points."""
import hashlib
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tridi.data.hoi_dataset import HOIDataset
from tridi.data.random_dataset import RandomDataset
from tridi.data.object_keypoints import load_object_keypoints


class ObjectKeypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'data'
        self.assets = Path(self.temp.name) / 'assets'
        (self.root / 'object_keypoints').mkdir(parents=True)
        (self.assets / 'object_keypoints').mkdir(parents=True)
        self.expected = np.array([[-.1, -.02, -.05], [.1, .02, .05], [0, .01, -.03]], dtype=np.float32)
        self.reference = self.assets / 'object_keypoints/grab.pkl'
        self.npz = self.root / 'object_keypoints/flute.npz'
        self.save_reference({'flute': {'cartesian': self.expected}})
        np.savez(self.npz, cartesian=np.tile([.099, 0, -.049], (3, 1)))

    def save_reference(self, value):
        with self.reference.open('wb') as fp:
            pickle.dump(value, fp)

    def load(self, name='grab'):
        return load_object_keypoints(name, self.root, {7: 'flute'}, self.assets)

    def test_bundled_reference_replaces_caps_without_mutating_files(self):
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (self.reference, self.npz)}
        with self.assertLogs('tridi.data.object_keypoints', level='WARNING') as captured:
            result = self.load()
        np.testing.assert_array_equal(result[7]['cartesian'], self.expected)
        self.assertEqual(result[7]['cartesian'].dtype, np.float32)
        self.assertIn('preprocessed object keypoints differ', captured.output[0])
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before})

    def test_both_dataset_loaders_share_reference_resolution(self):
        for dataset_type in (HOIDataset, RandomDataset):
            with self.subTest(dataset=dataset_type.__name__):
                dataset = dataset_type.__new__(dataset_type)
                dataset.name, dataset.root = 'grab', self.root
                dataset.classid2obj, dataset.assets_folder = {7: 'flute'}, self.assets
                with self.assertLogs('tridi.data.object_keypoints', level='WARNING'):
                    points = dataset._load_canonical_keypoints()[7]['cartesian']
                np.testing.assert_array_equal(points, self.expected)

    def test_custom_dataset_falls_back_only_when_reference_file_absent(self):
        expected = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float64)
        np.savez(self.npz, cartesian=expected, normals=np.ones_like(expected))
        entry = self.load('custom')[7]
        np.testing.assert_array_equal(entry['cartesian'], expected)
        np.testing.assert_array_equal(entry['normals'], np.ones_like(expected))

    def test_reference_does_not_require_regenerated_npz(self):
        self.npz.unlink()
        np.testing.assert_array_equal(self.load()[7]['cartesian'], self.expected)

    def test_present_reference_missing_object_fails_instead_of_mixing_sources(self):
        self.save_reference({'another_object': {'cartesian': self.expected}})
        with self.assertRaisesRegex(KeyError, 'missing requested object flute'):
            self.load()

    def test_invalid_reference_coordinates_fail_clearly(self):
        for points in (np.zeros((3, 2)), np.empty((0, 3)), np.full((3, 3), np.nan),
                       np.full((3, 3), np.inf), np.array([['x', 'y', 'z']])):
            with self.subTest(shape=points.shape, dtype=points.dtype):
                self.save_reference({'flute': {'cartesian': points}})
                with self.assertRaisesRegex(ValueError, 'finite nonempty Nx3'):
                    self.load()

    def test_invalid_fallback_coordinates_fail_clearly(self):
        np.savez(self.npz, cartesian=np.zeros((3, 2)))
        with self.assertRaisesRegex(ValueError, 'finite nonempty Nx3'):
            self.load('custom')

    def test_malformed_reference_entries_fail_clearly(self):
        for value in ([], {'flute': {}}, {'flute': self.expected}):
            with self.subTest(value_type=type(value).__name__):
                self.save_reference(value)
                with self.assertRaises(ValueError):
                    self.load()

    def test_matching_reference_and_npz_do_not_warn(self):
        np.savez(self.npz, cartesian=self.expected.astype(np.float64))
        with self.assertNoLogs('tridi.data.object_keypoints', level='WARNING'):
            result = self.load()
        np.testing.assert_array_equal(result[7]['cartesian'], self.expected)

    def test_different_point_count_warns_and_uses_reference(self):
        np.savez(self.npz, cartesian=self.expected[:1])
        with self.assertLogs('tridi.data.object_keypoints', level='WARNING'):
            result = self.load()
        self.assertEqual(result[7]['cartesian'].shape, (3, 3))


if __name__ == '__main__':
    unittest.main()
