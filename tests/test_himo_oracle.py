import unittest

import numpy as np

from tridi.model.ordered_scene.oracle import contact_agreement, maximum_error


class OracleChecks(unittest.TestCase):
    def test_vertex_roundtrip_detects_shift_and_layout_changes(self):
        vertices = np.arange(12, dtype=float).reshape(4, 3) / 10
        self.assertEqual(maximum_error(vertices, vertices), 0.)
        self.assertAlmostEqual(maximum_error(vertices, vertices + .8), .8)
        with self.assertRaises(ValueError):
            maximum_error(vertices, vertices.T)

    def test_empty_maps_match_and_boundary_disagreements_remain_visible(self):
        empty = contact_agreement(np.zeros(3), np.zeros(3))
        self.assertTrue(empty["strict_match"])
        self.assertEqual(empty["mismatch_points"], 0)
        difference = contact_agreement([1, 0, 0], [0, 1, 0], [.050001, .1, np.inf])
        self.assertEqual(difference["mismatch_points"], 2)
        self.assertEqual(difference["threshold_boundary_points"], 1)
        self.assertEqual(difference["unexplained_points"], 1)
        self.assertFalse(difference["strict_match"])


if __name__ == "__main__":
    unittest.main()
