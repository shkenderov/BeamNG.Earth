"""A fine terrain model's street tunnel is solved by the grading at the samples the meshes' profile solves it at
(grading.tunnel_samples): its open section, drawn on the ground the grading gives it, then starts where the tube
it leaves ends."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grading  # noqa: E402
import roads  # noqa: E402


class SamplesTest(unittest.TestCase):
    def line(self):
        return np.c_[[0.0, 50.0, 78.3, 86.0], np.zeros(4)]      # (a vertex at 78.3: not on any 4 m grid)

    def test_a_fine_model_samples_a_tunnel_where_the_profile_does(self):
        xy = self.line()
        want = roads.densify(roads._dedupe(xy), roads.DECK_STEP)
        got, s = grading.tunnel_samples(xy, True)
        np.testing.assert_allclose(got, want)
        self.assertAlmostEqual(float(s[-1]), 86.0)
        self.assertTrue(np.isclose(got[:, 0], 78.3).any(), "the map's own vertices are samples")
        self.assertLessEqual(float(np.diff(s).max()), roads.DECK_STEP + 1e-9)

    def test_elsewhere_as_before(self):
        xy = self.line()
        got, s = grading.tunnel_samples(xy, False)
        want, ws = grading.resample(xy, grading.VERTEX_STEP)
        np.testing.assert_allclose(got, want)
        np.testing.assert_allclose(s, ws)
        self.assertFalse(np.isclose(got[:, 0], 78.3).any())

    def test_a_pit_at_a_mouth_is_seen_at_the_profile_samples_only(self):
        # the model's pit a metre wide at the tube's end (550.4 in ground of 553.1): in the profile's samples it is
        # the floor the tube ends at, between the grading's it was not there and the grading solved another end
        def ground(x):
            return np.where(np.abs(x - 78.3) < 0.6, 549.9, 553.1)
        fine, _ = grading.tunnel_samples(self.line(), True)
        coarse, _ = grading.tunnel_samples(self.line(), False)
        self.assertAlmostEqual(float(ground(fine[:, 0]).min()), 549.9)
        self.assertAlmostEqual(float(ground(coarse[:, 0]).min()), 553.1)


if __name__ == "__main__":
    unittest.main()
