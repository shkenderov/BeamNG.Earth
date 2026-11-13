"""A tunnel's open section on a fine terrain model is drawn on the ground the grading gives it, so that ground must be
graded to the section's floor, where the tube it leaves ends, and the tube's and the grading's sections must be the
same (grading._Planner._tunnel_items samples the tunnels where roads.TunnelProfiles does; _on_open_sections)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grading  # noqa: E402
import roads  # noqa: E402


class OpenSectionTest(unittest.TestCase):
    class P(grading._Planner):
        def __init__(self, opens):
            import threading
            self.lock = threading.RLock()
            self.opens = {"cell": opens}
            self.covers = {}

    def spec(self, width=6.0):
        return type("S", (), {"width": width})()

    def test_the_ends_and_vertices_of_an_open_section_are_on_it(self):
        # (over_tunnel's projection is strictly inside each segment: a section's vertices, which are where its
        # road's height is read, were never on it)
        xy = np.array([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0], [8.0, 4.0]])
        p = self.P([(xy, self.spec())])
        pts = np.array([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0], [8.0, 4.0], [6.0, 2.9], [4.5, 3.5], [-0.1, 0.0], [-3.1, 0.0]])
        got = p._on_open_sections(pts[:, 0], pts[:, 1])
        self.assertEqual(got.tolist(), [True, True, True, True, True, False, True, False])

    def test_no_open_sections_no_points(self):
        p = self.P([])
        self.assertFalse(p._on_open_sections(np.array([0.0, 1.0]), np.array([0.0, 1.0])).any())
        p = self.P([(np.array([[0.0, 0.0], [4.0, 0.0]]), self.spec())])
        self.assertEqual(len(p._on_open_sections(np.zeros(0), np.zeros(0))), 0)

    def test_an_open_section_beside_a_covered_tunnel_is_graded_to_its_floor(self):
        # a service ramp running 4 m beside an underpass: inside the underpass's keep zone (its half width and
        # TUNNEL_KEEP_M), the ground there was left as the terrain model has it, 4 m over the ramp's floor
        ramp = np.c_[np.arange(5.0, 36.0, 4.0), np.full(8, 4.0)]      # (between the tube's vertices)
        tube = np.c_[np.arange(0.0, 41.0, 4.0), np.zeros(11)]
        p = self.P([(ramp, self.spec())])
        p.covers = {"cell": [(tube, 3.0 + grading.TUNNEL_KEEP_M)]}
        e = np.r_[ramp[:, 0], 21.0]
        n = np.r_[ramp[:, 1], 0.0]
        keep = p._over_tunnels(e, n)
        self.assertTrue(keep.all(), "the ramp and a point over the tube are in the keep zone")
        keep &= ~p._on_open_sections(e, n)
        self.assertFalse(keep[:-1].any(), "the ramp's own points are graded")
        self.assertTrue(keep[-1], "ground over the tube stays")


class ApplyTest(unittest.TestCase):
    """_Planner.apply over a ramp (an open section) running 4 m beside a covered tunnel, in Sofia's fine model or not."""

    def planner(self, ramp, tube):
        import threading
        import geodesy
        frame = geodesy.EnuFrame(42.7, 23.3)
        plan = grading.Plan(ramp, np.full(len(ramp), 560.0), 4.0)
        plan.rank = 100

        class P(grading._Planner):
            def __init__(self):
                self.frame, self.lock = frame, threading.RLock()
                self.covers = {"c": [(tube, 3.0 + grading.TUNNEL_KEEP_M)]}
                self.opens = {"c": [(ramp, type("S", (), {"width": 6.0})())]}
                self.keeps = {}

            def _plan_set(self, keys, roads_only):
                return [("ramp", plan)], grading.PlanPack([plan])

            def _on_kept(self, e, n):
                return None
        return P(), frame

    def grade(self, fine):
        ramp = np.c_[np.arange(5.0, 36.0, 4.0), np.full(8, 4.0)]
        tube = np.c_[np.arange(0.0, 41.0, 4.0), np.zeros(11)]
        p, frame = self.planner(ramp, tube)
        x = np.arange(8.0, 33.0, 2.0)
        lat, lon, _ = frame.enu_to_geodetic(x, np.full(len(x), 4.0), 0.0)
        orig = roads.fine_model_on
        roads.fine_model_on = lambda: fine
        try:
            return p.apply(np.asarray(lat), np.asarray(lon), np.full(len(x), 565.0))
        finally:
            roads.fine_model_on = orig

    def test_a_fine_model_grades_the_open_section_inside_the_keep_zone(self):
        got = self.grade(True)
        self.assertTrue((np.abs(got - 560.0) < 0.3).all(), got)

    def test_elsewhere_the_ground_there_is_kept_as_before(self):
        got = self.grade(False)
        self.assertTrue((got == 565.0).all(), got)


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
