"""A street tunnel on a fine terrain model (roads.FINE_*): covered where the ground rises over the line between its
mouths, its floor sunk under the street; a flat 'tunnel' (a road under a bridge) and every tunnel elsewhere as before."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import roads  # noqa: E402


class FineTunnelTest(unittest.TestCase):
    def ground(self, n=98, up=3.0, east=1.5):
        """A 388 m tunnel under a square: its west mouth `up` m under the square, its east one `east` m."""
        s = np.arange(n) * 4.0
        g = np.full(n, 557.0)
        g[0], g[-1] = 557.0 - up, 557.0 - east
        return s, g

    def test_as_before_without_sink(self):
        s, g = self.ground()
        floor, kinds = roads.tunnel_sections(s, g)
        self.assertFalse(kinds.any())                   # 3 m of cover: an open road
        np.testing.assert_allclose(floor[[0, -1]], g[[0, -1]])

    def test_covered_and_sunk_under_the_street(self):
        s, g = self.ground()
        hull, _ = roads.tunnel_sections(s, g)
        floor, kinds = roads.tunnel_sections(s, g, sink=6.0)
        self.assertGreater((kinds > 0).mean(), 0.8)
        np.testing.assert_allclose(floor[[0, -1]], g[[0, -1]])          # the mapped ends where the ground has them
        self.assertTrue((floor <= hull + 1e-9).all())
        mid = slice(len(s) // 3, 2 * len(s) // 3)
        self.assertTrue((g[mid] - floor[mid] >= 5.9).all(), (g - floor)[mid].min())
        self.assertLessEqual(np.abs(np.diff(floor) / np.diff(s)).max(), roads.FINE_RAMP_GRADE + 1e-6)

    def test_the_tube_never_stands_out_of_the_ground(self):
        # 2026-11-12, the NDK's park: mounds up to 6 m over the tube's first and last 80 m. Covered only where the
        # 5 m tube and its soil fit under the ground; before and after it an open ramp down from the mapped end
        s, g = self.ground(up=3.0, east=0.0)            # (its east end on the square: no ramp in the model there)
        floor, kinds = roads.tunnel_sections(s, g, sink=6.0)
        cov = kinds > 0
        need = 6.0 - roads.TUNNEL_SOIL_M + roads.ARCH_SOIL_MIN
        self.assertTrue((g[cov] - floor[cov] >= need - 0.06).all(), (g - floor)[cov].min())
        self.assertFalse(kinds[0] or kinds[-1], "open ramps at both ends")
        first, last = np.flatnonzero(cov)[[0, -1]]
        self.assertLess(s[first], 40.0)                 # 2.6 m more at 10 %
        self.assertLess(s[-1] - s[last], 70.0)          # 5.6 m at 10 %
        self.assertTrue(cov[first:last + 1].all())

    def test_a_pool_over_the_tunnel_opens_no_gap_in_it(self):
        # the NDK's fountains: pools carved 2 m into the square over the tunnel (2026-11-12: an open road in a trench
        # under them, in the middle of the tunnel)
        s, g = self.ground()
        hull, _ = roads.tunnel_sections(s, g)
        g[58:66] -= 2.5
        floor, kinds = roads.tunnel_sections(s, g, sink=6.0)
        cov = np.flatnonzero(kinds > 0)
        self.assertTrue((kinds[cov[0]:cov[-1] + 1] > 0).all(), kinds)
        self.assertTrue((floor[55:70] < hull[55:70] - 3.0).all(), "the floor under the pools as deep as beside them")

    def test_a_short_underpass_is_covered_as_before(self):
        s, g = self.ground(n=20)                        # 76 m: no room for the ramps down to a tube under the street
        self.assertIsNone(roads._fine_under_street(s, g, 6.0))
        floor, kinds = roads.tunnel_sections(s, g, sink=6.0)
        self.assertGreater((kinds > 0).mean(), 0.8)
        g[8:11] -= 2.5                                  # ... and a pool over it opens no gap in it either
        floor, kinds = roads.tunnel_sections(s, g, sink=6.0)
        cov = np.flatnonzero(kinds > 0)
        self.assertTrue(len(cov) and (kinds[cov[0]:cov[-1] + 1] > 0).all(), kinds)

    def test_flat_road_under_a_bridge_stays_open(self):
        s, g = self.ground(n=30, up=0.6, east=0.4)
        floor, kinds = roads.tunnel_sections(s, g, sink=6.0)
        self.assertFalse(kinds.any())
        np.testing.assert_allclose(floor, roads.tunnel_sections(s, g)[0])

    def test_only_streets_on_the_model(self):
        class Spec:
            cls = "service"
        self.assertIsNone(roads.fine_tunnel_sink(Spec(), np.array([42.7]), np.array([23.3])))
        self.assertIsNone(roads.fine_tunnel_sink(None, np.array([42.7]), np.array([23.3])))


class FineGroupTest(unittest.TestCase):
    """The mouths at the mapped ends, at the depth the tube needs: the roads to them sunk (roads.fine_tunnel_group,
    grading._sink_fine_approaches). 2026-11-12, the tunnel under the NDK's park."""

    def tunnel(self, x0=0.0, n=98, up=3.0, east=3.0):
        s = np.arange(n) * 4.0
        g = np.full(n, 557.0)
        g[0], g[-1] = 557.0 - up, 557.0 - east
        return np.c_[x0 + s, np.zeros(n)], s, g

    def test_without_a_fine_tunnel_as_before(self):
        xy, s, g = self.tunnel()
        (floor, kinds, used), = roads.fine_tunnel_group([(xy, s, g, g[[0, -1]], None)])[0]
        f0, k0 = roads.tunnel_sections(s, g)
        np.testing.assert_allclose(floor, f0)
        np.testing.assert_array_equal(kinds, k0)
        self.assertEqual(roads.fine_tunnel_group([(xy, s, g, g[[0, -1]], None)])[1], [])

    def test_the_mouths_wanted_at_depth_and_covered_from_them_once_the_roads_are_sunk(self):
        xy, s, g = self.tunnel()
        res, nodes = roads.fine_tunnel_group([(xy, s, g, g[[0, -1]], 6.0)])
        self.assertEqual(len(nodes), 2)
        for x, y, z in nodes:
            self.assertAlmostEqual(z, 551.0, places=1)      # the street's 557 less the tube and its soil
        floor, kinds, _ = res[0]
        self.assertFalse(kinds[0] or kinds[-1], "the roads not sunk yet: the ramps inside, as before")
        (floor, kinds, _), = roads.fine_tunnel_group([(xy, s, g, np.array([551.0, 551.0]), 6.0)])[0]
        self.assertTrue((kinds > 0).all(), kinds)           # the roads' plans sunk to the nodes: its ends' ground
        self.assertLess(np.abs(floor - 551.0).max(), 0.1)

    def test_open_parts_joined_to_it_fall_to_its_mouth(self):
        # the NDK tunnel's east end: its carriageway, then a 24 m 'tunnel' in the map that is an open road, then 60 m more
        xy, s, g = self.tunnel(east=0.0)
        s1 = np.arange(7) * 4.0
        s2 = np.arange(16) * 4.0
        link = (np.c_[xy[-1, 0] + s1, np.zeros(7)], s1, np.full(7, 557.0))
        road = (np.c_[link[0][-1, 0] + s2, np.zeros(16)], s2, np.full(16, 557.0))
        items = [(xy, s, g, g[[0, -1]], 6.0)] + [(a, b, c, c[[0, -1]], 6.0) for a, b, c in (link, road)]
        (f0, k0, _), (f1, k1, _), (f2, k2, _) = roads.fine_tunnel_group(items)[0]
        self.assertTrue(k0[-1] and k0[-20:].all(), "covered to its mapped east end")
        self.assertAlmostEqual(float(f0[-1]), 551.0, places=1)
        self.assertFalse(k1.any() or k2.any())
        self.assertAlmostEqual(float(f1[0]), 551.0, places=1)
        self.assertAlmostEqual(float(f1[-1]), 551.0 + 2.4, places=1)       # 24 m at 10 %
        self.assertAlmostEqual(float(f2[0]), float(f1[-1]), places=6)      # ... and the next goes on climbing
        self.assertAlmostEqual(float(f2[-1]), 557.0)
        self.assertLessEqual(np.abs(np.diff(f2) / 4.0).max(), roads.FINE_RAMP_GRADE + 1e-6)

    def test_a_tunnel_forking_off_an_open_road_is_open_until_clear_of_it(self):
        # the NDK tunnel's east node: its carriageway ends, an open road goes on, and another street's tunnel forks off
        # at 30 degrees: covered from the node, the ground kept over it stood across the open road
        xy, s, g = self.tunnel(east=0.0)
        s1 = np.arange(11) * 4.0
        link = (np.c_[xy[-1, 0] + s1, np.zeros(11)], s1, np.full(11, 557.0))
        n = 60
        sf = np.arange(n) * 4.0
        gf = np.full(n, 557.0)
        gf[-1] = 554.0
        fork = (np.c_[xy[-1, 0] + sf * np.cos(np.radians(30)), -sf * np.sin(np.radians(30))], sf, gf)
        items = [(xy, s, g, g[[0, -1]], 6.0), (link[0], link[1], link[2], link[2][[0, -1]], 6.0),
                 (fork[0], fork[1], fork[2], fork[2][[0, -1]], 6.0)]
        (f0, k0, _), (f1, k1, _), (f2, k2, _) = roads.fine_tunnel_group(items)[0]
        self.assertTrue(k0[-20:].all(), "the tunnel the open road goes on from: covered to its end")
        self.assertFalse(k1.any())
        first = int(np.flatnonzero(k2)[0])
        self.assertTrue(16.0 <= sf[first] <= 28.0, sf[first])       # 10 m clear at 30 degrees: 20 m along
        self.assertLess(float(f2[first]), 551.6)                    # ... at depth all the way (an open cut)

    def test_the_roads_and_tracks_at_a_mouth_are_sunk_to_it(self):
        import grading

        class P(grading._Planner):
            def __init__(self):
                pass

            def _tunnel_items(self, feats, ts, key=None, road_plans=(), portals=None):
                return [0], [(0, xy, s, g, g[[0, -1]], 6.0)]
        xy, s, g = self.tunnel()
        d = np.arange(0.0, 101.0, 4.0)
        road = grading.Plan(np.c_[-d[::-1], np.zeros(len(d))], np.r_[np.full(len(d) - 1, 557.0), 554.0], 4.0)
        side = grading.Plan(np.c_[np.full(len(d), -100.0), d], np.full(len(d), 557.0), 3.0)     # joined 100 m out
        away = grading.Plan(np.c_[-d[::-1], np.full(len(d), 40.0)], np.full(len(d), 557.0), 4.0)
        rail = grading.Plan(np.c_[-d[::-1], np.full(len(d), 3.0)], np.full(len(d), 558.0), 3.0)
        plans = [("r", road), ("s", side), ("a", away), (grading.RAIL_TAG + "t", rail)]
        orig = roads.tunnel_specs
        roads.tunnel_specs = lambda feats: [object()]
        try:
            nodes, lines = P()._sink_fine_approaches(None, None, plans)
        finally:
            roads.tunnel_specs = orig
        self.assertAlmostEqual(float(road.z[-1]), 551.0, places=1)         # at the mouth: the tube's depth
        self.assertAlmostEqual(float(road.z[-6]), 551.0 + 2.0, places=1)   # 20 m out: 10 % up
        self.assertAlmostEqual(float(road.z[0]), 557.0)                    # 100 m out: its own level
        self.assertTrue((side.z == 557.0).all() and (away.z == 557.0).all())
        P._sink_fine_rails(nodes, lines, plans)
        self.assertAlmostEqual(float(rail.z[-1]), 551.0, places=1)         # the track ending there: down with the road
        self.assertTrue((rail.z <= road.z + 1e-9).all(), "no bank beside the sunk road")
        self.assertEqual(float(rail.z[0]), 557.0)


if __name__ == "__main__":
    unittest.main()
