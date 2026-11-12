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
        self.assertGreater((kinds > 0).mean(), 0.9)     # ... to the east mouth, 1.5 m under the square
        np.testing.assert_allclose(floor[[0, -1]], g[[0, -1]])          # the mouths where the ground has them
        self.assertTrue((floor <= hull + 1e-9).all())
        mid = slice(len(s) // 3, 2 * len(s) // 3)
        self.assertTrue((g[mid] - floor[mid] >= 5.9).all(), (g - floor)[mid].min())
        self.assertLessEqual(np.abs(np.diff(floor) / np.diff(s)).max(), roads.FINE_FLOOR_GRADE + 1e-6)

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


if __name__ == "__main__":
    unittest.main()
