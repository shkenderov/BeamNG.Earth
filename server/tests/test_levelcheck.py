"""tools/levelcheck.py's checks on synthetic road parts (no pack, no server): a road 3 m over flat ground is exactly one
finding; roads on the ground, tunnels with cover, flush joints and gentle grades are none; each of the six checks fires
once on its own fault; the report is the same on two runs."""
import csv
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import levelcheck as lc  # noqa: E402

M_LAT = 111320.0


def piece(kind, x0, y0, x1, y1, z0, z1=None, terrain=None, step=2.0, **kw):
    """A straight part from (x0, y0) to (x1, y1) metres, its surface z0 -> z1 (constant by default), over a terrain."""
    n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step) + 1)
    t = np.linspace(0.0, 1.0, n)
    xy = np.c_[x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]
    z = z0 + ((z0 if z1 is None else z1) - z0) * t
    kw.setdefault("way", "1")
    kw.setdefault("fid", f"{kind}-{x0:g}-{y0:g}")
    return lc.Piece(kind, xy=xy, lat=42.68 + xy[:, 1] / M_LAT, lon=23.31 + xy[:, 0] / 82000.0, z=z,
                    terrain=None if terrain is None else np.full(n, float(terrain)), **kw)


def checks(findings):
    return sorted(f["check"] for f in findings)


class LevelCheck(unittest.TestCase):
    def test_road_on_ground_is_clean(self):
        self.assertEqual(lc.run_checks([piece("ground", 0, 0, 200, 0, 100.04, terrain=100.0)]), [])

    def test_road_3m_over_flat_ground_is_one_finding(self):
        f = lc.run_checks([piece("ground", 0, 0, 200, 0, 103.0, terrain=100.0)])
        self.assertEqual(checks(f), ["ground"])
        self.assertAlmostEqual(f[0]["value"], 3.0)
        self.assertEqual(f[0]["limit"], lc.GROUND_LIMIT_M)

    def test_road_3m_under_ground_is_one_finding(self):
        f = lc.run_checks([piece("ground", 0, 0, 200, 0, 97.0, terrain=100.0)])
        self.assertEqual(checks(f), ["ground"])
        self.assertAlmostEqual(f[0]["value"], -3.0)

    def test_two_separate_bad_stretches_are_two_findings(self):
        p = piece("ground", 0, 0, 200, 0, 100.0, terrain=100.0)
        p.z[5:8] += 2.0
        p.z[60:63] -= 2.0
        self.assertEqual(checks([f for f in lc.run_checks([p]) if f["check"] == "ground"]), ["ground", "ground"])

    def test_within_the_limit_is_clean(self):
        self.assertEqual(lc.run_checks([piece("ground", 0, 0, 100, 0, 100.9, terrain=100.0)]), [])

    def test_tunnel_with_cover_is_clean(self):
        t = piece("tunnel", 0, 0, 100, 0, 100.0, terrain=108.0, th=5.0, mouth=(True, True))
        self.assertEqual(lc.run_checks([t]), [])

    def test_tunnel_thin_cover_is_one_finding_and_portals_are_exempt(self):
        t = piece("tunnel", 0, 0, 100, 0, 100.0, terrain=108.0, th=5.0, mouth=(True, True))
        t.terrain[:3] = 100.0                       # (the ground falls to the floor at the portals: the first 6 m)
        t.terrain[-3:] = 100.0
        self.assertEqual(lc.run_checks([t]), [])
        t.terrain[20:25] = 105.2                    # (0.2 m over the vault in the middle)
        f = lc.run_checks([t])
        self.assertEqual(checks(f), ["cover"])
        self.assertAlmostEqual(f[0]["value"], 0.2)

    def test_tunnel_road_above_terrain_is_flagged(self):
        t = piece("tunnel", 0, 0, 100, 0, 100.0, terrain=108.0, th=5.0, mouth=(True, True))
        t.terrain[40:44] = 99.0
        f = lc.run_checks([t])
        self.assertEqual(checks(f), ["cover"])
        self.assertEqual(f[0]["note"], "road above terrain")

    def test_grade(self):
        steep = piece("ground", 0, 0, 60, 0, 100.0, 109.0, terrain=None)         # 15 %
        gentle = piece("ground", 0, 100, 60, 100, 100.0, 103.0, terrain=None)     # 5 %
        f = lc.run_checks([steep, gentle])
        self.assertEqual(checks(f), ["grade"])
        self.assertAlmostEqual(f[0]["value"], 0.15, places=2)
        dirt = piece("ground", 0, 0, 60, 0, 100.0, 109.0, paved=False)
        self.assertEqual(lc.run_checks([dirt]), [])

    def test_seam_between_parts_of_one_way(self):
        a = piece("ground", 0, 0, 50, 0, 100.0, fid="a")
        b = piece("ground", 50, 0, 100, 0, 100.3, fid="b")
        f = lc.run_checks([a, b])
        self.assertEqual(checks(f), ["seam"])
        self.assertAlmostEqual(f[0]["value"], 0.3)
        b2 = piece("ground", 50, 0, 100, 0, 100.05, fid="b")
        self.assertEqual(lc.run_checks([a, b2]), [])
        other = piece("ground", 50, 0, 100, 0, 100.3, fid="b", way="2")          # (another way: a junction, not a seam)
        self.assertEqual(lc.run_checks([a, other]), [])

    def test_portal_step_between_a_bridge_and_its_approach(self):
        road = piece("ground", 0, 0, 50, 0, 100.0, fid="road")
        deck = piece("bridge", 50, 0, 90, 0, 100.5, fid="deck", depth=1.0)
        f = lc.run_checks([road, deck])
        self.assertEqual(checks(f), ["portal"])
        self.assertAlmostEqual(f[0]["value"], 0.5)
        flush = piece("bridge", 50, 0, 90, 0, 100.1, fid="deck", depth=1.0)
        self.assertEqual(lc.run_checks([road, flush]), [])

    def test_portal_step_between_a_tunnel_and_its_open_approach(self):
        road = piece("open", 0, 0, 50, 0, 100.0, fid="road")
        tun = piece("tunnel", 50, 0, 150, 0, 99.0, terrain=110.0, th=5.0, fid="tun", mouth=(True, True))
        self.assertEqual(checks(lc.run_checks([road, tun])), ["portal"])

    def test_underpass_clearance(self):
        road = piece("ground", -50, 0, 50, 0, 100.0, way="1", fid="road")
        low = piece("bridge", 0, -40, 0, 40, 104.5, way="2", fid="deck", depth=1.0)     # underside 3.5 m over the road
        ok = piece("bridge", 0, -40, 0, 40, 106.0, way="2", fid="deck", depth=1.0)      # 5 m
        f = lc.run_checks([road, low])
        self.assertEqual(checks(f), ["clearance"])
        self.assertAlmostEqual(f[0]["value"], 3.5)
        self.assertEqual(lc.run_checks([road, ok]), [])

    def test_a_road_meeting_a_deck_end_is_no_underpass(self):
        road = piece("ground", -50, 0, 50, 0, 100.0, way="1", fid="road")
        deck = piece("bridge", 0, 0, 0, 80, 100.2, way="2", fid="deck", depth=1.0)      # starts on the road: a junction
        self.assertEqual(checks([f for f in lc.run_checks([road, deck]) if f["check"] == "clearance"]), [])

    def test_the_report_is_the_same_on_two_runs_and_sorted_worst_first(self):
        ps = [piece("ground", 0, 0, 200, 0, 103.0, terrain=100.0, way="1", fid="a"),
              piece("ground", 0, 50, 200, 50, 105.0, terrain=100.0, way="2", fid="b")]
        info = {"bbox": [0, 0, 1, 1]}
        outs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as d:
                f = lc.run_checks(ps)
                t = lc.write_report(d, f, info)
                raw = []
                for name in ("levelcheck.json", "levelcheck.csv"):
                    with open(os.path.join(d, name), "rb") as fh:
                        raw.append(fh.read())
                outs.append(tuple(raw))
                with open(os.path.join(d, "levelcheck.csv"), encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
        self.assertEqual(outs[0], outs[1])
        self.assertEqual([r["way"] for r in rows], ["2", "1"])
        self.assertEqual(t["ground"], 2)
        self.assertEqual(json.loads(outs[0][0])["totals"]["ground"], 2)
        self.assertIn("ground 2", lc.total_line(t))


if __name__ == "__main__":
    unittest.main()
