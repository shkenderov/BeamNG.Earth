"""tools/levelcheck.py's checks on synthetic road parts (no pack, no server): a road 3 m over flat ground is exactly one
finding; roads on the ground, tunnels with cover, flush joints and gentle grades are none; each of the six checks fires
once on its own fault; the report is the same on two runs. Classes: a ground offset a structure explains (a ramp that
tapers to the terrain from a bridge's end, a tunnel's mouth, under a deck) is an `earthwork`; the same offset with no
structure, or one that does not taper, is a `defect`; a steep road on a steep hillside is `terrain`, on flat ground a
`defect`."""
import csv
import json
import math
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


def ramp(kind, x0, x1, z_at, terrain=100.0, step=2.0, **kw):
    """A straight part along x from x0 to x1 (y = 0), its surface z_at(x) over a flat terrain."""
    n = int(abs(x1 - x0) / step) + 1
    x = np.linspace(x0, x1, n)
    z = np.array([z_at(v) for v in x], np.float64)
    kw.setdefault("way", "1")
    kw.setdefault("fid", f"{kind}-{x0:g}-{x1:g}")
    xy = np.c_[x, np.zeros(n)]
    return lc.Piece(kind, xy=xy, lat=42.68 + xy[:, 1] / M_LAT, lon=23.31 + xy[:, 0] / 82000.0, z=z,
                    terrain=np.full(n, float(terrain)), **kw)


def of(findings, check, cls=None):
    return [f for f in findings if f["check"] == check and (cls is None or f["class"] == cls)]


def embankment(height=6.0, run=150.0, tail=100.0):
    """A bridge ending at x = 0 and the road from it: `height` m over the flat terrain, falling to it over `run` m."""
    deck = ramp("bridge", -60, 0, lambda x: 100.0 + height, way="2", fid="deck", depth=1.0)
    road = ramp("ground", 0, run + tail, lambda x: 100.0 + max(0.0, height * (1.0 - x / run)), way="1", fid="road")
    return deck, road


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


class Classes(unittest.TestCase):
    """What a structure explains is set apart from the defects; everything else stays a defect."""

    def test_embankment_to_a_bridge_end_is_an_earthwork(self):
        f = lc.run_checks(list(embankment()))
        g = of(f, "ground")
        self.assertEqual([x["class"] for x in g], ["earthwork"])
        self.assertEqual(g[0]["earth_by"], "bridge deck end")
        self.assertAlmostEqual(g[0]["taper_m"], 125.0, delta=2.0)        # (within GROUND_LIMIT_M of it from here)
        self.assertEqual((g[0]["near_kind"], g[0]["near_id"], g[0]["near_basis"], g[0]["near_role"]),
                         ("bridge", "deck", "network", "end"))
        self.assertLess(g[0]["near_m"], 2.0)
        self.assertEqual(lc.totals(f)["ground"], 0)
        self.assertEqual(lc.set_apart(f), {"ground": 1, "grade": 0})
        self.assertIn("ground 0 (+1 earthwork)", lc.total_line(lc.totals(f), lc.set_apart(f)))

    def test_the_same_offset_with_no_structure_near_is_a_defect(self):
        _deck, road = embankment()
        g = of(lc.run_checks([road]), "ground")
        self.assertEqual([x["class"] for x in g], ["defect"])
        self.assertIsNone(g[0]["near_kind"])
        self.assertEqual(lc.totals(lc.run_checks([road]))["ground"], 1)

    def test_the_same_offset_with_a_far_structure_is_a_defect(self):
        deck = ramp("bridge", 1000, 1060, lambda x: 106.0, way="2", fid="deck", depth=1.0)
        _d, road = embankment()
        g = of(lc.run_checks([deck, road]), "ground")
        self.assertEqual([x["class"] for x in g], ["defect"])
        self.assertEqual(g[0]["near_basis"], "straight")
        self.assertAlmostEqual(g[0]["near_m"], 1000.0, delta=6.0)

    def test_an_offset_that_does_not_taper_to_the_ground_is_a_defect(self):
        deck = ramp("bridge", -60, 0, lambda x: 106.0, way="2", fid="deck", depth=1.0)
        road = ramp("ground", 0, 600, lambda x: 106.0 if x < 450 else 106.0 - (x - 450) * 0.04, way="1", fid="road")
        self.assertEqual([x["class"] for x in of(lc.run_checks([deck, road]), "ground")], ["defect"])      # (reach)

    def test_an_offset_that_grows_away_from_the_structure_is_a_defect(self):
        deck = ramp("bridge", -60, 0, lambda x: 103.0, way="2", fid="deck", depth=1.0)

        def z(x):                                   # down to 1.2 m, up to 5.2 m again, then down
            return 103.0 - 0.03 * x if x < 60 else 101.2 + 0.05 * (x - 60) if x < 140 else 105.2 - 0.05 * (x - 140)
        road = ramp("ground", 0, 250, z, way="1", fid="road")
        self.assertEqual([x["class"] for x in of(lc.run_checks([deck, road]), "ground")], ["defect"])

    def test_a_road_that_does_not_meet_the_deck_is_a_defect(self):
        deck = ramp("bridge", -60, 0, lambda x: 112.0, way="2", fid="deck", depth=1.0)       # (6 m over the road's end)
        _d, road = embankment()
        f = lc.run_checks([deck, road])
        self.assertEqual([x["class"] for x in of(f, "ground")], ["defect"])
        self.assertEqual(len(of(f, "portal")), 1)                  # (the step is the portal's finding)

    def test_a_step_at_a_bridge_end_stays_a_defect(self):
        deck = ramp("bridge", -60, 0, lambda x: 108.0, way="2", fid="deck", depth=1.0)
        road = ramp("ground", 0, 200, lambda x: 100.0, way="1", fid="road")                  # (flat at the ground: no ramp)
        f = lc.run_checks([deck, road])
        self.assertEqual([(x["check"], x["class"]) for x in f], [("portal", "defect")])
        self.assertAlmostEqual(f[0]["value"], 8.0)

    def test_a_road_below_the_ground_at_a_bridge_end_is_a_defect(self):
        deck = ramp("bridge", -60, 0, lambda x: 94.0, way="2", fid="deck", depth=1.0)
        road = ramp("ground", 0, 250, lambda x: 94.0 + 0.024 * x, way="1", fid="road")
        g = of(lc.run_checks([deck, road]), "ground")
        self.assertEqual([x["class"] for x in g], ["defect"])      # (a bridge explains a road above the ground)

    def test_a_cutting_to_a_tunnel_mouth_is_an_earthwork(self):
        tun = ramp("open", -60, 0, lambda x: 95.0, way="2", fid="open")
        road = ramp("ground", 0, 140, lambda x: 95.0 + 5.0 * min(1.0, x / 100.0), way="1", fid="road")
        g = of(lc.run_checks([tun, road]), "ground")
        self.assertEqual([x["class"] for x in g], ["earthwork"])
        self.assertEqual(g[0]["earth_by"], "open open end")
        self.assertEqual([x["class"] for x in of(lc.run_checks([road]), "ground")], ["defect"])

    def test_a_dip_under_a_deck_is_an_earthwork(self):
        deck = lc.Piece("bridge", xy=[(100, -40), (100, 40)], lat=42.68 + np.array([-40, 40.0]) / M_LAT,
                        lon=np.full(2, 23.31 + 100 / 82000.0), z=[107.0, 107.0], way="2", fid="deck", depth=1.0)
        road = ramp("ground", 0, 200, lambda x: 100.0 - 4.0 * max(0.0, min(1.0, (60.0 - abs(x - 100.0)) / 48.0)),
                    way="1", fid="road")                          # (4 m down under the deck, flat 12 m, then up over 48 m)
        g = of(lc.run_checks([deck, road]), "ground")
        self.assertEqual([x["class"] for x in g], ["earthwork"])
        self.assertEqual((g[0]["earth_by"], g[0]["near_role"]), ("bridge deck over", "over"))
        self.assertEqual([x["class"] for x in of(lc.run_checks([road]), "ground")], ["defect"])

    def test_the_run_must_be_wholly_explained(self):
        deck, road = embankment()
        road.z[60:63] += 3.0                                       # (a hump 120 m out: no part of the ramp)
        g = of(lc.run_checks([deck, road]), "ground")
        self.assertEqual([x["class"] for x in g], ["defect"])

    def test_without_a_context_every_ground_finding_is_a_defect_and_has_no_context(self):
        _deck, road = embankment()
        g = lc.check_ground(road)
        self.assertEqual([x["class"] for x in g], ["defect"])
        self.assertNotIn("near_kind", g[0])

    def test_defects_sort_before_the_findings_set_apart(self):
        deck, road = embankment(height=9.0)                        # (the earthwork is the worse offset)
        other = ramp("ground", 0, 200, lambda x: 103.0, terrain=100.0, way="3", fid="other")
        other.xy[:, 1] += 500.0
        other.lat = other.lat + 500.0 / M_LAT
        f = lc.run_checks([deck, road, other])
        g = of(f, "ground")
        self.assertEqual([x["class"] for x in g], ["defect", "earthwork"])
        self.assertGreater(g[1]["severity"], g[0]["severity"])

    def test_road_following_a_15_percent_hillside_is_terrain(self):
        hill = piece("ground", 0, 0, 60, 0, 100.0, 109.0, terrain=None)
        hill.terrain = hill.z - 0.04                               # (the ground under it: the same 15 %)
        f = lc.run_checks([hill])
        g = of(f, "grade")
        self.assertEqual([x["class"] for x in g], ["terrain"])
        self.assertAlmostEqual(g[0]["value"], 0.15, places=2)
        self.assertAlmostEqual(g[0]["terrain_grade"], 0.15, places=2)
        self.assertLess(g[0]["excess"], 0.01)
        self.assertEqual(lc.totals(f)["grade"], 0)
        self.assertEqual(lc.set_apart(f)["grade"], 1)

    def test_a_15_percent_road_on_flat_ground_is_a_defect(self):
        steep = piece("ground", 0, 0, 60, 0, 100.0, 109.0, terrain=100.0)
        g = of(lc.run_checks([steep]), "grade")
        self.assertEqual([x["class"] for x in g], ["defect"])
        self.assertAlmostEqual(g[0]["excess"], 0.15, places=2)
        self.assertEqual(lc.totals(lc.run_checks([steep]))["grade"], 1)

    def test_a_road_much_steeper_than_its_hillside_is_a_defect_and_a_little_steeper_is_not(self):
        def on_hill(road_rise, hill_rise):
            p = piece("ground", 0, 0, 60, 0, 100.0, 100.0 + road_rise, terrain=None)
            p.terrain = np.linspace(100.0, 100.0 + hill_rise, len(p.z))
            return of(lc.run_checks([p]), "grade")
        self.assertEqual([x["class"] for x in on_hill(12.0, 7.2)], ["defect"])      # 20 % on a 12 % hill: 8 more
        self.assertEqual([x["class"] for x in on_hill(10.2, 7.2)], ["terrain"])     # 17 % on a 12 % hill: 5 more

    def test_a_road_past_the_ceiling_is_a_defect_whatever_the_ground_does(self):
        p = piece("ground", 0, 0, 60, 0, 100.0, 118.0, terrain=None)               # 30 %, on a hill of the same
        p.terrain = p.z.copy()
        self.assertEqual([x["class"] for x in of(lc.run_checks([p]), "grade")], ["defect"])

    def test_a_steep_deck_is_a_defect_whatever_is_under_it(self):
        deck = piece("bridge", 0, 0, 60, 0, 100.0, 109.0, terrain=None, depth=1.0)
        deck.terrain = deck.z - 0.04
        self.assertEqual([x["class"] for x in of(lc.run_checks([deck]), "grade")], ["defect"])

    def test_the_report_carries_the_classes_and_the_version(self):
        deck, road = embankment()
        f = lc.run_checks([deck, road])
        with tempfile.TemporaryDirectory() as d:
            t = lc.write_report(d, f, {"bbox": [0, 0, 1, 1]})
            with open(os.path.join(d, "levelcheck.json"), encoding="utf-8") as fh:
                doc = json.load(fh)
            with open(os.path.join(d, "levelcheck.csv"), encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        self.assertEqual(doc["version"], lc.VERSION)
        self.assertGreaterEqual(lc.VERSION, 2)
        self.assertEqual(t["ground"], 0)
        self.assertEqual(doc["set_apart"], {"ground_earthwork": 1, "grade_terrain": 0})
        self.assertEqual(rows[0]["class"], "earthwork")
        self.assertEqual(rows[0]["near_kind"], "bridge")
        self.assertEqual(doc["thresholds"]["grade_ceiling"], lc.GRADE_CEILING)

    def test_the_constants_are_the_generators(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        import grading
        import roads
        self.assertEqual(lc.BRIDGE_RAMP_REACH_M, grading.RAMP_REACH_M)
        self.assertGreater(lc.BRIDGE_RAMP_REACH_M, grading.APPROACH_MAX_RUN)
        self.assertAlmostEqual(lc.TUNNEL_RAMP_REACH_M, roads.FINE_APPROACH_MAX_M / roads.TUNNEL_GRADE, delta=1.0)
        self.assertAlmostEqual(lc.UNDER_RAMP_REACH_M, grading.UNDER_FLAT_M
                               + grading.UNDER_MAX_DROP_M * math.pi / (2 * grading.UNDER_GRADE), delta=1.0)


if __name__ == "__main__":
    unittest.main()
