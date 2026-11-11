"""Sofia's surveyed streets in the roads' specs (roads._sofia_specs, sofiastreets.py; 2026-11-10): a road's
carriageway is its surveyed width, one width along the road (a short piece between two others takes theirs), with
parking lanes only where it has room for them, its sidewalks on the sides the survey has them, the map's lane count
kept; a road of the survey's area without a carriageway under it loses the parking lanes alone; a bridge's deck is as
wide as its road. Off without BNE_SOFIA_STREETS (the specs as before)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).parent)]
import roads  # noqa: E402
import sofiastreets as ss  # noqa: E402
from test_roads import LAT, LON, M_LAT, M_LON, features, rules  # noqa: E402


def ring(x0, y0, x1, y1):
    """A counter-clockwise ring in test_roads' east / north metres about (LAT, LON)."""
    return [[LON + x / M_LON, LAT + y / M_LAT] for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]


def poly(kind, *rings):
    return {"type": "Feature", "properties": {"type": kind, "street_name": ""},
            "geometry": {"type": "Polygon", "coordinates": list(rings)}}


SURVEY = [
    poly("улично платно", ring(-200, -2.75, 200, 2.75)),            # A: a 5.5 m street along y = 0 ...
    poly("тротоар", ring(-200, 2.75, 200, 5.0)),                    # ... a sidewalk on its north side alone
    poly("улично платно", ring(-200, 93, 200, 107)),                # B: a 14 m road along y = 100, sidewalks both
    poly("тротоар", ring(-200, 107, 200, 110)),
    poly("тротоар", ring(-200, 90, 200, 93)),
    poly("улично платно на мост", ring(-200, 196, 200, 204)),       # C: an 8 m road along y = 200, a bridge in it
    poly("улично платно", ring(-200, 296, 200, 300)),               # E: a lot's aisle along y = 298 (a service road)
    poly("улично платно", ring(-200, 493, 200, 507)),               # G: a 14 m street along y = 500
    poly("зелени площи", ring(-900, -900, 900, 900)),               # (the survey's area: its box)
]
LINES = [
    [(-180, 0), (180, 0)],          # 0 A residential
    [(-180, 100), (180, 100)],      # 1 B primary, 4 lanes mapped
    [(-180, 200), (180, 200)],      # 2 C tertiary with a bridge
    [(-180, 400), (180, 400)],      # 3 D residential, nothing surveyed under it
    [(-180, 298), (180, 298)],      # 4 E service
    [(5000, 0), (5400, 0)],         # 5 F residential, outside the survey's area
    [(-180, 500), (180, 500)],      # 6 G residential, wide
]
CLASSES = ["residential", "primary", "tertiary", "residential", "service", "residential", "residential"]


def layer():
    f = features(LINES, CLASSES, flags=[(2, "is_bridge", 0.4, 0.6)], levels=[(2, 1.0, 0.4, 0.6)])
    f.attrs["osm_lanes"] = np.array([np.nan, 4.0, np.nan, np.nan, np.nan, np.nan, np.nan])
    return f


def measure(width, stations=40, spread=0.2):
    return ss.Measure(width, stations, spread, True, True, False, False, 0.0, 0.0, 0.0, 0.0)


def streets_of_pieces(rows):
    """A layer of streets, each three pieces end to end along its own y: rows of (name, [(x0, x1, Measure or None)
    x 3]). Its measures are set on it (as _sofia_measures_local keeps them)."""
    lines, names, con, got = [], [], [], []
    for s, (name, pieces) in enumerate(rows):
        for k, (x0, x1, m) in enumerate(pieces):
            i = len(lines)
            lines.append([(x0, 100.0 * s), (x1, 100.0 * s)])
            names.append(name)
            got.append(m)
            if k:
                con += [(i - 1, f"j{s}-{k}", 1.0, 1.0), (i, f"j{s}-{k}", 0.0, 0.0)]
    f = features(lines, ["residential"] * len(lines))
    f.attrs["name"] = np.array(names)
    f.rules["connectors"] = rules(sorted(con))
    f.__dict__["_sofia_measures"] = got
    return f


def town(lat, lon):
    return np.ones(np.shape(lat), bool)


class SofiaRoadsTest(unittest.TestCase):
    def setUp(self):
        self.old = os.environ.get(ss.ENV)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "streets_curbs.geojson")
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": SURVEY}, f)
        ss._opened.clear()
        roads._FACTS_MEM.clear()

    def tearDown(self):
        if self.old is None:
            os.environ.pop(ss.ENV, None)
        else:
            os.environ[ss.ENV] = self.old
        ss._opened.clear()
        roads._FACTS_MEM.clear()
        self.tmp.cleanup()

    def specs(self, on):
        if on:
            os.environ[ss.ENV] = self.path
        else:
            os.environ.pop(ss.ENV, None)
        ss._opened.clear()
        f = layer()
        return f, roads.road_specs(f, town)

    def test_off_as_before(self):
        self.assertEqual(roads._sofia_key(), ())
        _f, sp = self.specs(False)
        self.assertAlmostEqual(sp[0].width, 6.0 + 2 * roads.PARKING_M)
        self.assertTrue(sp[0].parking)
        self.assertIsNone(sp[0].sides)
        f = layer()
        self.assertEqual(roads.sofia_measures(f), [None] * len(f))

    def test_surveyed_street(self):
        f, sp = self.specs(True)
        self.assertEqual(roads._sofia_key()[0][0], "sofia")
        a = sp[0]
        self.assertAlmostEqual(a.width, 5.5, delta=ss.RES_M)
        self.assertFalse(a.parking, "the survey's carriageway holds no parking lanes")
        self.assertEqual(a.sides, (1,), "a sidewalk on its north side: left of a line going east")
        self.assertEqual(roads.parking_width(a), 0.0)
        drive, lanes = roads.lane_layout(a)
        self.assertAlmostEqual(drive, a.width)
        self.assertEqual(lanes, 1)
        m = roads.sofia_measures(f)
        self.assertIsNotNone(m[0])
        self.assertIsNone(m[5])

    def test_mapped_lanes_stay(self):
        _f, sp = self.specs(True)
        b = sp[1]
        self.assertAlmostEqual(b.width, 14.0, delta=ss.RES_M)
        self.assertEqual(b.lanes, 4)
        self.assertEqual(sorted(b.sides), [-1, 1])
        self.assertEqual(roads.lane_layout(b)[1], 4)
        _f, off = self.specs(False)
        self.assertAlmostEqual(off[1].width, 4 * roads.LANE_W["primary"])

    def test_no_carriageway_under_it_no_parking_lanes(self):
        _f, sp = self.specs(True)
        d = sp[3]
        self.assertAlmostEqual(d.width, 6.0, msg="the class's width, the parking lanes off it (as Sweden's streets)")
        self.assertFalse(d.parking)
        self.assertIsNone(d.sides, "nothing surveyed: the class and land-cover rule decides its sidewalks")

    def test_service_roads_and_elsewhere_untouched(self):
        _f, on = self.specs(True)
        _f, off = self.specs(False)
        self.assertEqual(on[4], off[4], "a service road: not asked")
        self.assertEqual(on[5], off[5], "outside the survey's area")

    def test_deck_as_wide_as_its_road(self):
        f, sp = self.specs(True)
        deck = roads.bridge_specs(f)[2]
        self.assertIsNotNone(deck)
        self.assertAlmostEqual(deck.width, 8.0, delta=ss.RES_M)
        self.assertAlmostEqual(sp[2].width, deck.width, msg="the road goes on over its bridge as wide as it came")
        f2, _sp = self.specs(False)
        self.assertNotAlmostEqual(roads.bridge_specs(f2)[2].width, 8.0, delta=ss.RES_M)

    def test_a_spread_of_widths_is_not_one_width(self):
        f = streets_of_pieces([("A", [(-300, -100, measure(9.0, spread=6.0)), (-100, 100, measure(9.0, spread=1.0)),
                                      (100, 300, measure(2.2))])])
        self.assertEqual(roads._sofia_widths_local(f), [9.0, 9.0, 9.0],
                         "lanes joining along it, and a sliver: no width of their own, the road's they go on in")
        sp = roads.Spec(7.5, "paved", True, "secondary", False)
        self.assertEqual(roads._sofia_width(sp, 9.0), 9.0)
        self.assertIsNone(roads._sofia_width(sp, None))
        self.assertIsNone(roads._sofia_width(sp, 2.8), "narrower than a two-way road")
        self.assertEqual(roads._sofia_width(sp._replace(oneway=True), 2.8), 2.8)
        self.assertIsNone(roads._sofia_width(sp._replace(cls="service"), 9.0))

    def test_one_width_along_a_road(self):
        f = streets_of_pieces([
            ("dip", [(-300, -100, measure(8.0)), (-100, -40, measure(4.0, 9)), (-40, 300, measure(8.0))]),
            ("gap", [(-300, -100, measure(7.0)), (-100, 0, None), (0, 300, measure(7.5))]),
            ("long", [(-300, -150, measure(8.0)), (-150, 50, measure(4.0)), (50, 300, measure(8.0))]),
            ("", [(-300, -100, measure(8.0)), (-100, 0, None), (0, 300, measure(8.0))]),
            ("bulge", [(-300, -100, measure(6.0)), (-100, -40, measure(12.0, 9)), (-40, 300, measure(7.0))]),
            ("near", [(-300, -100, measure(8.0)), (-100, -40, measure(7.2, 9)), (-40, 300, measure(8.0))]),
            ("few", [(-300, -100, measure(8.0)), (-100, 100, measure(5.0, 4)), (100, 300, measure(8.0))]),
        ])
        got = roads._sofia_widths_local(f)
        self.assertEqual(got[0:3], [8.0, 8.0, 8.0], "a short piece narrower than both it goes on in: theirs")
        self.assertEqual(got[3:6], [7.0, 7.25, 7.5], "nothing surveyed under it: between its neighbours'")
        self.assertEqual(got[6:9], [8.0, 4.0, 8.0], "a long piece is as narrow as surveyed")
        self.assertEqual(got[9:12], [8.0, 8.0, 8.0], "nameless: the only other road at the joint")
        self.assertEqual(got[12:15], [6.0, 7.0, 7.0], "a short piece wider than both: the nearer one's")
        self.assertEqual(got[15:18], [8.0, 7.2, 8.0], "within the step: its own")
        self.assertEqual(got[18:21], [8.0, 8.0, 8.0], "measured at a few stations: as a short one")

    def test_another_roads_piece_is_not_its_neighbour(self):
        f = streets_of_pieces([("A", [(-300, -100, measure(8.0)), (-100, -40, measure(4.0, 9)), (-40, 300, measure(8.0))])])
        f.attrs["name"] = np.array(["A", "A", "B"])
        f.rules["connectors"] = rules([(0, "j1", 1.0, 1.0), (1, "j1", 0.0, 0.0), (1, "j2", 1.0, 1.0),
                                       (2, "j2", 0.0, 0.0), (2, "j3", 1.0, 1.0)])
        self.assertEqual(roads._sofia_widths_local(f)[1], 8.0, "another name, the only road it goes on in")
        g = features([[(-300, 0), (-100, 0)], [(-100, 0), (-40, 0)], [(-40, 0), (300, 0)], [(-40, 0), (-40, 200)]],
                     ["residential"] * 4)
        g.attrs["name"] = np.array(["A", "A", "B", "C"])
        g.rules["connectors"] = rules([(0, "j1", 1.0, 1.0), (1, "j1", 0.0, 0.0), (1, "j2", 1.0, 1.0),
                                       (2, "j2", 0.0, 0.0), (3, "j2", 0.0, 0.0)])
        g.__dict__["_sofia_measures"] = [measure(8.0), measure(4.0, 9), measure(8.0), measure(8.0)]
        self.assertEqual(roads._sofia_widths_local(g)[1], 4.0, "a junction of other streets: no road to go on in")

    def test_parking_lanes_where_they_fit_and_no_cars_in_the_lanes(self):
        f, sp = self.specs(True)
        g = sp[6]
        self.assertAlmostEqual(g.width, 14.0, delta=ss.RES_M)
        self.assertTrue(g.parking, "a carriageway with room for the parking lanes beside two driving lanes")
        self.assertEqual(roads.lane_layout(g)[1], 2)
        self.assertIsNone(sp[2].sides, "no sidewalk surveyed on either side: the rule's, not a street without any")
        seen = {"a": (f, 0, sp[0]), "g": (f, 6, g), "f": (f, 5, sp[5]._replace(parking=False))}
        self.assertEqual(roads._sofia_no_curb_cars(seen), {"a"})
        os.environ.pop(ss.ENV, None)
        ss._opened.clear()
        self.assertEqual(roads._sofia_no_curb_cars(seen), set())

    def test_specs_keyed_by_the_survey(self):
        os.environ.pop(ss.ENV, None)
        ss._opened.clear()
        k0 = roads._street_spec_key(None, None, False)
        os.environ[ss.ENV] = self.path
        ss._opened.clear()
        self.assertNotEqual(roads._street_spec_key(None, None, False), k0)


if __name__ == "__main__":
    unittest.main()
