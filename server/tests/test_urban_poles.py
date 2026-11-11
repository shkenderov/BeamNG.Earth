"""Utility poles in the towns of a country whose town lines are buried (furniture.URBAN_BURIED_LV: Bulgaria), every
pole style kept off the carriageways (furniture.collision_shape: the regional styles had no collision boxes), and the
mapped poles checked against the roads too (mappedpoles.place's clearance). Sofia, 2026-11-10."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).parent)]
import furniture  # noqa: E402
import geodesy  # noqa: E402
import mappedpoles  # noqa: E402
import poleprofiles  # noqa: E402
import roads  # noqa: E402
from test_furniture import road  # noqa: E402
from test_mapped_poles import infrastructure, pole, line  # noqa: E402

BOX = (0, 0, 256, 256)


def poles(rows):
    return [r for r in rows if poleprofiles.is_pole_group(r[0])]


class UrbanBuriedTest(unittest.TestCase):
    def test_shares(self):
        self.assertEqual(furniture.urban_buried_share("bg"), 1.0)
        self.assertEqual(furniture.urban_buried_share("BG"), 1.0)
        self.assertEqual(furniture.urban_buried_share("US"), 0.0)
        self.assertEqual(furniture.urban_buried_share(None), 0.0)
        self.assertEqual(furniture.buried_share("BG"), 0.0, "the country roads keep their poles")

    def test_town_street_lamps_on_posts_no_poles(self):
        r = road("d", "residential", True)
        before = furniture.place([r], [], BOX)
        self.assertTrue(poles(before), "elsewhere: lamps on utility poles")
        rows = furniture.place([r], [], BOX, country="BG")
        self.assertEqual(poles(rows), [])
        self.assertFalse(any(x[0] == furniture.G_POLELAMP for x in rows))
        self.assertGreater(sum(x[0] == furniture.G_LAMP for x in rows), 2, "lit all the same, from posts")

    def test_town_sidewalk_run(self):
        r = road("a", "residential", True)
        fid, spec, pts, _ = r
        walks = [{"fid": fid, "pts": roads.offset_polyline(pts, side * (spec.width / 2 + 1.25)), "side": side,
                  "w": 2.5, "caps": (False, False)} for side in (1, -1)]
        wires = []
        self.assertTrue(poles(furniture.place([r], walks, BOX, town=True)))
        rows = furniture.place([r], walks, BOX, town=True, wires=wires, country="BG")
        self.assertEqual(poles(rows), [])
        self.assertEqual(wires, [])
        self.assertTrue(any(x[0] == furniture.G_LAMP for x in rows))

    def test_country_road_keeps_its_poles(self):
        r = road("c", "tertiary", False)
        self.assertEqual(len(poles(furniture.place([r], [], BOX, country="BG"))),
                         len(poles(furniture.place([r], [], BOX))))
        self.assertTrue(poles(furniture.place([r], [], BOX, country="BG")))


class PoleStyleClearanceTest(unittest.TestCase):
    def test_every_style_has_the_poles_boxes(self):
        stock = furniture.COLLISION_BOXES[furniture.G_POLE]
        for group in ("eu_pole_concrete_open", "eu_pole_concrete_three", "eu_wood_pole", "fr_wood_pole",
                      "pole_concrete_recessed"):
            self.assertTrue(poleprofiles.is_pole_group(group), group)
            self.assertIs(furniture.collision_shape(group), stock, group)
        self.assertTrue(poleprofiles.is_pole_group(furniture.G_POLE))
        self.assertFalse(poleprofiles.is_pole_group(furniture.G_LAMP))
        self.assertIsNone(furniture.collision_shape("streettree"), "no box: not tested, as before")

    def test_a_regional_pole_in_a_carriageway_is_refused(self):
        lane = np.array([(-100.0, 50.0, 0.0), (300.0, 50.0, 0.0)])
        c = furniture.RoadClearance([(4.0, lane)])
        for group in (furniture.G_POLE, "eu_pole_concrete_open", "eu_wood_pole"):
            self.assertFalse(c.clear([group, 100.0, 50.0, 0.0, 0.0, 1.0, 8.5, 0.0]), f"{group} on the centre line")
            self.assertFalse(c.clear([group, 100.0, 53.0, 0.0, 0.0, 1.0, 8.5, 0.0]), f"{group} in the lane")
            self.assertTrue(c.clear([group, 100.0, 57.0, 0.0, 0.0, 1.0, 8.5, 0.0]), f"{group} 3 m off the edge")

    def test_bulgarias_guessed_poles_stay_off_another_road(self):
        # a country road's poles 2.4 m off its edge, and a street crossing it: none of Bulgaria's two styles in the street
        profile = poleprofiles.profile_for("BG", "europe")
        self.assertIsNotNone(profile)
        main = road("c", "tertiary", False)
        cross = ("x", roads.Spec(7.0, "paved", False, "residential", False),
                 np.array([(100.0, -200.0, 0.0), (100.0, 300.0, 0.0)]), 0.0)
        rows = furniture.place([main, cross], [], BOX, pole_profile=profile,
                               pole_frame=geodesy.EnuFrame(42.6965, 23.3215))
        got = poles(rows)
        self.assertTrue(got)
        self.assertTrue(all(r[0].startswith("eu_pole_") for r in got), {r[0] for r in got})
        for r in got:
            self.assertGreater(abs(r[1] - 100.0), 3.5 + 0.25, f"{r[0]} at x {r[1]}: in the crossing street")


class MappedClearanceTest(unittest.TestCase):
    def setUp(self):
        self.frame = geodesy.EnuFrame(42.6965, 23.3215)
        self.box = (-100., -100., 100., 100.)

    def build(self, records, clearance=None):
        props, wires = [], []
        mappedpoles.place(self.frame, self.box, [infrastructure(self.frame, records)],
                          lambda la, lo: np.zeros(np.shape(la)), props, wires, None, everywhere=True,
                          clearance=clearance)
        return props, wires

    def test_a_mapped_pole_in_the_road_is_left_out_with_its_spans(self):
        records = [pole("n1", -40.), pole("n2", 0.), pole("n3", 40.), line([[-40., 10.], [40., 10.]])]
        props, wires = self.build(records)
        self.assertEqual(len(props), 3)
        self.assertEqual(len(wires), 2 * len(furniture.wire_lines([(0, 0, 0, 0), (40, 0, 0, 0)], max_span=250.)))
        # a street across the line, through the middle pole
        street = furniture.RoadClearance([(4.0, np.array([(0.0, -80.0, 0.0), (0.0, 80.0, 0.0)]))])
        props, wires = self.build(records, street)
        self.assertEqual(sorted(round(r[1]) for r in props), [-40, 40])
        self.assertEqual(wires, [], "no span to or over the pole left out")

    def test_no_clearance_as_before(self):
        records = [pole("n1", 0.)]
        self.assertEqual(len(self.build(records)[0]), 1)


if __name__ == "__main__":
    unittest.main()
