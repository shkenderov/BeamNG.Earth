"""Sofia's surveyed streets (sofiastreets.py, 2026-11-10): the file's parsing and its .npz, a cell's raster, a road's
carriageway width from it (the line beside its carriageway, a divided road drawn as one carriageway or with a median,
sidewalks and parking bays per side, bridges), the points no pole stands on, and the furniture's use of it. Off
without BNE_SOFIA_STREETS."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).parent)]
import furniture  # noqa: E402
import mapcells  # noqa: E402
import poleprofiles  # noqa: E402
import roads  # noqa: E402
import sofiastreets as ss  # noqa: E402

KEY = mapcells.cell_of(42.6965, 23.3215)
FRAME = ss.cell_frame(KEY)


def ll(x, y):
    """lon / lat (k, 2) of east / north metres in the cell's frame."""
    la, lo, _ = FRAME.enu_to_geodetic(np.asarray(x, float), np.asarray(y, float), 0.0)
    return np.c_[np.atleast_1d(lo), np.atleast_1d(la)]


def rect(x0, y0, x1, y1):
    """A counter-clockwise closed ring, as GeoJSON has its outer rings."""
    return ll([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0]).tolist()


def poly(kind, *rings, name=""):
    return {"type": "Feature", "properties": {"rn": "1", "type": kind, "street_name": name, "region": "", "area": "1"},
            "geometry": {"type": "Polygon", "coordinates": list(rings)}}


def line(x0, x1, y):
    return ll([x0, x1], [y, y])


SCENE = [
    # A: a two-way street along x, 7 m, parking bays on its north side, sidewalks beyond both
    poly("улично платно", rect(-200, -3.5, 200, 3.5), name="A"),
    poly("паркинг на платното", rect(-150, 3.5, 150, 5.5), name="A"),
    poly("тротоар", rect(-200, 5.5, 200, 7.5), name="A"),
    poly("тротоар", rect(-200, -5.5, 200, -3.5), name="A"),
    # B: a divided road drawn as one 16 m carriageway (no median)
    poly("улично платно", rect(-200, 92, 200, 108), name="B"),
    # C: a divided road with its median strip
    poly("улично платно", rect(-200, 200, 200, 207), name="C"),
    poly("разделителна ивица", rect(-200, 207, 200, 211), name="C"),
    poly("улично платно", rect(-200, 211, 200, 218), name="C"),
    # D: a 6 m street whose map line lies 1.5 m beside it
    poly("улично платно", rect(-200, 300, 200, 306), name="D"),
    # F: a bridge's carriageway, a tram bed beside it
    poly("улично платно на мост", rect(-200, 400, 200, 408), name="F"),
    poly("трамвайно трасе", rect(-200, 408, 200, 414), name="F"),
    # a square: carriageway everywhere (no edge within the ray)
    poly("улично платно", rect(-200, 500, 200, 600), name="G"),
    {"type": "Feature", "properties": {"type": "нещо ново"}, "geometry": {"type": "MultiPolygon", "coordinates": [
        [rect(300, 0, 310, 10)], [rect(320, 0, 330, 10)]]}},
    {"type": "Feature", "properties": {"type": "тротоар"}, "geometry": {"type": "Point", "coordinates": [23.3, 42.7]}},
]


def write(folder, feats=SCENE, name="streets_curbs.geojson"):
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "name": "streets_curbs", "features": feats}, f)
    return path


class Scene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = write(cls.tmp.name)
        cls.s = ss.Streets(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def one(self, *lines):
        return self.s.measure_cell(KEY, list(lines))


class FileTest(Scene):
    def test_parsed(self):
        self.assertEqual(len(self.s), len(SCENE))          # (the MultiPolygon's two parts, the Point left out)
        self.assertTrue(os.path.exists(ss.npz_path(self.path)))
        self.assertEqual(int(self.s.kind[-1]), ss.OTHER, "a kind this module does not know")
        self.assertEqual(int(self.s.kind[9]), ss.CARRIAGEWAY | ss.BRIDGE)
        s, w, n, e = self.s.bounds
        lat_c, lon_c = KEY.center()                         # (the scene lies round the cell's centre)
        self.assertTrue(s < lat_c < n and w < lon_c < e)
        self.assertTrue(self.s.covers(KEY))
        self.assertFalse(self.s.covers(mapcells.cell_of(43.9, 23.3215)))
        self.assertIsNone(self.s.raster(mapcells.cell_of(43.9, 23.3215)))

    def test_npz_kept_until_the_file_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p = write(d)
            a = ss.Streets(p)
            t = os.path.getmtime(a.npz)
            b = ss.Streets(p)
            self.assertEqual(os.path.getmtime(b.npz), t, "read again, not parsed again")
            time.sleep(0.05)
            write(d, SCENE[:4])
            os.utime(p, (time.time() + 5, time.time() + 5))
            self.assertEqual(len(ss.Streets(p)), 4, "a newer file: built again")
            self.assertFalse(os.path.exists(a.npz + ".lock"))

    def test_not_a_collection(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.geojson")
            with open(p, "w") as f:
                f.write('{"type": "Topology"}')
            with self.assertRaises(ValueError):
                ss.Streets(p)
            self.assertFalse(os.path.exists(ss.npz_path(p) + ".lock"), "the lock is gone after a failed build")


class MaskTest(unittest.TestCase):
    def test_union_and_hole(self):
        outer = np.array([[2, 2], [18, 2], [18, 18], [2, 18], [2, 2]], float)           # counter-clockwise
        hole = np.array([[8, 8], [8, 12], [12, 12], [12, 8], [8, 8]], float)            # clockwise
        other = np.array([[16, 4], [26, 4], [26, 8], [16, 8], [16, 4]], float)          # overlaps the first
        m = ss.winding_mask([outer, hole, other], 0.0, 20.0, 1.0, 30, 20)
        self.assertEqual(m.shape, (20, 30))
        self.assertTrue(m[20 - 5, 5] and m[20 - 6, 22], "inside each")
        self.assertFalse(m[20 - 10, 10], "the hole")
        self.assertTrue(m[20 - 6, 17], "the overlap: once, not cancelled")
        self.assertFalse(m[20 - 1, 5] or m[20 - 15, 25])
        self.assertEqual(int(m.sum()), 16 * 16 - 16 + 8 * 4)

    def test_raster(self):
        with tempfile.TemporaryDirectory() as d:
            r = ss.Streets(write(d)).raster(KEY)
        self.assertEqual(r.res, ss.RES_M)
        self.assertEqual(int(r.at(0.0, 0.0)), ss.CARRIAGEWAY)
        self.assertEqual(int(r.at(0.0, 4.5)), ss.PARKING)
        self.assertEqual(int(r.at(0.0, 6.5)), ss.SIDEWALK)
        self.assertEqual(int(r.at(0.0, 404.0)), ss.CARRIAGEWAY | ss.BRIDGE)
        self.assertEqual(int(r.at(0.0, 410.0)), ss.TRACK)
        self.assertEqual(int(r.at(0.0, 209.0)), ss.OTHER)
        self.assertEqual(int(r.at(0.0, 50.0)), 0)
        self.assertEqual(int(r.at(1e6, 0.0)), 0, "outside the raster")


class MeasureTest(Scene):
    def test_street_with_parking_and_sidewalks(self):
        m, = self.one(line(-180, 180, 0.0))
        self.assertAlmostEqual(m.width, 7.0, delta=ss.RES_M)
        self.assertGreater(m.stations, 50)
        self.assertLess(m.spread, 0.3)
        self.assertTrue(m.walk_left and m.walk_right)
        self.assertTrue(m.park_left, "bays on the north side: left of a line going east")
        self.assertFalse(m.park_right)
        self.assertAlmostEqual(m.park_m, 2.0, delta=ss.RES_M)
        self.assertEqual((m.bridge, m.tunnel, m.shared), (0.0, 0.0, 0.0))
        # the same street drawn the other way: its sides swap
        back, = self.one(line(180, -180, 0.0))
        self.assertAlmostEqual(back.width, m.width, delta=0.26)
        self.assertTrue(back.park_right and not back.park_left)

    def test_off_centre_line_is_the_same_width(self):
        m, = self.one(line(-180, 180, 2.0))
        self.assertAlmostEqual(m.width, 7.0, delta=ss.RES_M)

    def test_divided_road_in_one_carriageway_is_cut_half_way(self):
        a, b = self.one(line(-180, 180, 96.0), line(180, -180, 104.0))
        for m in (a, b):
            self.assertAlmostEqual(m.width, 8.0, delta=ss.RES_M + 0.26)
            self.assertEqual(m.shared, 1.0)
            self.assertFalse(m.walk_left or m.walk_right)
        alone, = self.one(line(-180, 180, 96.0))
        self.assertAlmostEqual(alone.width, 16.0, delta=ss.RES_M, msg="no other line: the whole carriageway")

    def test_divided_road_with_a_median(self):
        a, b = self.one(line(-180, 180, 203.5), line(180, -180, 214.5))
        for m in (a, b):
            self.assertAlmostEqual(m.width, 7.0, delta=ss.RES_M)
            self.assertEqual(m.shared, 0.0)

    def test_a_crossing_line_is_no_twin(self):
        cross = ll([0.0, 0.0], [-60.0, 60.0])
        m, _ = self.one(line(-180, 180, 0.0), cross)
        self.assertAlmostEqual(m.width, 7.0, delta=ss.RES_M)
        self.assertLess(m.shared, 0.05)

    def test_line_beside_its_carriageway(self):
        m, = self.one(line(-180, 180, 298.5))
        self.assertAlmostEqual(m.width, 6.0, delta=ss.RES_M)
        far, = self.one(line(-180, 180, 300.0 - ss.SNAP_M - 2.0))
        self.assertIsNone(far, "further off than SNAP_M: not this road's")

    def test_bridge(self):
        m, = self.one(line(-180, 180, 404.0))
        self.assertAlmostEqual(m.width, 8.0, delta=ss.RES_M)
        self.assertEqual(m.bridge, 1.0)

    def test_nothing_to_measure(self):
        none = self.one(line(-180, 180, 50.0),              # no carriageway there
                        line(-180, 180, 550.0),             # a square: no edge within the ray
                        line(0, 6, 0.0),                    # too short for three stations
                        np.zeros((1, 2)))                   # no line at all
        self.assertEqual(none, [None, None, None, None])
        self.assertEqual(self.s.measure_cell(KEY, []), [])
        self.assertEqual(self.s.measure_cell(mapcells.cell_of(43.9, 23.3215), [line(-180, 180, 0.0)]), [None])

    def test_same_answer_whatever_else_is_asked(self):
        a, = self.one(line(-180, 180, 0.0))
        b = self.one(line(-180, 180, 203.5), line(-180, 180, 0.0), line(-180, 180, 404.0))[1]
        self.assertEqual(a, b)


class BlockedTest(Scene):
    def at(self, x, y, **kw):
        p = ll(x, y)
        return bool(self.s.blocked(p[:, 1], p[:, 0], **kw)[0])

    def test_kinds(self):
        self.assertTrue(self.at(0.0, 0.0), "a carriageway")
        self.assertTrue(self.at(0.0, 4.5), "a parking bay")
        self.assertTrue(self.at(0.0, 410.0), "a tram bed")
        self.assertFalse(self.at(0.0, 6.9), "a sidewalk")
        self.assertFalse(self.at(0.0, 209.0), "a median strip")
        self.assertFalse(self.at(0.0, 50.0), "nothing mapped")

    def test_margin(self):
        self.assertTrue(self.at(0.0, -3.5 - 0.1), "0.1 m off the edge: within the pole's foot")
        self.assertFalse(self.at(0.0, -3.5 - 1.2))
        self.assertFalse(self.at(0.0, -3.5 - 0.1, margin=0.0))

    def test_arrays_and_elsewhere(self):
        p = np.vstack([ll(0.0, 0.0), ll(0.0, 50.0), [[2.35, 48.85]], [[np.nan, np.nan]]])
        self.assertEqual(self.s.blocked(p[:, 1], p[:, 0]).tolist(), [True, False, False, False])


class EnvTest(unittest.TestCase):
    def setUp(self):
        self.old = os.environ.get(ss.ENV)
        ss._opened.clear()

    def tearDown(self):
        if self.old is None:
            os.environ.pop(ss.ENV, None)
        else:
            os.environ[ss.ENV] = self.old
        ss._opened.clear()

    def test_unset_off_missing(self):
        os.environ.pop(ss.ENV, None)
        self.assertIsNone(ss.from_env())
        os.environ[ss.ENV] = "off"
        self.assertIsNone(ss.from_env())
        os.environ[ss.ENV] = os.path.join(tempfile.gettempdir(), "no_such_streets.geojson")
        with self.assertLogs("earthd.sofiastreets", "WARNING"):
            self.assertIsNone(ss.from_env())
        self.assertIsNone(ss.from_env(), "said once")

    def test_named(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ[ss.ENV] = write(d)
            s = ss.from_env()
            self.assertIsInstance(s, ss.Streets)
            self.assertIs(ss.from_env(), s, "one per process")

    def test_furniture_keeps_lamps_and_poles_off_the_surveyed_streets(self):
        # a country road (it guesses poles 2.4 m off its edge) drawn 6 m wide where the survey has a 16 m
        # carriageway: its poles would stand in the lanes
        road = ("r", roads.Spec(6.0, "paved", False, "tertiary", False),
                np.array([(-190.0, 100.0, 0.0), (190.0, 100.0, 0.0)]), 0.0)
        box = (-200, 0, 200, 200)
        os.environ.pop(ss.ENV, None)
        before = [r for r in furniture.place([road], [], box, pole_frame=FRAME) if poleprofiles.is_pole_group(r[0])]
        self.assertTrue(before)
        self.assertTrue(all(abs(r[2] - 100.0) < 8.0 for r in before), "inside the real carriageway")
        with tempfile.TemporaryDirectory() as d:
            os.environ[ss.ENV] = write(d)
            ss._opened.clear()
            wires = []
            rows = furniture.place([road], [], box, pole_frame=FRAME, wires=wires)
            self.assertEqual([r for r in rows if poleprofiles.is_pole_group(r[0])], [])
            self.assertEqual(wires, [])
            # a town street: its lamp posts 1.2 m off the guessed edge, on the real carriageway too
            street = ("t", roads.Spec(6.0, "paved", True, "residential", False),
                      np.array([(-190.0, 100.0, 0.0), (190.0, 100.0, 0.0)]), 0.0)
            lamps = [r for r in furniture.place([street], [], box, pole_frame=FRAME, country="BG")
                     if r[0] == furniture.G_LAMP]
            self.assertEqual(lamps, [])
            # without a frame nothing is asked (the positions are the anchor's: no place on the Earth)
            self.assertTrue([r for r in furniture.place([road], [], box) if poleprofiles.is_pole_group(r[0])])


if __name__ == "__main__":
    unittest.main()
