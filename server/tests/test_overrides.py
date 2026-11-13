"""overrides.py: per-way corrections from the file BNE_OVERRIDES names (lanes, width, sidewalk, level, skip), their
validation, the bbox confinement, a way split over a segment's src_ways, and the cache keys (nothing without a file;
the variant suffix follows the content)."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import overrides  # noqa: E402
import plancache  # noqa: E402
import roads  # noqa: E402
from test_roads import features, rules, LAT, LON, M_LON  # noqa: E402


def deg(e, n):
    return LON + e / M_LON, LAT + n / 111320.0


def streets(records=("w100", "w200@1", "w300"), flags=()):
    """Three 200 m east-west residential streets 100 m apart, with their source ways."""
    f = features([[(-100, 0), (100, 0)], [(-100, 100), (100, 100)], [(-100, 200), (100, 200)]],
                 ["residential"] * 3, flags=list(flags))
    f.attrs["src_record"] = np.array(list(records))
    return f


class Env(unittest.TestCase):
    def setUp(self):
        self.old = os.environ.get(overrides.ENV)
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        os.environ.pop(overrides.ENV, None)

    def tearDown(self):
        os.environ.pop(overrides.ENV, None)
        if self.old is not None:
            os.environ[overrides.ENV] = self.old

    def use(self, ways, **top):
        path = os.path.join(self.dir.name, "ov.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict({"version": 1, "ways": ways}, **top), f)
        os.environ[overrides.ENV] = path
        return path


class NoFile(Env):
    def test_unset_or_missing_changes_nothing(self):
        self.assertIsNone(overrides.from_env())
        self.assertEqual(overrides.key(), ())
        os.environ[overrides.ENV] = os.path.join(self.dir.name, "missing.json")
        self.assertIsNone(overrides.from_env())
        self.assertEqual(overrides.key(), ())

    def test_output_is_what_it_was(self):
        f = streets(flags=[(0, "is_tunnel", 0.2, 0.4)])
        before = roads._tag_ranges_all(f)
        specs = roads.road_specs(streets())
        self.assertEqual(roads._lane_overrides(f), {})
        self.assertEqual([(round(a, 3), round(b, 3), k) for a, b, k in before[0]], [(0.2, 0.4, "tunnel")])
        self.assertEqual(roads.unbuilt_mask(f).tolist(), [False] * 3)
        self.assertEqual(roads._sofia_key(), ())
        self.assertIsNone(plancache.plans_variant())
        self.assertEqual([s.width for s in roads.road_specs(streets())], [s.width for s in specs])

    def test_an_empty_file_changes_no_key(self):
        self.use({})
        self.assertEqual(overrides.key(), ())
        self.assertIsNone(plancache.plans_variant())


class Fields(Env):
    def test_lanes_win_over_rampoverrides_and_stay_on_their_way(self):
        self.use({"100": {"lanes": 3}})
        f = streets()
        self.assertEqual(roads._lane_overrides(f), {0: 3})
        import rampoverrides
        from unittest import mock
        with mock.patch.object(rampoverrides, "lane_counts", lambda feats: {0: 1, 2: 1}):
            self.assertEqual(roads._lane_overrides(streets()), {0: 3, 2: 1})

    def test_width_and_sidewalk(self):
        self.use({"200": {"width_m": 9.0, "sidewalk": "left"}, "300": {"sidewalk": "none"}})
        f = streets()
        base = roads.road_specs(streets())          # (a fresh layer: specs are kept per feats)
        specs = roads.road_specs(f)
        self.assertEqual(specs[1].width, 9.0)
        self.assertEqual(specs[1].sides, (1,))
        self.assertEqual(specs[2].sides, ())
        self.assertEqual(specs[2].width, base[2].width)
        self.assertEqual(specs[0].width, base[0].width)

    def test_skip_is_unbuilt(self):
        self.use({"300": {"skip": True}, "100": {"skip": False}})
        self.assertEqual(roads.unbuilt_mask(streets()).tolist(), [False, False, True])

    def test_level_replaces_the_tag_ranges(self):
        self.use({"100": {"level": [{"from": 0.25, "to": 0.5, "kind": "tunnel"}]},
                  "200": {"level": [{"from": 0.0, "to": 1.0, "kind": "ground"}]}})
        f = streets(flags=[(0, "is_bridge", 0.6, 0.9), (1, "is_tunnel", 0.1, 0.3)])
        got = roads._tag_ranges_all(f)
        self.assertEqual(got[0], [(0.25, 0.5, "tunnel")])      # (the file's, not the map's bridge)
        self.assertEqual(got[1], [])                           # (ground: no range)
        self.assertEqual(got[2], [])
        self.assertEqual(roads.tag_level_ranges(f, 0), [(0.25, 0.5, "tunnel")])

    def test_bbox_confines_to_features_wholly_inside(self):
        w, s = deg(-150, -10)
        e, n = deg(150, 110)
        self.use({"100": {"lanes": 4, "bbox": [w, s, e, n]}, "200": {"lanes": 4, "bbox": [w, s, e, n]},
                  "300": {"lanes": 4, "bbox": [w, s, e, n]}})
        f = streets(records=("w100", "w200", "w300"))
        self.assertEqual(roads._lane_overrides(f), {0: 4, 1: 4})        # (y = 200 lies outside)
        w2, s2 = deg(-50, -10)
        self.use({"100": {"lanes": 4, "bbox": [w2, s2, e, n]}})
        self.assertEqual(roads._lane_overrides(streets(records=("w100", "w200", "w300"))), {})    # (reaches out west)

    def test_a_split_way_maps_its_fractions_onto_each_segment(self):
        # way 7 is the last 40 % of segment 0 (0.6-1.0) and the first half of segment 1 (0.0-0.5); its level
        # range 0.5-1.0 maps onto each share: 0.8-1.0 of segment 0, 0.25-0.5 of segment 1
        self.use({"7": {"level": [{"from": 0.5, "to": 1.0, "kind": "tunnel"}], "lanes": 2}})
        f = streets(records=("w1@1", "w7@1", "w9"))
        f.rules["src_ways"] = rules([(0, "w1@1", 0.0, 0.6), (0, "w7@1", 0.6, 1.0),
                                     (1, "w7@1", 0.0, 0.5), (1, "w8@1", 0.5, 1.0)])
        ov = overrides.from_env()
        got = ov.levels(f, [[], [], []])
        self.assertEqual([(round(a, 3), round(b, 3), k) for a, b, k in got[0]], [(0.8, 1.0, "tunnel")])
        self.assertEqual([(round(a, 3), round(b, 3), k) for a, b, k in got[1]], [(0.25, 0.5, "tunnel")])
        self.assertEqual(got[2], [])
        self.assertEqual(roads._lane_overrides(f), {0: 2, 1: 2})

    def test_a_level_only_replaces_the_ways_share(self):
        self.use({"7": {"level": [{"from": 0.0, "to": 1.0, "kind": "ground"}]}})
        f = streets(records=("w1@1", "w9", "w9"))
        f.rules["src_ways"] = rules([(0, "w1@1", 0.0, 0.5), (0, "w7@1", 0.5, 1.0)])
        got = overrides.from_env().levels(f, [[(0.1, 0.3, "bridge"), (0.4, 0.8, "tunnel")], [], []])
        self.assertEqual([(round(a, 3), round(b, 3), k) for a, b, k in got[0]],
                         [(0.1, 0.3, "bridge"), (0.4, 0.5, "tunnel")])


class Validation(Env):
    def bad(self, doc, *words):
        path = os.path.join(self.dir.name, "bad.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc if isinstance(doc, str) else json.dumps(doc))
        os.environ[overrides.ENV] = path
        with self.assertRaises(overrides.OverridesError) as cm:
            overrides.from_env()
        for w in words:
            self.assertIn(w, str(cm.exception))

    def test_errors_name_the_way(self):
        self.bad({"version": 1, "ways": {"42": {"lanez": 2}}}, "way 42", "lanez")
        self.bad({"version": 1, "ways": {"42": {"lanes": "two"}}}, "way 42", "lanes")
        self.bad({"version": 1, "ways": {"42": {"lanes": True}}}, "way 42")
        self.bad({"version": 1, "ways": {"43": {"width_m": -1}}}, "way 43", "width_m")
        self.bad({"version": 1, "ways": {"44": {"sidewalk": "front"}}}, "way 44", "sidewalk")
        self.bad({"version": 1, "ways": {"45": {"skip": "yes"}}}, "way 45", "skip")
        self.bad({"version": 1, "ways": {"46": {"bbox": [1, 2, 3]}}}, "way 46", "bbox")
        self.bad({"version": 1, "ways": {"47": {"bbox": [5, 0, 1, 1]}}}, "way 47", "bbox")
        self.bad({"version": 1, "ways": {"48": {"level": [{"from": 0.5, "to": 0.2, "kind": "tunnel"}]}}}, "way 48")
        self.bad({"version": 1, "ways": {"49": {"level": [{"from": 0, "to": 1, "kind": "tube"}]}}}, "way 49", "kind")
        self.bad({"version": 1, "ways": {"50": {"level": [{"from": 0, "to": 1}]}}}, "way 50")
        self.bad({"version": 1, "ways": {"x": {}}}, "way key")
        self.bad({"version": 1, "ways": {"51": 3}}, "way 51")

    def test_file_level_errors(self):
        self.bad("{not json", "JSON")
        self.bad({"version": 2, "ways": {}}, "version")
        self.bad({"version": 1, "ways": {}, "extra": 1}, "extra")

    def test_a_good_file_with_every_field_loads(self):
        self.use({"123456789": {"note": "why", "bbox": [23.31, 42.69, 23.33, 42.70], "lanes": 2, "width_m": 7.0,
                                "sidewalk": "both", "level": [{"from": 0.0, "to": 1.0, "kind": "tunnel"}],
                                "skip": False}})
        self.assertIn(123456789, overrides.from_env().ways)


class Keys(Env):
    def test_variant_follows_the_content(self):
        self.use({"1": {"lanes": 2}})
        a = plancache.plans_variant()
        self.assertRegex(a, r"\.ov[0-9a-f]{8}$")
        self.use({"1": {"lanes": 3}})
        b = plancache.plans_variant()
        self.assertRegex(b, r"\.ov[0-9a-f]{8}$")
        self.assertNotEqual(a, b)
        self.use({"1": {"lanes": 3}})
        self.assertEqual(plancache.plans_variant(), b)
        self.assertEqual(overrides.key(), (("ov", overrides.from_env().digest),))

    def test_the_spec_keys_follow_it_too(self):
        plain = roads._street_spec_key(None, None, False)
        self.use({"1": {"lanes": 2}})
        self.assertNotEqual(roads._street_spec_key(None, None, False), plain)
        os.environ.pop(overrides.ENV)
        self.assertEqual(roads._street_spec_key(None, None, False), plain)


if __name__ == "__main__":
    unittest.main()
