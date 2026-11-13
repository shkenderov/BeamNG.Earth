"""tools/bake_area.py: a bake from a cache folder that holds an earlier bake's plans and facts gives the pack a cold
cache does. The plans and facts read back would not be made again (nor the neighbour cells' facts making them asks
for), so the folder's are dropped first (forget_made) and a pack in it is not read; what is made is recorded once."""
import json
import os
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import bake_area  # noqa: E402


class Feats:
    def __init__(self, cell):
        self.__dict__["_bnge_cell"] = (cell, "roads", "{}")


def stubs(root):
    class PlanCache:
        signature = "w1s1"

        def _path(self, key):
            return os.path.join(root, "run_%d_%d_%d.npz" % tuple(key))

        def store(self, key, plans, keeps, covers, opens, persist=False):
            return True

        def _baked(self, key):
            return b"from a pack"

    def facts_store(feats, kind, extra, data):
        pass

    return (types.SimpleNamespace(PlanCache=PlanCache),
            types.SimpleNamespace(_facts_store=facts_store, _facts_baked=lambda feats, kind, extra: b"[1]"))


class BakeAreaTest(unittest.TestCase):
    def test_forget_made_keeps_the_source_data(self):
        with tempfile.TemporaryDirectory() as cache:
            for d in ("plans", "cellfacts", "roadwidth", "elevation"):
                os.makedirs(os.path.join(cache, "revisit", d, "ab"))
                open(os.path.join(cache, "revisit", d, "ab", "x.zip"), "w").close()
            os.makedirs(os.path.join(cache, "mapdata"))
            bake_area.forget_made(cache)
            left = sorted(os.listdir(os.path.join(cache, "revisit")))
            self.assertEqual(left, ["elevation", "roadwidth"])
            self.assertTrue(os.path.isdir(os.path.join(cache, "mapdata")))
            bake_area.forget_made(os.path.join(cache, "nowhere"))       # (a cold cache: nothing to drop)

    def test_records_made_once_and_reads_no_pack(self):
        with tempfile.TemporaryDirectory() as root:
            pc, rd = stubs(root)
            out = os.path.join(root, "p0w0.jsonl")
            with open(out, "a") as rec:
                bake_area.record(pc, rd, rec)
                cache = pc.PlanCache()
                for _ in range(3):                              # (a cell's plans and facts asked again and again)
                    cache.store((14, 1, 2), [], [], [], [], persist=True)
                    cache.store((14, 1, 3), [], [], [], [], persist=False)      # (incomplete: not recorded)
                    rd._facts_store(Feats("14/1/2"), "ranges", None, {"raw": []})
                    rd._facts_store(Feats("14/1/2"), "specs", (True, False), [None])
                self.assertIsNone(cache._baked((14, 1, 2)))
                self.assertIsNone(rd._facts_baked(Feats("14/1/2"), "ranges", None))
            with open(out) as f:
                got = [json.loads(line) for line in f.read().splitlines()]
            self.assertEqual([("plans" in r, r.get("facts")) for r in got],
                             [(True, None), (False, "ranges"), (False, "specs")])


if __name__ == "__main__":
    unittest.main()
