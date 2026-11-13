"""tools/bake_area.py: a bake from a cache folder that holds an earlier bake's plans and facts gives the pack a cold
cache does. The plans and facts read back would not be made again (nor the neighbour cells' facts making them asks
for), so the folder's are dropped first (forget_made) and a pack in it is not read; what is made is recorded once. A bake that fails (no items, or workers raised on cells no pass planned) writes no pack
and exits non-zero, naming the cells."""
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


class FakePool:
    """multiprocessing.get_context("spawn").Pool(n) as a context manager running the jobs here."""

    def __init__(self, n):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, jobs):
        return [fn(j) for j in jobs]


class FakeContext:
    Pool = FakePool


class BakeOutcomeTest(unittest.TestCase):
    """What a bake does when the workers fail: no pack, and a non-zero exit saying which cells (a bake that planned
    nothing used to write an empty pack and exit 0)."""

    BBOX = "42.680,23.310,42.690,23.320"

    def run_bake(self, behaviour):
        """behaviour(cell, pass_) -> None (plans made), "raise" (the worker raises on it) or "skip" (nothing, no raise).
        Returns (the SystemExit's text or None, the pack's path, the folder's file names)."""
        import mapcells
        root = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        cache, out = os.path.join(root, "cache"), os.path.join(root, "p.bnge")
        plan = os.path.join(root, "plan.npz")
        with open(plan, "wb") as f:
            f.write(b"plans")

        def worker(args):
            _cache, cells, out_jsonl, _wid = args
            pass_ = int(os.path.basename(out_jsonl)[1])
            errors = []
            with open(out_jsonl, "a") as rec:
                for z, x, y, _la, _lo in cells:
                    what = behaviour((z, x, y), pass_)
                    if what == "raise":
                        errors.append([z, x, y, "ValueError: boom"])
                    elif what is None:
                        rec.write(json.dumps({"plans": [z, x, y], "sig": "w1s1", "path": plan}) + "\n")
            return {"mapdata": 1}, errors

        argv, old = sys.argv, (bake_area._worker, bake_area.multiprocessing.get_context)
        sys.argv = ["bake_area.py", "--bbox", self.BBOX, "--cache", cache, "--out", out, "--procs", "2"]
        bake_area._worker = worker
        bake_area.multiprocessing.get_context = lambda _name: FakeContext
        say = None
        try:
            try:
                bake_area.main()
            except SystemExit as ex:
                say = str(ex.code)
        finally:
            sys.argv = argv
            bake_area._worker, bake_area.multiprocessing.get_context = old
        keys = mapcells.cells_for_bbox(*[float(v) for v in self.BBOX.split(",")], max_cells=200000)
        self.cells = [(k.z, k.x, k.y) for k in keys]
        return say, out, sorted(n for n in os.listdir(root) if n.startswith("p.bnge"))

    def test_a_bake_that_works_writes_its_pack(self):
        say, out, files = self.run_bake(lambda cell, pass_: None)
        self.assertIsNone(say)
        self.assertEqual(files, ["p.bnge"])
        import bakepack
        self.assertEqual(len(bakepack.Pack(out).items), len(self.cells))

    def test_a_bake_with_no_items_writes_no_pack_and_fails(self):
        say, out, files = self.run_bake(lambda cell, pass_: "skip")
        self.assertIn("no pack written", say)
        self.assertIn("no items", say)
        self.assertEqual(files, [])

    def test_cells_a_worker_raised_on_fail_the_bake_and_are_named(self):
        bad = {}

        def behaviour(cell, pass_):
            bad.setdefault("cell", cell)
            return "raise" if cell == bad["cell"] else None
        say, out, files = self.run_bake(behaviour)
        z, x, y = bad["cell"]
        self.assertIn("no pack written", say)
        self.assertIn(f"{z}/{x}/{y} (ValueError: boom)", say)
        self.assertEqual(files, [])

    def test_a_cell_a_later_pass_planned_is_no_failure(self):
        bad = {}

        def behaviour(cell, pass_):
            bad.setdefault("cell", cell)
            return "raise" if cell == bad["cell"] and pass_ == 0 else None
        say, out, files = self.run_bake(behaviour)
        self.assertIsNone(say)
        self.assertEqual(files, ["p.bnge"])

    def test_cells_left_incomplete_without_a_raise_are_still_the_clients_to_plan(self):
        bad = {}

        def behaviour(cell, pass_):
            bad.setdefault("cell", cell)
            return "skip" if cell == bad["cell"] else None
        say, out, files = self.run_bake(behaviour)
        self.assertIsNone(say)
        import bakepack
        self.assertEqual(len(bakepack.Pack(out).items), len(self.cells) - 1)

    def test_fatal_problems(self):
        todo = [(14, 1, 2, 0, 0), (14, 1, 3, 0, 0)]
        self.assertEqual(bake_area.fatal_problems([], {}, 5), [])
        self.assertEqual(bake_area.fatal_problems(todo, {}, 5), [])                    # (short, no raise: the client's)
        self.assertEqual(bake_area.fatal_problems([], {(14, 1, 2): "x"}, 5), [])        # (raised, planned later)
        p = bake_area.fatal_problems(todo, {(14, 1, 3): "KeyError: 'a'"}, 5)
        self.assertEqual(len(p), 1)
        self.assertIn("14/1/3 (KeyError: 'a')", p[0])
        self.assertNotIn("14/1/2", p[0])
        self.assertEqual(len(bake_area.fatal_problems([], {}, 0)), 1)
        self.assertEqual(len(bake_area.fatal_problems(todo, {(14, 1, 2): "x"}, 0)), 2)


if __name__ == "__main__":
    unittest.main()
