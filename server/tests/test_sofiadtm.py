"""Sofia's terrain model (sofiadtm.py, 2026-11-10): the EPSG:7801 projection, cog.py's ZSTD tiles through the
zstandard package, the tiles read from a small file written here (BigTIFF, ZSTD, an overview, nodata), and
dem.Dem's use of it (the near tiles, the far rings' shift, coarse_at, the decoded mirror's folder). Off without
BNE_SOFIA_DTM."""
import math
import os
import struct
import sys
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cog  # noqa: E402
import credits  # noqa: E402
import dem as demmod  # noqa: E402
import hrdem  # noqa: E402
import rangeread  # noqa: E402
import sofiadtm  # noqa: E402

try:
    import zstandard
except ImportError:
    zstandard = None

X0, Y0 = 320000.0, 4732000.0        # the fixture's north-west corner (EPSG:7801): Sofia's centre, about
W, H, T = 1024, 768, 256
NODATA = -9999.0


def plane(x, y):
    """The fixture's terrain: a tilted plane (m) of EPSG:7801 coordinates."""
    return 550.0 + 0.01 * (np.asarray(x) - X0) + 0.02 * (np.asarray(y) - (Y0 - H))


def fixture_heights():
    c, r = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    h = plane(X0 + c, Y0 - r).astype(np.float32)
    h[:200, :300] = NODATA              # outside the "municipality": its north-west corner
    return h


def write_tiff(levels, nodata=NODATA, epsg=sofiadtm.EPSG, compress=True):
    """A BigTIFF of ZSTD-compressed 256 px float32 tiles: levels[0] the image (1 m pixels at X0, Y0), the others
    its overviews; bytes. The tags cog.py reads, as GDAL's COG driver writes them."""
    cctx = zstandard.ZstdCompressor(level=3)
    out = bytearray(b"II" + struct.pack("<HHHQ", 43, 8, 0, 0))     # (the first IFD's offset patched below)
    tiles = []
    for a in levels:
        h, w = a.shape
        offs, cnts = [], []
        for ty in range(-(-h // T)):
            for tx in range(-(-w // T)):
                t = np.full((T, T), nodata, np.float32)
                part = a[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T]
                t[:part.shape[0], :part.shape[1]] = part
                data = cctx.compress(t.tobytes()) if compress else t.tobytes()
                offs.append(len(out))
                cnts.append(len(data))
                out += data
        tiles.append((offs, cnts))
    nd = (repr(float(nodata)) + "\0").encode()
    geokeys = (1, 1, 0, 1, 3072, 0, 1, epsg)
    prev_next = 8                       # where the offset of the next IFD goes
    for k, (a, (offs, cnts)) in enumerate(zip(levels, tiles)):
        h, w = a.shape
        ents = [(256, 4, [w]), (257, 4, [h]), (258, 3, [32]), (259, 3, [50000 if compress else 1]), (262, 3, [1]),
                (277, 3, [1]), (322, 3, [T]), (323, 3, [T]), (324, 16, offs), (325, 16, cnts), (339, 3, [3]),
                (42113, 2, nd)]
        if k == 0:
            ents += [(33550, 12, [1.0, 1.0, 0.0]), (33922, 12, [0.0, 0.0, 0.0, X0, Y0, 0.0]), (34735, 3, geokeys)]
        else:
            ents.append((254, 4, [1]))
        ents.sort()
        fmt = {2: "s", 3: "H", 4: "I", 12: "d", 16: "Q"}
        size = {2: 1, 3: 2, 4: 4, 12: 8, 16: 8}
        ifd_at = len(out)
        out[prev_next:prev_next + 8] = struct.pack("<Q", ifd_at)
        extra_at = ifd_at + 8 + 20 * len(ents) + 8
        body, extra = bytearray(struct.pack("<Q", len(ents))), bytearray()
        for tag, typ, val in ents:
            raw = val if typ == 2 else struct.pack("<" + fmt[typ] * len(val), *val)
            cnt = len(val)
            if len(raw) <= 8:
                body += struct.pack("<HHQ", tag, typ, cnt) + raw + b"\0" * (8 - len(raw))
            else:
                body += struct.pack("<HHQQ", tag, typ, cnt, extra_at + len(extra))
                extra += raw
        prev_next = ifd_at + len(body)
        body += struct.pack("<Q", 0)
        out += body + extra
    return bytes(out)


def fixture_file(folder):
    h = fixture_heights()
    half = h[::2, ::2].copy()           # (an overview's pixels: every other one, enough for these tests)
    path = os.path.join(folder, "dtm.tif")
    with open(path, "wb") as f:
        f.write(write_tiff([h, half]))
    return path


def tile_of(lat, lon, z):
    n = 2 ** z
    return z, int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


class ProjTest(unittest.TestCase):
    def test_false_origin(self):
        x, y = sofiadtm.forward(42.0 + 40.0 / 60.0 + 4.35138 / 3600.0, 25.5)
        self.assertAlmostEqual(float(x), 500000.0, places=6)
        self.assertAlmostEqual(float(y), 4725824.3591, places=5)

    def test_round_trip(self):
        lat = np.array([42.40, 42.5637, 42.6965, 42.90, 42.65])
        lon = np.array([23.06, 23.2783, 23.3215, 23.68, 23.37])
        la, lo = sofiadtm.inverse(*sofiadtm.forward(lat, lon))
        np.testing.assert_allclose(la, lat, atol=1e-9)
        np.testing.assert_allclose(lo, lon, atol=1e-9)

    def test_true_scale_on_the_standard_parallels(self):
        # a conformal conic's scale is 1 on its two standard parallels (42 N and 43 20' N), under 1 between them:
        # a step along the parallel there is the ellipsoid's own arc (independent of this module's constants)
        a, f = 6378137.0, 1 / 298.257222101
        e2 = f * (2 - f)
        d = 1e-4
        for lat, want in ((42.0, 1.0), (43.0 + 20.0 / 60.0, 1.0)):
            x0, y0 = sofiadtm.forward(lat, 23.3)
            x1, y1 = sofiadtm.forward(lat, 23.3 + d)
            arc = a * math.cos(math.radians(lat)) / math.sqrt(1 - e2 * math.sin(math.radians(lat)) ** 2) * math.radians(d)
            self.assertAlmostEqual(float(np.hypot(x1 - x0, y1 - y0)) / arc, want, places=7)
        x0, y0 = sofiadtm.forward(42.67, 23.3)
        x1, y1 = sofiadtm.forward(42.67, 23.3 + d)
        arc = a * math.cos(math.radians(42.67)) / math.sqrt(1 - e2 * math.sin(math.radians(42.67)) ** 2) * math.radians(d)
        k = float(np.hypot(x1 - x0, y1 - y0)) / arc
        self.assertTrue(0.9999 < k < 1.0, k)

    def test_sofia_is_where_the_file_says(self):
        # the published file's box (x 299392..351114, y 4699268..4753611) is Sofia Municipality
        la, lo = sofiadtm.inverse(np.array([299392.0, 351114.0]), np.array([4699268.0, 4753611.0]))
        np.testing.assert_allclose(la, [42.4028, 42.9036], atol=5e-4)
        np.testing.assert_allclose(lo, [23.0628, 23.6767], atol=5e-4)


@unittest.skipUnless(zstandard, "the zstandard package")
class CogZstdTest(unittest.TestCase):
    def test_tiles_and_overview(self):
        h = fixture_heights()
        data = write_tiff([h, h[::2, ::2].copy()])
        c = cog.Cog(rangeread.MemoryRange(data))
        self.assertTrue(c.big)
        self.assertEqual((c.width, c.height, c.epsg, c.nodata), (W, H, sofiadtm.EPSG, NODATA))
        self.assertEqual(c.compression_code, 50000)
        np.testing.assert_array_equal(c.read_window(250, 190, 300, 280), h[190:470, 250:550])
        o = cog.Cog(rangeread.MemoryRange(data), level=1)
        self.assertEqual((o.sx, o.sy), (2.0, 2.0))
        np.testing.assert_array_equal(o.read_window(0, 0, W // 2, H // 2), h[::2, ::2])

    def test_a_damaged_block_is_an_error(self):
        with self.assertRaises(cog.CogError):
            cog._zstd(b"not a zstd frame", 1024)


@unittest.skipUnless(zstandard, "the zstandard package")
class TileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = sofiadtm.SofiaDtm(fixture_file(self.tmp.name))

    def tearDown(self):
        self.src.file._f.close()
        self.tmp.cleanup()

    def ll(self, x, y):
        la, lo = sofiadtm.inverse(x, y)
        return float(la), float(lo)

    def test_opened(self):
        self.assertEqual(self.src.levels, 1)
        s, w, n, e = self.src.bounds
        la, lo = self.ll(X0 + W / 2, Y0 - H / 2)
        self.assertTrue(s < la < n and w < lo < e)
        self.assertIn(f"{sofiadtm.VERSION}-", self.src.tag)

    def test_tile_is_the_terrain(self):
        la, lo = self.ll(X0 + 700, Y0 - 500)
        key = tile_of(la, lo, 18)                           # (~115 m across: inside the file)
        got = self.src.tile(*key)
        self.assertIsNotNone(got)
        h, ok = got
        self.assertTrue(ok.all())
        lat, lon = hrdem.tile_lat_lon(*key)
        np.testing.assert_allclose(h, plane(*sofiadtm.forward(lat, lon)), atol=2e-3)

    def test_nodata_and_the_edge_are_not_valid(self):
        la, lo = self.ll(X0 + 300, Y0 - 200)                # the nodata corner's corner
        h, ok = self.src.tile(*tile_of(la, lo, 18))
        self.assertTrue(ok.any() and not ok.all())
        lat, lon = hrdem.tile_lat_lon(*tile_of(la, lo, 18))
        x, y = sofiadtm.forward(lat, lon)
        in_corner = (x < X0 + 299) & (y > Y0 - 199)
        self.assertFalse(ok[in_corner].any(), "no height where the file has none")
        np.testing.assert_allclose(h[ok], plane(x, y)[ok], atol=2e-3)
        z, tx, ty = tile_of(*self.ll(X0 + W / 2, Y0 - H / 2), 14)   # a tile far larger than the file
        h, ok = self.src.tile(z, tx, ty)
        self.assertTrue(ok.any() and not ok.all())
        self.assertIsNone(self.src.tile(*tile_of(la + 1.0, lo, 16)), "a degree north: not asked")

    def test_coarse_tile_reads_the_overview(self):
        z, tx, ty = tile_of(*self.ll(X0 + W / 2, Y0 - H / 2), 13)
        h, ok = self.src.tile(z, tx, ty, size=64, min_px=32.0)
        lat, lon = hrdem.tile_lat_lon(z, tx, ty, 64)
        self.assertTrue(ok.any())
        np.testing.assert_allclose(h[ok], plane(*sofiadtm.forward(lat, lon))[ok], atol=0.05)   # (2 m pixels' halves)

    def test_valid_at(self):
        self.assertTrue(self.src.valid_at(*self.ll(X0 + 700, Y0 - 500)))
        self.assertFalse(self.src.valid_at(*self.ll(X0 + 100, Y0 - 100)), "the nodata corner")
        self.assertFalse(self.src.valid_at(*self.ll(X0 - 500, Y0 - 500)), "outside the file")
        la, lo = sofiadtm.inverse(np.array([X0 + 700, X0 + 100]), np.array([Y0 - 500, Y0 - 100]))
        self.assertEqual(self.src.valid_at(la, lo).tolist(), [True, False])


class StubSofia:
    tag = "1-test"

    def __init__(self, h, ok, valid=True, fail=False):
        self.h, self.ok, self.valid, self.fail, self.asked = h, ok, valid, fail, []

    def covers(self, z, x, y):
        return True

    def valid_at(self, lat, lon):
        return self.valid

    def tile(self, z, x, y, size=512, min_px=1.0):
        self.asked.append((z, size, min_px))
        if self.fail:
            raise OSError("unreadable")
        if size != 512:
            return np.full((size, size), 540.0, np.float32), np.ones((size, size), bool)
        return self.h.copy(), self.ok.copy()


class DemTest(unittest.TestCase):
    def make(self, sofia):
        d = demmod.Dem.__new__(demmod.Dem)
        d._lock = threading.Lock()
        d._local = threading.local()
        d.stats = Counter()
        d.lidarbc = d.hrdem = d.lidarbc_sparse = None
        d.lidarbc_served, d.hrdem_served, d.sofia_served = set(), set(), set()
        d.sofia = sofia
        d.cache_dir = "cache"
        return d

    KEY = tile_of(42.6965, 23.3215, 16)

    def test_near_tile_is_the_terrain_model_feathered_at_its_edge(self):
        a = np.full((512, 512), 550.0, np.float32)
        ok = np.zeros((512, 512), bool)
        ok[:, 256:] = True
        d = self.make(StubSofia(np.full((512, 512), 545.0, np.float32), ok))
        out = d._canada(self.KEY, a)
        self.assertAlmostEqual(float(out[100, 500]), 545.0, places=4)
        self.assertAlmostEqual(float(out[100, 100]), 550.0, places=4)
        row = out[200, 250:340]
        self.assertTrue(np.all(np.diff(row) <= 1e-5), "a slope from 550 down to 545, no step")
        self.assertLess(float(row[6] - row[7]), 1.0)
        self.assertIn(self.KEY, d.sofia_served)
        self.assertIn("dem:sofiaplan", credits.SHORT)

    def test_far_tile_is_shifted_not_replaced(self):
        z, x, y = self.KEY
        key = (13, x >> 3, y >> 3)
        a = (555.0 + np.linspace(0, 3, 512)[None, :] * np.ones((512, 1))).astype(np.float32)   # the global model: 15+ m up
        src = StubSofia(None, None)
        out = self.make(src)._canada(key, a)
        self.assertEqual(src.asked, [(13, demmod.HRDEM_BIAS_PX, demmod.HRDEM_BIAS_M)])
        # at the terrain model's level (540 m): the smooth offset between them taken off, the tile's own detail kept
        self.assertLess(float(np.abs(out - 540.0).max()), 0.5)
        self.assertEqual(self.make(src)._canada((11, x >> 5, y >> 5), a) is a, True, "z11 and coarser: as they are")

    def test_failure_keeps_the_heights_and_skips_the_mirror(self):
        a = np.full((512, 512), 550.0, np.float32)
        d = self.make(StubSofia(None, None, fail=True))
        d._local.partial = False
        np.testing.assert_array_equal(d._canada(self.KEY, a), a)
        self.assertTrue(d._local.partial)

    def test_off_without_the_file(self):
        a = np.full((512, 512), 550.0, np.float32)
        d = self.make(None)
        self.assertIs(d._canada(self.KEY, a), a)
        z, x, y = self.KEY
        self.assertIn(os.path.join("decoded", f"v{demmod.DECODED_VERSION}", "g1", str(z)), d._mirror_path(self.KEY))

    def test_mirror_folder_of_its_own(self):
        d = self.make(StubSofia(None, None))
        self.assertIn("g1-sofia1-test", d._mirror_path(self.KEY))
        z, x, y = self.KEY
        self.assertNotIn("sofia", d._mirror_path((11, x >> 5, y >> 5)), "tiles it never changes: the shared ones")

    def test_not_coarse_where_it_has_heights(self):
        d = self.make(StubSofia(None, None, valid=True))
        d._coverage = lambda: {}                # (Mapterhorn: nothing deeper than z12 anywhere)
        self.assertFalse(d.coarse_at(42.6965, 23.3215))
        d = self.make(StubSofia(None, None, valid=False))
        d._coverage = lambda: {}
        self.assertTrue(d.coarse_at(42.6965, 23.3215), "outside the municipality: the global model's rules")


class EnvTest(unittest.TestCase):
    def setUp(self):
        self.old = os.environ.get(sofiadtm.ENV)
        sofiadtm._opened.clear()

    def tearDown(self):
        if self.old is None:
            os.environ.pop(sofiadtm.ENV, None)
        else:
            os.environ[sofiadtm.ENV] = self.old
        sofiadtm._opened.clear()

    def test_unset_and_off(self):
        os.environ.pop(sofiadtm.ENV, None)
        self.assertIsNone(sofiadtm.from_env())
        os.environ[sofiadtm.ENV] = "off"
        self.assertIsNone(sofiadtm.from_env())

    def test_a_missing_file_is_a_warning(self):
        os.environ[sofiadtm.ENV] = os.path.join(tempfile.gettempdir(), "no_such_sofia_dtm.tif")
        with self.assertLogs("earthd.sofiadtm", "WARNING"):
            self.assertIsNone(sofiadtm.from_env())
        self.assertIsNone(sofiadtm.from_env(), "said once")

    @unittest.skipUnless(zstandard, "the zstandard package")
    def test_the_file_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[sofiadtm.ENV] = fixture_file(tmp)
            s = sofiadtm.from_env()
            try:
                self.assertIsInstance(s, sofiadtm.SofiaDtm)
                self.assertIs(sofiadtm.from_env(), s, "one per process")
            finally:
                s.file._f.close()

    @unittest.skipUnless(zstandard, "the zstandard package")
    def test_another_projection_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "utm.tif")
            with open(path, "wb") as f:
                f.write(write_tiff([fixture_heights()], epsg=32634))
            os.environ[sofiadtm.ENV] = path
            with self.assertLogs("earthd.sofiadtm", "WARNING"):
                self.assertIsNone(sofiadtm.from_env())


if __name__ == "__main__":
    unittest.main()
