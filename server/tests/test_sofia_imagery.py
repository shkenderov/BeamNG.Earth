"""GIS-Sofia's 2024 orthophoto (imagery_regional.SOFIA_ORTHO, BNE_SOFIA_ORTHO; 2026-11-10) through the real imagery
cache and fallback chain: a WMS that answers CRS:84 alone (the tile's extent longitude first), black outside its
raster, asked only inside its box, and not a source at all without the switch."""
import io
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import credits  # noqa: E402
import geodesy  # noqa: E402
import imagery  # noqa: E402
import imagery_regional  # noqa: E402
import maptiles  # noqa: E402
import netfetch  # noqa: E402

SERVER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CENTRE = (42.6965, 23.3215)


class SofiaTransport:
    """The municipality's WMS (a photo, or all black: outside its raster) and a stand-in for the next source."""

    def __init__(self, photo=True, colour=(60, 110, 160)):
        self.photo, self.colour, self.requests = photo, colour, []

    def request(self, url, headers, timeout):
        query = parse_qs(urlsplit(url).query)
        if "gis-sofia" in url:
            assert url.startswith("http://nospire.gis-sofia.bg/open_wms?"), url
            assert query["LAYERS"] == ["aerophoto_2024"]
            assert query["CRS"] == ["CRS:84"]
            assert query["FORMAT"] == ["image/jpeg"]
            assert query["BGCOLOR"] == ["0x000000"]
            w, s, e, n = map(float, query["BBOX"][0].split(","))
            assert 23.0 < w < e < 23.7 and 42.5 < s < n < 42.9, "longitude first, a box in Sofia"
            size = int(query["WIDTH"][0])
            self.requests.append("sofia")
            pixels = np.zeros((size, size, 3), np.uint8)
            if self.photo:
                pixels[:] = self.colour
                pixels[::2, ::2] += 4
        else:
            size = 256
            self.requests.append("other")
            pixels = np.full((size, size, 3), (50, 150, 100), np.uint8)
            pixels[::2, ::2] += 4
        out = io.BytesIO()
        Image.fromarray(pixels).save(out, "JPEG", quality=95)
        return 200, {}, out.getvalue()


def sofia_source():
    url, attribution, bounds, kind, leaf_off = imagery_regional.SOFIA_ORTHO
    return replace(imagery.SOURCES["poland"], name="bg_sofia", url=url, attribution=attribution, bounds=bounds,
                   leaf_off=leaf_off)


class SofiaImageryTest(unittest.TestCase):
    def make(self, photo=True):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        transport = SofiaTransport(photo)
        policies = {h: replace(p, rate=1e9, burst=1e9) for h, p in netfetch.POLICIES.items()}
        policies.setdefault("nospire.gis-sofia.bg", replace(policies["mapy.geoportal.gov.pl"]))
        fetcher = netfetch.Fetcher(transport=transport, policies=policies, sleep=lambda _: None)
        # (the next source: the world's Sentinel-2, as in the default chain; Spain's box does not reach Bulgaria)
        im = imagery.Imagery(temp.name, (sofia_source(), "eox2023"), fetcher=fetcher, tone=0)
        self.addCleanup(im.close)
        return im, transport

    def warm(self, im, lat, lon):
        frame = geodesy.EnuFrame(lat, lon)
        im.warm(frame, [(0, 0)], 150, 1)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with im._warm_cv:
                if all(im._warm_done(k) for k in im._warm_want):
                    return frame
            time.sleep(.01)
        self.fail("the warm fetch did not finish")

    def map_source(self, im, lat, lon):
        z = 17
        x = (lon + 180) / 360 * (1 << z)
        y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * (1 << z)
        got = maptiles.MapTiles(im).tile(z, int(x), int(y))
        self.assertIsNotNone(got)
        return got[1]

    def test_the_source_as_declared(self):
        url, attribution, bounds, kind, _leaf = imagery_regional.SOFIA_ORTHO
        self.assertEqual(kind, "export")
        self.assertIn("{bbox84}", url)
        self.assertNotIn("{bbox}", url)
        self.assertTrue(attribution.isascii())
        s, w, n, e = bounds
        self.assertTrue(s < CENTRE[0] < n and w < CENTRE[1] < e)
        self.assertEqual(imagery_regional.PLACE["bg_sofia"], "Sofia")
        self.assertEqual(credits.SHORT["img:bg_sofia"], "GIS-Sofia")

    def test_photo_used_by_ground_and_map(self):
        im, transport = self.make()
        frame = self.warm(im, *CENTRE)
        _, used = im._render(imagery.EnuSquare(frame, -64, -64, 128), 128)
        self.assertIn("bg_sofia", used)
        self.assertEqual(self.map_source(im, *CENTRE), "bg_sofia")
        self.assertIn("sofia", transport.requests)

    def test_black_outside_its_raster_falls_through_and_stays_cached(self):
        im, transport = self.make(photo=False)
        self.warm(im, *CENTRE)
        self.assertNotEqual(self.map_source(im, *CENTRE), "bg_sofia")
        self.assertIn("sofia", transport.requests)
        before = transport.requests.count("sofia")
        self.assertNotEqual(self.map_source(im, *CENTRE), "bg_sofia")
        self.assertEqual(transport.requests.count("sofia"), before, "asked once: the miss is remembered")

    def test_not_asked_outside_its_box(self):
        im, transport = self.make()
        self.warm(im, 42.1354, 24.7453)                     # Plovdiv
        self.map_source(im, 42.1354, 24.7453)
        self.assertNotIn("sofia", transport.requests)

    def test_off_without_the_switch(self):
        code = ("import sys; sys.path.insert(0, r'%s'); import imagery, imagery_regional, netfetch, credits;"
                "print('bg_sofia' in imagery_regional.REGIONAL, 'bg_sofia' in imagery.DEFAULT_CHAIN,"
                " 'nospire.gis-sofia.bg' in netfetch.POLICIES, 'bg_sofia' in credits.IMAGERY)" % SERVER)
        for value, want in ((None, "False False False False"), ("on", "True True True True")):
            env = {k: v for k, v in os.environ.items() if k != imagery_regional.SOFIA_ORTHO_ENV}
            if value:
                env[imagery_regional.SOFIA_ORTHO_ENV] = value
            got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            self.assertEqual(got.stdout.strip(), want, got.stderr[-800:])

    def test_crs84_box(self):
        mb = imagery.mercator_bbox(15, 18506, 12078)        # (a z15 tile over Sofia)
        w, s, e, n = map(float, imagery.lon_lat_bbox(mb).split(","))
        s4, w4, n4, e4 = map(float, imagery.lat_lon_bbox(mb).split(","))
        self.assertEqual((w, s, e, n), (w4, s4, e4, n4), "the same numbers, longitude first")
        self.assertTrue(w < e and s < n)
        self.assertTrue(23.2 < w < 23.4 and 42.6 < s < 42.8, (w, s))


if __name__ == "__main__":
    unittest.main()
