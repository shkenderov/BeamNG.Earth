"""Sofia Municipality's terrain model: Sofiaplan's "DTM SO 2024", 1 m bare earth, read from a file on this computer.

Why: Mapterhorn has nothing deeper than z12 for Bulgaria, so Sofia stands on the 30 m global surface model (the
ground filter's version of it): bumpy streets, embankments and underpass ramps smeared away. The municipality
publishes its own terrain model as open data (https://sofiaplan.bg/api/, dataset 656, DTM_SO_2024.zip, 4.4 GB:
"the data can be used freely"). Its host answers no range requests, so the file cannot be read in place over the
network as HRDEM's is: the player downloads it once and names it in BNE_SOFIA_DTM (off without it; the variable is a
build switch like every BNE_*, so the cached roads and tiles made without the file are not reused).

Data (its _readme_so_dtm_1m_2024.md): one Cloud-Optimised GeoTIFF, dtm_so_1m_cog_2024.tif, BigTIFF, 51 722 x 54 343
px at 1 m, 512 px tiles, float32, ZSTD, nodata -9999 (the rectangle outside the municipality's border: about half of
it), 8 overviews (2 .. 256 m); BGS2005 / CCS2005 (EPSG:7801), heights in metres above Baltic 1982 (EPSG:5786).
Interpolated from GIS-Sofia's 1 m terrain contours (2024), not measured per cell: smooth facets between the
contours, crisp where they are drawn (road cuttings and embankments, underpass ramps, the rivers' channels).

    s = SofiaDtm(path)
    got = s.tile(z, x, y)       # None, or (heights float32 [512, 512], valid bool [512, 512]) on dem.py's grid

The contract is hrdem.Hrdem's, so dem.py blends it the same way (feathered into Mapterhorn at its edge, the far
rings shifted to its ground from an overview). Pure numpy: the Lambert projection is the EPSG formula (conformal
conic, two standard parallels, GRS80; tests/test_sofiadtm.py). ZSTD tiles need the zstandard package on Windows
(cog.py: Pillow's libtiff has no ZSTD there).

Horizontal datum: BGS2005 is ETRS89, taken as WGS84 (under a metre apart). Vertical: Baltic 1982 against Mapterhorn's
EGM2008 differs by decimetres: nothing is added, the seams are feathered and the far rings come to it.
"""
import logging
import math
import os
import threading

import numpy as np

import cog as cogmod
import hrdem as hrdemmod

log = logging.getLogger("earthd.sofiadtm")

ENV = "BNE_SOFIA_DTM"       # the file's path; unset or "off": Sofia as before
VERSION = 1                 # in the decoded mirror's folder (dem.Dem._mirror_path): raise it when the heights read change
EPSG = 7801
NODATA_LIMIT = -1000.0      # heights below this are nodata (the file's -9999, whatever the tag says)
TILE = 512                  # dem.py's tile size
TILE_CACHE_BYTES = 96 * 1024 * 1024     # decoded file tiles kept per process (1 MB each)
LATTICE_M = 1.0             # m: the terrain's lattice; z16's 0.9 m pixels read the 1 m image
MAX_LEVEL = 8               # overviews asked down to 2^8 = 256 m pixels

# EPSG:7801: BGS2005 / CCS2005 (Lambert conformal conic, 2 standard parallels, GRS80)
_A = 6378137.0
_F = 1 / 298.257222101
_E = math.sqrt(_F * (2 - _F))
_LAT1, _LAT2 = math.radians(42.0), math.radians(43.0 + 20.0 / 60.0)
_LAT0 = math.radians(42.0 + 40.0 / 60.0 + 4.35138 / 3600.0)
_LON0 = math.radians(25.5)
_FE, _FN = 500000.0, 4725824.3591


def _m(phi):
    return math.cos(phi) / math.sqrt(1 - (_E * math.sin(phi)) ** 2)


def _t(phi):
    s = np.sin(phi)
    return np.tan(np.pi / 4 - phi / 2) / ((1 - _E * s) / (1 + _E * s)) ** (_E / 2)


_N = (math.log(_m(_LAT1)) - math.log(_m(_LAT2))) / (math.log(float(_t(_LAT1))) - math.log(float(_t(_LAT2))))
_FF = _m(_LAT1) / (_N * float(_t(_LAT1)) ** _N)
_R0 = _A * _FF * float(_t(_LAT0)) ** _N


def forward(lat, lon):
    """EPSG:7801 easting / northing (m) of WGS84 lat / lon (degrees; arrays)."""
    phi = np.radians(np.asarray(lat, np.float64))
    lam = np.radians(np.asarray(lon, np.float64))
    r = _A * _FF * _t(phi) ** _N
    th = _N * (lam - _LON0)
    return _FE + r * np.sin(th), _FN + _R0 - r * np.cos(th)


def inverse(x, y):
    """WGS84 lat / lon (degrees) of EPSG:7801 easting / northing (m; arrays)."""
    dx = np.asarray(x, np.float64) - _FE
    dy = _R0 - (np.asarray(y, np.float64) - _FN)
    r = np.hypot(dx, dy)
    t = (r / (_A * _FF)) ** (1 / _N)
    phi = np.pi / 2 - 2 * np.arctan(t)
    for _ in range(8):
        s = np.sin(phi)
        phi = np.pi / 2 - 2 * np.arctan(t * ((1 - _E * s) / (1 + _E * s)) ** (_E / 2))
    return np.degrees(phi), np.degrees(np.arctan2(dx, dy) / _N + _LON0)


class LocalFile:
    """A file on disk as cog.Cog reads its sources: read(start, end, kind) and size."""

    def __init__(self, path):
        self.path = path
        self.size = os.path.getsize(path)
        self._f = open(path, "rb")
        self._lock = threading.Lock()

    def read(self, start, end, kind="demand"):
        end = min(int(end), self.size)
        start = int(start)
        if end <= start:
            return b""
        with self._lock:
            self._f.seek(start)
            return self._f.read(end - start)


class SofiaDtm:
    def __init__(self, path):
        """Raises OSError (no such file) or cog.CogError (a ValueError: not the file this module reads, or its tiles
        cannot be decoded here)."""
        self.path = path
        self.file = LocalFile(path)
        self._tiles = cogmod.TileCache(TILE_CACHE_BYTES)
        self._cogs = {}                     # level -> Cog, or None: no such overview
        self._lock = threading.Lock()
        self._mask = None                   # (valid [h, w], the coarsest overview's Cog): where the file has heights
        self.stats = {"tiles": 0, "empty": 0}
        try:
            c = self._cog(0)
            if c.epsg not in (None, EPSG) or c.sx is None or c.bands != 1:
                raise cogmod.CogError(f"{path}: not the Sofia terrain model (EPSG {c.epsg}, wanted {EPSG})")
            self.levels = 0
            while self.levels < MAX_LEVEL and self._cog(self.levels + 1) is not None:
                self.levels += 1
            # where to ask at all: the raster's box as lat / lon (south, west, north, east), its edges' points
            ex = np.linspace(c.x0, c.x0 + c.width * c.sx, 9)
            ny = np.linspace(c.y0 - c.height * c.sy, c.y0, 9)
            X, Y = np.meshgrid(ex, ny)
            la, lo = inverse(X.ravel(), Y.ravel())
            self.bounds = (float(la.min()), float(lo.min()), float(la.max()), float(lo.max()))
            self.tag = f"{VERSION}-{self.file.size:x}"
            self._valid_mask()              # (decodes a tile: a file whose compression cannot be read fails here)
        except Exception:
            self.close()
            raise

    def close(self):
        self.file._f.close()

    def _cog(self, level):
        with self._lock:
            if level in self._cogs:
                return self._cogs[level]
        try:
            c = cogmod.Cog(self.file, "demand", level=level, tiles=self._tiles)
        except cogmod.CogError as e:
            if level > 0 and "no overview" in str(e):
                c = None
            else:
                raise
        with self._lock:
            self._cogs[level] = c
        return c

    def _valid_mask(self):
        if self._mask is None:
            c = self._cog(self.levels)
            a = c.read_window(0, 0, c.width, c.height).astype(np.float32)
            ok = np.isfinite(a) & (a > NODATA_LIMIT)
            if c.nodata is not None:
                ok &= a != np.float32(c.nodata)
            self._mask = (ok, c)
        return self._mask

    def valid_at(self, lat, lon):
        """Has the file a height at (lat, lon) (the coarsest overview's pixel there: the municipality, not the
        rectangle round it)? A point or arrays."""
        ok, c = self._valid_mask()
        x, y = forward(lat, lon)
        col, row = c.pixel_of(x, y)
        ci = np.floor(col).astype(np.int64)
        ri = np.floor(row).astype(np.int64)
        inside = (ci >= 0) & (ci < c.width) & (ri >= 0) & (ri < c.height)
        out = np.zeros(np.shape(ci), bool)
        if np.ndim(out) == 0:
            return bool(inside and ok[ri, ci])
        out[inside] = ok[ri[inside], ci[inside]]
        return out

    def covers(self, z, x, y):
        """Whether tile (z, x, y) touches the raster's box (where the file is asked at all)."""
        s, w, n, e = hrdemmod.tile_bounds(z, x, y)
        S, W, N, E = self.bounds
        return s < N and n > S and w < E and e > W

    def tile(self, z, x, y, size=TILE, min_px=LATTICE_M):
        """Heights and their validity on dem.py's tile (z, x, y), or None where the file has nothing there."""
        if not self.covers(z, x, y):
            return None
        lat, lon = hrdemmod.tile_lat_lon(z, x, y, size)
        ex, ny = forward(lat, lon)
        # the tile's ground pixel (m) at its centre: the overview that is as fine as that or one finer
        clat = math.radians(float(lat[size // 2, 0]))
        px = 40075016.686 * math.cos(clat) / 2 ** z / size
        level = max(0, min(self.levels, int(math.floor(math.log2(max(px, min_px, 1.0))))))
        c = self._cog(level)
        col, row = c.pixel_of(ex, ny)
        col, row = col - 0.5, row - 0.5                 # pixel centres at integers
        # only the part of the tile over the raster is read (a tile at its edge: not the whole tile's box)
        over = (col > -1.0) & (col < c.width) & (row > -1.0) & (row < c.height)
        if not over.any():
            return None
        c0, r0 = int(math.floor(col[over].min())) - 1, int(math.floor(row[over].min())) - 1
        c1, r1 = int(math.floor(col[over].max())) + 3, int(math.floor(row[over].max())) + 3
        win = c.read_window(c0, r0, c1 - c0, r1 - r0).astype(np.float32)
        bad = ~np.isfinite(win) | (win < NODATA_LIMIT)
        if c.nodata is not None:
            bad |= win == np.float32(c.nodata)
        with self._lock:
            self.stats["tiles"] += 1
        if bad.all():
            with self._lock:
                self.stats["empty"] += 1
            return None
        u, v = col[over] - c0, row[over] - r0
        u0, v0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        fu, fv = (u - u0).astype(np.float32), (v - v0).astype(np.float32)
        a00, a01 = win[v0, u0], win[v0, u0 + 1]
        a10, a11 = win[v0 + 1, u0], win[v0 + 1, u0 + 1]
        good = ~(bad[v0, u0] | bad[v0, u0 + 1] | bad[v0 + 1, u0] | bad[v0 + 1, u0 + 1])
        h = (a00 * (1 - fu) + a01 * fu) * (1 - fv) + (a10 * (1 - fu) + a11 * fu) * fv
        out = np.zeros((size, size), np.float32)
        ok = np.zeros((size, size), bool)
        out[over] = np.where(good, h, 0.0)
        ok[over] = good
        if not ok.any():
            with self._lock:
                self.stats["empty"] += 1
            return None
        return out, ok


_opened = {}                # path -> SofiaDtm, or None: it could not be read (said once per process)
_open_lock = threading.Lock()


def from_env():
    """The process's SofiaDtm of the file BNE_SOFIA_DTM names, or None (unset, "off", or a file that cannot be read:
    a warning, once)."""
    path = os.environ.get(ENV, "").strip().strip('"')
    if not path or path.lower() == "off":
        return None
    path = os.path.abspath(os.path.expanduser(path))
    with _open_lock:
        if path in _opened:
            return _opened[path]
        try:
            got = SofiaDtm(path)
            log.info("Sofia terrain model: %s (%d overviews, lat %.3f..%.3f, lon %.3f..%.3f)", path, got.levels,
                     got.bounds[0], got.bounds[2], got.bounds[1], got.bounds[3])
        except (OSError, ValueError) as e:
            log.warning("Sofia terrain model: %s cannot be read (%s: %s); Sofia keeps the global elevation. "
                        "ZSTD tiles need the zstandard package (server/requirements.txt).", path, type(e).__name__, e)
            got = None
        _opened[path] = got
        return got
