"""Sofia's streets as the municipality surveyed them: GIS-Sofia's open dataset "Улици и тротоари" (streets and
sidewalks; https://urbandata.sofia.bg/dataset/streets_curbs-zip, CC BY, current 2026-08-28), read from a file on this
computer.

Why: a road's carriageway width is a guess here (its class, a lane count where OpenStreetMap has one: 37 % of
Sofia's segments, a building-to-building measurement elsewhere), made per map segment: streets narrowed to one lane
and widened again from one junction to the next, a residential street was drawn 10.6 m wide (6 m and parking both
sides) where its carriageway is 5.8 m, and lamps and poles stood on carriageways the guess did not know (the player,
2026-11-10). The dataset has the outline of every carriageway, sidewalk, parking bay, median and tram bed of Sofia
Municipality: 91 244 polygons in WGS84 with a type (29 values), the street's name, the district and the area.

    s = from_env()                              # BNE_SOFIA_STREETS names streets_curbs.geojson (off without it)
    m = s.measure_cell(key, lines, oneway)      # per road line of a map cell: Measure or None
    s.blocked(lat, lon)                         # is a point on a carriageway, a parking bay, a tram bed, a bike lane

The file is parsed once (json: ~3 s, ~800 MB at its peak) into <file>.v<VERSION>.npz beside it (10 MB, 0.1 s to
load); every process reads that. Per map cell (mapcells, z14) the polygons are rasterised by kind on a RES_M grid in
the cell's own ENU frame (the frame the cell's road features carry as `en`), MARGIN_M past its box, and kept (an
LRU). Pure numpy and Pillow.

The width of a road (measure_cell): at stations every STEP_M along its line, END_SKIP_M in from its ends (the
junctions: their carriageways merge), the carriageway's run across the road through the station (the line may lie
SNAP_M beside it: the map's lines are 0.75 m off the carriageway's middle at the median, 1.9 m at the 90th
percentile). Where another road's line runs along inside the same run (a divided road drawn as one carriageway:
42 % of Sofia's one-way secondary roads), the run ends half way to it. The road's width is the median of its
stations', given when MIN_STATIONS of them have one. Per side: whether a sidewalk lies within WALK_REACH_M past the
carriageway's edge, and whether a parking bay adjoins it. Everything is a function of the cell's polygons and road
lines alone: measure a road with its home cell's lines and every cell and process gets the same answer.
"""
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict, namedtuple

import numpy as np

import geodesy
import mapcells

log = logging.getLogger("earthd.sofiastreets")

ENV = "BNE_SOFIA_STREETS"   # streets_curbs.geojson's path (or its .npz); unset or "off": the roads as before
VERSION = 1                 # the .npz's layout and the kinds below: a new number is a new file
MEASURE_VERSION = 1         # measure_cell's rules and constants: a new number where its answers change (roads.py keeps
                            # each cell's on disk under it)
RES_M = 0.5                 # m: the rasters' pixels (the polygons' corners are surveyed to centimetres)
MARGIN_M = 150.0            # m past a cell's box its raster reaches (a road's line runs on past the cell)
RASTERS_MAX = 3             # cells' rasters kept (18 MB each)
STEP_M = 4.0                # m between a road's stations
END_SKIP_M = 12.0           # m from a road's ends without stations (a third of a shorter road's length)
RAY_M = 30.0                # m either side of the line a station looks
RAY_STEP_M = 0.25
SNAP_M = 3.0                # m beside its carriageway a road's line may lie
MIN_STATIONS = 3
MIN_SHARE = 0.4             # ... and this share of its stations
WIDTH_MIN_M, WIDTH_MAX_M = 2.5, 40.0
TWIN_COS = math.cos(math.radians(30.0))     # another line within 30 degrees of the road's: running along it
TWIN_MIN_M = 2.0            # m: a line nearer than this is the road's own next piece, not a twin
WALK_REACH_M = 3.0          # m past the carriageway's edge a sidewalk counts as the road's
PARK_REACH_M = 0.75         # m past the edge a parking bay adjoins it
SIDE_SHARE = 0.5            # a side has a sidewalk / parking when this share of its stations do
PROP_MARGIN_M = 0.3         # m: blocked()'s reach round a point (a pole's foot)

# the kinds of ground, a bit each in a cell's raster (uint8)
CARRIAGEWAY, BRIDGE, TUNNEL, PARKING, SIDEWALK, WALK_PARKING, TRACK, OTHER = (1 << k for k in range(8))
KINDS = {
    "улично платно": CARRIAGEWAY, "улично платно без покритие": CARRIAGEWAY, "улично платно на обръщало": CARRIAGEWAY,
    "тупик с паркинг": CARRIAGEWAY,
    "улично платно на мост": CARRIAGEWAY | BRIDGE, "улично платно в тунел": CARRIAGEWAY | TUNNEL,
    "паркинг на платното": PARKING, "паркинг на тротоара": WALK_PARKING,
    "тротоар": SIDEWALK, "тротоар без покритие": SIDEWALK, "тротоар на мост": SIDEWALK, "тротоар в тунел": SIDEWALK,
    "тротоар на обръщало": SIDEWALK, "тротоар стълбище": SIDEWALK,
    "трамвайно трасе": TRACK, "трамвайно трасе на мост": TRACK, "велоалея": TRACK, "велоалея на мост": TRACK,
}                           # (the rest: medians, green strips, pedestrian streets, walls ...: OTHER)
NO_PROPS = CARRIAGEWAY | PARKING | TRACK        # blocked(): no lamp or pole stands on these

Measure = namedtuple("Measure", "width stations spread walk_left walk_right park_left park_right park_m "
                                "bridge tunnel shared")
Measure.__doc__ = """A road's carriageway as surveyed: width (m, the median of its stations'), stations (how many
had one), spread (m, their interquartile range), walk_left / walk_right (a sidewalk along that side: left of the
line's direction), park_left / park_right (parking bays adjoining that side), park_m (their depth across, the median;
0 without), bridge / tunnel (the share of its stations on a bridge's / a tunnel's carriageway), shared (the share
whose carriageway is another line's too: cut half way to it)."""


# ---- the file ------------------------------------------------------------------------------------------------

def npz_path(path):
    """The .npz made of a .geojson path (path itself when it is an .npz already)."""
    return path if path.lower().endswith(".npz") else f"{path}.v{VERSION}.npz"


def build(geojson, out=None):
    """Parse the GeoJSON into out (default npz_path(geojson)), atomically; returns out. Polygons and
    MultiPolygons; every ring closed; a feature without rings or with a type that is no string is left out."""
    out = out or npz_path(geojson)
    with open(geojson, encoding="utf-8-sig") as f:
        doc = json.load(f)
    feats = doc.get("features") if isinstance(doc, dict) else None
    if not isinstance(feats, list):
        raise ValueError(f"{geojson}: not a GeoJSON FeatureCollection")
    kind, ring_off, coord_off, coords = [], [0], [0], []
    for f in feats:
        g = f.get("geometry") if isinstance(f, dict) else None
        p = (f.get("properties") or {}) if isinstance(f, dict) else {}
        if not isinstance(g, dict):
            continue
        if g.get("type") == "Polygon":
            polys = [g.get("coordinates")]
        elif g.get("type") == "MultiPolygon":
            polys = g.get("coordinates") or []
        else:
            continue
        bits = KINDS.get(str(p.get("type") or "").strip(), OTHER)
        for rings in polys:                 # (a MultiPolygon's parts: polygons of their own, the same kind)
            n0 = len(coord_off)
            for r in rings or ():
                try:
                    a = np.asarray(r, np.float64)[:, :2]
                except (ValueError, IndexError, TypeError):
                    continue
                if len(a) < 3 or not np.isfinite(a).all():
                    continue
                if (a[0] != a[-1]).any():
                    a = np.vstack([a, a[:1]])
                coords.append(a)
                coord_off.append(coord_off[-1] + len(a))
            if len(coord_off) > n0:
                kind.append(bits)
                ring_off.append(len(coord_off) - 1)
    if not kind:
        raise ValueError(f"{geojson}: no polygons")
    ll = np.concatenate(coords)
    arrays = {"version": np.array(VERSION), "kind": np.asarray(kind, np.uint8),
              "ring_off": np.asarray(ring_off, np.int64), "coord_off": np.asarray(coord_off, np.int64),
              # (1e-7 degrees: a centimetre; int32 packs to a third of the float64's size)
              "lon7": np.round(ll[:, 0] * 1e7).astype(np.int32), "lat7": np.round(ll[:, 1] * 1e7).astype(np.int32)}
    tmp = f"{out}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)
    return out


def _ensure_npz(path, wait_s=180.0):
    """The .npz of path, built from the .geojson when it is missing or older than it. One process builds (a lock
    file beside it); the others wait for its file."""
    out = npz_path(path)
    if out == path:
        return out

    def fresh():
        try:
            return os.path.getmtime(out) >= os.path.getmtime(path)
        except OSError:
            return False
    if fresh():
        return out
    lock = out + ".lock"
    end = time.monotonic() + wait_s
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            if fresh():
                return out
            try:
                if time.time() - os.path.getmtime(lock) > 600.0:    # (a build that died)
                    os.remove(lock)
                    continue
            except OSError:
                pass
            if time.monotonic() > end:
                raise OSError(f"{out}: still being built by another process")
            time.sleep(0.5)
    try:
        if not fresh():
            t0 = time.monotonic()
            build(path, out)
            log.info("Sofia streets: %s built in %.1f s", out, time.monotonic() - t0)
        return out
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


# ---- rasters -------------------------------------------------------------------------------------------------

def winding_mask(rings, e0, n1, res, width, height):
    """The union (nonzero winding) of closed rings ((k, 2) east / north m; outer rings counter-clockwise, holes
    clockwise, as GeoJSON has them) at the pixels' centres: bool (height, width); pixel (j, i)'s centre is at east
    e0 + (i + .5) res, north n1 - (j + .5) res. A scanline fill in numpy: every edge's crossings of the rows it
    spans, summed along each row."""
    if not rings:
        return np.zeros((height, width), bool)
    a = np.concatenate([r[:-1] for r in rings])
    b = np.concatenate([r[1:] for r in rings])
    x0, y0 = (a[:, 0] - e0) / res, (n1 - a[:, 1]) / res
    x1, y1 = (b[:, 0] - e0) / res, (n1 - b[:, 1]) / res
    sign = np.where(y1 > y0, 1, -1).astype(np.int16)
    ylo, yhi = np.minimum(y0, y1), np.maximum(y0, y1)
    j0 = np.clip(np.ceil(ylo - 0.5).astype(np.int64), 0, height)       # the first row whose centre is >= ylo
    j1 = np.clip(np.ceil(yhi - 0.5).astype(np.int64), 0, height)       # ... >= yhi (not included)
    cnt = np.maximum(j1 - j0, 0)
    keep = cnt > 0
    if not keep.any():
        return np.zeros((height, width), bool)
    x0, y0, x1, y1, sign, j0, cnt = x0[keep], y0[keep], x1[keep], y1[keep], sign[keep], j0[keep], cnt[keep]
    e = np.repeat(np.arange(len(cnt)), cnt)
    j = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt) + j0[e]
    x = x0[e] + (j + 0.5 - y0[e]) * (x1[e] - x0[e]) / (y1[e] - y0[e])
    c = np.clip(np.ceil(x - 0.5), 0, width).astype(np.int64)            # the first column right of the crossing
    d = np.zeros(height * (width + 1), np.int16)
    np.add.at(d, j * (width + 1) + c, sign[e])
    d = d.reshape(height, width + 1)
    np.cumsum(d, axis=1, out=d)
    return d[:, :width] != 0


class Raster:
    """A cell's ground by kind: bits (height, width) uint8 on a res grid in frame (the cell's ENU frame); pixel
    (j, i)'s centre is at east e0 + (i + .5) res, north n1 - (j + .5) res."""

    def __init__(self, key, frame, e0, n1, res, bits):
        self.key, self.frame, self.e0, self.n1, self.res, self.bits = key, frame, e0, n1, res, bits
        self.height, self.width = bits.shape

    def at(self, e, n):
        """The bits at east / north (arrays): 0 outside the raster."""
        i = np.floor((np.asarray(e, np.float64) - self.e0) / self.res).astype(np.int64)
        j = np.floor((self.n1 - np.asarray(n, np.float64)) / self.res).astype(np.int64)
        ok = (i >= 0) & (i < self.width) & (j >= 0) & (j < self.height)
        out = np.zeros(np.shape(i), np.uint8)
        out[ok] = self.bits[j[ok], i[ok]]
        return out

    def inside(self, e, n):
        e, n = np.asarray(e, np.float64), np.asarray(n, np.float64)
        return ((e >= self.e0) & (e < self.e0 + self.width * self.res)
                & (n <= self.n1) & (n > self.n1 - self.height * self.res))


def cell_frame(key):
    """The ENU frame of a map cell: the tangent plane at its centre (mapdata's, the frame of its features' `en`)."""
    lat, lon = key.center()
    return geodesy.EnuFrame(lat, lon)


def _sample(en, step, first):
    """Stations every step m from first m along polyline en (k, 2): (points (m, 2), unit tangents (m, 2))."""
    d = np.diff(en, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    ok = seg > 1e-9
    if not ok.all():
        en = np.concatenate([en[:1], en[1:][ok]])
        d = np.diff(en, axis=0)
        seg = np.hypot(d[:, 0], d[:, 1])
    if not len(seg):
        return np.zeros((0, 2)), np.zeros((0, 2))
    cs = np.r_[0.0, np.cumsum(seg)]
    length = float(cs[-1])
    first = min(first, length / 3.0)
    s = np.arange(first, length - first + 1e-6, step)
    if not len(s):
        return np.zeros((0, 2)), np.zeros((0, 2))
    k = np.clip(np.searchsorted(cs, s, side="right") - 1, 0, len(seg) - 1)
    t = d[k] / seg[k][:, None]
    return en[k] + t * (s - cs[k])[:, None], t


# ---- the dataset ---------------------------------------------------------------------------------------------

class Streets:
    def __init__(self, path):
        """path: streets_curbs.geojson (its .npz is made beside it when missing) or the .npz. Raises OSError or
        ValueError when it cannot be read."""
        self.path = path
        self.npz = _ensure_npz(path)
        with np.load(self.npz, allow_pickle=False) as z:
            if int(z["version"]) != VERSION:
                raise ValueError(f"{self.npz}: version {int(z['version'])}, wanted {VERSION}")
            self.kind = z["kind"]
            self.ring_off = z["ring_off"]
            self.coord_off = z["coord_off"]
            self.lon = z["lon7"].astype(np.float64) * 1e-7
            self.lat = z["lat7"].astype(np.float64) * 1e-7
        c0 = self.coord_off[self.ring_off[:-1]]         # each polygon's first coordinate
        self.box = np.stack([np.minimum.reduceat(self.lon, c0), np.minimum.reduceat(self.lat, c0),
                             np.maximum.reduceat(self.lon, c0), np.maximum.reduceat(self.lat, c0)], 1)
        self.bounds = (float(self.box[:, 1].min()), float(self.box[:, 0].min()),
                       float(self.box[:, 3].max()), float(self.box[:, 2].max()))      # south, west, north, east
        self.tag = f"{VERSION}-{len(self.kind):x}-{len(self.lon):x}"
        self._rasters = OrderedDict()
        self._lock = threading.Lock()
        self._cell_locks = {}

    def __len__(self):
        return len(self.kind)

    def covers(self, key):
        """Does map cell key touch the dataset's box (where it is asked at all)?"""
        lon0, lat0, lon1, lat1 = key.bounds()
        s, w, n, e = self.bounds
        return lat0 < n and lat1 > s and lon0 < e and lon1 > w

    def raster(self, key):
        """Cell key's Raster (made once, kept: RASTERS_MAX), or None where the dataset has nothing near it."""
        key = mapcells.CellKey(int(key[0]), int(key[1]), int(key[2]))
        k = str(key)
        with self._lock:
            got = self._rasters.get(k, False)
            if got is not False:
                self._rasters.move_to_end(k)
                return got
            lock = self._cell_locks.setdefault(k, threading.Lock())
        with lock:
            with self._lock:
                got = self._rasters.get(k, False)
            if got is False:
                got = self._make(key) if self.covers(key) else None
                with self._lock:
                    self._rasters[k] = got
                    while len(self._rasters) > RASTERS_MAX:
                        self._rasters.popitem(last=False)
                    self._cell_locks.pop(k, None)
        return got

    def _make(self, key):
        frame = cell_frame(key)
        lon0, lat0, lon1, lat1 = key.bounds()
        ce, cn, _ = frame.geodetic_to_enu(np.array([lat0, lat0, lat1, lat1]), np.array([lon0, lon1, lon0, lon1]), 0.0)
        e0 = math.floor(float(np.min(ce)) - MARGIN_M)
        n1 = math.ceil(float(np.max(cn)) + MARGIN_M)
        width = int(math.ceil((float(np.max(ce)) + MARGIN_M - e0) / RES_M))
        height = int(math.ceil((n1 - (float(np.min(cn)) - MARGIN_M)) / RES_M))
        dlat = (MARGIN_M + 5.0) / 110000.0
        dlon = dlat / max(0.05, math.cos(math.radians(lat0)))
        b = self.box
        idx = np.flatnonzero((b[:, 2] >= lon0 - dlon) & (b[:, 0] <= lon1 + dlon)
                             & (b[:, 3] >= lat0 - dlat) & (b[:, 1] <= lat1 + dlat))
        bits = np.zeros((height, width), np.uint8)
        if not len(idx):
            return None
        # every ring of the cell's polygons in its frame, one conversion
        r0, r1 = self.ring_off[idx], self.ring_off[idx + 1]
        rings = np.concatenate([np.arange(a, b_) for a, b_ in zip(r0, r1)])
        kind_of_ring = np.repeat(self.kind[idx], r1 - r0)
        c0, c1 = self.coord_off[rings], self.coord_off[rings + 1]
        sel = np.concatenate([np.arange(a, b_) for a, b_ in zip(c0, c1)])
        e, n, _ = frame.geodetic_to_enu(self.lat[sel], self.lon[sel], 0.0)
        en = np.stack([np.asarray(e, np.float64), np.asarray(n, np.float64)], 1)
        cut = np.cumsum(c1 - c0)[:-1]
        parts = np.split(en, cut)
        for bit in (CARRIAGEWAY, BRIDGE, TUNNEL, PARKING, SIDEWALK, WALK_PARKING, TRACK, OTHER):
            pick = [parts[i] for i in np.flatnonzero(kind_of_ring & bit)]
            if pick:
                bits[winding_mask(pick, e0, n1, RES_M, width, height)] |= np.uint8(bit)
        return Raster(key, frame, e0, n1, RES_M, bits)

    # ---- a cell's roads ----
    def measure_cell(self, key, lines, oneway=None):
        """Per road line of map cell key: its Measure, or None (no carriageway under it, too few stations, off the
        raster). lines: [(k, 2) lon / lat arrays], every drivable road line of the cell (they cut one another's
        carriageways where two run along in one); oneway: per line, True for a one-way road (unused yet: the cut is
        the same). Pure given the dataset and the lines."""
        out = [None] * len(lines)
        r = self.raster(key)
        if r is None or not lines:
            return out
        from PIL import Image, ImageDraw
        ens = []
        for ll in lines:
            ll = np.asarray(ll, np.float64).reshape(-1, 2)
            if len(ll) < 2:
                ens.append(None)
                continue
            e, n, _ = r.frame.geodetic_to_enu(ll[:, 1], ll[:, 0], 0.0)
            ens.append(np.stack([np.asarray(e, np.float64), np.asarray(n, np.float64)], 1))
        # the lines' own raster: which line passes a pixel (its index + 1) and its direction there (degrees mod 180
        # + 1), two pixels wide: a station's ray knows another road's line when it meets one
        ids = Image.new("I", (r.width, r.height), 0)
        ang = Image.new("L", (r.width, r.height), 0)
        d_id, d_ang = ImageDraw.Draw(ids), ImageDraw.Draw(ang)
        for k, en in enumerate(ens):
            if en is None:
                continue
            px = np.c_[(en[:, 0] - r.e0) / r.res, (r.n1 - en[:, 1]) / r.res]
            for a in range(len(px) - 1):
                if not (np.isfinite(px[a]).all() and np.isfinite(px[a + 1]).all()):
                    continue
                deg = int(round(math.degrees(math.atan2(en[a + 1, 1] - en[a, 1], en[a + 1, 0] - en[a, 0])))) % 180
                seg = [(float(px[a, 0]), float(px[a, 1])), (float(px[a + 1, 0]), float(px[a + 1, 1]))]
                d_id.line(seg, fill=k + 1, width=2)
                d_ang.line(seg, fill=deg + 1, width=2)
        ids = np.asarray(ids).astype(np.int32)
        ang = np.asarray(ang)
        nray = int(round(RAY_M / RAY_STEP_M))
        off = np.arange(-nray, nray + 1) * RAY_STEP_M           # m across: negative right, positive left
        mid = nray
        for k, en in enumerate(ens):
            if en is None:
                continue
            pts, tan = _sample(en, STEP_M, END_SKIP_M)
            if len(pts) < MIN_STATIONS:
                continue
            left = np.stack([-tan[:, 1], tan[:, 0]], 1)
            ex = pts[:, 0, None] + left[:, 0, None] * off[None, :]
            ny = pts[:, 1, None] + left[:, 1, None] * off[None, :]
            b = r.at(ex, ny)
            i = np.floor((ex - r.e0) / r.res).astype(np.int64)
            j = np.floor((r.n1 - ny) / r.res).astype(np.int64)
            ok = (i >= 0) & (i < r.width) & (j >= 0) & (j < r.height)
            who = np.zeros(ex.shape, np.int32)
            who[ok] = ids[j[ok], i[ok]]
            dirs = np.zeros(ex.shape, np.uint8)
            dirs[ok] = ang[j[ok], i[ok]]
            mine = np.degrees(np.arctan2(tan[:, 1], tan[:, 0])) % 180.0
            delta = np.abs(((dirs.astype(np.float64) - 1.0) - mine[:, None] + 90.0) % 180.0 - 90.0)
            other = (who > 0) & (who != k + 1) & (dirs > 0) & (np.cos(np.radians(delta)) >= TWIN_COS)
            m = self._stations(b, other, off, mid)
            if m is not None:
                out[k] = m
        return out

    @staticmethod
    def _stations(b, other, off, mid):
        """A road's Measure from its stations' rays: b (stations, 2 * rays + 1) the ground's bits across the road
        (index mid at its line, to the left upward), other: another road's line running along there."""
        cw = (b & CARRIAGEWAY) != 0
        nst, n = cw.shape
        step = float(off[1] - off[0])
        snap = int(round(SNAP_M / step))
        rows, ar = np.arange(nst), np.arange(n)
        # (every station at once: a loop over them was most of a cell's 0.6 s, and a road tile asks nine cells)
        # the line beside its carriageway: the nearest of it within SNAP_M (the right side at a tie)
        d = np.where(cw[:, mid - snap:mid + snap + 1], np.abs(np.arange(-snap, snap + 1))[None, :], snap + 1)
        k = np.argmin(d, axis=1)
        ok = d[rows, k] <= snap
        c = mid - snap + k
        # the carriageway's run through it: the nearest ground that is none, either way
        lo = np.maximum.accumulate(np.where(cw, -1, ar[None, :]), axis=1)[rows, c] + 1
        hi = np.minimum.accumulate(np.where(cw, n, ar[None, :])[:, ::-1], axis=1)[:, ::-1][rows, c] - 1
        ok &= (lo > 0) & (hi < n - 1)       # (past the ray: a square, a junction's fan, no edge to measure)
        # another line along inside the run, each side: the carriageway is the two roads', cut half way
        near_m = int(round(TWIN_MIN_M / step))
        right = np.maximum.accumulate(np.where(other, ar[None, :], -1), axis=1)[:, mid - near_m - 1]
        left = np.minimum.accumulate(np.where(other, ar[None, :], n)[:, ::-1], axis=1)[:, ::-1][:, mid + near_m + 1]
        has_r, has_l = right >= lo, left <= hi
        a0 = np.where(has_r, np.maximum(lo, (right + mid) // 2), lo)
        a1 = np.where(has_l, np.minimum(hi, (left + mid + 1) // 2), hi)
        w = (a1 - a0 + 1) * step
        ok &= (w >= WIDTH_MIN_M) & (w <= WIDTH_MAX_M)
        if int(ok.sum()) < MIN_STATIONS or int(ok.sum()) < MIN_SHARE * nst:
            return None

        def beyond(bits, start, side, reach):
            """Per station: any of bits within reach pixels past index start, going side (+1 left, -1 right)?"""
            hit = np.zeros(nst, bool)
            first = np.full(nst, -1, np.int64)
            for q in range(1, reach + 1):
                j = start + side * q
                at = (j >= 0) & (j < n) & ((b[rows, np.clip(j, 0, n - 1)] & bits) != 0)
                first = np.where(at & ~hit, j, first)
                hit |= at
            return hit, first

        reach_w, reach_p = int(round(WALK_REACH_M / step)), max(1, int(round(PARK_REACH_M / step)))
        # (past the carriageway's own edges, not a cut's: the other road's side has no sidewalk of this one's)
        wl = beyond(SIDEWALK | WALK_PARKING, hi, 1, reach_w)[0] & (a1 == hi)
        wr = beyond(SIDEWALK | WALK_PARKING, lo, -1, reach_w)[0] & (a0 == lo)
        pl, pl0 = beyond(PARKING, hi, 1, reach_p)
        pr, pr0 = beyond(PARKING, lo, -1, reach_p)
        pl &= a1 == hi
        pr &= a0 == lo
        depth = []
        for side, has, first in ((1, pl & ok, pl0), (-1, pr & ok, pr0)):
            if has.any():                   # the bays' depth across: their run from the edge outward
                run, alive = np.zeros(nst, np.int64), has.copy()
                for q in range(int(round(15.0 / step))):
                    j = first + side * q
                    alive &= (j >= 0) & (j < n) & ((b[rows, np.clip(j, 0, n - 1)] & PARKING) != 0)
                    if not alive.any():
                        break
                    run += alive
                depth.append(run[has] * step)
        ws = w[ok]
        q1, med, q3 = np.percentile(ws, [25, 50, 75])
        depth = np.concatenate(depth) if depth else np.zeros(0)
        return Measure(round(float(med), 2), int(ok.sum()), round(float(q3 - q1), 2),
                       float(np.mean(wl[ok])) >= SIDE_SHARE, float(np.mean(wr[ok])) >= SIDE_SHARE,
                       float(np.mean(pl[ok])) >= SIDE_SHARE, float(np.mean(pr[ok])) >= SIDE_SHARE,
                       round(float(np.median(depth)), 2) if len(depth) else 0.0,
                       round(float(np.mean((b[rows, c] & BRIDGE)[ok] != 0)), 3),
                       round(float(np.mean((b[rows, c] & TUNNEL)[ok] != 0)), 3),
                       round(float(np.mean((has_r | has_l)[ok])), 3))

    # ---- props ----
    def blocked(self, lat, lon, margin=PROP_MARGIN_M):
        """Per point (arrays of degrees): is it, or anything within margin m of it, on a carriageway, a parking bay
        on one, a tram bed or a bike lane (NO_PROPS)? False where the dataset has nothing."""
        lat = np.atleast_1d(np.asarray(lat, np.float64))
        lon = np.atleast_1d(np.asarray(lon, np.float64))
        out = np.zeros(lat.shape, bool)
        s, w, n, e = self.bounds
        near = np.flatnonzero((lat > s) & (lat < n) & (lon > w) & (lon < e) & np.isfinite(lat) & np.isfinite(lon))
        if not len(near):
            return out
        keys = {}
        for k in near.tolist():
            keys.setdefault(tuple(mapcells.cell_of(float(lat[k]), float(lon[k]))), []).append(k)
        ring = [(0.0, 0.0)] + ([(margin, 0.0), (-margin, 0.0), (0.0, margin), (0.0, -margin),
                                (margin * 0.7, margin * 0.7), (-margin * 0.7, margin * 0.7),
                                (margin * 0.7, -margin * 0.7), (-margin * 0.7, -margin * 0.7)] if margin > 0 else [])
        for key, ks in sorted(keys.items()):
            r = self.raster(key)
            if r is None:
                continue
            ks = np.asarray(ks)
            pe, pn, _ = r.frame.geodetic_to_enu(lat[ks], lon[ks], 0.0)
            pe, pn = np.asarray(pe, np.float64), np.asarray(pn, np.float64)
            hit = np.zeros(len(ks), bool)
            for de, dn in ring:
                hit |= (r.at(pe + de, pn + dn) & NO_PROPS) != 0
            out[ks] = hit
        return out


_opened = {}                # path -> Streets, or None: it could not be read (said once per process)
_open_lock = threading.Lock()


def from_env():
    """The process's Streets of the file BNE_SOFIA_STREETS names, or None (unset, "off", or a file that cannot be
    read: a warning, once)."""
    path = os.environ.get(ENV, "").strip().strip('"')
    if not path or path.lower() == "off":
        return None
    path = os.path.abspath(os.path.expanduser(path))
    with _open_lock:
        if path in _opened:
            return _opened[path]
        try:
            got = Streets(path)
            log.info("Sofia streets: %s (%d polygons, lat %.3f..%.3f, lon %.3f..%.3f)", path, len(got),
                     got.bounds[0], got.bounds[2], got.bounds[1], got.bounds[3])
        except (OSError, ValueError, KeyError) as e:
            log.warning("Sofia streets: %s cannot be read (%s: %s); Sofia's roads keep their guessed widths.", path,
                        type(e).__name__, e)
            got = None
        _opened[path] = got
        return got


if __name__ == "__main__":      # python server/sofiastreets.py <streets_curbs.geojson>: build its .npz now
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.monotonic()
    s = Streets(sys.argv[1])
    print(f"{s.npz}: {len(s)} polygons, {len(s.lon)} points, bounds {tuple(round(v, 4) for v in s.bounds)}, "
          f"{time.monotonic() - t0:.1f} s")
