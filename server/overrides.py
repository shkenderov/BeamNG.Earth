"""Per-way corrections from a JSON file (env BNE_OVERRIDES), the generic input for what a map's data gets wrong at one
place. Unset, or a file that does not exist: no overrides, nothing changes (and no cache key).

File format, version 1:

  {"version": 1,
   "ways": {"123456789": {"note": "why, for humans",
                          "bbox": [west, south, east, north],     only features wholly inside it
                          "lanes": 2, "width_m": 7.0,
                          "sidewalk": "both" | "left" | "right" | "none",
                          "level": [{"from": 0.0, "to": 1.0, "kind": "tunnel" | "bridge" | "ground"}],
                          "skip": false}}}

The key is an OpenStreetMap way id; every field but the key is optional; an unknown field or a wrong type is an
OverridesError naming the way, raised when the file is loaded. A way's features are found as rampoverrides does
(feats.attrs['src_record'] "w<id>" / "w<id>@...", feats.attrs['lane_osm_way']), and by the "src_ways" rule table
where a segment is made of several ways. "level" fractions run along the source way: on a segment of several ways
they are mapped onto the way's share of the segment (its src_ways range); a feature that is the way alone takes them
as its own fractions. The content's short hash (key) joins the plans' and the facts' cache keys, so plans made with
other overrides are never mixed.
"""
import hashlib
import json
import os
import threading

import numpy as np

ENV = "BNE_OVERRIDES"
VERSION = 1
SIDEWALKS = {"both": (1, -1), "left": (1,), "right": (-1,), "none": ()}     # (walk_sides: +1 left, -1 right)
KINDS = ("tunnel", "bridge", "ground")
_FIELDS = {"note", "bbox", "lanes", "width_m", "sidewalk", "level", "skip"}


class OverridesError(ValueError):
    pass


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class Way:
    __slots__ = ("id", "note", "bbox", "lanes", "width_m", "sidewalk", "level", "skip")

    def __init__(self, way, d):
        def bad(msg):
            raise OverridesError(f"way {way}: {msg}")
        if not isinstance(d, dict):
            bad("must be an object")
        unknown = sorted(set(d) - _FIELDS)
        if unknown:
            bad(f"unknown field {unknown[0]!r}")
        self.id = way
        self.note = d.get("note")
        if self.note is not None and not isinstance(self.note, str):
            bad("note must be a string")
        self.bbox = d.get("bbox")
        if self.bbox is not None:
            if not isinstance(self.bbox, list) or len(self.bbox) != 4 or not all(_num(v) for v in self.bbox):
                bad("bbox must be [west, south, east, north] numbers")
            if not (self.bbox[0] < self.bbox[2] and self.bbox[1] < self.bbox[3]):
                bad("bbox must have west < east and south < north")
        self.lanes = d.get("lanes")
        if self.lanes is not None and (not isinstance(self.lanes, int) or isinstance(self.lanes, bool)
                                       or self.lanes < 1):
            bad("lanes must be an integer >= 1")
        self.width_m = d.get("width_m")
        if self.width_m is not None and (not _num(self.width_m) or not 0 < self.width_m <= 100):
            bad("width_m must be a number in (0, 100]")
        self.sidewalk = d.get("sidewalk")
        if self.sidewalk is not None and self.sidewalk not in SIDEWALKS:
            bad(f"sidewalk must be one of {sorted(SIDEWALKS)}")
        lv = d.get("level")
        self.level = None
        if lv is not None:
            if not isinstance(lv, list):
                bad("level must be a list")
            out = []
            for r in lv:
                if not isinstance(r, dict) or set(r) != {"from", "to", "kind"}:
                    bad("each level range needs exactly from, to, kind")
                a, b = r["from"], r["to"]
                if not _num(a) or not _num(b) or not 0 <= a < b <= 1:
                    bad("level from/to must be numbers with 0 <= from < to <= 1")
                if r["kind"] not in KINDS:
                    bad(f"level kind must be one of {list(KINDS)}")
                out.append((float(a), float(b), r["kind"]))
            self.level = out
        self.skip = d.get("skip")
        if self.skip is not None and not isinstance(self.skip, bool):
            bad("skip must be true or false")


class Overrides:
    def __init__(self, text):
        try:
            doc = json.loads(text)
        except ValueError as e:
            raise OverridesError(f"not valid JSON: {e}")
        if not isinstance(doc, dict) or doc.get("version") != VERSION:
            raise OverridesError(f"needs an object with \"version\": {VERSION}")
        extra = sorted(set(doc) - {"version", "ways"})
        if extra:
            raise OverridesError(f"unknown top-level field {extra[0]!r}")
        ways = doc.get("ways", {})
        if not isinstance(ways, dict):
            raise OverridesError("\"ways\" must be an object")
        self.ways = {}
        for k, d in ways.items():
            try:
                wid = int(k)
                if str(wid) != k.strip() or wid <= 0:
                    raise ValueError
            except ValueError:
                raise OverridesError(f"way key {k!r} is not an OpenStreetMap way id")
            self.ways[wid] = Way(wid, d)
        self.digest = hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()[:8]

    def key(self):
        """The cache keys' part: (("ov", hash),), or () when there is nothing to override (the keys as they were)."""
        return (("ov", self.digest),) if self.ways else ()

    # ---- finding a way's features ---------------------------------------------------------------------------------

    def _matches(self, feats):
        """{way id: [(feature, a, b)]}: the features of each way and its share (a, b) of each. Kept on feats."""
        memo = feats.__dict__.setdefault("_overrides_match", {})
        got = memo.get(self.digest)
        if got is not None:
            return got
        n = len(feats)
        got = {w: [] for w in self.ways}
        if n:
            rec = np.asarray(feats.attrs.get("src_record", [""] * n)).astype(str)
            lane_way = np.asarray(feats.attrs.get("lane_osm_way", [0] * n))
            table = feats.rules.get("src_ways")
            listed = {}
            if table is not None and len(table.feature):
                vals = table.value.tolist() if hasattr(table.value, "tolist") else list(table.value)
                for j, v in enumerate(vals):
                    v = str(v)
                    if v.startswith("w"):
                        try:
                            listed.setdefault(int(v[1:].split("@")[0]), []).append(
                                (int(table.feature[j]), float(table.start[j]), float(table.end[j])))
                        except ValueError:
                            pass
            inside = {}
            for wid, way in self.ways.items():
                key = f"w{wid}"
                near = (rec == key) | np.char.startswith(rec, key + "@")
                try:
                    near |= lane_way == wid
                except (TypeError, ValueError):
                    pass
                shares = {}
                for i, a, b in listed.get(wid, ()):
                    shares.setdefault(i, []).append((a, b))
                for i in sorted(set(np.flatnonzero(near).tolist()) | set(shares)):
                    if way.bbox is not None:
                        bk = tuple(way.bbox)
                        if bk not in inside:
                            inside[bk] = _inside_mask(feats, bk)
                        if not inside[bk][i]:
                            continue
                    for a, b in shares.get(i, [(0.0, 1.0)]):
                        got[wid].append((i, a, b))
        memo[self.digest] = got
        return got

    # ---- what each field gives ------------------------------------------------------------------------------------

    def _field(self, feats, name):
        out = {}
        for wid, rows in self._matches(feats).items():
            v = getattr(self.ways[wid], name)
            if v is not None:
                for i, _a, _b in rows:
                    out[i] = v
        return out

    def lane_counts(self, feats):
        """{feature: lanes}."""
        return self._field(feats, "lanes")

    def widths(self, feats):
        """{feature: carriageway width m}."""
        return self._field(feats, "width_m")

    def sidewalks(self, feats):
        """{feature: sides} (+1 left, -1 right of the geometry; () none)."""
        return {i: SIDEWALKS[v] for i, v in self._field(feats, "sidewalk").items()}

    def skipped(self, feats):
        """The features not to be built."""
        return {i for i, v in self._field(feats, "skip").items() if v}

    def levels(self, feats, ranges):
        """ranges: per feature [(start, end, kind)] (roads._tag_ranges_all). A new list with the "level" overrides
        applied: inside the way's share of a feature its ranges are replaced by the file's."""
        out = None
        for wid, rows in self._matches(feats).items():
            lv = self.ways[wid].level
            if lv is None:
                continue
            for i, a, b in rows:
                if out is None:
                    out = [list(r) for r in ranges]
                keep = []
                for s, e, k in out[i]:
                    if s < a:
                        keep.append((s, min(e, a), k))
                    if e > b:
                        keep.append((max(s, b), e, k))
                new = [(a + f0 * (b - a), a + f1 * (b - a), k) for f0, f1, k in lv if k != "ground"]
                out[i] = sorted(keep + new, key=lambda r: (r[0], r[1]))
        return ranges if out is None else out


def _inside_mask(feats, bbox):
    """Per feature: lies wholly inside bbox (west, south, east, north)."""
    w, s, e, n = bbox
    ll = np.asarray(feats.lonlat, np.float64)
    out = np.zeros(len(feats), bool)
    for i in range(len(feats)):
        a = feats.coord_off[feats.ring_off[feats.part_off[i]]]
        b = feats.coord_off[feats.ring_off[feats.part_off[i + 1]]]
        if b > a:
            p = ll[a:b]
            out[i] = bool((p[:, 0] >= w).all() and (p[:, 0] <= e).all() and (p[:, 1] >= s).all()
                          and (p[:, 1] <= n).all())
    return out


_lock = threading.Lock()
_loaded = {}        # path -> ((mtime_ns, size), Overrides)


def load(path):
    """The Overrides of a file (OverridesError for a bad one)."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise OverridesError(f"{path}: not UTF-8 ({e})")
    try:
        got = Overrides(text)
    except OverridesError as e:
        raise OverridesError(f"{path}: {e}")
    got.digest = hashlib.sha256(raw).hexdigest()[:8]
    return got


def from_env():
    """The process's Overrides of the file BNE_OVERRIDES names, or None (unset, or no such file)."""
    path = os.environ.get(ENV, "").strip().strip('"')
    if not path:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    try:
        st = os.stat(path)
    except OSError:
        return None
    stamp = (st.st_mtime_ns, st.st_size)
    with _lock:
        hit = _loaded.get(path)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        got = load(path)
        _loaded[path] = (stamp, got)
        return got


def key():
    """The cache keys' part of the process's overrides (Overrides.key), () without."""
    ov = from_env()
    return () if ov is None else ov.key()
