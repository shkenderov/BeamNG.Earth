#!/usr/bin/env python3
"""levelcheck.py PACK --cache DIR [--bbox S,W,N,E] [--out DIR] [--online]: where the drawn road disagrees with the
ground, offline (no game), for a baked pack (server/bakepack.py) and the terrain model the pack was made with.

    BNE_SOFIA_DTM=... BNE_SOFIA_STREETS=... BNE_OVERRIDES=... \
        python tools/levelcheck.py dist/sofia.bnge --cache cache/full --out reports/levelcheck

Run it with the env vars the pack was baked with (they are part of its meta: a pack whose "variant" differs from the
running server's is refused). It writes levelcheck.json (every finding: check, way, lat, lon, value, limit, ...) and
levelcheck.csv (the same, worst first) into --out and prints a one-line total per check. Two runs over the same pack
and inputs write byte-identical files (no times in them).

What is compared with what (heights are geodetic metres in the terrain model's own datum: the plans, the profiles and
the terrain all read the one dem.Dem, so no datum offset exists between them; "z" below is that height):

  parts       Every road piece the game draws, classified the way the tile builder classifies it:
              ground   a Plan of the pack's road plans (grading._Planner._plan_cell; the pack's own, not planned here),
                       excluding the rail beds, rounded corners and the tunnels' own plans (TUNNEL_TAG);
              open     a tunnel's open section (roads.TunnelProfiles.open_parts: a road in the open the map calls a
                       tunnel); both ground and open are drawn at the graded ground:
                         surface z = Grader.apply(terrain, ROAD_LIFT) at the part's vertices, which is the height the
                       road mesh gets (roadmesh.drawn_ground);
              bridge   a deck of roads.BridgeProfiles (the solved profile, z = the road surface; its structure under it
                       is bridges.depth_of);
              tunnel   a closed section of roads.TunnelProfiles (z = the floor / road surface; the vault is
                       tunnels.height above it, TUNNEL_H for a full tube, a low arch's own).
              The bridge and tunnel profiles are solved at run time from the map data and the pack's plans (they are
              not stored in a pack); the pack's plans are read, never planned, unless one is missing (reported).
  terrain     blockworker.ground(dem.sample(lat, lon, grading.PROFILE_ZOOM)): the ungraded terrain model under the
              vertex, at the zoom the plans, the profiles and the terrain blocks use, as the game shapes it before the
              road grading: with the rivers' and lakes' beds carved in (the sea carver) and the runways graded in.

  ground      surface z of a ground part minus terrain, more than GROUND_LIMIT_M either way, per run of vertices.
  cover       a tunnel: terrain - (floor + vault height) under COVER_MIN_M, or the floor above the terrain; the first
              PORTAL_ZONE_M from a portal are not judged (the portal face rises to the ground there).
  clearance   a bridge deck over a road: (deck z - structure depth) - road surface z at the crossing of their
              centrelines, under CLEARANCE_MIN_M. Crossings with the same way, and junctions (a crossing within
              JUNCTION_END_M of an end of either, the two surfaces less than JUNCTION_DZ_M apart) are skipped.
  portal      a bridge or tunnel end and the surface part (or another structure) it meets within APPROACH_M: the
              height step over PORTAL_STEP_M.
  grade       the steepest rise over GRADE_WINDOW_M along a paved part's own surface, over GRADE_MAX.
  seam        two parts of one way of the same kind meeting (ends within JOIN_M): the height step over SEAM_STEP_M.
"""
import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field

import numpy as np

# ---- thresholds ----------------------------------------------------------------------------------------------------
GROUND_LIMIT_M = 1.0        # ground road: surface this far above or below the terrain
COVER_MIN_M = 0.5           # tunnel: ground over the vault at least
CLEARANCE_MIN_M = 4.0       # underpass: deck underside over the lower road at least
PORTAL_STEP_M = 0.2         # a bridge / tunnel end against its approach
GRADE_MAX = 0.12            # paved road: rise over run
GRADE_WINDOW_M = 10.0       # ... over this run
SEAM_STEP_M = 0.1           # consecutive parts of one way

# ---- how the parts are read (not limits) -----------------------------------------------------------------------------
PORTAL_ZONE_M = 12.0        # m from a portal not judged for cover (roads.PORTAL_LOOK_M: where the face reads the ground)
JOIN_M = 0.5                # part ends this close are one joint
APPROACH_M = 2.5            # a bridge / tunnel end looks for its approach this far
RUN_GAP_M = 10.0            # findings of one part closer than this along it are one
GRADE_MIN_LEN_M = 4.0       # parts shorter than this have no grade
JUNCTION_END_M = 10.0       # a crossing this near an end of either road, at nearly one height, is a junction
JUNCTION_DZ_M = 1.0
NON_VEHICLE = {"footway", "path", "steps", "cycleway", "bridleway", "pedestrian", "track"}   # (not judged as a road under a deck)
CHECKS = ("ground", "cover", "clearance", "portal", "grade", "seam")
LIMITS = {"ground": GROUND_LIMIT_M, "cover": COVER_MIN_M, "clearance": CLEARANCE_MIN_M, "portal": PORTAL_STEP_M,
          "grade": GRADE_MAX, "seam": SEAM_STEP_M}
SURFACE = ("ground", "open")


@dataclass
class Piece:
    """One drawn road part. xy: (n, 2) metres in one common plane (only distances and crossings use it); lat, lon: (n,);
    z: (n,) the drawn road surface (geodetic m); terrain: (n,) the ungraded terrain under it, or None."""
    kind: str                       # "ground" | "open" | "bridge" | "tunnel"
    way: str                        # the OpenStreetMap way id, "" when the feature has none
    fid: str
    xy: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    z: np.ndarray
    terrain: np.ndarray = None
    cls: str = ""
    paved: bool = True
    name: str = ""
    th: float = 0.0                 # tunnel: floor to vault
    mouth: tuple = (True, True)     # tunnel: its start / end is a portal
    depth: float = 0.0              # bridge: structure under the road
    arc: np.ndarray = field(init=False)

    def __post_init__(self):
        self.xy = np.asarray(self.xy, np.float64).reshape(-1, 2)
        self.lat, self.lon = np.asarray(self.lat, np.float64), np.asarray(self.lon, np.float64)
        self.z = np.asarray(self.z, np.float64)
        if self.terrain is not None:
            self.terrain = np.asarray(self.terrain, np.float64)
        d = np.hypot(*np.diff(self.xy, axis=0).T) if len(self.xy) > 1 else np.zeros(0)
        self.arc = np.r_[0.0, np.cumsum(d)]

    @property
    def length(self):
        return float(self.arc[-1])

    @property
    def group(self):
        return self.way or self.fid


def _finding(check, p, i, value, note="", at=None):
    """A finding of check on piece p at vertex i (or at = (lat, lon))."""
    la, lo = (float(p.lat[i]), float(p.lon[i])) if at is None else at
    lim = LIMITS[check]
    v = float(value)
    if check == "ground":
        sev = abs(v) / lim
    elif check in ("cover", "clearance"):
        sev = 1.0 + (lim - v) / lim
    else:
        sev = abs(v) / lim
    return {"check": check, "way": p.way, "fid": p.fid, "kind": p.kind, "cls": p.cls, "name": p.name,
            "lat": round(la, 6), "lon": round(lo, 6), "value": round(v, 3), "limit": lim,
            "severity": round(sev, 3), "note": note}


def _runs(bad, arc):
    """[(i0, i1)] runs of the True vertices of bad, a gap of less than RUN_GAP_M along arc joining two."""
    idx = np.flatnonzero(bad)
    if not len(idx):
        return []
    out, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if arc[i] - arc[b] <= RUN_GAP_M:
            b = i
        else:
            out.append((a, b))
            a = b = i
    out.append((a, b))
    return out


# ---- the per-part checks ---------------------------------------------------------------------------------------------

def check_ground(p):
    if p.kind != "ground" or p.terrain is None:
        return []
    d = p.z - p.terrain
    out = []
    for a, b in _runs(np.isfinite(d) & (np.abs(d) > GROUND_LIMIT_M), p.arc):
        seg = d[a:b + 1]
        i = a + int(np.nanargmax(np.abs(seg)))
        out.append(_finding("ground", p, i, d[i], "above terrain" if d[i] > 0 else "below terrain"))
    return out


def check_cover(p):
    if p.kind != "tunnel" or p.terrain is None or len(p.xy) < 2:
        return []
    cover = p.terrain - (p.z + p.th)
    judged = np.ones(len(cover), bool)
    for end, arc_at in ((0, p.arc), (1, p.length - p.arc)):
        if p.mouth[end]:
            judged &= arc_at > PORTAL_ZONE_M
    bad = judged & np.isfinite(cover) & ((cover < COVER_MIN_M) | (p.z > p.terrain))
    out = []
    for a, b in _runs(bad, p.arc):
        i = a + int(np.nanargmin(cover[a:b + 1]))
        out.append(_finding("cover", p, i, cover[i], "road above terrain" if p.z[i] > p.terrain[i] else "thin cover"))
    return out


def _resample(p, step=1.0):
    """(arc, z) of the part's surface every step m (linear between its vertices)."""
    s = np.arange(0.0, p.length + 1e-9, step)
    return s, np.interp(s, p.arc, p.z)


def check_grade(p):
    if not p.paved or p.length < GRADE_MIN_LEN_M or len(p.xy) < 2:
        return []
    w = min(GRADE_WINDOW_M, p.length)
    s, z = _resample(p)
    k = max(1, int(round(w)))
    if len(z) <= k:
        s, z, k = np.array([0.0, p.length]), np.interp([0.0, p.length], p.arc, p.z), 1
        w = p.length
    g = np.abs(z[k:] - z[:-k]) / (s[k:] - s[:-k])
    out = []
    # (a window is placed at its middle vertex of the 1 m grid; runs along it)
    mid = (s[:-k] + s[k:]) / 2
    for a, b in _runs(g > GRADE_MAX, mid):
        i = a + int(np.argmax(g[a:b + 1]))
        at = float(mid[i])
        j = int(np.argmin(np.abs(p.arc - at)))
        la = float(np.interp(at, p.arc, p.lat))
        lo = float(np.interp(at, p.arc, p.lon))
        out.append(_finding("grade", p, j, g[i], f"over {w:.0f} m", at=(la, lo)))
    return out


# ---- the checks between parts ----------------------------------------------------------------------------------------

def _cls(kind):
    return "surface" if kind in SURFACE else kind


def _ends(pieces):
    """[(piece, end, xy, z)] of the ends of every part."""
    out = []
    for pi, p in enumerate(pieces):
        if len(p.xy) < 2:
            continue
        out.append((pi, 0, p.xy[0], p.z[0]))
        out.append((pi, 1, p.xy[-1], p.z[-1]))
    return out


def _grid(points, cell):
    g = {}
    for k, (x, y) in enumerate(points):
        g.setdefault((int(math.floor(x / cell)), int(math.floor(y / cell))), []).append(k)
    return g


def _near(grid, cell, x, y, r):
    cx, cy = int(math.floor(x / cell)), int(math.floor(y / cell))
    n = int(math.ceil(r / cell))
    for i in range(cx - n, cx + n + 1):
        for j in range(cy - n, cy + n + 1):
            yield from grid.get((i, j), ())


def check_joints(pieces):
    """The seams (same way, same kind, ends within JOIN_M: the step over SEAM_STEP_M) and the portals (a bridge or
    tunnel end with the surface part or other structure it meets within APPROACH_M: the step over PORTAL_STEP_M)."""
    ends = _ends(pieces)
    cell = max(APPROACH_M, 1.0)
    grid = _grid([e[2] for e in ends], cell)
    out, seen = [], set()
    for ai, (pa, ea, xa, za) in enumerate(ends):
        A = pieces[pa]
        deck = _cls(A.kind) != "surface"
        best = None                                  # (priority, distance, B end) of A's approach
        for bi in _near(grid, cell, xa[0], xa[1], APPROACH_M):
            pb, eb, xb, zb = ends[bi]
            if pb == pa:
                continue
            B = pieces[pb]
            d = float(np.hypot(*(xa - xb)))
            if d > APPROACH_M:
                continue
            same_way = bool(A.group) and A.group == B.group
            if _cls(A.kind) == _cls(B.kind):                                    # a seam
                if same_way and d <= JOIN_M and (min(pa, pb), min(ea, eb), max(pa, pb)) not in seen:
                    seen.add((min(pa, pb), min(ea, eb), max(pa, pb)))
                    step = abs(float(za) - float(zb))
                    if step > SEAM_STEP_M:
                        out.append(_finding("seam", A, 0 if ea == 0 else len(A.xy) - 1, step,
                                            f"{A.kind} {A.fid[:8]} / {B.kind} {B.fid[:8]}"))
            elif deck:                                                          # a portal: A's approach
                rank = (0 if same_way else 1, d)
                if best is None or rank < best[0]:
                    best = (rank, bi)
        if best is not None:
            pb, eb, xb, zb = ends[best[1]]
            B = pieces[pb]
            key = ("p", min(pa, pb), max(pa, pb), min(ea, eb), max(ea, eb))
            if key not in seen:
                seen.add(key)
                step = abs(float(za) - float(zb))
                if step > PORTAL_STEP_M:
                    out.append(_finding("portal", A, 0 if ea == 0 else len(A.xy) - 1, step,
                                        f"{A.kind} {A.fid[:8]} meets {B.kind} {B.fid[:8]}"))
    return out


def _seg_cross(a0, a1, b0, b1):
    """(t, u) where segments a0-a1 and b0-b1 cross, or None."""
    r, s = a1 - a0, b1 - b0
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return None
    q = b0 - a0
    t = (q[0] * s[1] - q[1] * s[0]) / den
    u = (q[0] * r[1] - q[1] * r[0]) / den
    return (t, u) if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0 else None


def check_clearance(pieces):
    """Every bridge deck over a road (a surface part, or a lower deck): clearance under its structure at the crossing."""
    cell = 40.0
    segs = []                       # (piece, segment index)
    for pi, p in enumerate(pieces):
        if p.kind in ("tunnel",) or len(p.xy) < 2 or p.cls in NON_VEHICLE:
            continue
        for k in range(len(p.xy) - 1):
            segs.append((pi, k))
    mids = [(pieces[pi].xy[k] + pieces[pi].xy[k + 1]) / 2 for pi, k in segs]
    grid = _grid(mids, cell)
    out, kept = [], {}
    for pa, A in enumerate(pieces):
        if A.kind != "bridge" or len(A.xy) < 2:
            continue
        for ka in range(len(A.xy) - 1):
            a0, a1 = A.xy[ka], A.xy[ka + 1]
            m = (a0 + a1) / 2
            la = float(np.hypot(*(a1 - a0)))
            for si in _near(grid, cell, m[0], m[1], cell + la):
                pb, kb = segs[si]
                B = pieces[pb]
                if pb == pa or (A.group and A.group == B.group) or A.fid == B.fid:
                    continue
                c = _seg_cross(a0, a1, B.xy[kb], B.xy[kb + 1])
                if c is None:
                    continue
                t, u = c
                za = A.z[ka] + t * (A.z[ka + 1] - A.z[ka])
                zb = B.z[kb] + u * (B.z[kb + 1] - B.z[kb])
                if not (za > zb):
                    continue                    # (B is the upper one: judged from its own side when it is a deck)
                sa = float(A.arc[ka] + t * la)
                sb = float(B.arc[kb] + u * np.hypot(*(B.xy[kb + 1] - B.xy[kb])))
                if za - zb < JUNCTION_DZ_M and (min(sa, A.length - sa) < JUNCTION_END_M
                                                or min(sb, B.length - sb) < JUNCTION_END_M):
                    continue                    # (a road meeting the deck's abutment, a ramp: not an underpass)
                clear = float(za - A.depth - zb)
                if clear >= CLEARANCE_MIN_M:
                    continue
                at = (float(A.lat[ka] + t * (A.lat[ka + 1] - A.lat[ka])), float(A.lon[ka] + t * (A.lon[ka + 1] - A.lon[ka])))
                key = (pa, pb)
                prev = kept.get(key)
                if prev is not None and abs(prev[0] - sa) < RUN_GAP_M:     # (the same crossing met again on a bend)
                    if clear < prev[1]["value"]:
                        out[prev[2]] = _finding("clearance", A, ka, clear, f"over {B.kind} {B.fid[:8]} {B.cls}", at=at)
                        kept[key] = (sa, out[prev[2]], prev[2])
                    continue
                out.append(_finding("clearance", A, ka, clear, f"over {B.kind} {B.fid[:8]} {B.cls}", at=at))
                kept[key] = (sa, out[-1], len(out) - 1)
    return out


def run_checks(pieces):
    """Every finding (a list of dicts), worst first."""
    out = []
    for p in pieces:
        out += check_ground(p) + check_cover(p) + check_grade(p)
    out += check_joints(pieces) + check_clearance(pieces)
    out.sort(key=lambda f: (-f["severity"], f["check"], f["way"], f["fid"], f["lat"], f["lon"], f["value"]))
    return out


def totals(findings):
    t = {c: 0 for c in CHECKS}
    for f in findings:
        t[f["check"]] += 1
    return t


def total_line(t):
    return "levelcheck: " + ", ".join(f"{c} {t[c]}" for c in CHECKS) + f" (total {sum(t.values())})"


# ---- reading a pack ---------------------------------------------------------------------------------------------------

def load_pieces(pack_path, cache, bbox=None, offline=True, log=print):
    """(pieces, info): every road part of the pack's area as the game would draw it (see the module's docstring)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(here, "server"))
    import logging
    import bakepack
    import revisit
    revisit.ON = False                      # (plans and facts from the pack alone, not the cache's earlier copies)
    pk = bakepack.Pack(pack_path)
    bakepack.packs = lambda _cache_dir: [pk]
    import plancache
    run = f"levelcheck_{os.getpid()}"
    os.environ[plancache.RUN_ENV] = run
    variant = plancache.plans_variant()
    if (variant or None) != (pk.meta.get("variant") or None):
        raise SystemExit(f"the pack was made with variant {pk.meta.get('variant')!r}, this run is {variant!r}: set the "
                         "env vars it was baked with (BNE_SOFIA_DTM, BNE_SOFIA_STREETS, BNE_OVERRIDES, ...)")
    import blockworker as bw
    import grading
    import mapcells
    import roads
    bw.init(cache, offline, True, True, True, True, None, logging.ERROR, True, True, 0, None, False)
    s = bw._s
    if bbox is None:
        bbox = pk.meta.get("bbox")
    if not bbox:
        raise SystemExit("the pack has no bbox: pass --bbox S,W,N,E")
    S, W, N, E = bbox
    g, _ = s.anchor(1, (S + N) / 2, (W + E) / 2)
    frame = g.frame
    zoom = grading.PROFILE_ZOOM
    plan_dir = os.path.join(cache, "plans", run)
    try:
        keys = mapcells.cells_for_bbox(S, W, N, E, max_cells=200000)
        # (the features' index: the plans' ids to ways and specs; the cells round the box too: a road's record is
        # in each cell it crosses)
        ring = 0.01
        fkeys = mapcells.cells_for_bbox(S - ring, W - ring, N + ring, E + ring, max_cells=200000)
        feats_of = {k: s.md.cell(k, layers=("roads",), kind="demand").layers.get("roads") for k in fkeys}
        index = {}
        for k, f in feats_of.items():
            if f is None or not len(f):
                continue
            specs = roads.road_specs(f, None, s.widths)
            rec = np.asarray(f.attrs.get("src_record", [""] * len(f))).astype(str)
            lw = np.asarray(f.attrs.get("lane_osm_way", [0] * len(f)))
            names = f.attrs.get("name")
            for i in range(len(f)):
                fid = str(f.ids[i])
                if fid in index:
                    continue
                way = ""
                if rec[i].startswith("w") and rec[i][1:].split("@")[0].isdigit():
                    way = rec[i][1:].split("@")[0]
                elif int(lw[i]) > 0:
                    way = str(int(lw[i]))
                elif fid.startswith("osm-busway-") and fid.rsplit("-", 1)[1].isdigit():
                    way = str(int(fid.rsplit("-", 1)[1]))
                index[fid] = (specs[i], way, str(names[i]) if names is not None else "")

        def info_of(fid):
            base = fid.split(".")[0].split("#")[0]
            return index.get(base)

        def terrain_at(la, lo):
            return np.asarray(bw.ground(frame, la, lo, s.dem.sample(la, lo, zoom)), np.float64)

        def drawn_at(la, lo):
            return np.asarray(g.apply(la, lo, terrain_at(la, lo), grading.ROAD_LIFT), np.float64)

        pieces, skipped = [], {"unknown_feature": 0}
        pending = []                                # (kind, fid, spec, way, name, xy, z_enu or None, extras)
        seen = set()
        for k in keys:
            for fid, p in g._cell_plans(k):
                if grading.TUNNEL_TAG in fid or fid.startswith(grading.RAIL_TAG) or fid.startswith("corner:"):
                    continue
                key = (fid, round(float(p.xy[0, 0]), 3), round(float(p.xy[0, 1]), 3))
                if key in seen:
                    continue
                seen.add(key)
                inf = info_of(fid)
                if inf is None:
                    skipped["unknown_feature"] += 1
                    continue
                pending.append(("ground", fid, inf, np.asarray(p.xy, np.float64)[:, :2], None, {}))
        e0, n0, _ = frame.geodetic_to_enu(np.array([S, S, N, N]), np.array([W, E, W, E]), 0.0)
        box = (float(min(e0)) - 50, float(min(n0)) - 50, float(max(e0)) + 50, float(max(n0)) + 50)
        cells = [f for f in (feats_of.get(k) for k in keys) if f is not None]
        bridges = bw._bridges(1, frame, lambda la, lo: drawn_at(la, lo))
        tunnels = bw.tunnel_profiles(1, frame, lambda la, lo: drawn_at(la, lo), g)
        for fid, spec, xy, z, _lon in bridges.parts(cells, box):
            pending.append(("bridge", str(fid), (spec, *(info_of(str(fid)) or (None, "", ""))[1:]),
                            np.asarray(xy, np.float64)[:, :2], np.asarray(z, np.float64),
                            {"depth": float(bridges.depth_of(fid))}))
        for fid, spec, xy, z, _lon in tunnels.parts(cells, box):
            pending.append(("tunnel", str(fid), (spec, *(info_of(str(fid)) or (None, "", ""))[1:]),
                            np.asarray(xy, np.float64)[:, :2], np.asarray(z, np.float64),
                            {"th": float(tunnels.height(fid)), "mouth": tuple(bool(v) for v in tunnels.mouth(fid))}))
        for fid, spec, xy, z, _lon in tunnels.open_parts(cells, box):
            pending.append(("open", str(fid), (spec, *(info_of(str(fid)) or (None, "", ""))[1:]),
                            np.asarray(xy, np.float64)[:, :2], None, {}))
        # lat/lon of every vertex at once, terrain and the drawn ground (apply) in one call each
        allxy = np.concatenate([q[3] for q in pending]) if pending else np.zeros((0, 2))
        la, lo, _ = frame.enu_to_geodetic(allxy[:, 0], allxy[:, 1], 0.0)
        la, lo = np.asarray(la, np.float64), np.asarray(lo, np.float64)
        terr = terrain_at(la, lo)
        drawn = drawn_at(la, lo)
        at = 0
        for kind, fid, (spec, way, name), xy, zenu, extra in pending:
            n = len(xy)
            sl = slice(at, at + n)
            at += n
            if kind in ("bridge", "tunnel"):
                z = np.asarray(frame.enu_to_geodetic(xy[:, 0], xy[:, 1], zenu)[2], np.float64)
            else:
                z = drawn[sl]
            pieces.append(Piece(kind, way, fid, xy, la[sl], lo[sl], z, terr[sl],
                                cls=str(getattr(spec, "cls", "")), paved=getattr(spec, "kind", "paved") in roads.BUILT,
                                name=name, **extra))
        stats = dict(getattr(g.shared, "stats", {}) or {})
        info = {"bbox": [S, W, N, E], "cells": len(keys), "pieces": {k: sum(1 for p in pieces if p.kind == k)
                for k in ("ground", "open", "bridge", "tunnel")}, "skipped": skipped,
                "plans_read_from_pack": stats.get("baked", 0), "plans_made_here": stats.get("written", 0),
                "terrain_zoom": zoom, "variant": variant,
                "pack_meta": {k: pk.meta.get(k) for k in ("bbox", "variant", "signature", "plans_format", "facts_format")}}
        return pieces, info
    finally:
        shutil.rmtree(plan_dir, ignore_errors=True)


def inside(f, bbox):
    S, W, N, E = bbox
    return S <= f["lat"] <= N and W <= f["lon"] <= E


FIELDS = ("check", "way", "fid", "kind", "cls", "name", "lat", "lon", "value", "limit", "severity", "note")


def write_report(out_dir, findings, info):
    os.makedirs(out_dir, exist_ok=True)
    t = totals(findings)
    doc = {"version": 1, "thresholds": {**{f"{c}_limit": LIMITS[c] for c in CHECKS}, "grade_window_m": GRADE_WINDOW_M,
                                       "portal_zone_m": PORTAL_ZONE_M, "approach_m": APPROACH_M, "join_m": JOIN_M},
           "area": info, "totals": t, "findings": findings}
    with open(os.path.join(out_dir, "levelcheck.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    with open(os.path.join(out_dir, "levelcheck.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in findings:
            w.writerow({k: r[k] for k in FIELDS})
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", help="a .bnge pack (tools/bake_area.py)")
    ap.add_argument("--cache", help="the cache folder the pack was baked with (its map data and terrain); default: the "
                                    "pack's own folder")
    ap.add_argument("--bbox", help="S,W,N,E to report (default: the pack's box)")
    ap.add_argument("--out", default="levelcheck", help="folder for levelcheck.json / .csv")
    ap.add_argument("--online", action="store_true", help="allow downloads of missing map data (default: offline)")
    a = ap.parse_args(argv)
    t0 = time.time()
    bbox = [float(v) for v in a.bbox.split(",")] if a.bbox else None
    pieces, info = load_pieces(os.path.abspath(a.pack), os.path.abspath(a.cache or os.path.dirname(os.path.abspath(a.pack))),
                               bbox, not a.online)
    box = info["bbox"]
    findings = [f for f in run_checks(pieces) if inside(f, box)]
    t = write_report(a.out, findings, info)
    print(total_line(t))
    if info["plans_made_here"]:
        print(f"warning: {info['plans_made_here']} cells' plans were not in the pack and were planned here")
    print(f"{sum(info['pieces'].values())} parts {info['pieces']}, {len(findings)} findings, "
          f"{time.time() - t0:.1f} s -> {os.path.join(a.out, 'levelcheck.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
