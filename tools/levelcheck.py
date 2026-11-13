#!/usr/bin/env python3
"""levelcheck.py PACK --cache DIR [--bbox S,W,N,E] [--out DIR] [--online]: where the drawn road disagrees with the
ground, offline (no game), for a baked pack (server/bakepack.py) and the terrain model the pack was made with.

    BNE_SOFIA_DTM=... BNE_SOFIA_STREETS=... BNE_OVERRIDES=... \
        python tools/levelcheck.py dist/sofia.bnge --cache cache/full --out reports/levelcheck

Run it with the env vars the pack was baked with (they are part of its meta: a pack whose "variant" differs from the
running server's is refused). It writes levelcheck.json (every finding: check, way, lat, lon, value, limit, class, ...)
and levelcheck.csv (the same: the defects first, worst first, then the findings set apart as `earthwork` or `terrain`)
into --out and prints a one-line total per check: the totals count the defects, the others are counted apart. Two
runs over the same pack and inputs write byte-identical files (no times in them).

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

  ground      surface z of a ground part minus terrain, more than GROUND_LIMIT_M either way, per run of vertices. A run is
              an `earthwork` (counted apart, listed after the defects) when a structure explains it, else a `defect`:
                above the terrain: the road climbs to a bridge's end; below it: the road falls to a tunnel's mouth (or the
              open section of one) or dips under a bridge deck crossing it; and in each case the road, followed from the
              structure along the road network, meets its end z within GROUND_LIMIT_M, and the offset from the terrain
              never grows by more than TAPER_TOL_M while it falls to GROUND_LIMIT_M within the reach the generator builds
              such ramps (BRIDGE_RAMP_REACH_M, TUNNEL_RAMP_REACH_M, UNDER_RAMP_REACH_M). Every vertex of the run
              must be so explained: a run that is only near a structure, or that stops short of the ground, is a defect.
              Every ground finding also names the nearest bridge, tunnel or tunnel open section: near_kind, near_id,
              near_way, near_m, near_basis ("network": metres along the roads to its end or to where its deck crosses
              the road, within NET_REACH_M; "straight": metres in a straight line to its line where no such way leads
              to one), near_role, and near_straight_m, the straight-line distance to the nearest of them; run_m is the
              length of the run along the road (0 for one vertex).
  cover       a tunnel: terrain - (floor + vault height) under COVER_MIN_M, or the floor above the terrain; the first
              PORTAL_ZONE_M from a portal are not judged (the portal face rises to the ground there).
  clearance   a bridge deck over a road: (deck z - structure depth) - road surface z at the crossing of their
              centrelines, under CLEARANCE_MIN_M. Crossings with the same way, and junctions (a crossing within
              JUNCTION_END_M of an end of either, the two surfaces less than JUNCTION_DZ_M apart) are skipped.
  portal      a bridge or tunnel end and the surface part (or another structure) it meets within APPROACH_M: the
              height step over PORTAL_STEP_M.
  grade       the steepest rise over GRADE_WINDOW_M along a paved part's own surface, over GRADE_MAX. A run is `terrain`
              (counted apart) where the road is no steeper than the ground it runs on: the terrain along the same centre
              line, smoothed over TERRAIN_SMOOTH_M, is read over the same windows, and the road's largest excess over it,
              taken over the run's windows past GRADE_MAX, is at most GRADE_TERRAIN_MARGIN and the steepest window at
              most GRADE_CEILING (a paved road steeper than that is a defect whatever the ground does). A bridge or tunnel
              part, or a part with no terrain, is a defect: the ground under a deck says nothing of its grade.
  seam        two parts of one way of the same kind meeting (ends within JOIN_M): the height step over SEAM_STEP_M.
"""
import argparse
import csv
import heapq
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

# ---- how findings are classed (measured on the Sofia full area; see the report that came with them) --------------------
# A ground finding is an `earthwork` when a structure explains it (see the docstring). The reaches are the generator's:
BRIDGE_RAMP_REACH_M = 400.0     # a road climbing to a bridge's end: grading.RAMP_REACH_M (the envelope goes at most this far
                                # along the roads from a raised one; grading.APPROACH_MAX_RUN, the longest ramp, is 320)
TUNNEL_RAMP_REACH_M = 130.0     # a road falling to a tunnel's mouth: roads.FINE_APPROACH_MAX_M / roads.TUNNEL_GRADE (6.5 m at
                                # 5 %; the street tunnels' own ramps fall at roads.FINE_RAMP_GRADE, 10 %, so 65 m)
UNDER_RAMP_REACH_M = 221.0      # a road dipping under a deck: grading.UNDER_FLAT_M + UNDER_MAX_DROP_M * pi / (2 * UNDER_GRADE) (12 + 8 * pi / .12)
TAPER_TOL_M = 1.5               # the offset may rise this much going away from the structure (terrain relief: of the 458 runs
                                # a 5 m tolerance would explain, 433 need 1.0 m or less, 440 need 1.5 m, the rest 2 to 5 m)
NODE_M = 2.5                    # part ends this close are one node of the road network (= APPROACH_M)
NET_REACH_M = 400.0             # nearest structure along the roads: looked for this far

# A grade finding is `terrain` when the road is no steeper than the ground it runs on (see the docstring):
TERRAIN_SMOOTH_M = 5.0          # terrain smoothed over this (box) along the centre line: measured, see below
GRADE_TERRAIN_MARGIN = 0.07     # road steeper than the terrain by more than this in any window of the run past GRADE_MAX ...
GRADE_CEILING = 0.25            # ... or over this anywhere: a defect (grade finding, any ground)

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


def _finding(check, p, i, value, note="", at=None, **extra):
    """A finding of check on piece p at vertex i (or at = (lat, lon)); extra: more fields (class, context)."""
    la, lo = (float(p.lat[i]), float(p.lon[i])) if at is None else at
    lim = LIMITS[check]
    v = float(value)
    if check == "ground":
        sev = abs(v) / lim
    elif check in ("cover", "clearance"):
        sev = 1.0 + (lim - v) / lim
    else:
        sev = abs(v) / lim
    out = {"check": check, "way": p.way, "fid": p.fid, "kind": p.kind, "cls": p.cls, "name": p.name,
           "lat": round(la, 6), "lon": round(lo, 6), "value": round(v, 3), "limit": lim,
           "severity": round(sev, 3), "note": note, "class": "defect"}
    out.update(extra)
    return out


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

def check_ground(p, ctx=None):
    """The ground findings of a part. With ctx (a Context of all the parts) each says what structure is nearest and is
    classed `earthwork` where one explains it; without, every finding is a `defect` with no context."""
    if p.kind != "ground" or p.terrain is None:
        return []
    d = p.z - p.terrain
    out = []
    for a, b in _runs(np.isfinite(d) & (np.abs(d) > GROUND_LIMIT_M), p.arc):
        seg = d[a:b + 1]
        i = a + int(np.nanargmax(np.abs(seg)))
        extra = ctx.ground_context(p, a, b, i) if ctx is not None else {}
        extra["run_m"] = round(float(p.arc[b] - p.arc[a]), 1)
        out.append(_finding("ground", p, i, d[i], "above terrain" if d[i] > 0 else "below terrain", **extra))
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


def _smooth(z, width_m, step=1.0):
    """z (every step m) averaged over width_m (a box; the ends reflected about their end points, so a slope keeps its
    grade to the end)."""
    h = int(round(width_m / step)) // 2
    if h < 1 or len(z) < 2:
        return z
    zp = np.pad(z, h, mode="reflect", reflect_type="odd")
    c = np.r_[0.0, np.cumsum(zp)]
    return (c[2 * h + 1:] - c[:-(2 * h + 1)]) / (2 * h + 1)


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
    # the terrain's grade over the same windows, on the terrain smoothed along the centre line: only where there is a
    # ground to follow (a deck or a tube says nothing of it)
    gt = None
    if p.kind in SURFACE and p.terrain is not None and np.all(np.isfinite(p.terrain)):
        t = _smooth(np.interp(s, p.arc, p.terrain), TERRAIN_SMOOTH_M)
        gt = np.abs(t[k:] - t[:-k]) / (s[k:] - s[:-k])
    out = []
    # (a window is placed at its middle vertex of the 1 m grid; runs along it)
    mid = (s[:-k] + s[k:]) / 2
    for a, b in _runs(g > GRADE_MAX, mid):
        i = a + int(np.argmax(g[a:b + 1]))
        at = float(mid[i])
        j = int(np.argmin(np.abs(p.arc - at)))
        la = float(np.interp(at, p.arc, p.lat))
        lo = float(np.interp(at, p.arc, p.lon))
        extra = {}
        if gt is not None:
            steep = g[a:b + 1] > GRADE_MAX
            excess = float(np.max((g[a:b + 1] - gt[a:b + 1])[steep]))
            extra = {"terrain_grade": round(float(gt[i]), 4), "excess": round(excess, 4)}
            if excess <= GRADE_TERRAIN_MARGIN and float(g[i]) <= GRADE_CEILING:
                extra["class"] = "terrain"
        out.append(_finding("grade", p, j, g[i], f"over {w:.0f} m", at=(la, lo), **extra))
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


def deck_crossings(pieces):
    """Every crossing of a bridge deck over a road (a surface part, or a lower deck) that is no junction, in the order
    check_clearance judges them: (pa, ka, pb, kb, t, u, za, zb, sa, sb), the deck A's segment ka at t, the road B's
    segment kb at u, their heights there and the arcs along each."""
    cell = 40.0
    segs = []                       # (piece, segment index)
    for pi, p in enumerate(pieces):
        if p.kind in ("tunnel",) or len(p.xy) < 2 or p.cls in NON_VEHICLE:
            continue
        for k in range(len(p.xy) - 1):
            segs.append((pi, k))
    mids = [(pieces[pi].xy[k] + pieces[pi].xy[k + 1]) / 2 for pi, k in segs]
    grid = _grid(mids, cell)
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
                yield pa, ka, pb, kb, t, u, float(za), float(zb), sa, sb


def check_clearance(pieces):
    """Every bridge deck over a road (a surface part, or a lower deck): clearance under its structure at the crossing."""
    out, kept = [], {}
    for pa, ka, pb, kb, t, u, za, zb, sa, sb in deck_crossings(pieces):
        A, B = pieces[pa], pieces[pb]
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


class Context:
    """The parts as a road network, for what a single part cannot say: which structure is nearest a ground finding, and
    whether one explains it (an earthwork). Nodes: part ends within NODE_M are one node; ground parts are the roads
    between them; bridge, tunnel and open parts are structures standing at nodes (or, a deck, over a road)."""

    def __init__(self, pieces):
        self.pieces = pieces
        self.index = {id(p): i for i, p in enumerate(pieces)}
        self.node_of = {}                           # (piece, end) -> node
        self.ground_at = {}                         # node -> [(ground piece, end)]
        self.struct_at = {}                         # node -> [(structure piece, end)]
        self._nodes()
        self.over = {}                              # ground piece -> [(arc of a deck's crossing, deck piece, segment)]
        for pa, ka, pb, kb, t, u, za, zb, sa, sb in deck_crossings(pieces):
            if pieces[pb].kind == "ground":
                self.over.setdefault(pb, []).append((sb, pa, kb))
        self.explained = {}                         # (ground piece, vertex) -> (origin, taper length m)
        self._mark_earthworks()
        self._structure_lines()

    # -- the network
    def _nodes(self):
        ends = _ends(self.pieces)
        cell = max(NODE_M, 1.0)
        grid = _grid([e[2] for e in ends], cell)
        root = list(range(len(ends)))

        def find(a):
            while root[a] != a:
                root[a] = root[root[a]]
                a = root[a]
            return a

        for ai, (_pa, _ea, xa, _za) in enumerate(ends):
            for bi in _near(grid, cell, xa[0], xa[1], NODE_M):
                if bi > ai and float(np.hypot(*(xa - ends[bi][2]))) <= NODE_M:
                    ra, rb = find(ai), find(bi)
                    if ra != rb:
                        root[max(ra, rb)] = min(ra, rb)
        for ai, (pa, ea, _xa, _za) in enumerate(ends):
            n = find(ai)
            self.node_of[(pa, ea)] = n
            at = self.ground_at if self.pieces[pa].kind == "ground" else self.struct_at
            at.setdefault(n, []).append((pa, ea))

    # -- earthworks: what a structure's ramps explain
    def _mark_earthworks(self):
        P = self.pieces
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))      # (a walk goes part by part, a few metres each)
        for pi, p in enumerate(P):
            if p.kind == "ground" or len(p.xy) < 2:
                continue
            if p.kind == "bridge":
                sgn, reach = 1, BRIDGE_RAMP_REACH_M
            else:
                sgn, reach = -1, TUNNEL_RAMP_REACH_M
            for end in (0, 1):
                if p.kind == "tunnel" and not p.mouth[end]:
                    continue                        # (a joint inside a tunnel, not a mouth)
                z_end = p.z[0] if end == 0 else p.z[-1]
                for pj, ej in self.ground_at.get(self.node_of[(pi, end)], ()):
                    q = P[pj]
                    if q.terrain is None or abs((q.z[0] if ej == 0 else q.z[-1]) - z_end) > GROUND_LIMIT_M:
                        continue                    # (the road does not rise to the structure: no ramp of it)
                    ks, ss = self._from_end(q, ej, 0.0)
                    self._walk((p.kind, p.fid, p.way, "end"), sgn, reach, pj, ks, ss, ej == 0, math.inf, {pj})
        for pb, lst in sorted(self.over.items()):
            q = P[pb]
            if q.terrain is None:
                continue
            for sb, pa, kb in lst:
                A = P[pa]
                origin = (A.kind, A.fid, A.way, "over")
                back = np.arange(kb, -1, -1)
                self._walk(origin, -1, UNDER_RAMP_REACH_M, pb, back, sb - q.arc[back], False, math.inf, {pb})
                fwd = np.arange(kb + 1, len(q.xy))
                self._walk(origin, -1, UNDER_RAMP_REACH_M, pb, fwd, q.arc[fwd] - sb, True, math.inf, {pb})

    @staticmethod
    def _from_end(q, end, s0):
        """(vertices, distances along the road) of a part walked from its end."""
        if end == 0:
            return np.arange(len(q.xy)), s0 + q.arc
        return np.arange(len(q.xy) - 1, -1, -1), s0 + (q.length - q.arc[::-1])

    def _walk(self, origin, sgn, reach, pj, ks, ss, fwd, runmin, onpath):
        """Follow the road along part pj's vertices ks (distances ss from the structure), then on through its far end's
        node into every road there. sgn * (z - terrain) is the offset the structure explains (bridge: above, tunnel and
        deck over it: below). The walk holds while the offset does not grow past TAPER_TOL_M and ends (the ground is
        reached) where it is within GROUND_LIMIT_M; it fails when it grows, changes sign, runs past `reach` or comes to
        a road's end first. The vertices of a walk that reaches the ground are marked explained. Returns the distance
        at which the ground was reached on the first branch that did, or None."""
        q = self.pieces[pj]
        d = q.z - q.terrain
        for n, k in enumerate(ks):
            dk = float(d[k])
            if not math.isfinite(dk):
                return None
            if abs(dk) <= GROUND_LIMIT_M:
                self._mark(pj, ks[:n + 1], origin, float(ss[n]))
                return float(ss[n])
            x = sgn * dk
            if x < 0.0 or x > runmin + TAPER_TOL_M or ss[n] > reach:
                return None
            runmin = min(runmin, x)
        if not len(ks):
            return None
        hit = None
        for pk, ek in self.ground_at.get(self.node_of[(pj, 1 if fwd else 0)], ()):
            r = self.pieces[pk]
            if pk in onpath or r.terrain is None:
                continue
            ks2, ss2 = self._from_end(r, ek, float(ss[-1]))
            onpath.add(pk)
            res = self._walk(origin, sgn, reach, pk, ks2, ss2, ek == 0, runmin, onpath)
            onpath.discard(pk)
            if res is not None and hit is None:
                hit = res
        if hit is not None:
            self._mark(pj, ks, origin, hit)
        return hit

    def _mark(self, pj, ks, origin, taper):
        for k in ks:
            self.explained.setdefault((pj, int(k)), (origin, taper))

    # -- the nearest structure
    def _structure_lines(self):
        a, b, who = [], [], []
        for pi, p in enumerate(self.pieces):
            if p.kind != "ground" and len(p.xy) >= 2:
                a.append(p.xy[:-1])
                b.append(p.xy[1:])
                who += [pi] * (len(p.xy) - 1)
        self.line_a = np.concatenate(a) if a else np.zeros((0, 2))
        self.line_b = np.concatenate(b) if b else np.zeros((0, 2))
        self.line_of = np.array(who, np.int64)

    def straight(self, xy):
        """(distance m, structure piece) of the nearest structure line to xy, or (None, None)."""
        if not len(self.line_of):
            return None, None
        ab = self.line_b - self.line_a
        den = np.maximum((ab * ab).sum(axis=1), 1e-12)
        t = np.clip(((xy - self.line_a) * ab).sum(axis=1) / den, 0.0, 1.0)
        d = np.hypot(*(self.line_a + t[:, None] * ab - xy).T)
        j = int(np.argmin(d))
        return float(d[j]), int(self.line_of[j])

    def nearest(self, pi, k):
        """(distance m, structure piece, role) of the nearest structure along the roads from vertex k of ground part pi:
        to a structure's end at a node ("end") or to where a deck crosses over a road ("over"), within NET_REACH_M; or
        None."""
        P = self.pieces
        p = P[pi]
        best = None

        def offer(dist, si, role):
            nonlocal best
            key = (round(dist, 6), P[si].kind, P[si].fid, role)
            if best is None or key < best[0]:
                best = (key, dist, si, role)

        for sb, pa, _kb in self.over.get(pi, ()):
            offer(abs(sb - float(p.arc[k])), pa, "over")
        heap, seen = [], {}
        for end, dist in ((0, float(p.arc[k])), (1, float(p.length - p.arc[k]))):
            n = self.node_of.get((pi, end))
            if n is not None:
                heapq.heappush(heap, (dist, n))
        while heap:
            dist, n = heapq.heappop(heap)
            if n in seen or dist > NET_REACH_M or (best is not None and dist > best[1]):
                continue
            seen[n] = dist
            for si, _se in self.struct_at.get(n, ()):
                offer(dist, si, "end")
            for pj, ej in self.ground_at.get(n, ()):
                r = P[pj]
                for sb, pa, _kb in self.over.get(pj, ()):
                    offer(dist + (sb if ej == 0 else r.length - sb), pa, "over")
                far = self.node_of.get((pj, 1 - ej))
                if far is not None and far not in seen:
                    heapq.heappush(heap, (dist + r.length, far))
        return None if best is None else best[1:]

    def ground_context(self, p, a, b, i):
        """The fields a ground finding (run a..b of part p, worst vertex i) carries: its class and the nearest structure."""
        pi = self.index[id(p)]
        d = p.z - p.terrain
        bad = [k for k in range(a, b + 1) if np.isfinite(d[k]) and abs(d[k]) > GROUND_LIMIT_M]
        why = [self.explained.get((pi, k)) for k in bad]
        out = {"class": "defect", "earth_by": None, "taper_m": None, "near_kind": None, "near_id": None,
               "near_way": None, "near_m": None, "near_basis": None, "near_role": None, "near_straight_m": None}
        if bad and all(w is not None for w in why):
            (kind, fid, _way, role), taper = self.explained.get((pi, i)) or why[0]
            out.update({"class": "earthwork", "earth_by": f"{kind} {fid[:8]} {role}", "taper_m": round(taper, 1)})
        ds, sj = self.straight(p.xy[i])
        if ds is not None:
            out["near_straight_m"] = round(ds, 1)
        net = self.nearest(pi, i)
        if net is not None:
            dist, si, role = net
            basis = "network"
        elif ds is not None:
            dist, si, role, basis = ds, sj, "line", "straight"
        else:
            return out
        S = self.pieces[si]
        out.update({"near_kind": S.kind, "near_id": S.fid, "near_way": S.way, "near_m": round(dist, 1),
                    "near_basis": basis, "near_role": role})
        return out


def run_checks(pieces):
    """Every finding (a list of dicts): the defects first, worst first, then the findings set apart (earthworks, grades
    that follow the terrain), worst first."""
    out = []
    ctx = Context(pieces)
    for p in pieces:
        out += check_ground(p, ctx) + check_cover(p) + check_grade(p)
    out += check_joints(pieces) + check_clearance(pieces)
    out.sort(key=lambda f: (f["class"] != "defect", -f["severity"], f["check"], f["way"], f["fid"], f["lat"], f["lon"],
                            f["value"]))
    return out


SET_APART = {"ground": "earthwork", "grade": "terrain"}      # the class each check sets apart


def totals(findings):
    """{check: number of defects}: the findings set apart are counted by set_apart."""
    t = {c: 0 for c in CHECKS}
    for f in findings:
        if f["class"] == "defect":
            t[f["check"]] += 1
    return t


def set_apart(findings):
    """{check: number of findings of the class that check sets apart}."""
    t = {c: 0 for c in SET_APART}
    for f in findings:
        if f["class"] != "defect":
            t[f["check"]] += 1
    return t


def total_line(t, apart=None):
    apart = apart or {}
    parts = []
    for c in CHECKS:
        extra = f" (+{apart[c]} {SET_APART[c]})" if apart.get(c) else ""
        parts.append(f"{c} {t[c]}{extra}")
    tail = f"total {sum(t.values())} defects" + (f", {sum(apart.values())} set apart" if sum(apart.values()) else "")
    return "levelcheck: " + ", ".join(parts) + f" ({tail})"


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


VERSION = 2                 # the report's layout: 2 = classes (defect / earthwork / terrain), context of ground findings
FIELDS = ("check", "class", "way", "fid", "kind", "cls", "name", "lat", "lon", "value", "limit", "severity", "note",
          "earth_by", "taper_m", "near_kind", "near_id", "near_way", "near_m", "near_basis", "near_role",
          "near_straight_m", "run_m", "terrain_grade", "excess")


def write_report(out_dir, findings, info):
    os.makedirs(out_dir, exist_ok=True)
    t = totals(findings)
    apart = set_apart(findings)
    thresholds = {**{f"{c}_limit": LIMITS[c] for c in CHECKS}, "grade_window_m": GRADE_WINDOW_M,
                  "portal_zone_m": PORTAL_ZONE_M, "approach_m": APPROACH_M, "join_m": JOIN_M,
                  "bridge_ramp_reach_m": BRIDGE_RAMP_REACH_M, "tunnel_ramp_reach_m": TUNNEL_RAMP_REACH_M,
                  "under_ramp_reach_m": UNDER_RAMP_REACH_M, "taper_tol_m": TAPER_TOL_M, "node_m": NODE_M,
                  "net_reach_m": NET_REACH_M, "terrain_smooth_m": TERRAIN_SMOOTH_M,
                  "grade_terrain_margin": GRADE_TERRAIN_MARGIN, "grade_ceiling": GRADE_CEILING}
    doc = {"version": VERSION, "thresholds": thresholds, "area": info, "totals": t,
           "set_apart": {f"{c}_{SET_APART[c]}": apart[c] for c in SET_APART}, "findings": findings}
    with open(os.path.join(out_dir, "levelcheck.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    with open(os.path.join(out_dir, "levelcheck.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n", restval="")
        w.writeheader()
        for r in findings:
            w.writerow({k: r.get(k) for k in FIELDS})
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
    print(total_line(t, set_apart(findings)))
    if info["plans_made_here"]:
        print(f"warning: {info['plans_made_here']} cells' plans were not in the pack and were planned here")
    print(f"{sum(info['pieces'].values())} parts {info['pieces']}, {len(findings)} findings, "
          f"{time.time() - t0:.1f} s -> {os.path.join(a.out, 'levelcheck.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
