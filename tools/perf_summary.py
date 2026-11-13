#!/usr/bin/env python3
"""Summarise the server log's "game: perf minute" lines between timestamps, one column per run.

  python tools/perf_summary.py --run "before=2026-11-12 18:00,2026-11-12 18:25" --run "after=2026-11-12 19:00,2026-11-12 19:25"

Reads server.log, server.log.1 ... in --logs (default %LOCALAPPDATA%\\beamng-earth\\logs). A time is "YYYY-MM-DD HH:MM"
(seconds optional). Per run: minutes logged, median fps, median 1% low, hitches per minute (the logged counts summed over the
minutes), the worst single frame, and the number of "catch-up boost:" lines (the world builder behind the car for 10 s).
"""
import argparse
import glob
import os
import re
import statistics

PERF = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\S* .*game: perf minute (\d+) fps, 1% low (\d+), "
                  r"worst (\d+) ms, (\d+) hitches")
BOOST = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\S* .*catch-up boost: ")


def stamp(text):
    text = text.strip()
    return text + ":00" if len(text) == 16 else text


def lines(folder):
    for path in sorted(glob.glob(os.path.join(folder, "server.log*"))):
        with open(path, encoding="utf-8", errors="replace") as f:
            yield from f


def summarise(folder, start, end):
    fps, low, worst, hitches, boosts = [], [], [], 0, 0
    for ln in lines(folder):
        m = PERF.match(ln)
        if m and start <= m.group(1) <= end:
            fps.append(int(m.group(2)))
            low.append(int(m.group(3)))
            worst.append(int(m.group(4)))
            hitches += int(m.group(5))
            continue
        m = BOOST.match(ln)
        if m and start <= m.group(1) <= end:
            boosts += 1
    n = len(fps)
    if not n:
        return {"minutes": 0}
    return {"minutes": n, "median fps": statistics.median(fps), "median 1% low": statistics.median(low),
            "hitches / min": round(hitches / n, 1), "worst frame ms": max(worst), "catch-up boost lines": boosts}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=FROM,TO")
    ap.add_argument("--logs", default=os.path.join(os.environ.get("LOCALAPPDATA", ""), "beamng-earth", "logs"))
    a = ap.parse_args()
    runs = []
    for r in a.run:
        name, _, span = r.partition("=")
        start, _, end = span.partition(",")
        runs.append((name, summarise(a.logs, stamp(start), stamp(end))))
    keys = ["minutes", "median fps", "median 1% low", "hitches / min", "worst frame ms", "catch-up boost lines"]
    print(f"{'':24}" + "".join(f"{name:>14}" for name, _ in runs))
    for k in keys:
        print(f"{k:24}" + "".join(f"{s.get(k, '-')!s:>14}" for _, s in runs))


if __name__ == "__main__":
    main()
