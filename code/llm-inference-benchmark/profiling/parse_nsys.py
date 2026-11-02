"""Turn Nsight Systems CSV exports into the two numbers that explain latency:
  1. which kernels dominate GPU time (top-N share)
  2. how much of the wall-clock window the GPU sat idle (gaps between kernels
     -> CPU scheduling / launch overhead / waiting on the queue, not compute)
"""
import csv
import sys


def main(kern_csv, trace_csv):
    rows = list(csv.DictReader(open(kern_csv)))
    tcol = next(c for c in rows[0] if "Total Time" in c)
    ncol = next(c for c in rows[0] if c.strip() == "Name")
    tot = sum(float(r[tcol]) for r in rows)
    print("Top kernels by GPU time:")
    for r in sorted(rows, key=lambda r: -float(r[tcol]))[:10]:
        print(f"  {100*float(r[tcol])/tot:5.1f}%  {r[ncol][:90]}")
    tr = list(csv.DictReader(open(trace_csv)))
    scol = next(c for c in tr[0] if c.startswith("Start"))
    dcol = next(c for c in tr[0] if c.startswith("Duration"))
    ev = sorted((float(r[scol]), float(r[dcol])) for r in tr)
    window = ev[-1][0] + ev[-1][1] - ev[0][0]
    busy, end = 0.0, ev[0][0]
    for s, d in ev:                      # union of busy intervals
        if s > end:
            busy += d
            end = s + d
        elif s + d > end:
            busy += s + d - end
            end = s + d
    print(f"GPU busy {100*busy/window:.1f}% of the captured window; idle {100*(1-busy/window):.1f}%")


if __name__ == "__main__":
    main(*sys.argv[1:3])
