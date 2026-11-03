#!/usr/bin/env python
"""Bubble rate per GPU from torch-profiler chrome traces (EVALUATION §7).

  bubble = 1 - |union of model-kernel intervals| / span

Model kernels = every GPU kernel except NCCL (allreduce/send/recv spin while
waiting for the peer, so they are NOT model compute); memcpy/memset are not
counted as model compute either. Overlapping streams are merged (union).
The non-model time is split into
  nccl_only : some NCCL kernel runs, no model kernel (communication / waiting
              for a peer inside NCCL)
  idle      : no kernel at all (starved: waiting for input, host, wire)
Span = first kernel start .. last kernel end of the profiled window.

  python bubble.py $HET_RESULTS/bubble/ours_250/prof $HET_RESULTS/bubble/native_250/prof \
      [--csv out.csv] [--clip A:B]

--clip A:B keeps only [A, B] seconds after each trace's first kernel. Use it
when a profiler stop stalls the pipeline (synchronous trace export): the
stage still recording then sits in NCCL waits that are an artefact of the
measurement, not of the system (seen with Model Runner V2's ~10K kernels/s,
where the export took ~2 min).
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path


def union(iv):
    iv = sorted(iv)
    out, cur_s, cur_e = 0.0, None, None
    merged = []
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        merged.append((cur_s, cur_e))
    return merged


def length(m):
    return sum(e - s for s, e in m)


def intersect(a, b):
    i = j = 0
    out = 0.0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            out += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def is_nccl(name: str) -> bool:
    n = name.lower()
    return "nccl" in n or "cross_device_reduce" in n


CLIP: tuple[float, float] | None = None


def _clip(iv, lo, hi):
    return [(max(s, lo), min(e, hi)) for s, e in iv if e > lo and s < hi]


def analyze(path: Path) -> dict:
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        tr = json.load(f)
    model, nccl = [], []
    for e in tr.get("traceEvents", []):
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        s, d = float(e["ts"]), float(e.get("dur", 0))
        (nccl if is_nccl(e.get("name", "")) else model).append((s, s + d))
    allk = model + nccl
    if not allk:
        return {}
    if CLIP is not None:
        t0 = min(s for s, _ in allk)
        lo, hi = t0 + CLIP[0] * 1e6, t0 + CLIP[1] * 1e6
        model, nccl = _clip(model, lo, hi), _clip(nccl, lo, hi)
        allk = model + nccl
        span = min(hi, max(e for _, e in allk)) - lo
    else:
        span = max(e for _, e in allk) - min(s for s, _ in allk)
    m, n = union(model), union(nccl)
    model_t = length(m)
    nccl_only = length(n) - intersect(m, n)
    return {"span_s": span / 1e6, "model": model_t / span,
            "nccl_only": nccl_only / span,
            "idle": 1 - (model_t + nccl_only) / span,
            "bubble": 1 - model_t / span,
            "n_kernels": len(allk)}


def main():
    args = sys.argv[1:]
    csv_out = None
    if "--csv" in args:
        i = args.index("--csv")
        csv_out = args[i + 1]
        del args[i:i + 2]
    global CLIP
    if "--clip" in args:
        i = args.index("--clip")
        a, b = args[i + 1].split(":")
        CLIP = (float(a), float(b))
        del args[i:i + 2]
    import re
    rows = []
    for d in args:
        print(f"== {d}")
        files = [p for p in sorted(Path(d).iterdir())
                 if p.name.endswith((".json", ".json.gz"))]
        # "..._rank3.pt.trace.json.gz" (TP2) or "rank1.<id>.pt.trace.json.gz"
        ranks = {p: int(re.search(r"rank(\d+)", p.name).group(1))
                 for p in files}
        per_stage = max(1, len(files) // 2)   # PP2: first half = stage 0
        for p in files:
            r = analyze(p)
            if not r:
                continue
            rank = str(ranks[p])
            rows.append({"run": Path(d).parent.name, "gpu_rank": rank,
                         "stage": ranks[p] // per_stage,
                         **{k: round(v, 4) for k, v in r.items()}})
            print(f"  {p.name[:60]:60s} span {r['span_s']:6.1f}s  "
                  f"model {r['model']:6.1%}  nccl_only {r['nccl_only']:6.1%}"
                  f"  idle {r['idle']:6.1%}  BUBBLE {r['bubble']:6.1%}"
                  f"  ({r['n_kernels']} kernels)")
    if csv_out:
        import csv
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
