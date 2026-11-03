#!/usr/bin/env python
"""Where the non-model time goes, per NCCL kernel kind (round-6 bubble runs).

For each GPU trace: share of the window covered by model kernels, by each
NCCL kind (Broadcast / SendRecv / AllReduce / AllGather / other NCCL, union
per kind; kinds can overlap each other), and by no kernel at all.

  python nccl_split.py --clip 2:32 $HET_RESULTS/bubble/ours_250/prof ...
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bubble import _clip, intersect, is_nccl, length, union  # noqa: E402

KINDS = ("Broadcast", "SendRecv", "AllReduce", "AllGather")


def kind(name: str) -> str:
    for k in KINDS:
        if k.lower() in name.lower():
            return k
    if "cross_device_reduce" in name:
        return "AllReduce"
    return "otherNCCL"


def split(path: Path, clip):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        tr = json.load(f)
    model, per = [], {}
    for e in tr.get("traceEvents", []):
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        s, d = float(e["ts"]), float(e.get("dur", 0))
        n = e.get("name", "")
        if is_nccl(n):
            per.setdefault(kind(n), []).append((s, s + d))
        else:
            model.append((s, s + d))
    allk = model + [x for v in per.values() for x in v]
    t0 = min(s for s, _ in allk)
    lo, hi = t0 + clip[0] * 1e6, t0 + clip[1] * 1e6
    span = hi - lo
    m = union(_clip(model, lo, hi))
    out = {"model": length(m) / span}
    anyk = list(m)
    for k, iv in per.items():
        u = union(_clip(iv, lo, hi))
        out[k] = (length(u) - intersect(u, m)) / span  # NCCL time not under model
        anyk += u
    out["no_kernel"] = 1 - length(union(anyk)) / span
    return out


def main():
    args = sys.argv[1:]
    clip = (2.0, 32.0)
    if "--clip" in args:
        i = args.index("--clip")
        clip = tuple(float(x) for x in args[i + 1].split(":"))
        del args[i:i + 2]
    for d in args:
        print(f"== {d}")
        for p in sorted(Path(d).iterdir()):
            if not p.name.endswith((".json", ".json.gz")):
                continue
            r = split(p, clip)
            rank = re.search(r"rank(\d+)", p.name).group(1)
            print(f"  rank {rank}: " + "  ".join(
                f"{k} {v:5.1%}" for k, v in sorted(r.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
