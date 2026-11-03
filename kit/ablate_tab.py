#!/usr/bin/env python
"""Ablation summary ($HET_RESULTS/runs/{pfx}_{config}_{cell}_{band}.json from ablate.sh).

Per config: speedup vs native on each of the 12 throughput cells (decode:
out tok/s ratio; prefill/mix: native wall / config wall), geometric mean
overall and per band, plus the sat_long@0.5@250 SLO run (goodput, requests
passing, TTFT p50, raw out tok/s).

  python ablate_tab.py --pfx ab --base natv1 --tags nat2,best,bestsync [--csv out.csv]
  python ablate_tab.py --runs ../data/round6 --base r6:natv1 --tags r6:LQM2t   # shipped data
  tags/base may carry their own prefix as "pfx:tag" (e.g. --base r6:natv1
  --tags ab:best,r6:LQM2t) to mix runs from several matrices.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

R = Path(os.environ.get("HET_RESULTS", "./results")).resolve() / "runs"
BANDS = [100, 250, 1000]
CELLS = ["dec4", "dec32", "pre8192", "mix"]
CONFIGS = [  # tag, description (what is ON)
    ("nat", "原生 vLLM"),
    ("nat_moe0", "原生，去掉 MoE 调参"),
    ("tpc", "原生 + TP-C（NCCL 传输）"),
    ("L", "+ pp_link（bf16，两个向量）"),
    ("LQ", "+ int4 编码"),
    ("LQM", "+ 残差合并"),
    ("full", "+ TP-C = 终版（--best）"),
    ("full_noQ", "终版去掉 int4（bf16 合并）"),
    ("full_noM", "终版去掉合并（int4 两个向量）"),
]


def _opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if "--runs" in sys.argv:  # e.g. the shipped ../data/round6
    R = Path(_opt("--runs", "")).resolve()
PFX = _opt("--pfx", "ab")
BASE = _opt("--base", "nat")


def load(tag, cell, band):
    pfx, t = tag.split(":", 1) if ":" in tag else (PFX, tag)
    f = R / f"{pfx}_{t}_{cell}_{band}.json"
    return json.load(open(f)) if f.exists() else None


def geo(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def main():
    csv_out = _opt("--csv", None)
    rows = []
    known = dict(CONFIGS)
    tags = _opt("--tags", None)
    configs = ([(t, known.get(t, t)) for t in tags.split(",")] if tags
               else CONFIGS)
    for tag, desc in configs:
        if load(tag, "dec4", 100) is None:
            continue
        row = {"config": tag, "desc": desc}
        per_band = {b: [] for b in BANDS}
        for b in BANDS:
            for c in CELLS:
                o, n = load(tag, c, b), load(BASE, c, b)
                if not (o and n):
                    continue
                r = (o["out_tps_wall"] / n["out_tps_wall"] if c.startswith("dec")
                     else n["wall"] / o["wall"])
                row[f"{c}@{b}"] = round(r, 3)
                per_band[b].append(r)
        for b in BANDS:
            g = geo(per_band[b])
            row[f"geo@{b}"] = round(g, 3) if g else None
        g = geo(sum(per_band.values(), []))
        row["geo_all"] = round(g, 3) if g else None
        s = load(tag, "satlong05", 250)
        if s:
            row["slo_goodput"] = s["goodput"]
            row["slo_pass"] = s["per_cls"].get("long", "")
            row["slo_ttft_p50_s"] = round(s.get("ttft_p50_long", 0) / 1000, 1)
            row["slo_out_tps"] = s["out_tps_wall"]
        rows.append(row)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    for r in rows:
        print(f"{r['config']:9s} {r['desc']:24s} geo {r.get('geo_all')}  "
              + "  ".join(f"@{b} {r.get(f'geo@{b}')}" for b in BANDS)
              + f"  | SLO {r.get('slo_pass')} gp {r.get('slo_goodput')} "
                f"ttft {r.get('slo_ttft_p50_s')}s tps {r.get('slo_out_tps')}")
    if csv_out:
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
