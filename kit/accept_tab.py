"""Paired acceptance table from bench.sh runs (4 bands x 10 cells + SLO).

  python accept_tab.py ours_tag native_tag   # e.g. a3b_ours a3b_native
"""
from __future__ import annotations

import json
import os
import math
import sys
from pathlib import Path

R = Path(os.environ.get("HET_RESULTS", "./results")).resolve() / "runs"
BANDS = [100, 250, 500, 1000]


def cell(tag: str, name: str, band: int):
    f = R / f"{tag}_{name}_{band}.json"
    if not f.exists():
        return None
    return json.load(open(f))


def tps(tag: str, name: str, band: int, key: str = "out_tps_wall"):
    d = cell(tag, name, band)
    return d.get(key) if d else None


def main(ours: str, nat: str):
    print("== per-band decode saturation (out tok/s, ours/native) ==")
    geos = []
    for b in BANDS:
        rats = []
        cells = []
        for name in ["dec1", "dec4", "dec16", "dec32", "dec16in256",
                     "dec16in4K"]:
            o, n = tps(ours, name, b), tps(nat, name, b)
            if o and n:
                cells.append(f"{name.replace('dec','M')} {o:.0f}/{n:.0f}"
                             f"={o / n:.2f}")
                rats.append(o / n)
        g = math.prod(rats) ** (1 / len(rats)) if rats else float("nan")
        geos.append(g)
        print(f"  @{b:>4}: " + "  ".join(cells) + f"  || geo={g:.3f}")
    if all(not math.isnan(x) for x in geos):
        print(f"  4-band decode geo-mean = "
              f"{math.prod(geos) ** (1 / len(geos)):.3f}  "
              f"(min-win >=1.8, ambition >=2.0)")

    print("== prefill makespan (wall s, lower=better; ratio=native/ours) ==")
    for b in BANDS:
        cells = []
        for name in ["pre2048", "pre4096", "pre8192"]:
            o, n = cell(ours, name, b), cell(nat, name, b)
            if o and n:
                cells.append(f"{name[3:]} {o['wall']:.1f}s/{n['wall']:.1f}s"
                             f"={n['wall'] / o['wall']:.2f}")
        print(f"  @{b:>4}: " + "  ".join(cells))

    print("== fixed_mix makespan ==")
    for b in BANDS:
        o, n = cell(ours, "mix", b), cell(nat, "mix", b)
        if o and n:
            print(f"  @{b:>4}: {o['wall']:.1f}s/{n['wall']:.1f}s "
                  f"= {n['wall'] / o['wall']:.2f}  out "
                  f"{o.get('out_tps_wall', 0):.0f}/"
                  f"{n.get('out_tps_wall', 0):.0f}")

    print("== WAN bytes / token (A-side tc class counter, same instrument) ==")
    for tag in (ours, nat):
        f = R / f"{tag}_wan.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(x) for x in f.read_text().splitlines()]
        out = "  " + tag + ": "
        for r in rows:
            # tokens this band = sum over that band's cells
            tok = 0
            for name in ["dec1", "dec4", "dec16", "dec32", "dec16in256",
                         "dec16in4K", "pre2048", "pre4096", "pre8192",
                         "mix"]:
                d = cell(tag, name, r["band"])
                if d:
                    tok += d.get("in_tok", 0) + d.get("out_tok", 0)
            out += (f"@{r['band']} {r['tc_bytes'] / max(tok, 1):.0f}B/tok "
                    f"({r['tc_bytes'] / 1e6:.0f}MB)  ")
        print(out)

    print("== GPU-seconds / out token (4 GPUs x wall / out_tok, dec cells) ==")
    for tag in (ours, nat):
        tot_gs = tot_tok = 0.0
        for b in BANDS:
            for name in ["dec16", "dec32"]:
                d = cell(tag, name, b)
                if d:
                    tot_gs += 4 * d["wall"]
                    tot_tok += d.get("out_tok", 0)
        if tot_tok:
            print(f"  {tag}: {tot_gs / tot_tok * 1e3:.1f} ms-GPU/tok")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
