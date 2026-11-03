#!/usr/bin/env python
"""Figures + CSV tables from the raw paired runs.

Reads {fin6_o,fin6_n}_* (Qwen3-30B-A3B, --best vs stock native vLLM,
same window 2026-10-30 23:27 -> 10-31 01:35; fin6_n2 = native with Model
Runner V2 forced, same window) and the 32B pair (Qwen3-32B, both 32,32,
same window 2026-10-31 09:40 -> 13:40, tags d9_o/d9_n) from
data/raw, plus data/round6 and the summary CSVs, and writes
data/*.csv and figs/*.png. Every number in the
figures comes from those files.

  .venv/bin/python make_figs.py                  # shipped data
  .venv/bin/python make_figs.py --runs $HET_RESULTS/runs \
      --ours a3b_ours --native a3b_native                   # your bench.sh re-run
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.patches  # noqa: F401
import matplotlib.ticker  # noqa: F401

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "data/raw"
R6 = ROOT / "data/round6"  # round-6 raw runs (prefix cache off)
TAGS = ["fin6_o", "fin6_n"]  # ours / native run tags (A3B)
OUT = ROOT
BANDS = [100, 250, 500, 1000]
DEC = ["dec1", "dec4", "dec16", "dec32", "dec16in256", "dec16in4K"]
SPAN = ["pre2048", "pre4096", "pre8192", "mix"]
CELLS = DEC + SPAN
LABEL = {"dec1": "decode M=1", "dec4": "decode M=4", "dec16": "decode M=16",
         "dec32": "decode M=32", "dec16in256": "decode M=16, 256-in",
         "dec16in4K": "decode M=16, 4K-in", "pre2048": "prefill 48x2K",
         "pre4096": "prefill 48x4K", "pre8192": "prefill 48x8K",
         "mix": "fixed mix (24x8K + 48 short)"}

# validated palette (dataviz reference, light surface #fcfcfb)
OURS, NATIVE = "#2a78d6", "#eb6834"
CLS = {"decode": "#1baf7a", "prefill": "#6250d6", "mix": "#e87ba4"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIV = LinearSegmentedColormap.from_list(
    "div", ["#e34948", "#f0efec", "#2a78d6"])  # red <1 | gray 1 | blue >1

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 160,
    "legend.frameon": False, "lines.linewidth": 2,
})


def load(tag: str, cell: str, band: int) -> dict:
    return json.load(open(RUNS / f"{tag}_{cell}_{band}.json"))


def ratio(o: str, n: str, cell: str, band: int) -> float:
    """>1 = ours better. Decode: out tok/s ratio; makespan cells: n/o."""
    a, b = load(o, cell, band), load(n, cell, band)
    if cell.startswith("dec"):
        return a["out_tps_wall"] / b["out_tps_wall"]
    return b["wall"] / a["wall"]


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def wan(tag: str) -> dict[int, float]:
    """A-side tc class bytes per output+input token proxy: bytes per band
    divided by the band's processed tokens (same instrument both systems)."""
    out = {}
    for line in open(RUNS / f"{tag}_wan.jsonl"):
        d = json.loads(line)
        toks = 0
        for c in CELLS:
            r = load(tag, c, d["band"])
            toks += r["in_tok"] + r.get("out_tok", 0)
        out[d["band"]] = d["tc_bytes"] / toks
    return out


def write_tables(o: str, n: str, name: str) -> list[list[float]]:
    rows, mat = [], []
    for c in CELLS:
        mrow = []
        for b in BANDS:
            a, z = load(o, c, b), load(n, c, b)
            r = ratio(o, n, c, b)
            mrow.append(r)
            rows.append({"cell": c, "band_mbps": b,
                         "ours_out_tps": a["out_tps_wall"],
                         "native_out_tps": z["out_tps_wall"],
                         "ours_wall_s": a["wall"], "native_wall_s": z["wall"],
                         "speedup": round(r, 3)})
        mat.append(mrow)
    with open(OUT / "data" / f"{name}_cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return mat


def heatmap(mat, title: str, path: str):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.grid(False)
    lo = min(min(r) for r in mat)
    hi = max(max(r) for r in mat)
    m = max(hi, 1 / lo)
    im = ax.imshow(mat, cmap=DIV, norm=LogNorm(1 / m, m), aspect="auto")
    for i, r in enumerate(mat):
        for j, v in enumerate(r):
            ax.text(j, i, f"{v:.2f}x", ha="center", va="center",
                    fontsize=9, color=INK, fontweight=("bold" if v >= 1.5
                                                       else "normal"))
    ax.set_xticks(range(len(BANDS)), [f"{b} Mbps" for b in BANDS])
    ax.set_yticks(range(len(CELLS)), [LABEL[c] for c in CELLS])
    ax.tick_params(colors=INK2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # 2px surface gaps between cells
    for k in range(len(CELLS) + 1):
        ax.axhline(k - 0.5, color=SURF, lw=2)
    for k in range(len(BANDS) + 1):
        ax.axvline(k - 0.5, color=SURF, lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ticks = [t for t in (0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3, 5, 8)
             if 1 / m <= t <= m]
    cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cb.ax.yaxis.set_minor_locator(NullLocator())
    cb.ax.set_yticklabels([f"{t:g}x" for t in ticks])
    cb.set_label("ours / native  (>1 = ours faster)", color=INK2)
    cb.ax.tick_params(colors=MUTED)
    cb.outline.set_visible(False)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "figs" / path)
    plt.close(fig)


def fig_trend(mat):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    series = {
        "decode": [geo([mat[i][j] for i in range(6)]) for j in range(4)],
        "prefill": [geo([mat[i][j] for i in range(6, 9)]) for j in range(4)],
        "mix": [mat[9][j] for j in range(4)],
    }
    marks = {"decode": "o", "prefill": "s", "mix": "D"}
    for k, ys in series.items():
        ax.plot(BANDS, ys, color=CLS[k], marker=marks[k], ms=8,
                markeredgecolor=SURF, markeredgewidth=2, label=k)
        ax.annotate(f"{k} {ys[0]:.2f}x", (BANDS[0], ys[0]),
                    xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", color=INK2, fontsize=9)
    allg = [geo([mat[i][j] for i in range(10)]) for j in range(4)]
    ax.plot(BANDS, allg, color=INK2, ls="--", lw=1.5, marker="x", ms=8,
            label="all 10 cells (geo-mean)")
    ax.axhline(1.0, color=AXIS, lw=1.2)
    ax.text(60, 1.0, "native = 1.0", va="bottom", ha="left", color=MUTED,
            fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(BANDS, [str(b) for b in BANDS])
    ax.minorticks_off()
    ax.set_xlim(55, 1200)
    ax.set_xlabel("cross-machine bandwidth (Mbps, tc-shaped)")
    ax.set_ylabel("speedup vs native vLLM (x)")
    ax.legend(loc="upper right")
    ax.set_title("Speedup shrinks as the WAN stops being the bottleneck",
                 loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "fig2_speedup_vs_bandwidth.png")
    plt.close(fig)


def fig_decode():
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    ms = [1, 4, 16, 32]
    for ax, b in zip(axs, BANDS):
        end = {}
        for tag, col, lab in ((TAGS[0], OURS, "ours"),
                              (TAGS[1], NATIVE, "native vLLM")):
            ys = [load(tag, f"dec{m}", b)["out_tps_wall"] for m in ms]
            ax.plot(ms, ys, color=col, marker="o", ms=7,
                    markeredgecolor=SURF, markeredgewidth=2, label=lab)
            end[tag] = ys[-1]
        for tag, y in end.items():
            other = end[TAGS[1] if tag == TAGS[0] else TAGS[0]]
            dy = 0 if abs(y - other) > 60 else (6 if y >= other else -6)
            ax.annotate(f"{y:.0f}", (32, y), xytext=(4, dy),
                        textcoords="offset points", va="center",
                        color=INK2, fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ms, [str(m) for m in ms])
        ax.minorticks_off()
        ax.set_xlim(0.8, 52)
        ax.set_title(f"{b} Mbps", loc="left")
        ax.set_xlabel("concurrent streams M")
    axs[0].set_ylabel("output tokens/s")
    axs[0].legend(loc="upper left")
    fig.suptitle("Saturated decode throughput (2K-in / 1K-out)", x=0.01,
                 ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "fig3_decode_throughput.png")
    plt.close(fig)


def fig_makespan():
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.6), sharey=False)
    xs = ["pre2048", "pre4096", "pre8192", "mix"]
    xl = ["48x2K", "48x4K", "48x8K", "mix"]
    w = 0.38
    for ax, b in zip(axs, BANDS):
        o = [load(TAGS[0], c, b)["wall"] for c in xs]
        n = [load(TAGS[1], c, b)["wall"] for c in xs]
        idx = range(len(xs))
        ax.bar([i - w / 2 - 0.01 for i in idx], o, w, color=OURS,
               label="ours")
        ax.bar([i + w / 2 + 0.01 for i in idx], n, w, color=NATIVE,
               label="native vLLM")
        for i in idx:
            ax.text(i, max(o[i], n[i]) * 1.02, f"{n[i] / o[i]:.2f}x",
                    ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(list(idx), xl)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{b} Mbps", loc="left")
        ax.set_ylim(0, max(max(o), max(n)) * 1.32)
    axs[0].set_ylabel("makespan (s, lower is better)")
    axs[0].legend(loc="upper left", ncol=2)
    fig.suptitle("Fixed prefill / mixed workloads - completion time "
                 "(label = native/ours)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "fig4_prefill_mix_makespan.png")
    plt.close(fig)


def fig_slo():
    if not (RUNS / f"{TAGS[0]}_satlong05_250.json").exists():
        print("skip SLO figure: no *_satlong05_250.json for", TAGS)
        return
    o = json.load(open(RUNS / f"{TAGS[0]}_satlong05_250.json"))
    n = json.load(open(RUNS / f"{TAGS[1]}_satlong05_250.json"))
    rows = [
        ("SLO goodput (req/s)", o["goodput"], n["goodput"], False),
        ("requests meeting SLO", 155 * o["good_frac"],
         155 * n["good_frac"], False),
        ("TTFT p50 (s)", o["ttft_p50_long"] / 1e3,
         n["ttft_p50_long"] / 1e3, True),
        ("worst ITL p95 (ms)", o["itlp95_max_long"],
         n["itlp95_max_long"], True),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (t, a, b, logy) in zip(axs, rows):
        ax.bar([0], [a], 0.6, color=OURS)
        ax.bar([0.64], [max(b, 1e-9)], 0.6, color=NATIVE)
        for x, v in ((0, a), (0.64, b)):
            ax.text(x, (v if v > 0 else 0) * (1.08 if logy else 1) +
                    (0 if logy else max(a, b) * 0.02),
                    (f"{v:.0f}" if v >= 100 else f"{v:.3g}"), ha="center", va="bottom", fontsize=9,
                    color=INK)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(min(a, b) / 3, max(a, b) * 4)
        else:
            ax.set_ylim(0, max(a, b) * 1.2 if max(a, b) else 1)
        ax.set_xticks([0, 0.64], ["ours", "native"])
        ax.tick_params(axis="x", length=0, colors=INK2)
        ax.grid(axis="x", visible=False)
        ax.set_title(t, loc="left")
    fig.suptitle("Overload SLO run: sat_long @0.5 req/s, 250 Mbps, 300 s "
                 "open loop, 155 requests (gates TTFT<=12s, TPOT<=100ms, "
                 "ITL p95<=200ms)", x=0.01, ha="left", fontweight="bold",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "fig5_slo_overload.png")
    plt.close(fig)
    with open(OUT / "data" / "a3b_slo_satlong05_250.csv", "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "ours", "native"])
        for k in ("goodput", "good_frac", "n", "out_tps_wall", "wall",
                  "ttft_p50_long", "ttft_p95_long", "itlp95_max_long"):
            w.writerow([k, o[k], n[k]])


def fig_wan():
    wo, wn = wan(TAGS[0]), wan(TAGS[1])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    w = 0.38
    idx = range(len(BANDS))
    ax.bar([i - w / 2 - 0.01 for i in idx], [wo[b] for b in BANDS], w,
           color=OURS, label="ours")
    ax.bar([i + w / 2 + 0.01 for i in idx], [wn[b] for b in BANDS], w,
           color=NATIVE, label="native vLLM")
    for i, b in enumerate(BANDS):
        ax.text(i - w / 2, wo[b] * 1.04, f"{wo[b] / wn[b]:.2f}x",
                ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(list(idx), [f"{b} Mbps" for b in BANDS])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("WAN bytes per processed token")
    ax.set_ylim(0, max(wn.values()) * 1.22)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Cross-machine traffic (A-side tc counter, same instrument)",
                 loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "fig6_wan_bytes.png")
    plt.close(fig)
    with open(OUT / "data" / "a3b_wan_bytes.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["band_mbps", "ours_B_per_tok", "native_B_per_tok",
                     "ratio"])
        for b in BANDS:
            cw.writerow([b, round(wo[b], 1), round(wn[b], 1),
                         round(wo[b] / wn[b], 3)])


def fig_quant():
    """Boundary-quantization accuracy vs native (data/a3b_quant_accuracy.csv,
    produced by kit/qeval_cmp.py): paired deltas with 95% CI, and answer
    agreement with native against the native-rerun noise floor."""
    f = OUT / "data" / "a3b_quant_accuracy.csv"
    if not f.exists():
        return
    rows = {r["config"]: r for r in csv.DictReader(open(f))}
    tasks = [("mmlu", "MMLU 5-shot", 14042), ("arcc", "ARC-C 5-shot", 1165),
             ("gsm8k", "GSM8K strict", 1319)]
    cfgs = [("q6m16", "merged bf16", "#1baf7a"),
            ("q6m8", "merged int8", "#6250d6"),
            ("q6m4", "merged int4 (--best)", "#eb6834")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    w = 0.26
    for j, (c, lab, col) in enumerate(cfgs):
        xs, ds, es = [], [], []
        for i, (t, _, n) in enumerate(tasks):
            r = rows[c]
            b, g = (int(v) for v in r[f"{t}_lost_gained"].split("/"))
            d = (g - b) / n
            se = math.sqrt(max(b + g - (b - g) ** 2 / n, 0)) / n
            xs.append(i + (j - 1) * (w + 0.02))
            ds.append(100 * d)
            es.append(100 * 1.96 * se)
        ax1.bar(xs, ds, w, color=col, label=lab, yerr=es,
                error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
    ax1.axhline(0, color=AXIS, lw=1.2)
    ax1.set_xticks(range(3), [t[1] for t in tasks])
    ax1.tick_params(axis="x", length=0)
    ax1.grid(axis="x", visible=False)
    ax1.set_ylabel("accuracy change vs native (points)")
    ax1.set_title("Paired accuracy change, 95% CI", loc="left")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3,
               fontsize=8.5)
    for i, (t, _, _) in enumerate(tasks):
        y = float(rows["q6nat2"][f"{t}_agree_pct"])
        ax2.plot([i - 0.42, i + 0.42], [y, y], color=INK2, lw=1.2, ls="--",
                 zorder=1)
    for j, (c, lab, col) in enumerate(cfgs):
        xs = [i + (j - 1) * 0.25 for i in range(3)]
        ys = [float(rows[c][f"{t}_agree_pct"]) for t, _, _ in tasks]
        ax2.plot(xs, ys, ls="none", marker="o", ms=9, color=col, label=lab,
                 markeredgecolor=SURF, markeredgewidth=2, zorder=3)
        for x, y, (t, _, _) in zip(xs, ys, tasks):
            floor = float(rows["q6nat2"][f"{t}_agree_pct"])
            dy = -0.3 if abs(y - floor) < 0.25 else 0.0  # clear the line
            ax2.text(x + 0.07, y + dy, f"{y:.1f}", ha="left", va="center",
                     fontsize=8, color=INK2)
    ax2.plot([], [], color=INK2, ls="--", lw=1.2,
             label="native vs native rerun (noise floor)")
    ax2.set_ylim(92, 100.3)
    ax2.set_xticks(range(3), [t[1] for t in tasks])
    ax2.tick_params(axis="x", length=0)
    ax2.grid(axis="x", visible=False)
    ax2.set_ylabel("same answer as native (%)")
    ax2.set_title("Answer agreement with native", loc="left")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2,
               fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "fig9_quant_accuracy.png")
    plt.close(fig)


def fig_ablation():
    """Ablation (data/a3b_ablation.csv from kit/ablate_tab.py): 12-test
    throughput geo-mean vs native, and requests passing the SLO run."""
    f = OUT / "data" / "a3b_ablation.csv"
    if not f.exists():
        return
    rows = {r["config"]: r for r in csv.DictReader(open(f))}
    # groups (separated by rules): native baselines | our stack on top of
    # native V2 + tok-tail | variants of the final config
    groups = [[("r6:natv1", "native vLLM (V1 runner, default)"),
               ("r6:nat2c", "native, V2 runner"),
               ("r6:nat_moe0", "native V2, MoE configs off"),
               ("r6:nat2t", "native V2 + tok-tail")],
              [("r6:L2t", "+ pp_link (bf16, 2 vectors)"),
               ("r6:LQ2t", "+ pp_link + int4"),
               ("r6:LM2t", "+ pp_link + merge (bf16)"),
               ("r6:LQM2t", "+ pp_link + int4 + merge = --best")],
              [("r6:LQM2ts", "--best, synchronous link"),
               ("r6:LQM2nt", "--best without tok-tail"),
               ("r6:full2t", "--best + TP-C (--best-tpc)")]]
    groups = [[o for o in g if o[0] in rows] for g in groups]
    order = [o for g in groups for o in g]
    cuts, n = [], 0
    for g in groups[:-1]:
        n += len(g)
        cuts.append(len(order) - n - 0.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True,
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    ys = list(range(len(order)))[::-1]
    for y, (k, _) in zip(ys, order):
        g = float(rows[k]["geo_all"])
        ax1.barh(y, g, height=0.62, color=OURS)
        ax1.text(max(g, 1.0) + 0.03, y, f"{g:.2f}x", va="center",
                 fontsize=9, color=INK)  # clear of the 1.0 reference line
        p, n = rows[k]["slo_pass"].split("/")
        ax2.barh(y, int(p), height=0.62, color=OURS)
        ax2.text(int(p) + 2, y, f"{p}/{n}", va="center", fontsize=9,
                 color=INK)
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0, colors=INK2)
        for cut in cuts:
            ax.axhline(cut, color=AXIS, lw=0.8)
    ax1.axvline(1.0, color=INK2, lw=1.2, ls="--")
    ax1.set_yticks(ys, [lab for _, lab in order])
    ax1.set_xlim(0, 2.8)
    ax1.set_xlabel("speedup vs native V1 (geo-mean of 12 throughput tests)")
    ax1.set_title("Throughput: dec4/dec32/prefill 8K/mix x 100/250/1000 Mbps",
                  loc="left", fontsize=10)
    ax2.set_xlim(0, 175)
    ax2.set_xlabel("requests meeting all SLO gates (of 155)")
    ax2.set_title("SLO run: sat_long @0.5 req/s, 250 Mbps", loc="left",
                  fontsize=10)
    fig.suptitle("Ablation (prefix caching off, one fresh engine per row)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "fig10_ablation.png")
    plt.close(fig)


R6_CELLS = ["dec4", "dec32", "pre8192", "mix"]
R6_BANDS = [100, 250, 1000]
NAT_V2 = "#1baf7a"   # categorical slot 3: native vLLM with Model Runner V2


def r6_geo(o: str, n: str, band: int | None = None) -> float | None:
    """12-test (or per-band 4-test) geo-mean of ours/native, prefix cache
    off: decode = out tok/s ratio, prefill/mix = makespan ratio."""
    xs = []
    for b in ([band] if band else R6_BANDS):
        for c in R6_CELLS:
            fo, fn = R6 / f"{o}_{c}_{b}.json", R6 / f"{n}_{c}_{b}.json"
            if not (fo.exists() and fn.exists()):
                return None
            a, z = json.load(open(fo)), json.load(open(fn))
            xs.append(a["out_tps_wall"] / z["out_tps_wall"]
                      if c.startswith("dec") else z["wall"] / a["wall"])
    return geo(xs)


def fig_round6():
    """Round 6: the new default vs native vLLM in both runner versions,
    per band, for TP2xPP2 and PP4 (12 tests, prefix caching off)."""
    panels = [("TP2 x PP2  (ours: r6_LQM2t)", "r6_LQM2t", "r6_natv1", "r6_nat2c"),
              ("PP4 = 4090, 4090 | 3090, 3090  (ours: r7_o4f)", "r7_o4f",
               "p4_n4", "p4_n4v2")]
    if r6_geo(*panels[0][1:3]) is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    xs = R6_BANDS + ["all 12"]
    for ax, (title, o, n1, n2) in zip(axes, panels):
        for k, (n, col, lab) in enumerate(((n1, NATIVE, "vs native, V1 runner "
                                            "(vLLM default for MoE)"),
                                           (n2, NAT_V2, "vs native, V2 runner "
                                            "forced on"))):
            vals = [r6_geo(o, n, b) for b in R6_BANDS] + [r6_geo(o, n)]
            pos = [i + (k - 0.5) * 0.38 for i in range(len(xs))]
            ax.bar(pos, vals, 0.36, color=col, label=lab)
            for p, v in zip(pos, vals):
                ax.text(p, v + 0.06, f"{v:.2f}x", ha="center", fontsize=8.5,
                        color=INK)
        ax.axhline(1.0, color=INK2, lw=1.1, ls="--")
        ax.set_xticks(range(len(xs)), [f"{x} Mbps" if isinstance(x, int)
                                        else x for x in xs])
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].set_ylabel("speedup (geo-mean of dec4/dec32/prefill 8K/mix)")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle("--best (pp_link int4 + merge, Model Runner V2, "
                 "--tok-tail) vs native vLLM", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "fig11_round6_vs_native.png")
    plt.close(fig)


def fig_bubble_bands():
    """Bubble rate per band, common steady 30 s window (pp2_bubble.py
    --clip 2:32), data/a3b_bubble_bands.csv."""
    f = OUT / "data" / "a3b_bubble_bands.csv"
    if not f.exists():
        return
    rows = list(csv.DictReader(open(f)))
    series = [("native_v1", NATIVE, "native vLLM (V1 runner)"),
              ("native_v2", NAT_V2, "native vLLM (V2 runner)"),
              ("ours", OURS, "ours (--best-tpc)")]
    bands = sorted({int(r["band"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, st in zip(axes, ("A", "B")):
        for k, (sysn, col, lab) in enumerate(series):
            first = True
            for i, b in enumerate(bands):
                v = [float(r["bubble"]) for r in rows if r["system"] == sysn
                     and int(r["band"]) == b and r["stage"] == st]
                if not v:
                    continue
                x = i + (k - 1) * 0.26
                ax.bar(x, v[0], 0.24, color=col, label=lab if first else None)
                first = False
                ax.text(x, v[0] + 1, f"{v[0]:.0f}", ha="center", fontsize=7.5,
                        color=INK)
        ax.set_xticks(range(len(bands)), [f"{b} Mbps" for b in bands])
        ax.set_ylim(0, 108)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"machine {st} ({'2x 4090, stage 0' if st == 'A' else '2x 3090, stage 1'})",
                     loc="left", fontsize=10)
    axes[0].set_ylabel("bubble rate, % (1 - model-kernel time / window)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=3, fontsize=8.5)
    fig.suptitle("Bubble rate, saturated fixed mix, same 30 s window on every "
                 "GPU (gaps = run not available)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "figs" / "fig12_bubble_by_band.png")
    plt.close(fig)


def fig_arch():
    """System schematic: what runs where, what crosses the WAN."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    def box(x, y, w, h, title, lines, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#ffffff", ec=edge, lw=1.6))
        ax.text(x + 0.15, y + h - 0.3, title, fontsize=10.5,
                fontweight="bold", color=INK, va="top")
        for i, ln in enumerate(lines):
            ax.text(x + 0.15, y + h - 0.75 - i * 0.36, ln, fontsize=8.8,
                    color=INK2, va="top")

    box(0.2, 0.5, 3.6, 3.5, "Machine A  (2x RTX 4090)",
        ["API server + scheduler", "  (--tok-tail: lean control plane)",
         "Model Runner V2 (3 batches in flight)",
         "stage 0: layers 0-27, TP=2",
         "boundary h + r merged -> int4 codec",
         "(TP-C scheduling optional: --best-tpc)"],
        OURS)
    box(7.2, 0.5, 3.6, 3.5, "Machine B  (2x RTX 3090)",
        ["int4 decode (sliced, GPU)", "stage 1: layers 28-47, TP=2",
         "LM head + sampling", "sampled tokens back to A"], OURS)
    ax.add_patch(FancyArrowPatch((3.9, 3.3), (7.1, 3.3),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=INK2, lw=1.6))
    ax.add_patch(FancyArrowPatch((7.1, 1.5), (3.9, 1.5),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=MUTED, lw=1.2, ls="--"))
    ax.text(5.5, 3.5, "WAN 100-1000 Mbps (tc-shaped)", ha="center",
            fontsize=9.5, fontweight="bold", color=INK)
    ax.text(5.5, 3.12, "pp_link: ZMQ, async send/recv,\nhi (decode) / lo "
            "(prefill) lanes,\npinned buffer pools", ha="center",
            va="top", fontsize=8.6, color=INK2)
    ax.text(5.5, 2.0, "measured ~1.1 KB / token  (native ~7.6 KB)", ha="center",
            fontsize=9, color=INK)
    ax.text(5.5, 1.15, "sampled-token feedback", ha="center",
            fontsize=8.6, color=MUTED)
    ax.text(0.2, 0.15, "native vLLM baseline: same TP=2 x PP=2 layout and "
            "28/20 split, bf16 (h, r) over NCCL, stock scheduler",
            fontsize=8.6, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "fig0_architecture.png")
    plt.close(fig)


def main():
    global RUNS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--ours", default="fin6_o")
    ap.add_argument("--native", default="fin6_n")
    ap.add_argument("--native2", default="fin6_n2",
                    help="second native baseline (Model Runner V2 forced); "
                         "written to data/a3b_vs_native_v2_cells.csv")
    ap.add_argument("--out", default=str(OUT),
                    help="where figs/ and data/ go (default: repo root)")
    ap.add_argument("--d32", default="d9_o,d9_n",
                    help="ours,native tags for the 32B heatmap")
    ap.add_argument("--title", default="Qwen3-30B-A3B",
                    help="model name for the --ours/--native heatmap")
    args = ap.parse_args()
    OUT = Path(args.out)
    RUNS = Path(args.runs)
    TAGS[:] = [args.ours, args.native]
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    fig_arch()
    fig_quant()
    fig_ablation()
    fig_round6()
    fig_bubble_bands()
    a3b = write_tables(args.ours, args.native, "a3b")
    if (RUNS / f"{args.native2}_dec1_100.json").exists():
        v2 = write_tables(args.ours, args.native2, "a3b_vs_native_v2")
        heatmap(v2, f"{args.title}: ours / native vLLM with Model Runner V2, "
                "40 paired cells", "fig1b_a3b_vs_native_v2_heatmap.png")
        print("A3B vs native V2: geo all40 %.3f" % geo([v for r in v2 for v in r]))
    heatmap(a3b, f"{args.title}: ours / native vLLM, 40 paired cells",
            "fig1_a3b_speedup_heatmap.png")
    d32 = None
    d_o, d_n = args.d32.split(",")
    if (RUNS / f"{d_o}_dec1_100.json").exists():
        d32 = write_tables(d_o, d_n, "d32")
        heatmap(d32, "Qwen3-32B (dense): ours / native vLLM, "
                "40 paired cells", "fig7_d32_speedup_heatmap.png")
    fig_trend(a3b)
    fig_decode()
    fig_makespan()
    fig_slo()
    fig_wan()
    flat = [v for r in a3b for v in r]
    print("A3B geo all40 %.3f  <=250 %.3f" % (
        geo(flat), geo([r[j] for r in a3b for j in (0, 1)])))
    for j, b in enumerate(BANDS):
        print(f"  A3B @{b}: geo {geo([r[j] for r in a3b]):.3f}  "
              f"dec {geo([a3b[i][j] for i in range(6)]):.3f}")
    if d32:
        print("32B geo all40 %.3f  <=250 %.3f" % (
            geo([v for r in d32 for v in r]),
            geo([r[j] for r in d32 for j in (0, 1)])))


if __name__ == "__main__":
    main()
