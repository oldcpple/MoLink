#!/usr/bin/env python
"""Compare qeval runs against the native reference (paired, per item).

For every multiple-choice task and GSM8K (strict): accuracy delta, exact
two-sided McNemar test on the discordant pairs (same questions, so this is
far tighter than comparing two independent accuracies), and agreement =
share of items with the same prediction as native. GSM8K also reports mean
generated characters and the share of outputs that keep writing after the
"#### N" answer line. WikiText-2: perplexity delta.

  python qeval_cmp.py nat m4 m8 m16 [--csv out.csv]
"""
from __future__ import annotations

import csv
import json
import os
import math
import sys
from pathlib import Path

D = Path(os.environ.get("QEVAL_DIR")
         or Path(os.environ.get("HET_RESULTS", "./results")) / "qeval")


def items(tag):
    out = {}
    for line in open(D / f"{tag}_items.jsonl"):
        r = json.loads(line)
        out.setdefault(r["task"], []).append(r)
    return out


def mcnemar_p(b, c):
    """Exact two-sided binomial test on discordant pairs b vs c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    args = sys.argv[1:]
    csv_out = None
    if "--csv" in args:
        i = args.index("--csv")
        csv_out = args[i + 1]
        del args[i:i + 2]
    ref, others = args[0], args[1:]
    R = json.loads((D / f"{ref}.json").read_text())
    ri = items(ref)
    rows = []
    for tag in [ref] + others:
        S = json.loads((D / f"{tag}.json").read_text())
        si = items(tag)
        row = {"config": tag, "wikitext2_ppl": round(S["wikitext2"]["ppl"], 4),
               "ppl_delta_pct": round(100 * (S["wikitext2"]["ppl"]
                                             / R["wikitext2"]["ppl"] - 1), 2)}
        for task in ("mmlu", "arcc", "gsm8k"):
            a, b = ri[task], si[task]
            assert len(a) == len(b)
            if task == "gsm8k":
                ok_a = [x["ok_strict"] for x in a]
                ok_b = [x["ok_strict"] for x in b]
                same = [x["strict"] == y["strict"] for x, y in zip(a, b)]
                row["gsm8k_chars"] = round(
                    sum(len(y["out"]) for y in b) / len(b))
                row["gsm8k_runon_pct"] = round(100 * sum(
                    1 for y in b if y["strict"] is not None
                    and "\n" in y["out"].split("####")[-1].strip()) / len(b), 1)
            else:
                ok_a = [int(x["pred"] == x["gold"]) for x in a]
                ok_b = [int(y["pred"] == y["gold"]) for y in b]
                same = [x["pred"] == y["pred"] for x, y in zip(a, b)]
            nb = sum(1 for x, y in zip(ok_a, ok_b) if x and not y)
            nc = sum(1 for x, y in zip(ok_a, ok_b) if y and not x)
            row[f"{task}_acc"] = round(100 * sum(ok_b) / len(ok_b), 2)
            row[f"{task}_delta_pp"] = round(
                100 * (sum(ok_b) - sum(ok_a)) / len(ok_b), 2)
            row[f"{task}_lost_gained"] = f"{nb}/{nc}"
            row[f"{task}_mcnemar_p"] = float(f"{mcnemar_p(nb, nc):.2g}")
            row[f"{task}_agree_pct"] = round(100 * sum(same) / len(same), 1)
        rows.append(row)
    for r in rows:
        print(r["config"])
        for k, v in r.items():
            if k != "config":
                print(f"  {k:22s} {v}")
    if csv_out:
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
