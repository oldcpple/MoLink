"""Diff two greedy-capture files against each other token-prefix-wise.

  greedy_cmp.py native int4
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

EXP = Path(os.environ.get("HET_RESULTS", "./results")).resolve()


def toks(s: str) -> list[str]:
    # cheap word-level compare (tokenizer-agnostic enough for exact-prefix)
    return s.split()


def main(a_tag: str, b_tag: str):
    a = json.load(open(EXP / f"{a_tag}_gen_fixed.json"))
    b = json.load(open(EXP / f"{b_tag}_gen_fixed.json"))
    for k in sorted(a):
        if k not in b:
            print(k, "missing in", b_tag)
            continue
        ta, tb = toks(a[k]), toks(b[k])
        n = 0
        for x, y in zip(ta, tb):
            if x != y:
                break
            n += 1
        print(f"{k}: exact prefix {n}/{min(len(ta), len(tb))} word-toks "
              f"({'IDENTICAL' if ta == tb else 'diverges'})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
