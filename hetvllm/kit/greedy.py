"""pp2 greedy generation capture: fixed prompts, temperature 0, 64 tokens.

Saves outputs to $HET_RESULTS/{tag}_gen_fixed.json for cross-session diffing
against native_gen_fixed.json (token-exact prefix comparison).

  greedy.py --tag int4 [--port 26000]
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

EXP = Path(os.environ.get("HET_RESULTS", "./results")).resolve()
PROMPTS = {
    "p1": "The capital of France is",
    "p2": "def fibonacci(n):",
    "p3": ("Water boils at a lower temperature at high altitude because "
           "the atmospheric pressure decreases, which means"),
}


def main(tag: str, port: int):
    out = {}
    for k, p in PROMPTS.items():
        body = {"model": "m", "prompt": p, "max_tokens": 64,
                "temperature": 0}
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(),
            {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=120))
        out[k] = r["choices"][0]["text"]
    (EXP / f"{tag}_gen_fixed.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v[:40] for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, default=26000)
    a = ap.parse_args()
    main(a.tag, a.port)
