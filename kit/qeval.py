#!/usr/bin/env python
"""Accuracy of the boundary-quantization configs on public benchmarks.

Runs against whatever engine is up (OpenAI-compatible /v1/completions), all
greedy, so the only difference between runs is the engine config:

  wikitext2  perplexity of WikiText-2 raw test (2048-token windows, stride
             1024, each token scored once with >=1024 tokens of context)
  mmlu       5-shot (dev split exemplars per subject), next-token logprob
             of " A".." D" after "Answer:" (lm-eval-harness prompt format)
  arcc       ARC-Challenge, 5-shot (first 5 train items with 4 choices),
             same scoring
  gsm8k      5-shot chain of thought (first 5 train items), greedy up to 384
             tokens; strict = "#### N" answer, flexible = last number

  python qeval.py --tag nat [--tasks wikitext2,mmlu,arcc,gsm8k]
Results: $HET_RESULTS/qeval/{tag}.json (+ per-item predictions {tag}_items.jsonl)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import math
import re
import time
from pathlib import Path

import aiohttp
import pyarrow.parquet as pq

D = Path(os.environ.get("QEVAL_DIR")
         or Path(os.environ.get("HET_RESULTS", "./results")) / "qeval")
DATA = Path(os.environ.get("QEVAL_DATA") or D / "data")
URL = "http://127.0.0.1:26000"
CONC = 48
LETTERS = "ABCD"
LIMIT = 0  # >0: first N items per task (smoke test)


def table(name):
    return pq.read_table(DATA / f"{name}.parquet").to_pylist()


async def post(session, path, body, retries=3):
    for i in range(retries):
        try:
            async with session.post(URL + path, json=body) as r:
                if r.status == 200:
                    return await r.json()
                txt = await r.text()
                if i == retries - 1:
                    raise RuntimeError(f"{r.status}: {txt[:300]}")
        except aiohttp.ClientError:
            if i == retries - 1:
                raise
        await asyncio.sleep(1 + i)


async def gather(coros):
    sem = asyncio.Semaphore(CONC)

    async def one(c):
        async with sem:
            return await c
    return await asyncio.gather(*(one(c) for c in coros))


# ---------------------------------------------------------------- MC tasks
def mc_block(q, choices, ans=None):
    s = q.strip() + "\n" + "".join(
        f"{LETTERS[i]}. {c.strip()}\n" for i, c in enumerate(choices))
    s += "Answer:"
    if ans is not None:
        s += f" {LETTERS[ans]}\n\n"
    return s


async def score_mc(session, prompt):
    r = await post(session, "/v1/completions", {
        "model": "m", "prompt": prompt, "max_tokens": 1, "temperature": 0,
        "logprobs": 20})
    top = r["choices"][0]["logprobs"]["top_logprobs"][0] or {}
    lp = []
    for L in LETTERS:
        cands = [v for k, v in top.items() if k.strip() == L]
        lp.append(max(cands) if cands else -1e9)
    return max(range(4), key=lambda i: lp[i])


def mmlu_prompts():
    dev, test = table("mmlu_dev"), table("mmlu_test")
    shots = {}
    for r in dev:
        shots.setdefault(r["subject"], []).append(r)
    out = []
    for r in test:
        sub = r["subject"]
        head = ("The following are multiple choice questions (with answers) "
                f"about {sub.replace('_', ' ')}.\n\n")
        fs = "".join(mc_block(x["question"], x["choices"], x["answer"])
                     for x in shots[sub][:5])
        out.append((head + fs + mc_block(r["question"], r["choices"]),
                    r["answer"], sub))
    return out


def arc_prompts():
    def norm(r):
        labels = list(r["choices"]["label"])
        texts = list(r["choices"]["text"])
        return texts, labels.index(r["answerKey"])
    train = [r for r in table("arcc_train") if len(r["choices"]["label"]) == 4]
    head = ("The following are multiple choice science questions "
            "(with answers).\n\n")
    fs = "".join(mc_block(r["question"], *norm(r)) for r in train[:5])
    out = []
    for r in table("arcc_test"):
        texts, ans = norm(r)
        if len(texts) != 4:
            continue  # 1% of items have 3 or 5 choices; letter scoring is 4-way
        out.append((head + fs + mc_block(r["question"], texts), ans, "arc"))
    return out


async def run_mc(session, items):
    preds = await gather([score_mc(session, p) for p, _, _ in items])
    correct = [int(p == a) for p, (_, a, _) in zip(preds, items)]
    return preds, correct


# ---------------------------------------------------------------- GSM8K
NUM = re.compile(r"-?[\d,]*\.?\d+")


def gsm_answer(s):
    return s.split("####")[-1].strip().replace(",", "")


def gsm_extract(text):
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    strict = m.group(1).replace(",", "") if m else None
    nums = NUM.findall(text)
    flex = nums[-1].replace(",", "").rstrip(".") if nums else None
    return strict, flex


def same(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


async def run_gsm(session):
    train, test = table("gsm8k_train"), table("gsm8k_test")[:LIMIT or None]
    fs = "".join(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n"
                 for r in train[:5])

    async def one(r):
        res = await post(session, "/v1/completions", {
            "model": "m", "prompt": fs + f"Question: {r['question']}\nAnswer:",
            "max_tokens": 384, "temperature": 0,
            "stop": ["Question:", "\n\n\n"]})
        return res["choices"][0]["text"]
    outs = await gather([one(r) for r in test])
    rows = []
    for r, o in zip(test, outs):
        gold = gsm_answer(r["answer"])
        s, f = gsm_extract(o)
        rows.append({"gold": gold, "strict": s, "flex": f, "out": o,
                     "ok_strict": int(same(s, gold)),
                     "ok_flex": int(same(f, gold))})
    return rows


# ---------------------------------------------------------------- WikiText-2
async def run_ppl(session, win=2048, stride=1024):
    rows = table("wikitext2_test")
    text = "\n\n".join(r["text"] for r in rows[:LIMIT * 20 or None])
    tok = await post(session, "/tokenize", {"model": "m", "prompt": text,
                                            "add_special_tokens": False})
    ids = tok["tokens"]
    jobs = []  # (start, score_from)
    s = 0
    while True:
        e = min(s + win, len(ids))
        jobs.append((s, 1 if s == 0 else win - stride, e))
        if e == len(ids):
            break
        s += stride

    async def one(start, frm, end):
        r = await post(session, "/v1/completions", {
            "model": "m", "prompt": ids[start:end], "max_tokens": 1,
            "temperature": 0, "prompt_logprobs": 0})
        pl = r["choices"][0]["prompt_logprobs"]
        tot, n = 0.0, 0
        for pos in range(frm, end - start):
            d = pl[pos]
            want = str(ids[start + pos])
            tot += d[want]["logprob"]
            n += 1
        return tot, n
    # one window at a time: prompt logprobs materialize fp32 logits for every
    # scheduled prompt row on the last stage (2048 x 151936 x 4 B = 1.2 GB);
    # concurrent windows would OOM the 3090s
    res = [await one(*j) for j in jobs]
    nll = -sum(t for t, _ in res)
    n = sum(k for _, k in res)
    return {"ppl": math.exp(nll / n), "tokens": n, "windows": len(jobs)}


def se(xs):
    p = sum(xs) / len(xs)
    return math.sqrt(p * (1 - p) / len(xs))


async def main(tag, tasks):
    D.mkdir(parents=True, exist_ok=True)
    out = {"tag": tag}
    items_f = open(D / f"{tag}_items.jsonl", "w")
    to = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=to) as session:
        if "wikitext2" in tasks:
            t0 = time.time()
            out["wikitext2"] = await run_ppl(session)
            out["wikitext2"]["s"] = round(time.time() - t0)
            print("wikitext2", out["wikitext2"], flush=True)
        for name, gen in (("mmlu", mmlu_prompts), ("arcc", arc_prompts)):
            if name not in tasks:
                continue
            t0 = time.time()
            items = gen()[:LIMIT or None]
            preds, correct = await run_mc(session, items)
            acc = sum(correct) / len(correct)
            out[name] = {"acc": acc, "se": se(correct), "n": len(correct),
                         "s": round(time.time() - t0)}
            for (_, a, sub), p, c in zip(items, preds, correct):
                items_f.write(json.dumps({"task": name, "sub": sub,
                                          "gold": a, "pred": p}) + "\n")
            print(name, out[name], flush=True)
        if "gsm8k" in tasks:
            t0 = time.time()
            rows = await run_gsm(session)
            ks = [r["ok_strict"] for r in rows]
            kf = [r["ok_flex"] for r in rows]
            out["gsm8k"] = {"strict": sum(ks) / len(ks), "se_strict": se(ks),
                            "flex": sum(kf) / len(kf), "se_flex": se(kf),
                            "n": len(rows), "s": round(time.time() - t0)}
            for r in rows:
                items_f.write(json.dumps({"task": "gsm8k", **r}) + "\n")
            print("gsm8k", out["gsm8k"], flush=True)
    items_f.close()
    (D / f"{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", default="wikitext2,mmlu,arcc,gsm8k")
    ap.add_argument("--port", type=int, default=26000)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    URL = f"http://127.0.0.1:{a.port}"
    LIMIT = a.limit
    asyncio.run(main(a.tag, set(a.tasks.split(","))))
