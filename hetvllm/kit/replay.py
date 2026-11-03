"""pp2 trace replay: open-loop arrivals, fixed timestamps, SLO goodput.

Traces:
  sat_long   4K/8K in, 128 out; concurrency sweep via arrival rate
  mixed      70% (256 in/512 out) + 30% (8K in/128 out), long inputs
             arrive in groups of 4
  waves      bursts of 32/64 x (128 in/256 out) every 20 s
  bw_var     same as mixed; link shaped 250<->100 mbit every 15 s
             (shaping driven externally by this script via het.py tc)
  boundary   64 in/1024 out, pure long decode, 1 Gbit

SLO (per class): short TTFT<=2s TPOT<=100ms ITLp95<=200ms; long TTFT<=12s
(class 'long' covers sat_long/boundary; 'mixed' rows have per-part gates).

Usage:
  replay.py replay --trace mixed --rate 2.0 --dur 300 --out $HET_RESULTS/runs/x.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import aiohttp

URL = "http://127.0.0.1:26000"
TIMEOUT = 600.0
EXP = Path(os.environ.get("HET_RESULTS", "./results")).resolve()

SLOS = {
    "short": {"ttft_ms": 2000, "tpot_ms": 100, "itl_p95_ms": 200},
    "long": {"ttft_ms": 12000, "tpot_ms": 100, "itl_p95_ms": 200},
    "mixed_short": {"ttft_ms": 2000, "tpot_ms": 80, "itl_p95_ms": 150},
    "mixed_long": {"ttft_ms": 12000, "tpot_ms": 100, "itl_p95_ms": 300},
}


def tok_prompt(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(1000, 100000) for _ in range(n)]


def gen_trace(name: str, rate: float, dur: float, seed: int):
    """Returns list of dicts {t, in_len, out_len, cls} sorted by t."""
    rng = random.Random(seed)
    reqs = []

    def poisson(r):
        return rng.expovariate(r)

    if name == "sat_long":
        t = 0.0
        i = 0
        while t < dur:
            reqs.append(dict(t=t, in_len=rng.choice([4096, 8192]),
                             out_len=128, cls="long", wave=0))
            t += poisson(rate)
            i += 1
    elif name == "mixed":
        t, i = 0.0, 0
        while t < dur:
            long_now = rng.random() < 0.30
            if long_now:  # groups of 4 arriving together
                for g in range(4):
                    reqs.append(dict(t=t + g * 0.05, in_len=8192,
                                     out_len=128, cls="mixed_long",
                                     wave=0))
                t += poisson(rate / 4 * 4)  # group rate keeps total ~rate
            else:
                reqs.append(dict(t=t, in_len=256, out_len=512,
                                 cls="mixed_short", wave=0))
                t += poisson(rate)
            i += 1
    elif name == "bw_var":  # same arrival process as mixed; shaping driven
        t, i = 0.0, 0       # externally by --bw-var's tc alternation
        while t < dur:
            long_now = rng.random() < 0.30
            if long_now:
                for g in range(4):
                    reqs.append(dict(t=t + g * 0.05, in_len=8192,
                                     out_len=128, cls="mixed_long",
                                     wave=0))
                t += poisson(rate)
            else:
                reqs.append(dict(t=t, in_len=256, out_len=512,
                                 cls="mixed_short", wave=0))
                t += poisson(rate)
            i += 1
    elif name == "waves":
        t = 0.0
        i = 0
        while t < dur:
            n = rng.choice([32, 64])
            for k in range(n):
                reqs.append(dict(t=t + rng.random() * 0.5, in_len=128,
                                 out_len=256, cls="short", wave=n))
            t += 20.0
            i += n
    elif name == "boundary":
        t, i = 0.0, 0
        while t < dur:
            reqs.append(dict(t=t, in_len=64, out_len=1024, cls="long",
                             wave=0))
            t += poisson(rate)
            i += 1
    elif name == "pre_sat":
        # fixed prefill workload: all at t=0; headline = makespan and
        # prefill tokens/s (in_tok/makespan), identical set on every system
        for i in range(48):
            reqs.append(dict(t=0.0, in_len=args.in_len or 8192,
                             out_len=64, cls="long", wave=0))
    elif name == "dec_sat":
        # fixed decode workload: rate = concurrency M; headline = out tok/s
        # over makespan (M * 1024 / wall) and TPOT/ITL under that M
        for i in range(int(rate)):
            reqs.append(dict(t=0.0, in_len=args.in_len or 2048,
                             out_len=1024, cls="long", wave=0))
    elif name == "fixed_mix":
        # fixed mixed workload, identical set everywhere; headline =
        # makespan, req/s, out tok/s
        # --mix-x K: K copies (distinct prompts) so a fast system stays
        # saturated for the whole profiler window (bubble runs, §7.4)
        for i in range(24 * args.mix_x):
            reqs.append(dict(t=0.0, in_len=8192, out_len=128, cls="long",
                             wave=0))
        for i in range(48 * args.mix_x):
            reqs.append(dict(t=0.0, in_len=rng.choice([256, 512]),
                             out_len=512, cls="short", wave=0))
    else:
        raise ValueError(name)
    return sorted(reqs, key=lambda r: r["t"])


async def send_one(sess, idx, r, results):
    await asyncio.sleep(r["t"])  # absolute-ish: caller starts t0 at 0
    prompt = tok_prompt(r["in_len"], 777000 + idx)
    t0 = time.perf_counter()
    ttft = None
    itls = []
    ntok = 0
    err = None
    try:
        async with sess.post(f"{URL}/v1/completions", json={
                "model": "m", "prompt": prompt, "max_tokens": r["out_len"],
                "temperature": 0.0, "stream": True, "ignore_eos": True},
                timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
            last = t0
            async for line in resp.content:
                if line.startswith(b"data:") and b"[DONE]" not in line:
                    now = time.perf_counter()
                    try:
                        j = json.loads(line[5:])
                    except Exception:  # noqa: BLE001
                        continue
                    ch = j.get("choices") or []
                    if ch and ch[0].get("text") is not None:
                        if ttft is None:
                            ttft = now - t0
                        else:
                            itls.append((now - last) * 1000)
                        last = now
                        ntok += 1
    except Exception as e:  # noqa: BLE001
        err = repr(e)[:80]
    results.append(dict(
        idx=idx, cls=r["cls"], in_len=r["in_len"], out_len=r["out_len"],
        ttft_ms=round(ttft * 1000, 1) if ttft else None,
        itl_p95=round(sorted(itls)[int(0.95 * len(itls))], 1)
            if len(itls) > 3 else None,
        itl_max=round(max(itls), 1) if itls else None,
        tpot_ms=round(sum(itls) / len(itls), 2) if itls else None,
        n_tok=ntok, err=err,
        t_arrive=round(r["t"], 2)))


def score(results, dur, offered):
    ok = 0
    per_cls = {}
    for x in results:
        gate = SLOS.get(x["cls"], SLOS["short"])
        good = (x["err"] is None and x["ttft_ms"] is not None
                and x["ttft_ms"] <= gate["ttft_ms"]
                and (x["tpot_ms"] is None or
                     x["tpot_ms"] <= gate["tpot_ms"])
                and (x["itl_p95"] is None or
                     x["itl_p95"] <= gate["itl_p95_ms"]))
        x["good"] = bool(good)
        ok += good
        c = per_cls.setdefault(x["cls"], [0, 0])
        c[1] += 1
        c[0] += good
    out = {
        "offered": offered,
        "goodput": round(ok / dur, 3),
        "good_frac": round(ok / max(1, len(results)), 3),
        "per_cls": {k: f"{v[0]}/{v[1]}" for k, v in per_cls.items()},
        "n": len(results),
        # raw committed output tokens (what the server actually produced,
        # independent of SLO gates): the throughput-side口径
        "out_tok": sum(x.get("n_tok") or 0 for x in results),
        "out_tps_dur": round(sum(x.get("n_tok") or 0 for x in results) / dur, 1),
        "in_tok": sum(x["in_len"] for x in results),
        "done": sum(1 for x in results
                    if (x.get("n_tok") or 0) >= x["out_len"]),
    }
    for cls in per_cls:
        xs = [x["ttft_ms"] for x in results
              if x["cls"] == cls and x["ttft_ms"] is not None]
        if xs:
            xs.sort()
            out[f"ttft_p50_{cls}"] = round(xs[len(xs) // 2], 0)
            out[f"ttft_p95_{cls}"] = round(xs[int(0.95 * len(xs))], 0)
        it = [x["itl_p95"] for x in results
              if x["cls"] == cls and x["itl_p95"] is not None]
        if it:
            it.sort()
            out[f"itlp95_max_{cls}"] = round(it[int(0.9 * len(it))], 1)
    return out


def _bw_var_thread(hi_mbit: int, lo_mbit: int, period: float, stop_at: float,
                   adaptive_chunk: bool = False):
    """Alternate link shaping hi<->lo every `period` s (bw_var trace).

    With adaptive_chunk, also publish the TP-D net_chunk cap matching the
    current bandwidth (int8 ~4224 B/tok, TGT 80ms) via the scheduler ctrl
    file - the bandwidth-awareness input TP-D is designed to consume."""
    import subprocess
    import threading
    state = {"hi": True}

    def run():
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < stop_at:
            m = hi_mbit if state["hi"] else lo_mbit
            subprocess.run(
                [sys.executable, str(Path(__file__).with_name("het.py")), "tc", "set",
                 str(m)], capture_output=True, timeout=60)
            if adaptive_chunk:
                # mbit -> MB/s (x85% goodput), TGT 80ms, int8 4224 B/tok
                nc = int(m * 1e6 * 0.080 * 0.85 / 8 / 4224)
                Path(EXP / "sched_ctrl.json").write_text(
                    json.dumps({"net_chunk": max(1, nc)}))
            state["hi"] = not state["hi"]
            time.sleep(period)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


async def replay(args):
    global URL, TIMEOUT
    URL = f"http://127.0.0.1:{args.port}"
    TIMEOUT = args.timeout
    trace = gen_trace(args.trace, args.rate, args.dur, args.seed)
    results = []
    if args.bw_var:
        _bw_var_thread(args.bw_hi, args.bw_lo, args.bw_period, args.dur,
                       adaptive_chunk=args.bw_adaptive)
    t0 = time.perf_counter()
    async with aiohttp.ClientSession() as sess:
        await asyncio.gather(*[
            send_one(sess, i, r, results)
            for i, r in enumerate(trace)])
    wall = time.perf_counter() - t0
    if args.bw_var:
        import subprocess
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("het.py")), "tc", "set",
             str(args.bw_hi)], capture_output=True, timeout=60)
    out = score(results, wall, args.rate)
    out.update(trace=args.trace, rate=args.rate, dur=args.dur,
               wall=round(wall, 1),
               out_tps_wall=round(out["out_tok"] / max(0.1, wall), 1))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".reqs.json"), "w") as f:
        json.dump(results, f)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("replay")
    p.add_argument("--trace", required=True)
    p.add_argument("--rate", type=float, default=2.0)
    p.add_argument("--dur", type=float, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(EXP / "runs" / "x.json"))
    p.add_argument("--bw-var", action="store_true",
                   help="drive link shaping alternation during replay")
    p.add_argument("--bw-hi", type=int, default=250)
    p.add_argument("--bw-lo", type=int, default=100)
    p.add_argument("--bw-period", type=float, default=15.0)
    p.add_argument("--in-len", type=int, default=0,
                   help="override input length for pre_sat/dec_sat "
                        "(bench.sh: prefill 2K/4K/8K, "
                        "decode inputs 256/4K)")
    p.add_argument("--bw-adaptive", action="store_true",
                   help="publish matching net_chunk cap with each shape")
    p.add_argument("--port", type=int, default=26000)
    p.add_argument("--mix-x", type=int, default=1,
                   help="fixed_mix: number of copies of the 72-request set")
    p.add_argument("--timeout", type=float, default=600,
                   help="per-request total timeout (s)")
    args = ap.parse_args()
    asyncio.run(replay(args))
