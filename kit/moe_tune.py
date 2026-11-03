#!/usr/bin/env python
"""Tune vLLM Triton fused-MoE configs for Qwen3-30B-A3B shapes on local GPU.

Shapes (K=2048, topk=8, bf16):
  tp2: E=128 local, N=384  (experts intermediate-sharded across 2 GPUs)
  ep2: E=64 local,  N=768  (64 full experts per GPU, global 128 + expert_map)

Writes JSON files named exactly like vLLM's config lookup expects, e.g.
  E=128,N=384,device_name=NVIDIA_GeForce_RTX_3090.json
so dropping them into vllm/model_executor/layers/fused_moe/configs/ makes the
engine pick them up automatically (no code change needed).

Usage:
  CUDA_VISIBLE_DEVICES=0 python moe_tune.py --shape tp2 --out ./moe_cfg   (tuned JSONs already ship in the patch)
"""
import argparse
import itertools
import json
import os
import statistics
import time

import torch


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    for _ in range(iters):
        fn()
    ev[1].record()
    torch.cuda.synchronize()
    return ev[0].elapsed_time(ev[1]) / iters


def grid_for(m):
    if m <= 32:
        bm = [16, 32]
    elif m <= 128:
        bm = [16, 32, 64]
    elif m <= 512:
        bm = [32, 64, 128]
    else:
        bm = [64, 128]
    bn = [32, 64, 128]
    bk = [64, 128, 256] if m <= 128 else [64, 128]
    gm = [1, 16]
    warps = [4, 8]
    stages = [2, 3, 4]
    return bm, bn, bk, gm, warps, stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=["tp2", "ep2"], required=True)
    ap.add_argument("--out", default="/tmp/moe_cfg")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--ms", default="1,2,4,8,16,24,32,48,64,96,128,192,256,384,512,1024,2048,4096,8192")
    args = ap.parse_args()

    from vllm.model_executor.layers.fused_moe import fused_experts, override_config
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        try_get_optimal_moe_config,
    )
    from vllm.platforms import current_platform

    torch.manual_seed(0)
    dev = "cuda"
    K, topk, hidden = 2048, 8, 2048
    if args.shape == "tp2":
        E, N, ge, emap = 128, 384, 128, None
    else:
        E, N, ge, emap = 64, 768, 128, torch.arange(128, dtype=torch.int32, device=dev)
        emap[64:] = -1  # rank-0 view of an EP2 world

    w1 = torch.randn(E, 2 * N, K, device=dev, dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(E, K, N, device=dev, dtype=torch.bfloat16) * 0.02

    ms = [int(x) for x in args.ms.split(",")]

    # Pass 1: baseline (default config) for all M, before any sweep pollutes
    # the triton compile cache with oversized-tile failures.
    inputs = {}
    t_defaults = {}
    for m in ms:
        a = torch.randn(m, K, device=dev, dtype=torch.bfloat16)
        topk_ids = torch.randint(0, ge, (m, topk), device=dev, dtype=torch.int32)
        tw = torch.rand(m, topk, device=dev, dtype=torch.float32)
        inputs[m] = (a, topk_ids, tw)
        try:
            t_def = bench(lambda: fused_experts(a, w1, w2, tw, topk_ids,
                                                global_num_experts=ge,
                                                expert_map=emap), iters=args.iters)
        except Exception:
            t_def = float("nan")
            torch.cuda.synchronize()
        t_defaults[m] = t_def
        print(f"M={m:5d} default={t_def:8.3f}ms", flush=True)

    # Pass 2: sweep.
    results = {}
    report = []
    for m in ms:
        a, topk_ids, tw = inputs[m]
        t_def = t_defaults[m]

        best, best_t = None, float("inf")
        bm, bn, bk, gm, warps, stages = grid_for(m)
        for combo in itertools.product(bm, bn, bk, gm, warps, stages):
            cfg = {"BLOCK_SIZE_M": combo[0], "BLOCK_SIZE_N": combo[1],
                   "BLOCK_SIZE_K": combo[2], "GROUP_SIZE_M": combo[3],
                   "SPLIT_K": 1, "num_warps": combo[4], "num_stages": combo[5]}
            try:
                with override_config(cfg):
                    t = bench(lambda: fused_experts(a, w1, w2, tw, topk_ids,
                                                    global_num_experts=ge,
                                                    expert_map=emap),
                              warmup=1, iters=args.iters)
            except Exception:
                continue
            if t < best_t:
                best, best_t = cfg, t
        results[m] = best
        sp = t_def / best_t
        report.append((m, t_def, best_t, sp, best))
        print(f"M={m:5d} default={t_def:8.3f}ms best={best_t:8.3f}ms "
              f"speedup={sp:5.2f}x {best}", flush=True)

    dev_name = current_platform.get_device_name().replace(" ", "_")
    os.makedirs(args.out, exist_ok=True)
    out = {str(m): results[m] for m in ms}
    out["triton_version"] = ""
    path = os.path.join(args.out, f"E={E},N={N},device_name={dev_name}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)

    # human-readable summary alongside
    with open(path.replace(".json", ".summary.txt"), "w") as f:
        for m, td, tb, sp, cfg in report:
            f.write(f"M={m}\tdefault={td:.3f}\tbest={tb:.3f}\tspeedup={sp:.2f}x\n")


if __name__ == "__main__":
    main()
