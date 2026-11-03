#!/usr/bin/env python
"""Acceptance-kit lifecycle for the two-machine pipeline (A stage 0, B stage 1).

Everything cluster-specific comes from the environment (see
cluster.env.example). Same engine flags as the measured runs:
TP=2 x PP=2, ray executor, partition 28,20 (A3B) / 32,32 (32B),
max-num-batched-tokens 4096, max-num-seqs 64, max-model-len 16384,
prefix caching, gpu-memory-utilization 0.90.

  het.py up                      # (re)build the ray cluster A head + B
  het.py start --best            # ours (round 6): link + int4 + merge,
                                 #   Model Runner V2, --tok-tail
  het.py start --best-tpc        # + TP-C pred (strict ITL p95 tails)
  het.py start                   # native vLLM (all hetvllm features off)
  het.py start --runner v2       # native vLLM with Model Runner V2 forced
  het.py start --best --model d32      # Qwen3-32B, 32,32
  het.py start --best --tp 1 --pp 4    # PP4: A0, A1 | B0, B1 (14/14/10/10)
  het.py start --best --link-sync     # ablation control: synchronous link
  het.py start ... --prof t15:55 --prof-cuda-only   # bubble window (§7)
  het.py wait | stop
  het.py tc set 250 | tc clear   # symmetric egress shaping, verified
  het.py pull-prof DIR           # collect both machines' profiler traces
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def env(k: str) -> str:
    v = os.environ.get(k)
    if not v:
        sys.exit(f"missing {k}: source cluster.env first")
    return v


RES = Path(os.environ.get("HET_RESULTS", "./results")).resolve()
TRACE = "/tmp/het_trace"  # exists on both machines; trace on = measured setup


def sh(cmd: str, bg: bool = False) -> int:
    print("+", cmd[:200])
    if bg:
        subprocess.Popen(["bash", "-c", cmd], start_new_session=True)
        return 0
    return subprocess.run(["bash", "-c", cmd]).returncode


def onb(cmd: str, timeout: float = 60.0) -> tuple[int, str]:
    full = f"{env('B_SSH')} {shlex.quote('bash -lc ' + shlex.quote(cmd))}"
    p = subprocess.run(["bash", "-c", full], capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def up(args):
    a_ray = f"{env('A_VENV')}/bin/ray"
    b_ray = f"{env('B_VENV')}/bin/ray"
    sh(f"{a_ray} stop --force >/dev/null 2>&1; "
       f"CUDA_VISIBLE_DEVICES={env('GPUS_A')} RAY_AUTH_MODE=disabled "
       f"NCCL_SOCKET_IFNAME={env('A_IF')} GLOO_SOCKET_IFNAME={env('A_IF')} "
       f"HETVLLM_PP_TRACE={TRACE} "
       f"{a_ray} start --head --port=6399 --node-ip-address={env('A_IP')}")
    time.sleep(3)
    rc, out = onb(
        f"{b_ray} stop --force >/dev/null 2>&1; mkdir -p {TRACE}; "
        f"RAY_AUTH_MODE=disabled NCCL_SOCKET_IFNAME={env('B_IF')} "
        f"GLOO_SOCKET_IFNAME={env('B_IF')} HETVLLM_PP_TRACE={TRACE} "
        f"{b_ray} start --address={env('A_IP')}:6399 "
        f"--node-ip-address={env('B_IP')}"
        # B_NUM_GPUS=1 for a 1+1 GPU run: otherwise Ray packs both stages
        # onto B (the node with more free GPUs) and nothing crosses the WAN
        + (f" --num-gpus={os.environ['B_NUM_GPUS']}"
           if os.environ.get("B_NUM_GPUS") else ""), timeout=120)
    print(out[-400:])
    time.sleep(5)
    sh(f"RAY_AUTH_MODE=disabled {a_ray} status")


def start(args):
    if args.partition is None:
        # KV pool = min over stages: keep the 32B split symmetric
        args.partition = {("a3b", 2): "28,20", ("d32", 2): "32,32",
                          ("q8", 2): "20,16",
                          ("a3b", 4): "14,14,10,10"}[(args.model, args.pp)]
    if args.best or args.best_tpc:
        args.link, args.merge, args.tok_tail = True, True, True
        args.codec, args.runner = "int4", "v2"
        if args.best_tpc:
            args.net_tgt_ms, args.tpc = 80.0, "pred"
        else:
            args.net_bw = 0
    if args.tpc and args.model == "d32" and args.tpc_gate_ms is None:
        args.tpc_gate_ms = 400  # 32B decode step ~100 ms > A3B gate
    RES.mkdir(parents=True, exist_ok=True)
    model = env({"a3b": "MODEL_A3B", "d32": "MODEL_D32",
                 "q8": "MODEL_Q8"}[args.model])
    e = {"CUDA_VISIBLE_DEVICES": env("GPUS_A"), "RAY_AUTH_MODE": "disabled",
         "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
         "VLLM_PP_LAYER_PARTITION": args.partition,
         "HETVLLM_PP_TRACE": TRACE}
    if args.runner:
        e["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "v2" else "0"
    if args.tok_tail:
        e["HETVLLM_TOK_TAIL"] = "1"
    if args.link_sync:
        e["HETVLLM_PP_LINK_ASYNC"] = "0"
    prof = ""
    if args.prof:
        # torch profiler window, seconds after the first busy step
        # ("t15:55"); B writes to /tmp/het_prof if RES/prof does not exist
        # there. Collect with `het.py pull-prof`.
        e["HETVLLM_PROF_STEPS"] = args.prof
        if args.prof_cuda_only:
            e["HETVLLM_PROF_CUDA_ONLY"] = "1"
        cfg = ('{"profiler":"torch","torch_profiler_dir":"%s",'
               '"torch_profiler_with_stack":false}' % (RES / "prof"))
        prof = f"--profiler-config {shlex.quote(cfg)} "
    if args.link:
        # (h, r) pair: bf16 / int8 + fp16 per 64 / int4 + fp16 per 32
        H = {"a3b": 2048, "d32": 5120, "q8": 4096}[args.model]
        bpt = {"int8": 2 * H + H // 16, "int4": H + H // 8}.get(
            args.codec, 4 * H)
        if args.merge:
            bpt //= 2
        # the cross-machine hop leaves the last stage on A
        n_a = len(env("GPUS_A").split(",")) // args.tp
        e |= {"HETVLLM_PP_LINK": "1", "HETVLLM_PP_LINK_CODEC": args.codec,
              "HETVLLM_PP_LINK_FROM": str(n_a - 1),
              "HETVLLM_PP_LINK_BIND": env("A_IP"),
              "HETVLLM_PP_LINK_PEER": env("B_IP"),
              "HETVLLM_PP_LINK_SLICE_KIB": "32",
              "HETVLLM_PP_LINK_WIN_MB": "8",
              "HETVLLM_NET_BW_MBPS": str(args.net_bw),
              "HETVLLM_NET_BYTES_PER_TOK": str(bpt),
              "HETVLLM_NET_TGT_MS": str(args.net_tgt_ms)}
    if args.merge:
        e["HETVLLM_PP_MERGE"] = "1"
    if args.tpc:
        e |= {"HETVLLM_TPC": args.tpc, "HETVLLM_TPC_HI_ROWS": "256",
              "HETVLLM_TPC_SPLIT": "1", "HETVLLM_TPC_FASTLANE": "1",
              "HETVLLM_SCHED_CTRL": str(RES / "sched_ctrl.json")}
        if args.tpc_gate_ms:
            e["HETVLLM_TPC_GATE_MS"] = str(args.tpc_gate_ms)
    envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in e.items())
    sh(f"{envs} nohup {env('A_VENV')}/bin/python -m "
       f"vllm.entrypoints.openai.api_server --model {model} "
       f"--served-model-name m --tensor-parallel-size {args.tp} "
       f"--pipeline-parallel-size {args.pp} --distributed-executor-backend ray "
       f"--port {os.environ.get('PORT', '26000')} "
       f"--gpu-memory-utilization {args.gpu_util} --max-model-len 16384 "
       f"--max-num-seqs 64 --max-num-batched-tokens 4096 {prof}"
       f"{'--no-enable-prefix-caching' if args.no_prefix_cache else '--enable-prefix-caching'}"
       f" < /dev/null > {RES}/engine.log 2>&1 &",
       bg=True)


def wait(args):
    url = f"http://127.0.0.1:{os.environ.get('PORT', '26000')}/health"
    t0 = time.time()
    while time.time() - t0 < 3600:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    print(f"healthy in {time.time() - t0:.0f}s")
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    sys.exit("TIMEOUT")


def stop(args):
    # bracket trick: the pattern must not match this command line itself
    sh("pkill -f '[v]llm.entrypoints.openai.api_server'; "
       "pkill -f '[E]ngineCore'; "
       f"{env('A_VENV')}/bin/ray stop --force >/dev/null 2>&1; true")
    onb("pkill -f '[v]llm.entrypoints'; pkill -f '[E]ngineCore'; "
        f"{env('B_VENV')}/bin/ray stop --force >/dev/null 2>&1; true")


def pull_prof(args):
    """Copy profiler traces of both stages into DIR (A: RES/prof; B: RES/prof
    if that path exists on B, else /tmp/het_prof), then clear them."""
    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    src = RES / "prof"
    sh(f"cp {src}/* {d}/ 2>/dev/null; rm -rf {src}; true")
    tar = f"cd / && tar cf - {src} /tmp/het_prof 2>/dev/null"
    sh(f"{env('B_SSH')} {shlex.quote('bash -lc ' + shlex.quote(tar))} "
       f"| tar xf - -C {d} 2>/dev/null; "
       f"find {d} -mindepth 2 -type f -name '*.json*' -exec mv {{}} {d}/ \\; ; "
       f"find {d} -mindepth 1 -type d -empty -delete; true")
    onb(f"rm -rf {src} /tmp/het_prof; true")
    n = len(list(d.glob("*.json*")))
    print(f"{n} trace files in {d}")


def tc(args):
    sudo_a = os.environ.get("A_SUDO", "")
    sudo_b = os.environ.get("B_SUDO", "sudo -n")
    a_if, b_if = env("A_IF"), env("B_IF")
    if args.cmd2[0] == "clear":
        sh(f"{sudo_a} tc qdisc del dev {a_if} root 2>/dev/null; true")
        onb(f"{sudo_b} tc qdisc del dev {b_if} root 2>/dev/null; true")
        print("cleared")
        return
    rate = int(args.cmd2[-1])
    want = f"rate {rate}Mbit" if rate < 1000 else f"rate {rate // 1000}Gbit"
    for _ in range(3):  # 'replace' can fail under load; retry + verify
        q = (f"tc qdisc replace dev {{i}} root handle 1: htb default 10; "
             f"sleep 0.3; tc class replace dev {{i}} parent 1: classid 1:10 "
             f"htb rate {rate}mbit ceil {rate}mbit quantum 8000")
        sh(f"{sudo_a} bash -c {shlex.quote(q.format(i=a_if))}")
        onb(f"{sudo_b} bash -c {shlex.quote(q.format(i=b_if))}")
        got_a = subprocess.run(["tc", "class", "show", "dev", a_if],
                               capture_output=True, text=True).stdout
        _, got_b = onb(f"tc class show dev {b_if}")
        if want in got_a and want in got_b:
            print(f"shaped both egress to {rate} mbit (verified)")
            return
        time.sleep(1)
    sys.exit("tc verify failed")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("up").set_defaults(f=up)
    p = sub.add_parser("start")
    p.add_argument("--best", action="store_true",
                   help="ours: --link --codec int4 --merge --net-bw 0 "
                        "--runner v2 --tok-tail")
    p.add_argument("--best-tpc", action="store_true",
                   help="--best + TP-C pred (--net-tgt-ms 80 --tpc pred)")
    p.add_argument("--runner", default=None, choices=("v1", "v2"),
                   help="force the model runner (default: vLLM's choice)")
    p.add_argument("--tok-tail", action="store_true",
                   help="ship only the token ids the runner reads")
    p.add_argument("--gpu-util", type=float, default=0.90,
                   help="--gpu-memory-utilization (measured runs: 0.90)")
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--pp", type=int, default=2)
    p.add_argument("--model", default="a3b", choices=("a3b", "d32", "q8"))
    p.add_argument("--partition", default=None,
                   help="A,B layer split (default 28,20 a3b / 32,32 d32 / "
                        "20,16 q8)")
    p.add_argument("--tpc-gate-ms", type=float, default=None)
    p.add_argument("--link", action="store_true")
    p.add_argument("--codec", default="bf16", choices=("bf16", "int8", "int4"))
    p.add_argument("--merge", action="store_true")
    p.add_argument("--net-bw", type=float, default=26,
                   help="link bandwidth estimate MB/s; 0 disables the TP-D "
                        "chunk cap when TP-C is off")
    p.add_argument("--no-prefix-cache", action="store_true")
    p.add_argument("--net-tgt-ms", type=float, default=150)
    p.add_argument("--tpc", default=None, choices=("auto", "pred"))
    p.add_argument("--link-sync", action="store_true",
                   help="synchronous pp_link send/recv (ablation control)")
    p.add_argument("--prof", default="",
                   help="'tA:B' torch-profiler window, seconds A..B after "
                        "the first busy step (bubble measurement)")
    p.add_argument("--prof-cuda-only", action="store_true",
                   help="profiler records GPU kernels only (low host cost)")
    p.set_defaults(f=start)
    sub.add_parser("wait").set_defaults(f=wait)
    sub.add_parser("stop").set_defaults(f=stop)
    p = sub.add_parser("pull-prof")
    p.add_argument("dir")
    p.set_defaults(f=pull_prof)
    p = sub.add_parser("tc")
    p.add_argument("cmd2", nargs="+")
    p.set_defaults(f=tc)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
