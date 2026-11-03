<div align="center">
    <img src="resources/images/original.png" width="200" height="200">
</div>

# MoLink v2: Heterogeneous Pipeline Serving over Constrained Networks

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![vLLM 0.23.0](https://img.shields.io/badge/vLLM-0.23.0-green.svg)](https://github.com/vllm-project/vllm/tree/v0.23.0)

**MoLink** serves one large LLM across machines that are far apart or poorly connected, and across GPUs of different classes.
**MoLink v2** is a rewrite of that idea as a small, default-off patch on top of upstream vLLM v0.23.0,
instead of the gRPC-based engine fork of v1. It targets one specific problem: with pipeline parallelism over a
slow link (100 Mbps – 1 Gbps), the activation transfer between stages dominates every step, and the GPUs sit idle.

## What v2 changes

| Technique | What it does |
|---|---|
| **`pp_link`** | Cross-machine PP transport over ZMQ instead of NCCL `isend/irecv`: 32 KiB framing, two priority classes (decode can overtake in-flight prefill bytes), per-direction ack-window backpressure, fully asynchronous send/receive. |
| **int4 boundary encoding** | Stage-boundary activations are group-quantized to int4 (int8 also available). |
| **Residual merging** | Only `s = h + r` crosses the link; the next stage recovers `RMSNorm(s)` exactly with a fused add+RMSNorm. Together with int4, cross-machine bytes drop to 0.14× of stock. |
| **Model Runner V2** | Use upstream's V2 executor so decode no longer serializes across the two stages. |
| **Control-plane trimming** (`--tok-tail`) | The scheduler stops re-sending full prompt+output token lists for requests not scheduled in the previous step. |
| **Predictive chunk scheduler** (optional, `--best-tpc`) | Fits step latency online and sizes prefill chunks to protect decode latency. |

Every change is off unless a `HETVLLM_*` environment variable is set, so without them the patched tree behaves like upstream v0.23.0.

## Results (headline)

Testbed: A = 2×RTX 4090, B = 2×RTX 3090, link throttled with `tc`. Same machines, same split, same time window as the stock-vLLM baseline.

| Setup | Result |
|---|---|
| Qwen3-30B-A3B, TP2×PP2, 40 paired tests, 100 Mbps – 1 Gbps | **1.91×** geometric-mean speedup vs stock vLLM (1.54× vs stock with Model Runner V2 forced) |
| Long-prompt prefill, 100 Mbps, 48 × 8K prompts | 52.1 s vs 327.8 s (**6.29×**) |
| Overload at 250 Mbps (300 s of sustained arrivals) | 153/155 requests meet all latency targets vs 0/155; median TTFT 1.0 s vs 49.2 s |
| Qwen3-32B (dense), TP2×PP2 | 1.25× (1.67× at 100 Mbps) |
| Qwen3-30B-A3B, PP4 (4090, 4090 \| 3090, 3090) | 2.53× vs stock PP4 (1.90× vs stock V2) |
| Qwen3-8B, PP2 (1×4090 + 1×3090) | 1.97×; bubble rate 41%/39% vs 89%/81% |

The advantage grows as the network gets slower. Boundary int4 quantization costs some accuracy; see
[EVALUATION §8](docs/EVALUATION.md) for measured numbers and [§10](docs/EVALUATION.md) for limitations.
All numbers are from one cluster and one snapshot of the code; they do not transfer to other hardware without re-measuring.

## Quick start

```bash
# on BOTH machines
git clone https://github.com/vllm-project/vllm.git && cd vllm
git checkout 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665          # v0.23.0
git apply /path/to/MoLink/patch/hetvllm-vllm-v0.23.0.patch
uv venv --python 3.12 && . .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install pyzmq aiohttp matplotlib

# on machine A: describe the cluster, then start the engine with the final configuration
cd MoLink/kit && cp cluster.env.example cluster.env && $EDITOR cluster.env
set -a; . ./cluster.env; set +a
python het.py up                      # ray: A is head, B joins
python het.py start --best && python het.py wait
```

Machine B must be reachable by key-based, non-interactive ssh. The full procedure (one-shot paired benchmark, ablation,
bubble-rate and accuracy runs, expected results and tolerances) is in [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Repository layout

```
README.md              this file
docs/EVALUATION.md     full report: method, techniques, results, ablation, accuracy, limitations
docs/REPRODUCE.md      install, configure, run, expected results
patch/                 patch against vLLM v0.23.0 + per-file notes and env-var reference
kit/                   cluster launcher, benchmark/replay, ablation, bubble-rate and accuracy tools
data/                  raw replay JSON and summary CSVs behind every number and figure
figs/                  figures used in the report
make_figs.py           regenerates figs/ and data/*.csv from data/
resources/images/      logo
```

## Relationship to MoLink v1

v1 (gRPC pipeline between nodes, vLLM 0.19.0+, `molink/` and `molinkv1/` packages) is kept on the `main` branch and is
not part of this branch. v2 does not reuse its code; it keeps the goal (LLM serving across distributed, heterogeneous,
consumer-grade resources) and replaces the mechanism.

## License

Apache 2.0, see [LICENSE](LICENSE). The patch applies to vLLM, which is also Apache 2.0.

## Citation

If you find MoLink useful for your research or projects, please cite our paper:

Lewei Jin, Kui Zhang, Yongqi Chen, Zhuoyifan, Renjie Li, Yi Gao, Bowei Yang, Zhengong Cai, and Wei Dong. Distributed LLM Serving on Consumer-Grade GPUs by Reconciling Computation and Communication. In *Findings of the Association for Computational Linguistics: EMNLP 2025*, pages 17633–17642, Suzhou, China, November 2025. Association for Computational Linguistics.

```bibtex
@inproceedings{jin-etal-2025-distributed,
  title = {Distributed {LLM} Serving on Consumer-Grade {GPU}s by Reconciling Computation and Communication},
  author = {Jin, Lewei and Zhang, Kui and Chen, Yongqi and Zhuoyifan and Li, Renjie and Gao, Yi and Yang, Bowei and Cai, Zhengong and Dong, Wei},
  editor = {Christodoulopoulos, Christos and Chakraborty, Tanmoy and Rose, Carolyn and Peng, Violet},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  month = nov,
  year = {2025},
  address = {Suzhou, China},
  publisher = {Association for Computational Linguistics},
  pages = {17633--17642},
  doi = {10.18653/v1/2025.findings-emnlp.957/},
  url = {https://aclanthology.org/2025.findings-emnlp.957/},
  isbn = {979-8-89176-335-7}
}
```
