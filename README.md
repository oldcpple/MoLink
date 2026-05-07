<div align="center">
    <img src="resources/images/original.png" width="200" height="200">
</div>

# MoLink: Distributed Large Language Model Serving System

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![vLLM 0.19.0+](https://img.shields.io/badge/vLLM-0.19.0+-green.svg)](https://github.com/vllm-project/vllm)

**MoLink** (***Mo***del-***Link***) is a distributed LLM serving system that enables inference of large language models across geographically distributed and heterogeneous computing resources. By reconciling computation and communication overhead, MoLink delivers efficient LLM serving even when resources are spread across the Internet or connected via consumer-grade networks.

## Key Features

- **Distributed Architecture**: Deploy LLMs across multiple servers with automatic pipeline parallelism
- **Heterogeneous Support**: Run on mixed GPU configurations with different compute capabilities
- **Flexible Layer Partitioning**: Customizable model splitting with configurable layer distribution
- **API Compatibility**: Full compatibility with vLLM and OpenAI API standards
- **Multi-Backend Support**: Works with various LLM architectures (LLaMA, Qwen, etc.)

## Architecture Overview

MoLink v1 extends vLLM v0.19.0+ with cross-node pipeline parallelism via gRPC. Each node runs a subset of the model's transformer layers, and intermediate activations are transferred between nodes over gRPC.

### Two Node Roles

| Role | Description |
|------|-------------|
| **Head Node** | Runs the `MolinkEngine` (based on vLLM's `AsyncLLM`). Handles request scheduling, KV cache management, the first pipeline stage, and the HTTP API server. |
| **Worker Node** | Runs a lightweight `MolinkWorkerNode` with an in-process `MolinkWorker`. Receives intermediate tensors via gRPC, executes its assigned layers, and forwards results to the next stage or back to the head. |


### Source Code Structure

```
molinkv1/
├── core/scheduler.py        # MolinkScheduler / MolinkAsyncScheduler (thin wrappers)
├── engine/
│   ├── engine.py            # MolinkEngine — head node engine (extends AsyncLLM)
│   ├── core.py              # MolinkEngineCoreProc — engine core process override
│   └── worker_node.py       # MolinkWorkerNode — lightweight worker node
├── executor/executor.py     # MolinkExecutor — head node executor (extends MultiprocExecutor)
├── worker/worker.py         # MolinkWorker — GPU worker with MoLink tensor passing
├── service.py               # MolinkService — gRPC service for head node
├── config.py                # MolinkConfig, MolinkSchedulerConfig
├── parallel_state.py        # MoLink PP state and vLLM patching utilities
├── arg_utils.py             # CLI argument definitions
├── utils.py                 # IP detection, port finding, PipelineTopology
├── comm/
│   ├── molink.proto         # gRPC protocol definitions
│   ├── molink_pb2.py        # Generated protobuf code
│   └── molink_pb2_grpc.py   # Generated gRPC stubs
└── entrypoints/api_server.py # HTTP API server
```

### Communication Protocol

Inter-node communication uses gRPC with the following key RPCs defined in `molink.proto`:

| RPC | Purpose |
|-----|---------|
| `JoinPipeline` | Worker registers with head node; head returns its `num_gpu_blocks` so the worker can cap its KV cache accordingly |
| `PushIntermediateTensors` | Forward serialized intermediate activations + `SchedulerOutput` to the next stage |
| `PushSamplerOutput` | Last stage returns final `ModelRunnerOutput` to the head node |
| `HealthCheck` | Liveness probe |

Tensors are serialized with a compact binary wire format: `[ndim][shape...][dtype_len][dtype_str][raw_bytes]`. bfloat16 tensors are handled specially via uint8 view conversion.

## Prerequisites

MoLink is built on top of **vLLM v0.19.0+** and inherits its system requirements:

- **GPU**: NVIDIA GPUs with compute capability 8.0+ (e.g., RTX 3090, A100)
- **CUDA**: Version 11.8 or higher
- **Python**: Version 3.10 or higher

For detailed vLLM requirements, refer to the [official documentation](https://docs.vllm.ai/en/latest/).

## Installation

```bash
git clone https://github.com/oldcpple/MoLink.git
cd MoLink
pip install -e .
```

## Usage Guide

### Distributed Deployment

> **Note:** If you are using GPUs with low compute capability (e.g., lower than sm80), set the attention backend:
> ```bash
> export VLLM_ATTENTION_BACKEND=TRITON_ATTN
> ```

#### Example: Deploying Qwen3-14B on Two Servers

**Server 1** (Head Node — Layers 0-20):

```bash
python -m molinkv1.entrypoints.api_server \
    --model Qwen/Qwen3-14B \
    --molink-grpc-port 50061 \
    --molink-start-layer 0 \
    --molink-end-layer 20 \
    --port 8080 \
    --max-model-len 4096
```

After startup, copy the gRPC address from the logs (e.g., `10.130.151.15:50061`).

**Server 2** (Worker Node — Layers 20-end):

```bash
python -m molinkv1.entrypoints.api_server \
    --model Qwen/Qwen3-14B \
    --molink-grpc-port 50062 \
    --molink-start-layer 20 \
    --molink-end-layer -1 \
    --port 9095 \
    --max-model-len 4096 \
    --molink-initial-peer 10.130.151.15:50061
```

### Single-Node Deployment

When no layer splitting or peer is configured, MoLink falls back to standard vLLM:

```bash
python -m molinkv1.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-chat-hf \
    --port 8080 \
    --dtype half \
    --max-model-len 4096
```

### Heterogeneous Deployment

Different stages can use different tensor parallelism sizes:

```bash
# Stage 1: 2 GPUs with layers 0-20
CUDA_VISIBLE_DEVICES=0,1 python -m molinkv1.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-chat-hf \
    --tensor-parallel-size 2 \
    --molink-start-layer 0 \
    --molink-end-layer 20 \
    --port 8080

# Stage 2: 4 GPUs with layers 20-end
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m molinkv1.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-chat-hf \
    --tensor-parallel-size 4 \
    --molink-start-layer 20 \
    --molink-end-layer -1 \
    --molink-initial-peer <first-node-address> \
    --port 9095
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--molink-grpc-port` | Port for inter-node gRPC communication | `0` (auto) |
| `--molink-start-layer` | First transformer layer this node handles (inclusive) | `0` |
| `--molink-end-layer` | Last transformer layer this node handles (exclusive, `-1` = all remaining) | `-1` |
| `--molink-initial-peer` | gRPC address of the head node to join. If unset, this node is the head. | `None` |
| `--molink-max-message-size-mb` | Maximum gRPC message size in MB | `200` |
| `--molink-enable-metrics` | Enable communication metrics recording | `False` |
| `--molink-max-concurrent-batches` | Maximum concurrent batches in pipeline | `2` |
| `--tensor-parallel-size` | Number of GPUs for intra-node tensor parallelism | `1` |
| `--max-model-len` | Maximum sequence length | Model default |

## API Usage

### Generation API

```bash
curl http://localhost:8080/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "San Francisco is a",
        "max_tokens": 20,
        "temperature": 0
    }'
```

### OpenAI-Compatible API

The `molink` package provides an OpenAI-compatible server:

```bash
python -m molink.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --molink-grpc-port 50061 \
    --molink-start-layer 0 \
    --molink-end-layer -1
```

Use with cURL:

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer YOUR_API_KEY" \
    -d '{
        "model": "Qwen/Qwen3-14B",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }'
```

Use with OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-api-key-here"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-14B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the theory of relativity."}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

### Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/molink_metrics` | GET | Communication layer metrics (service metrics, delivery metrics) |
| `/molink_metrics/reset` | POST | Reset communication metrics |


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

MoLink is built upon the excellent [vLLM](https://github.com/vllm-project/vllm) project. We thank the vLLM team for their outstanding work on efficient LLM serving.

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
