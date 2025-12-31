<div align="center">
    <img src="resources/images/original.png" width="200" height="200">
</div>

# MoLink: Distributed Large Language Model Serving System

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![vLLM 0.11.2](https://img.shields.io/badge/vLLM-0.11.2-green.svg)](https://github.com/vllm-project/vllm)

**MoLink** (***Mo***del-***Link***) is an advanced distributed LLM serving system designed to enable high-performance inference of large language models across geographically distributed and heterogeneous computing resources. By reconciling computation and communication overhead, MoLink delivers efficient LLM serving even when resources are spread across the Internet or connected via consumer-grade networks.

## ✨ Key Features

- **🌐 Distributed Architecture**: Seamlessly deploy LLMs across multiple servers with automatic pipeline parallelism
- **⚡ Optimized Communication**: Advanced pipeline management with overlapped computation and communication
- **🎯 Heterogeneous Support**: Run on mixed GPU configurations with different compute capabilities
- **🔄 Flexible Layer Partitioning**: Intelligent model splitting with customizable layer distribution
- **🚀 High Throughput**: Batch processing with continuous batching for optimal resource utilization
- **🔌 API Compatibility**: Full compatibility with vLLM and OpenAI API standards
- **📊 Multi-Backend Support**: Works with various LLM architectures (LLaMA, Qwen, etc.)

## 🏗️ Architecture Overview

MoLink v1 introduces a redesigned architecture with enhanced scalability and performance:

- **Communication Layer**: gRPC-based efficient inter-node communication with DHT peer discovery
- **Pipeline Management**: Intelligent request routing and activation transfer between stages
- **Execution Engine**: Optimized model runner with KV cache management and attention mechanisms  
- **Scheduler**: Advanced request scheduling with continuous batching support
- **Worker Pool**: Distributed worker management with automatic load balancing

## 📋 Prerequisites

MoLink is built on top of **vLLM v0.11.2** and inherits its system requirements:

- **GPU**: NVIDIA GPUs with compute capability 8.0+ (3090, etc.)
- **CUDA**: Version 11.8 or higher
- **Python**: Version 3.8 or higher

For detailed vLLM requirements, refer to the [official documentation](https://docs.vllm.ai/en/latest/).

## 🚀 Installation

### Quick Installation

```bash
git clone https://github.com/oldcpple/MoLink.git
cd MoLink
pip install -e .
```


## 📖 Usage Guide

### Distributed Deployment

**NOTE:** if you are using GPUs with low compute capability(e.g. lower than sm80), please set the attn backend as follows:
```bash
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

#### Example: Deploying Qwen3-14B on Two Servers

**Server 1** (Layers 0-20):

```bash
python -m molinkv1.entrypoints.api_server \
    --model Qwen/Qwen3-14B \
    --molink-enabled \
    --molink-grpc-port 50061 \
    --molink-start-layer 0 \
    --molink-end-layer 20 \
    --port 8080 \
    --max-model-len 4096
```

After startup, copy the gRPC address from the logs (e.g., `10.130.151.15:50061`).

**Server 2** (Layers 20-end):

```bash
python -m molinkv1.entrypoints.api_server \
    --model Qwen/Qwen3-14B \
    --molink-enabled \
    --molink-grpc-port 50062 \
    --molink-start-layer 20 \
    --molink-end-layer -1 \
    --port 9095 \
    --max-model-len 4096 \
    --molink-initial-peer 10.130.151.15:50061
```

### Single-Node Deployment

For single-node deployment, MoLink gracefully falls back to standard vLLM operation:

```bash
python -m molinkv1.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-chat-hf \
    --port 8080 \
    --dtype half \
    --max-model-len 4096
```

### Heterogeneous Deployment

MoLink supports heterogeneous clusters where different stages have varying tensor parallelism:

```bash
# Stage 1: 2 GPUs with layers 0-20
CUDA_VISIBLE_DEVICES=0,1 python -m molinkv1.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-chat-hf \
    --molink-enabled \
    --tensor-parallel-size 2 \
    --molink-start-layer 0 \
    --molink-end-layer 20

# Stage 2: 4 GPUs with layers 20-end
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m molinkv1.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-chat-hf \
    --molink-enabled \
    --tensor-parallel-size 4 \
    --molink-start-layer 20 \
    --molink-end-layer -1 \
    --molink-initial-peer <first-node-address>
```



### Key Configuration Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--molink-enabled` | Enable distributed MoLink mode | - |
| `--molink-grpc-port` | Port for inter-node communication | `50061` |
| `--molink-start-layer` | Starting layer index (inclusive) | `0` |
| `--molink-end-layer` | Ending layer index (exclusive, -1 for last) | `20` or `-1` |
| `--molink-initial-peer` | Bootstrap peer address | `10.0.0.1:50061` |
| `--tensor-parallel-size` | Number of GPUs for tensor parallelism | `2` |
| `--max-model-len` | Maximum sequence length | `4096` |

## 🔌 API Usage

### Standard Generation API

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

Start the OpenAI-compatible server:

```bash
python -m molinkv1.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --molink-enabled \
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


## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

MoLink is built upon the excellent [vLLM](https://github.com/vllm-project/vllm) project. We thank the vLLM team for their outstanding work on efficient LLM serving.

## 📜 Citation

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
  doi = {10.18653/v1/2025.findings-emnlp.957},
  url = {https://aclanthology.org/2025.findings-emnlp.957/},
  isbn = {979-8-89176-335-7}
}
```
