<div align="center">
    <img src="resources/images/original.png" width="200" height="200">
</div>

# MoLink Project

MoLink (***Mo***del-***Link***) is a distributed LLM serving system, aiming to achieve high performance LLM inference services with distributed computing resources that might spread over the Internet. You can also run MoLink over heterogeneous devices. 

## Installation Guide

MoLink is built on top of vLLM, and will manage to keep compatible with its latest version, currently we support vLLM **v0.8.5post1**. Please ensure that your server meets the requirements for running vLLM, refer to [this](https://docs.vllm.ai/en/latest/).

you can install MoLink with the following steps:

```shell
git clone https://github.com/oldcpple/MoLink.git
cd MoLink
pip install -e .
pip install grpcio-tools==1.71.0 protobuf==5.29.0
```

We need to perform additional processing for the installation of **grpcio-tools** and **protobuf**, because of the conflicts with vLLM dependencies.

## Usage Guide

Once MoLink is successfully installed, you can follow this guide to deploy LLMs with GPU servers.

This is an example, assume that we have 2 servers and each with one GPU, and attempt to deploy a 70B LLaMA2 model. On the first server, simply run:

```shell
python -m molink.entrypoints.api_server --model meta-llama/Llama-2-70b-chat-hf --port 8080 --dtype=half --max_model_len 4096 --pipeline_parallel_size 2 --serving_layers 0,39
```

Two important arguments are  ***pipeline_parallel_size*** and ***serving_layers***. Set  ***pipeline_parallel_size*** to the number of servers used to serve the model, 2 in this example.  ***serving_layers*** claims the transformer layers this server will hold, please refer to ***config.json***  of your target model from Huggingface Hub to checkout how many layers it possesses in total before deciding how to split it (80 layers for 70B LLaMA2 in this example, we split it as 0-39 and 40-79 on two servers respectively).  Other arguments are inherited from vLLM and compatible with it.

During startup, the first server will print logs like the following:

```shell
DISTRIBUTED SERVICE INFO: MoLink gRPC server works at 172.17.0.17:50051
DISTRIBUTED SERVICE INFO: If this is the first node of the swarm, you can copy the DHT INFO as the initial peer of following nodes
```

Simply copy the first line, namely address of the DHT server,  ***172.17.0.17:50051*** in this example, and use it as the ***initial_peer*** in the following command to start the second server:

```shell
python -m molink.entrypoints.api_server --model meta-llama/Llama-2-70b-chat-hf --port 9090 --dtype=half --max_model_len 4096 --pipeline_parallel_size 2 --serving_layers 40,79 --initial_peer 172.17.0.15:50051
```

You can also serve the LLM with a single node, in this case the system falls back to vLLM:

```shell
python -m molink.entrypoints.api_server --model meta-llama/Llama-2-70b-chat-hf --port 8080 --dtype=half --max_model_len 4096
```

For multi-GPU nodes, you can use multiple GPUs for tensor-parallel by specifying argument **--tensor_parallel_size**. It's also supported to run on a hybrid pipeline, for example, the tensor parallelism size of each stage can be different, and devices can be heterogeneous.

The inference service usage are also compatible with vLLM's api server, for example you can simply run (change localhost to your server IP if you're not running at local ):

```shell
curl http://localhost:8080/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "San Francisco is a",
        "max_tokens": 20,
        "temperature": 0
    }'
```

MoLink also supports OpenAI-Compatible servers,  you can start one with:

```shell
python -m molink.entrypoints.openai.api_server --model XXXXX (same as examples above)
```

And access the API server like：

```
curl http://localhost:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "San Francisco is a",
        "max_tokens": 20,
        "temperature": 0
    }'
```

or

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
  model="meta-llama/Llama-2-70b-chat-hf",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
```



## Supported Model Architectures:

- **BaichuanForCausalLM**
- **BloomForCausalLM**
- **ChatGLMForCausalLM**
- **CohereForCausalLM**
- **DeepseekForCausalLM**
- **DeepseekV2ForCausalLM**
- **DeepseekV3ForCausalLM**
- **FalconForCausalLM**
- **GemmaForCausalLM**
- **Gemma2ForCausalLM**
- **GlmForCausalLM**
- **GPT2LMHeadModel**
- **LlamaForCausalLM**
- **MambaForCausalLM**
- **MixtralForCausalLM**
- **PhiForCausalLM**
- **Phi3ForCausalLM**
- **QWenLMHeadModel**
- **Qwen2MoeForCausalLM**
- **Qwen2ForCausalLM**
- **Qwen3MoeForCausalLM**
- **Qwen3ForCausalLM**
