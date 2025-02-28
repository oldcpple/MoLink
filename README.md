<div align="center">
    <img src="resources/images/original.png" width="200" height="200">
</div>

# MoLink Project

MoLink (***Mo***del-***Link***) is a distributed and LLM serving systems, aiming to achieve high performance LLM inference service with distributed computing resources that might spread over the Internet. You can also run MoLink over heterogeneous devices. 

## Installation Guide

MoLink is built on top of vLLM, and we will manage to keep it compatible with its latest version, currently we support vLLM v0.7.2. Please ensure that your server meets the requirements for running vLLM, refer to [this](https://docs.vllm.ai/en/latest/).

you can install MoLink with the following steps:

```
git clone https://github.com/oldcpple/MoLink.git
cd MoLink
pip install -e .
```

## Usage Guide

Once MoLink is successfully installed, you can follow this guide to deploy LLMs with GPU servers.

This is an example, assume that we have 2 servers and each with one GPU, and attempt to deploy a 70B LLaMA2 model. On the first server, simply run:

```
python -m molink.entrypoints.api_server --model meta-llama/Llama-2-70b-chat-hf --port 8080 --dtype=half --max_model_len 4096 --pipeline_parallel_size 2 --serving_layers 0,39
```

Two important arguments are  ***pipeline_parallel_size*** and ***serving_layers***. Set  ***pipeline_parallel_size*** to the number of servers used to serve the model, 2 in this example.  ***serving_layers*** claims the transformer layers this server will hold, please refer to ***config.json***  of your target model from Huggingface Hub to checkout how many layers it possesses in total before deciding how to split it (80 layers for 70B LLaMA2 in this example, we split it as 0-39 and 40-79 on two servers respectively).  Other arguments are inherited from vLLM and compatible with it.

During startup, the first server will print logs like the following::

```
DISTRIBUTED SERVICE INFO: MoLink gRPC server works at 172.17.0.17:50051
DISTRIBUTED SERVICE INFO: MoLink DHT server works at 172.17.0.17:8468
DISTRIBUTED SERVICE INFO: If this is the first node of the swarm, you can copy the DHT INFO as the initial peer of following nodes
```

Simply copy the second line, namely address of the DHT server,  ***172.17.0.17:8468*** in this example, and use it as the ***initial_peer*** in the following command to start the second server:

```
python -m molink.entrypoints.api_server --model meta-llama/Llama-2-70b-chat-hf --port 9090 --dtype=half --max_model_len 4096 --pipeline_parallel_size 2 --serving_layers 40,79 --initial_peer 172.17.0.15:8468
```

You can also serve the LLM with a single node, in this case the system falls back to vLLM:

```
python -m molink.entrypoints.api_server --model meta-llama/Llama-2-70b-chat-hf --port 8080 --dtype=half --max_model_len 4096
```

The inference service usage are also compatible with vLLM's api server, for example you can simply run (change localhost to your server IP if you're not running at local ):

```
curl http://localhost:8080/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "San Francisco is a",
        "max_tokens": 20,
        "temperature": 0
    }'
```

Support for OpenAI-Compatible Servers is under development.

## Supported Model Architectures:

- LlamaForCausalLM
