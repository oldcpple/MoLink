"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""

import asyncio
import json
import os
import ssl
import tempfile
from argparse import Namespace
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

import vllm.envs as envs
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.utils import with_cancellation
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.system_utils import set_ulimit
from vllm.version import __version__ as VLLM_VERSION
from molinkv1.arg_utils import MolinkEngineArgs
from molinkv1.config import MolinkConfig
from molinkv1.engine.engine import MolinkEngine
logger = init_logger("vllm.entrypoints.api_server")

app = FastAPI()
engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/molink_metrics")
async def get_molink_metrics() -> Response:
    """Get communication layer metrics from the executor process."""
    # The executor writes metrics to a temp file keyed by gRPC port.
    # We can determine the gRPC port from the engine's molink config.
    grpc_port = None
    if engine is not None:
        try:
            vllm_config = getattr(engine, "vllm_config", None)
            molink_config = getattr(vllm_config, "molink_config", None) if vllm_config else None
            grpc_port = getattr(molink_config, "grpc_port", None) if molink_config else None
        except Exception:
            pass

    if grpc_port:
        path = os.path.join(tempfile.gettempdir(), f"molink_metrics_{grpc_port}.json")
        try:
            with open(path) as fh:
                return JSONResponse(json.load(fh))
        except FileNotFoundError:
            return JSONResponse({"service_metrics": [], "delivery_metrics": [],
                                 "node": None, "is_head": None})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # Fallback: try any file (single-node case)
    import glob
    pattern = os.path.join(tempfile.gettempdir(), "molink_metrics_*.json")
    files = glob.glob(pattern)
    if not files:
        return JSONResponse({"service_metrics": [], "delivery_metrics": [],
                             "node": None, "is_head": None})
    try:
        with open(files[0]) as fh:
            return JSONResponse(json.load(fh))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/molink_metrics/reset")
async def reset_molink_metrics() -> Response:
    """Reset communication layer metrics by deleting this node's metrics file."""
    grpc_port = None
    if engine is not None:
        try:
            vllm_config = getattr(engine, "vllm_config", None)
            molink_config = getattr(vllm_config, "molink_config", None) if vllm_config else None
            grpc_port = getattr(molink_config, "grpc_port", None) if molink_config else None
        except Exception:
            pass

    if grpc_port:
        path = os.path.join(tempfile.gettempdir(), f"molink_metrics_{grpc_port}.json")
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    else:
        import glob
        pattern = os.path.join(tempfile.gettempdir(), "molink_metrics_*.json")
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except Exception:
                pass
    return JSONResponse({"status": "ok"})


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    return await _generate(request_dict, raw_request=request)


@with_cancellation
async def _generate(request_dict: dict, raw_request: Request) -> Response:
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            assert prompt is not None
            text_outputs = [prompt + output.text for output in request_output.outputs]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\n").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output
    except asyncio.CancelledError:
        return Response(status_code=499)

    assert final_output is not None
    prompt = final_output.prompt
    assert prompt is not None
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


def build_app(args: Namespace) -> FastAPI:
    global app

    app.root_path = args.root_path
    return app


async def init_app(
    args: Namespace,
    llm_engine: MolinkEngine | None = None,
) -> FastAPI:
    app = build_app(args)

    global engine

    engine_args = MolinkEngineArgs.from_cli_args(args)

    is_worker = bool(getattr(args, "molink_initial_peer", None))

    if llm_engine is not None:
        engine = llm_engine
    elif is_worker:
        from molinkv1.engine.worker_node import MolinkWorkerNode
        vllm_config = engine_args.create_engine_config(UsageContext.API_SERVER)
        # Attach molink_config to vllm_config so the worker node can read it.
        molink_config = MolinkConfig(
            enabled=True,
            initial_peer=engine_args.molink_initial_peer,
            grpc_port=engine_args.molink_grpc_port,
            start_layer=engine_args.molink_start_layer,
            end_layer=engine_args.molink_end_layer,
        )
        from molinkv1.config import VllmConfig1
        vllm_config.__class__ = VllmConfig1
        vllm_config._update_attr(molink_config)
        engine = MolinkWorkerNode(vllm_config)
    else:
        engine = MolinkEngine.from_engine_args(
            engine_args, usage_context=UsageContext.API_SERVER
        )

    app.state.engine_client = engine
    return app


async def run_server(
    args: Namespace, llm_engine: MolinkEngine | None = None, **uvicorn_kwargs: Any
) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    set_ulimit()

    app = await init_app(args, llm_engine)
    assert engine is not None

    shutdown_task = await serve_http(
        app,
        sock=None,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    await shutdown_task


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=parser.check_port, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--ssl-ca-certs", type=str, default=None, help="The CA certificates file"
    )
    parser.add_argument(
        "--enable-ssl-refresh",
        action="store_true",
        default=False,
        help="Refresh SSL Context when SSL certificate files change",
    )
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy",
    )
    parser.add_argument("--log-level", type=str, default="debug")
    parser = MolinkEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    asyncio.run(run_server(args))