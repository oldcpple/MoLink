#!/usr/bin/env python3
"""
MoLink v1 Automated Test and Performance Benchmarking Script.

Usage:
    /opt/conda/envs/molink/bin/python /home/MoLink/tests/test_molinkv1_benchmark.py
"""

import asyncio
import json
import logging
import os
import random
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np
import psutil

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

MODEL_PATH = "/gxq/Qwen3-14B"
HEAD_PORT = 8080
TAIL_PORT = 9095
HEAD_GRPC_PORT = 50061
TAIL_GRPC_PORT = 50062
HEAD_URL = f"http://localhost:{HEAD_PORT}"
TAIL_URL = f"http://localhost:{TAIL_PORT}"
MAX_MODEL_LEN = 4096
CONDA_PYTHON = "/opt/conda/envs/vllm/bin/python"
HEALTH_TIMEOUT = 600
HEALTH_INTERVAL = 5
REQUEST_TIMEOUT = 300
TESTS_DIR = Path(__file__).parent


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass
class BenchmarkConfig:
    model_path: str = MODEL_PATH
    head_port: int = HEAD_PORT
    tail_port: int = TAIL_PORT
    head_grpc_port: int = HEAD_GRPC_PORT
    tail_grpc_port: int = TAIL_GRPC_PORT
    max_model_len: int = MAX_MODEL_LEN
    warmup_requests: int = 3
    baseline_iterations: int = 3
    request_timeout: int = REQUEST_TIMEOUT
    # Test matrices
    output_token_sizes: list[int] = field(default_factory=lambda: [64, 512, 1024, 2048])
    prompt_sizes_full: list[int] = field(default_factory=lambda: [32, 128, 512, 1024, 2048, 4096])
    # Concurrent: (concurrency, prompt_tokens, output_tokens)
    concurrent_tests: list[tuple[int, int, int]] = field(default_factory=lambda: [
        # output=64
        (1, 128, 64), (5, 128, 64), (10, 128, 64), (20, 128, 64), (50, 128, 64), (100, 128, 64),
        # output=512
        (1, 128, 512), (5, 128, 512), (10, 128, 512), (20, 128, 512), (50, 128, 512), (100, 128, 512),
        # output=1024
        (1, 128, 1024), (5, 128, 1024), (10, 128, 1024), (20, 128, 1024), (50, 128, 1024), (100, 128, 1024),
    ])


# ── IP Detection ───────────────────────────────────────────────────────────


def extract_ip() -> str:
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        st.connect(("10.255.255.255", 1))
        ip = st.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        st.close()
    return ip


# ── Metrics ────────────────────────────────────────────────────────────────


@dataclass
class RequestMetrics:
    request_id: int
    prompt_tokens: int
    max_tokens: int
    timestamp_start: str = ""
    timestamp_end: str = ""
    generated_tokens: int = 0
    generated_text_length: int = 0
    ttft_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class BenchmarkResult:
    test_name: str
    concurrency: int
    prompt_tokens: int
    max_tokens: int
    num_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time_s: float = 0.0
    requests_per_second: float = 0.0
    metrics: list[RequestMetrics] = field(default_factory=list)
    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    min_e2e_ms: float = 0.0
    avg_e2e_ms: float = 0.0
    p50_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    total_output_tokens: int = 0
    output_tokens_per_second: float = 0.0


@dataclass
class SamplePoint:
    relative_time_s: float
    gpu_utils: dict[int, float] = field(default_factory=dict)
    gpu_mem_pct: dict[int, float] = field(default_factory=dict)
    gpu_mem_mb: dict[int, float] = field(default_factory=dict)
    cpu_pct: float = 0.0
    mem_pct: float = 0.0
    net_sent_bytes_per_s: float = 0.0
    net_recv_bytes_per_s: float = 0.0


@dataclass
class SystemMetrics:
    samples: list[SamplePoint] = field(default_factory=list)
    duration_s: float = 0.0

    def gpu_summary(self) -> dict[int, dict]:
        if not self.samples:
            return {}
        gpu_ids = set()
        for s in self.samples:
            gpu_ids.update(s.gpu_utils.keys())
        summary = {}
        for gid in sorted(gpu_ids):
            utils = [s.gpu_utils[gid] for s in self.samples if gid in s.gpu_utils]
            mems = [s.gpu_mem_pct[gid] for s in self.samples if gid in s.gpu_mem_pct]
            if utils:
                summary[gid] = {
                    "avg_util": float(np.mean(utils)),
                    "max_util": float(np.max(utils)),
                    "min_util": float(np.min(utils)),
                    "avg_mem_pct": float(np.mean(mems)),
                    "max_mem_pct": float(np.max(mems)),
                }
        return summary

    def cpu_summary(self) -> dict:
        if not self.samples:
            return {}
        vals = [s.cpu_pct for s in self.samples]
        return {"avg": float(np.mean(vals)), "max": float(np.max(vals))}

    def net_summary(self) -> dict:
        if not self.samples:
            return {}
        sents = [s.net_sent_bytes_per_s for s in self.samples]
        recvs = [s.net_recv_bytes_per_s for s in self.samples]
        return {
            "avg_sent_per_s": float(np.mean(sents)),
            "avg_recv_per_s": float(np.mean(recvs)),
            "avg_sent_MB_s": float(np.mean(sents)) / 1024 / 1024,
            "avg_recv_MB_s": float(np.mean(recvs)) / 1024 / 1024,
        }


# ── Communication & Pipeline Metrics ───────────────────────────────────────


@dataclass
class CommMetrics:
    gpu_to_cpu_ms: float = 0.0
    serialize_ms: float = 0.0
    grpc_push_intermediate_ms: float = 0.0
    grpc_push_sampler_ms: float = 0.0
    deserialize_ms: float = 0.0
    head_compute_ms: float = 0.0
    tail_compute_ms: float = 0.0
    intermediate_bytes: int = 0
    sampler_bytes: int = 0


@dataclass
class PipelineMetrics:
    prompt_tokens: int = 0
    output_tokens: int = 0
    head_compute_ms: float = 0.0
    tail_compute_ms: float = 0.0
    gpu_to_cpu_ms: float = 0.0
    serialize_ms: float = 0.0
    grpc_push_intermediate_ms: float = 0.0
    deserialize_ms: float = 0.0
    grpc_push_sampler_ms: float = 0.0
    total_comm_overhead_ms: float = 0.0
    pipeline_overhead_pct: float = 0.0
    stage_balance_ratio: float = 0.0
    total_bytes: int = 0
    bandwidth_mb_per_s: float = 0.0
    e2e_latency_ms: float = 0.0


def _aggregate_metrics(raw_service: list[dict], raw_delivery: list[dict]) -> CommMetrics:
    """Aggregate raw metric dicts into a CommMetrics instance."""
    cm = CommMetrics()
    for m in raw_delivery:
        t = m.get("type", "")
        if t == "gpu_to_cpu":
            cm.gpu_to_cpu_ms += m.get("copy_ms", 0)
        elif t == "push_intermediate":
            cm.serialize_ms += m.get("serialize_ms", 0)
            cm.grpc_push_intermediate_ms += m.get("grpc_ms", 0)
            cm.intermediate_bytes = max(cm.intermediate_bytes, m.get("bytes", 0))
        elif t == "push_sampler":
            cm.grpc_push_sampler_ms += m.get("grpc_ms", 0)
            cm.sampler_bytes = max(cm.sampler_bytes, m.get("bytes", 0))
    for m in raw_service:
        t = m.get("type", "")
        if t == "head_compute":
            cm.head_compute_ms += m.get("compute_ms", 0)
        elif t == "worker_step":
            cm.deserialize_ms += m.get("deserialize_ms", 0)
            cm.tail_compute_ms += m.get("compute_ms", 0)
        elif t == "receive_intermediate":
            cm.intermediate_bytes = max(cm.intermediate_bytes, m.get("bytes", 0))
    return cm


class CommMetricsCollector:
    """Collects communication metrics from head and tail nodes."""

    def __init__(self, head_url: str, tail_url: str):
        self._head_url = head_url
        self._tail_url = tail_url

    async def reset(self, session: aiohttp.ClientSession):
        for url in (self._head_url, self._tail_url):
            try:
                async with session.post(
                    f"{url}/molink_metrics/reset",
                    timeout=aiohttp.ClientTimeout(total=5),
                ):
                    pass
            except Exception:
                pass

    async def collect(self, session: aiohttp.ClientSession) -> tuple[CommMetrics, CommMetrics]:
        """Collect metrics from both nodes. Returns (head_metrics, tail_metrics)."""
        results = []
        for url in (self._head_url, self._tail_url):
            try:
                async with session.get(
                    f"{url}/molink_metrics",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        cm = _aggregate_metrics(
                            data.get("service_metrics", []),
                            data.get("delivery_metrics", []),
                        )
                    else:
                        cm = CommMetrics()
            except Exception:
                cm = CommMetrics()
            results.append(cm)
        return results[0], results[1]

    def compute_pipeline(
        self, head: CommMetrics, tail: CommMetrics, e2e_ms: float,
        prompt_tokens: int, output_tokens: int,
    ) -> PipelineMetrics:
        pm = PipelineMetrics(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            head_compute_ms=head.head_compute_ms,
            tail_compute_ms=tail.tail_compute_ms,
            gpu_to_cpu_ms=head.gpu_to_cpu_ms,
            serialize_ms=head.serialize_ms,
            grpc_push_intermediate_ms=head.grpc_push_intermediate_ms,
            deserialize_ms=tail.deserialize_ms,
            grpc_push_sampler_ms=tail.grpc_push_sampler_ms,
            total_bytes=head.intermediate_bytes + tail.sampler_bytes,
            e2e_latency_ms=e2e_ms,
        )
        pm.total_comm_overhead_ms = (
            pm.gpu_to_cpu_ms + pm.serialize_ms
            + pm.grpc_push_intermediate_ms + pm.deserialize_ms
            + pm.grpc_push_sampler_ms
        )
        total = pm.total_comm_overhead_ms + pm.head_compute_ms + pm.tail_compute_ms
        pm.pipeline_overhead_pct = (
            pm.total_comm_overhead_ms / total * 100 if total > 0 else 0
        )
        max_stage = max(pm.head_compute_ms, pm.tail_compute_ms)
        min_stage = min(pm.head_compute_ms, pm.tail_compute_ms)
        pm.stage_balance_ratio = min_stage / max_stage if max_stage > 0 else 1.0
        grpc_time = pm.grpc_push_intermediate_ms + pm.grpc_push_sampler_ms
        pm.bandwidth_mb_per_s = (
            pm.total_bytes / 1024 / 1024 / (grpc_time / 1000) if grpc_time > 0 else 0
        )
        return pm


# ── System Monitor ─────────────────────────────────────────────────────────


class SystemMonitor:
    def __init__(self, gpus_to_monitor: list[int] | None = None):
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._metrics = SystemMetrics()
        self._gpus = gpus_to_monitor or [0, 1]
        self._nvml_initialized = False
        self._prev_net = None
        self._prev_time = 0.0
        self._start_time = 0.0

    def start(self):
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception:
                logger.warning("pynvml init failed")
        self._stop_event.clear()
        self._start_time = time.time()
        self._prev_net = psutil.net_io_counters()
        self._prev_time = time.time()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.info("System monitor started")

    def stop(self) -> SystemMetrics:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._metrics.duration_s = time.time() - self._start_time
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        logger.info("System monitor stopped (%.1fs, %d samples)",
                     self._metrics.duration_s, len(self._metrics.samples))
        return self._metrics

    def _sample_loop(self):
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(1.0)

    def _sample(self):
        now = time.time()
        pt = SamplePoint(relative_time_s=now - self._start_time)
        if self._nvml_initialized:
            for gid in self._gpus:
                try:
                    h = pynvml.nvmlDeviceGetHandleByIndex(gid)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    pt.gpu_utils[gid] = util.gpu
                    pt.gpu_mem_pct[gid] = mem.used / mem.total * 100
                    pt.gpu_mem_mb[gid] = mem.used / 1024 / 1024
                except Exception:
                    pass
        pt.cpu_pct = psutil.cpu_percent(interval=0)
        pt.mem_pct = psutil.virtual_memory().percent
        net = psutil.net_io_counters()
        dt = now - self._prev_time
        if self._prev_net is not None and dt > 0:
            pt.net_sent_bytes_per_s = (net.bytes_sent - self._prev_net.bytes_sent) / dt
            pt.net_recv_bytes_per_s = (net.bytes_recv - self._prev_net.bytes_recv) / dt
        self._prev_net = net
        self._prev_time = now
        self._metrics.samples.append(pt)


# ── Prompt Generator ───────────────────────────────────────────────────────


class PromptGenerator:
    """Generates diverse, random prompts using tokenizer vocabulary and templates."""

    TOPICS = {
        "science": ["molecule", "experiment", "hypothesis", "reaction", "element",
                     "quantum", "neutron", "protein", "genome", "catalyst",
                     "particle", "spectrum", "velocity", "entropy", "synthesis"],
        "technology": ["algorithm", "database", "network", "protocol", "interface",
                       "compiler", "framework", "middleware", "container", "pipeline",
                       "registry", "scheduler", "dispatcher", "encoder", "transformer"],
        "history": ["civilization", "dynasty", "revolution", "empire", "artifact",
                    "monument", "treaty", "colony", "parliament", "constitution",
                    "renaissance", "pharaoh", "legion", "medieval", "feudal"],
        "nature": ["ecosystem", "rainfall", "volcanic", "glacier", "coral",
                   "migration", "predator", "habitat", "tundra", "plateau",
                   "erosion", "canopy", "wetland", "reef", "savanna"],
        "society": ["democracy", "legislation", "education", "migration", "community",
                    "infrastructure", "institution", "regulation", "population", "economy",
                    "commerce", "governance", "tradition", "heritage", "innovation"],
    }

    SENTENCE_TEMPLATES = [
        "The {adj} {noun} {verb} {adv} during the {period}.",
        "{noun_c} {verb} a {adj} {noun2} in {location}.",
        "Research shows that {noun} can {verb} when {condition}.",
        "The {noun} of {location} {verb} {adj} {noun2} in {period}.",
        "{adj_c} {noun} {adv} {verb} the {noun2} of {noun3}.",
        "In {period}, {noun_c} began to {verb} {adj} {noun2}.",
        "The {noun} {verb} {adv} because the {noun2} {verb2} {adj}.",
        "Scientists discovered that {adj} {noun} {verb} under {condition}.",
        "The relationship between {noun} and {noun2} {verb} {adv}.",
        "{noun_c} from {location} {verb} {adj} {noun2} for {noun3}.",
        "Recent {noun} in {location} {verb} the {adj} {noun2}.",
        "The {adj} {noun} of {location} {verb} {noun2} through {noun3}.",
        "When {noun} {verb} {adv}, the {adj} {noun2} {verb2} to {noun3}.",
        "A {adj} {noun} was {verb2} by {noun_c} in {location}.",
        "The {noun} {verb} {adv} while {noun2} {verb2} the {noun3}.",
        "Through {adj} {noun}, {noun_c} {verb} {noun2} across {location}.",
        "The {noun} system {verb} {adj} {noun2} after {period}.",
        "Analysis of {noun} reveals that {adj} {noun2} {verb} {adv}.",
        "The development of {noun} {verb} {noun2} in {adj} ways during {period}.",
        "{adj_c} {noun} and {noun2} {verb} together to form {noun3}.",
    ]

    ADJECTIVES = [
        "significant", "complex", "ancient", "modern", "remarkable",
        "fundamental", "emerging", "traditional", "innovative", "critical",
        "subtle", "profound", "dynamic", "intricate", "unexpected",
        "extensive", "compelling", "progressive", "conventional", "prominent",
        "notable", "diverse", "extraordinary", "substantial", "elaborate",
    ]

    VERBS = [
        "transformed", "revealed", "established", "influenced", "demonstrated",
        "examined", "generated", "modified", "accelerated", "integrated",
        "expanded", "reduced", "enhanced", "disrupted", "synthesized",
        "explored", "challenged", "redefined", "illuminated", "facilitated",
        "contributed", "revolutionized", "validated", "deteriorated", "emerged",
    ]

    ADVERBS = [
        "rapidly", "gradually", "significantly", "unexpectedly",
        "systematically", "indirectly", "consistently", "partially",
        "substantially", "remarkably", "consequently", "simultaneously",
    ]

    PERIODS = [
        "the 1990s", "recent years", "the 21st century", "the early period",
        "the late century", "the modern era", "the post-war period",
        "the 19th century", "the digital age", "the medieval period",
    ]

    LOCATIONS = [
        "Europe", "Asia", "North America", "the region", "the area",
        "the Pacific", "the Mediterranean", "the continent", "the highlands",
        "the coastal zones", "the northern territories", "the southern basin",
    ]

    CONDITIONS = [
        "exposed to light", "under pressure", "heated", "isolated", "combined",
        "stimulated", "measured", "analyzed", "reproduced", "tested",
    ]

    def __init__(self, model_path: str):
        self._tokenizer = None
        self._word_pool: list[str] = []
        self._rng = random.Random(42)

        if TOKENIZER_AVAILABLE:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )
                vocab = self._tokenizer.get_vocab()
                words = [
                    k[1:] for k in vocab
                    if k.startswith('Ġ') and len(k) > 3
                    and k[1:].isalpha() and k[1:].islower()
                ]
                self._word_pool = words
                logger.info("Tokenizer loaded, word pool: %d words", len(words))
            except Exception as e:
                logger.warning("Failed to load tokenizer: %s", e)

        if not self._word_pool:
            self._word_pool = [w for ws in self.TOPICS.values() for w in ws]
            logger.info("Using fallback word pool: %d words", len(self._word_pool))

    def _fill_template(self) -> str:
        rng = self._rng
        template = rng.choice(self.SENTENCE_TEMPLATES)
        wp = self._word_pool
        return template.format(
            adj=rng.choice(self.ADJECTIVES),
            adj_c=rng.choice(self.ADJECTIVES).capitalize(),
            noun=rng.choice(wp),
            noun_c=rng.choice(wp).capitalize(),
            noun2=rng.choice(wp),
            noun3=rng.choice(wp),
            verb=rng.choice(self.VERBS),
            verb2=rng.choice(self.VERBS),
            adv=rng.choice(self.ADVERBS),
            period=rng.choice(self.PERIODS),
            location=rng.choice(self.LOCATIONS),
            condition=rng.choice(self.CONDITIONS),
        )

    def generate_prompt(self, num_tokens: int, seed: int | None = None) -> str:
        if seed is not None:
            self._rng = random.Random(seed)
        if self._tokenizer is not None:
            sentences = []
            current_tokens = 0
            while current_tokens < num_tokens:
                sentence = self._fill_template()
                tokens = self._tokenizer.encode(sentence)
                sentences.append(sentence)
                current_tokens += len(tokens)
            full_text = " ".join(sentences)
            full_tokens = self._tokenizer.encode(full_text)
            if len(full_tokens) > num_tokens:
                full_tokens = full_tokens[:num_tokens]
            return self._tokenizer.decode(full_tokens, skip_special_tokens=True)
        sentences = []
        while sum(len(s.split()) for s in sentences) < num_tokens:
            sentences.append(self._fill_template())
        text = " ".join(sentences)
        return text[:num_tokens * 4]

    def generate_diverse_prompts(self, num_tokens: int, count: int) -> list[str]:
        prompts = []
        for i in range(count):
            prompts.append(self.generate_prompt(num_tokens, seed=100 + i))
        return prompts

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        return len(text) // 4


# ── Service Manager ────────────────────────────────────────────────────────


class ServiceManager:
    def __init__(self, config: BenchmarkConfig):
        self._config = config
        self._head_proc: Optional[subprocess.Popen] = None
        self._tail_proc: Optional[subprocess.Popen] = None
        self._managed = False
        # Head: layers 0-21, Tail: layers 21-40
        self._head_end_layer = 21
        self._tail_start_layer = 21

    async def check_existing(self) -> bool:
        return await self._check_health(HEAD_URL) and await self._check_health(TAIL_URL)

    async def _check_health(self, url: str, timeout: int = 5) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{url}/health",
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def start_if_needed(self) -> bool:
        """Always kill residual processes and restart fresh for clean benchmarks."""
        if await self.check_existing():
            logger.info("Services detected running — killing for a clean start")
        else:
            logger.info("No running services detected — starting fresh")
        return await self.start_services()

    def _kill_port_processes(self):
        """Kill any processes listening on our ports (stale from previous crashes)."""
        for port in [self._config.head_port, self._config.tail_port,
                     self._config.head_grpc_port, self._config.tail_grpc_port]:
            try:
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True, text=True, timeout=5,
                )
                pids = result.stdout.strip().split()
                for pid in pids:
                    pid = pid.strip()
                    if pid and pid.isdigit():
                        logger.info("  Killing stale process pid=%s on port %d", pid, port)
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            except Exception:
                pass

    def _stop_managed_procs(self):
        """Stop processes we launched."""
        for name, proc in [("HEAD", self._head_proc), ("TAIL", self._tail_proc)]:
            if proc and proc.poll() is None:
                logger.info("  Stopping %s node (pid=%d)", name, proc.pid)
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        pass
                except Exception:
                    pass
        self._head_proc = None
        self._tail_proc = None

    async def start_services(self) -> bool:
        local_ip = extract_ip()
        logger.info("Starting services with local IP: %s (layers 0-%d / %d-40)",
                     local_ip, self._head_end_layer, self._tail_start_layer)

        # Clean up any stale processes
        self._stop_managed_procs()
        self._kill_port_processes()
        await asyncio.sleep(2)

        # Start head node
        head_cmd = [
            CONDA_PYTHON, "-m", "molinkv1.entrypoints.api_server",
            "--model", self._config.model_path,
            "--molink-enabled",
            "--molink-grpc-port", str(self._config.head_grpc_port),
            "--molink-start-layer", "0",
            "--molink-end-layer", str(self._head_end_layer),
            "--port", str(self._config.head_port),
            "--max-model-len", str(self._config.max_model_len),
        ]
        self._head_proc = subprocess.Popen(
            head_cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        self._managed = True
        if not await self._wait_for_healthy(HEAD_URL):
            logger.error("Head node failed to start")
            return False
        logger.info("Head node healthy (layers 0-%d)", self._head_end_layer)

        # Start tail node
        tail_cmd = [
            CONDA_PYTHON, "-m", "molinkv1.entrypoints.api_server",
            "--model", self._config.model_path,
            "--molink-enabled",
            "--molink-grpc-port", str(self._config.tail_grpc_port),
            "--molink-start-layer", str(self._tail_start_layer),
            "--molink-end-layer", "-1",
            "--port", str(self._config.tail_port),
            "--max-model-len", str(self._config.max_model_len),
            "--molink-initial-peer", f"{local_ip}:{self._config.head_grpc_port}",
        ]
        self._tail_proc = subprocess.Popen(
            tail_cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        if not await self._wait_for_healthy(TAIL_URL):
            logger.error("Tail node failed to start")
            return False
        logger.info("Tail node healthy (layers %d-40)", self._tail_start_layer)
        return True

    async def restart_after_oom(self) -> bool:
        """Restart both nodes after an OOM crash. Cleans up all stale processes first."""
        logger.warning("OOM detected - restarting both nodes...")
        self._stop_managed_procs()
        self._kill_port_processes()
        logger.info("Waiting 15s for GPU memory to be released...")
        await asyncio.sleep(15)
        return await self.start_services()

    async def _wait_for_healthy(self, url: str) -> bool:
        start = time.time()
        while time.time() - start < HEALTH_TIMEOUT:
            if await self._check_health(url):
                return True
            await asyncio.sleep(HEALTH_INTERVAL)
            logger.info("Waiting for %s ... (%ds)", url, int(time.time() - start))
        return False

    def stop_services(self):
        if not self._managed:
            return
        self._stop_managed_procs()
        logger.info("Services stopped")


# ── Request Functions ──────────────────────────────────────────────────────


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    request_id: int,
    prompt_tokens: int,
    prompt_gen: PromptGenerator | None = None,
) -> RequestMetrics:
    m = RequestMetrics(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        max_tokens=payload.get("max_tokens", 64),
        timestamp_start=datetime.now().isoformat(),
    )
    t_start = time.perf_counter()
    try:
        async with session.post(
            f"{url}/generate", json=payload,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
        ) as resp:
            t_end = time.perf_counter()
            m.e2e_latency_ms = (t_end - t_start) * 1000
            m.timestamp_end = datetime.now().isoformat()
            if resp.status != 200:
                body = await resp.text()
                m.error_message = f"HTTP {resp.status}: {body[:200]}"
                return m
            body = await resp.json()
            texts = body.get("text", [])
            if not texts:
                m.error_message = "Empty response"
                return m
            full_text = texts[0]
            prompt_text = payload.get("prompt", "")
            generated = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else full_text
            m.generated_text_length = len(generated)
            m.generated_tokens = prompt_gen.count_tokens(generated) if prompt_gen else max(1, len(generated) // 4)
            m.tokens_per_second = m.generated_tokens / (m.e2e_latency_ms / 1000) if m.e2e_latency_ms > 0 else 0
            m.success = True
    except asyncio.TimeoutError:
        m.e2e_latency_ms = (time.perf_counter() - t_start) * 1000
        m.timestamp_end = datetime.now().isoformat()
        m.error_message = "Timeout"
    except Exception as e:
        m.e2e_latency_ms = (time.perf_counter() - t_start) * 1000
        m.timestamp_end = datetime.now().isoformat()
        m.error_message = str(e)
    return m


async def send_streaming_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    request_id: int,
    prompt_tokens: int,
    prompt_gen: PromptGenerator | None = None,
) -> RequestMetrics:
    m = RequestMetrics(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        max_tokens=payload.get("max_tokens", 64),
        timestamp_start=datetime.now().isoformat(),
    )
    t_start = time.perf_counter()
    first_token = True
    accumulated_text = ""
    try:
        async with session.post(
            f"{url}/generate", json={**payload, "stream": True},
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                m.e2e_latency_ms = (time.perf_counter() - t_start) * 1000
                m.timestamp_end = datetime.now().isoformat()
                m.error_message = f"HTTP {resp.status}"
                return m
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    texts = chunk.get("text", [])
                    if texts:
                        accumulated_text = texts[0]  # latest full text
                except json.JSONDecodeError:
                    continue
                if first_token:
                    m.ttft_ms = (time.perf_counter() - t_start) * 1000
                    first_token = False
        t_end = time.perf_counter()
        m.e2e_latency_ms = (t_end - t_start) * 1000
        m.timestamp_end = datetime.now().isoformat()
        prompt_text = payload.get("prompt", "")
        generated = accumulated_text[len(prompt_text):] if accumulated_text.startswith(prompt_text) else accumulated_text
        m.generated_text_length = len(generated)
        m.generated_tokens = prompt_gen.count_tokens(generated) if prompt_gen else max(1, len(generated) // 4)
        m.tokens_per_second = m.generated_tokens / (m.e2e_latency_ms / 1000) if m.e2e_latency_ms > 0 else 0
        m.success = True
    except asyncio.TimeoutError:
        m.e2e_latency_ms = (time.perf_counter() - t_start) * 1000
        m.timestamp_end = datetime.now().isoformat()
        m.error_message = "Timeout"
    except Exception as e:
        m.e2e_latency_ms = (time.perf_counter() - t_start) * 1000
        m.timestamp_end = datetime.now().isoformat()
        m.error_message = str(e)
    return m


# ── Stats ──────────────────────────────────────────────────────────────────


def compute_stats(metrics: list[RequestMetrics]) -> dict:
    ok = [m for m in metrics if m.success]
    if not ok:
        return {}
    e2e = [m.e2e_latency_ms for m in ok]
    ttft = [m.ttft_ms for m in ok if m.ttft_ms > 0]
    tps = [m.tokens_per_second for m in ok]
    gen = [m.generated_tokens for m in ok]
    s: dict = {
        "successful": len(ok), "failed": len(metrics) - len(ok),
        "min_e2e_ms": float(np.min(e2e)),
        "avg_e2e_ms": float(np.mean(e2e)),
        "p50_e2e_ms": float(np.percentile(e2e, 50)),
        "p95_e2e_ms": float(np.percentile(e2e, 95)),
        "p99_e2e_ms": float(np.percentile(e2e, 99)),
        "std_e2e_ms": float(np.std(e2e)),
        "avg_tps": float(np.mean(tps)),
        "total_output_tokens": sum(gen),
    }
    if ttft:
        s["avg_ttft_ms"] = float(np.mean(ttft))
        s["p50_ttft_ms"] = float(np.percentile(ttft, 50))
        s["p95_ttft_ms"] = float(np.percentile(ttft, 95))
        s["p99_ttft_ms"] = float(np.percentile(ttft, 99))
    return s


def build_result(name: str, conc: int, pt: int, mt: int,
                 metrics: list[RequestMetrics], t_total: float) -> BenchmarkResult:
    s = compute_stats(metrics)
    ok_n = s.get("successful", 0)
    tot_gen = s.get("total_output_tokens", 0)
    return BenchmarkResult(
        test_name=name, concurrency=conc, prompt_tokens=pt, max_tokens=mt,
        num_requests=len(metrics), successful_requests=ok_n,
        failed_requests=len(metrics) - ok_n, total_time_s=t_total,
        requests_per_second=ok_n / t_total if t_total > 0 else 0,
        metrics=metrics,
        avg_ttft_ms=s.get("avg_ttft_ms", 0), p50_ttft_ms=s.get("p50_ttft_ms", 0),
        p95_ttft_ms=s.get("p95_ttft_ms", 0), p99_ttft_ms=s.get("p99_ttft_ms", 0),
        min_e2e_ms=s.get("min_e2e_ms", 0), avg_e2e_ms=s.get("avg_e2e_ms", 0),
        p50_e2e_ms=s.get("p50_e2e_ms", 0), p95_e2e_ms=s.get("p95_e2e_ms", 0),
        p99_e2e_ms=s.get("p99_e2e_ms", 0),
        avg_tokens_per_second=s.get("avg_tps", 0),
        total_output_tokens=tot_gen,
        output_tokens_per_second=tot_gen / t_total if t_total > 0 else 0,
    )


# ── Test Orchestration ─────────────────────────────────────────────────────


async def test_functional(session: aiohttp.ClientSession) -> bool:
    logger.info("=== Functional Test ===")
    payload = {"prompt": "San Francisco is a", "max_tokens": 50, "temperature": 0}
    try:
        async with session.post(f"{HEAD_URL}/generate", json=payload,
                                timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                logger.error("Functional test: HTTP %d", resp.status)
                return False
            body = await resp.json()
            texts = body.get("text", [])
            if not texts or len(texts[0]) <= len("San Francisco is a"):
                logger.error("Functional test: no generation")
                return False
            logger.info("Functional test PASSED: %s", texts[0][:120])
            return True
    except Exception as e:
        logger.error("Functional test error: %s", e)
        return False


async def test_streaming(session: aiohttp.ClientSession) -> bool:
    logger.info("=== Streaming Test ===")
    payload = {"prompt": "Hello, world!", "max_tokens": 30, "temperature": 0, "stream": True}
    try:
        chunks = 0
        async with session.post(f"{HEAD_URL}/generate", json=payload,
                                timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                return False
            async for raw in resp.content:
                if raw.decode("utf-8").strip():
                    chunks += 1
        logger.info("Streaming test PASSED (%d chunks)", chunks)
        return chunks > 0
    except Exception as e:
        logger.error("Streaming test error: %s", e)
        return False


async def warmup(session: aiohttp.ClientSession, n: int):
    logger.info("=== Warmup (%d) ===", n)
    payload = {"prompt": "The quick brown fox jumps over the lazy dog.", "max_tokens": 20, "temperature": 0}
    for i in range(n):
        try:
            async with session.post(f"{HEAD_URL}/generate", json=payload,
                                    timeout=aiohttp.ClientTimeout(total=60)) as resp:
                status = "OK" if resp.status == 200 else f"HTTP {resp.status}"
                logger.info("  Warmup %d/%d %s", i + 1, n, status)
        except Exception as e:
            logger.warning("  Warmup %d/%d error: %s", i + 1, n, e)
        await asyncio.sleep(0.5)


async def benchmark_output_scaling(
    session: aiohttp.ClientSession, config: BenchmarkConfig, pg: PromptGenerator,
) -> list[BenchmarkResult]:
    """Single-request, fixed prompt=128, varying output token sizes."""
    results = []
    prompt = pg.generate_prompt(128)
    for out_tok in config.output_token_sizes:
        name = f"output_scaling_{out_tok}"
        logger.info("=== Output Scaling (prompt=128, output=%d) ===", out_tok)
        payload = {"prompt": prompt, "max_tokens": out_tok, "temperature": 0}
        ms = []
        t0 = time.perf_counter()
        for i in range(config.baseline_iterations):
            m = await send_request(session, HEAD_URL, payload, i, 128, pg)
            ms.append(m)
            logger.info("  %d/%d: %s e2e=%.0fms gen=%dtok",
                        i + 1, config.baseline_iterations,
                        "OK" if m.success else m.error_message,
                        m.e2e_latency_ms, m.generated_tokens)
        r = build_result(name, 1, 128, out_tok, ms, time.perf_counter() - t0)
        results.append(r)
    return results


async def benchmark_prompt_scaling(
    session: aiohttp.ClientSession, config: BenchmarkConfig, pg: PromptGenerator,
) -> list[BenchmarkResult]:
    """Single-request, fixed output=64, varying prompt sizes."""
    results = []
    for pt in config.prompt_sizes_full:
        name = f"prompt_scaling_{pt}"
        logger.info("=== Prompt Scaling (prompt=%d, output=64) ===", pt)
        prompt = pg.generate_prompt(pt)
        payload = {"prompt": prompt, "max_tokens": 64, "temperature": 0}
        ms = []
        t0 = time.perf_counter()
        for i in range(config.baseline_iterations):
            m = await send_request(session, HEAD_URL, payload, i, pt, pg)
            ms.append(m)
            logger.info("  %d/%d: %s e2e=%.0fms",
                        i + 1, config.baseline_iterations,
                        "OK" if m.success else m.error_message, m.e2e_latency_ms)
        r = build_result(name, 1, pt, 64, ms, time.perf_counter() - t0)
        results.append(r)
    return results


async def benchmark_ttft(
    session: aiohttp.ClientSession, config: BenchmarkConfig, pg: PromptGenerator,
) -> list[BenchmarkResult]:
    """Streaming TTFT measurement, output=64, varying prompt sizes."""
    results = []
    for pt in config.prompt_sizes_full:
        name = f"ttft_{pt}"
        logger.info("=== TTFT (prompt=%d, output=64) ===", pt)
        prompt = pg.generate_prompt(pt)
        payload = {"prompt": prompt, "max_tokens": 64, "temperature": 0}
        ms = []
        t0 = time.perf_counter()
        for i in range(config.baseline_iterations):
            m = await send_streaming_request(session, HEAD_URL, payload, i, pt, pg)
            ms.append(m)
            logger.info("  %d/%d: %s ttft=%.1fms e2e=%.0fms",
                        i + 1, config.baseline_iterations,
                        "OK" if m.success else m.error_message,
                        m.ttft_ms, m.e2e_latency_ms)
        r = build_result(name, 1, pt, 64, ms, time.perf_counter() - t0)
        results.append(r)
    return results


async def benchmark_concurrent(
    session: aiohttp.ClientSession, config: BenchmarkConfig, pg: PromptGenerator,
    svc: "ServiceManager",
) -> list[BenchmarkResult]:
    """Concurrent benchmarks from config.concurrent_tests matrix.
    Detects OOM (all requests fail) and restarts services before retrying."""
    results = []
    total = len(config.concurrent_tests)
    for idx, (conc, pt, mt) in enumerate(config.concurrent_tests):
        name = f"concurrent_c{conc}_p{pt}_o{mt}"
        logger.info("=== Concurrent [%d/%d] c=%d p=%d o=%d ===", idx + 1, total, conc, pt, mt)

        # Check service health before each test
        if not await svc.check_existing():
            logger.warning("Service unhealthy before test, attempting restart...")
            if not await svc.restart_after_oom():
                logger.error("Failed to restart services, skipping remaining tests")
                break
            # Re-create session after restart
            session = aiohttp.ClientSession()

        prompt = pg.generate_prompt(pt)
        payload = {"prompt": prompt, "max_tokens": mt, "temperature": 0}
        t0 = time.perf_counter()
        tasks = [send_request(session, HEAD_URL, payload, i, pt, pg) for i in range(conc)]
        ms_list = await asyncio.gather(*tasks)
        t_total = time.perf_counter() - t0
        r = build_result(name, conc, pt, mt, list(ms_list), t_total)
        results.append(r)

        all_failed = r.successful_requests == 0 and r.num_requests > 0
        if all_failed:
            logger.warning("  ALL %d requests failed (likely OOM) - will restart services", r.num_requests)
            # Try restart
            if not await svc.restart_after_oom():
                logger.error("Failed to restart services after OOM, skipping remaining tests")
                break
            # Retry this test once after restart
            logger.info("  Retrying test after restart...")
            session = aiohttp.ClientSession()
            t0 = time.perf_counter()
            tasks = [send_request(session, HEAD_URL, payload, i, pt, pg) for i in range(conc)]
            ms_list = await asyncio.gather(*tasks)
            t_total = time.perf_counter() - t0
            r = build_result(name + "_retry", conc, pt, mt, list(ms_list), t_total)
            results.append(r)
            if r.successful_requests == 0:
                logger.error("  Retry also failed, skipping remaining output=%d tests", mt)
                break

        logger.info("  Done: %d/%d ok | RPS=%.2f | e2e avg=%.0fms p95=%.0fms | out=%.0ftok/s",
                    r.successful_requests, r.num_requests, r.requests_per_second,
                    r.avg_e2e_ms, r.p95_e2e_ms, r.output_tokens_per_second)
        await asyncio.sleep(1)
    return results


async def benchmark_communication(
    session: aiohttp.ClientSession, pg: PromptGenerator,
    comm_collector: CommMetricsCollector,
) -> list[tuple[BenchmarkResult, CommMetrics, CommMetrics]]:
    """Measure communication layer: single request per config, collect per-step metrics."""
    results = []
    prompt_sizes = [128, 512, 1024, 2048]
    output_sizes = [64, 512]

    for pt in prompt_sizes:
        for ot in output_sizes:
            name = f"comm_p{pt}_o{ot}"
            logger.info("=== Communication (prompt=%d, output=%d) ===", pt, ot)

            await comm_collector.reset(session)
            await asyncio.sleep(0.3)

            prompt = pg.generate_prompt(pt, seed=200 + pt + ot)
            payload = {"prompt": prompt, "max_tokens": ot, "temperature": 0}
            t0 = time.perf_counter()
            m = await send_request(session, HEAD_URL, payload, 0, pt, pg)
            t_total = time.perf_counter() - t0

            await asyncio.sleep(0.5)
            head_cm, tail_cm = await comm_collector.collect(session)

            r = build_result(name, 1, pt, ot, [m], t_total)
            results.append((r, head_cm, tail_cm))

            logger.info(
                "  %s | gpu2cpu=%.1fms ser=%.1fms grpc=%.1fms deser=%.1fms "
                "| head=%.1fms tail=%.1fms | bytes=%d",
                "OK" if m.success else m.error_message,
                head_cm.gpu_to_cpu_ms, head_cm.serialize_ms,
                head_cm.grpc_push_intermediate_ms, tail_cm.deserialize_ms,
                head_cm.head_compute_ms, tail_cm.tail_compute_ms,
                head_cm.intermediate_bytes,
            )
    return results


async def benchmark_pipeline_breakdown(
    session: aiohttp.ClientSession, config: BenchmarkConfig, pg: PromptGenerator,
    comm_collector: CommMetricsCollector,
) -> list[tuple[BenchmarkResult, PipelineMetrics]]:
    """Full pipeline breakdown: per-stage compute, comm overhead, balance."""
    results = []
    # Prompt scaling
    for pt in config.prompt_sizes_full:
        name = f"pipeline_p{pt}_o64"
        logger.info("=== Pipeline Breakdown (prompt=%d, output=64) ===", pt)

        await comm_collector.reset(session)
        await asyncio.sleep(0.3)

        prompt = pg.generate_prompt(pt, seed=300 + pt)
        payload = {"prompt": prompt, "max_tokens": 64, "temperature": 0}
        t0 = time.perf_counter()
        m = await send_request(session, HEAD_URL, payload, 0, pt, pg)
        t_total = time.perf_counter() - t0
        e2e_ms = m.e2e_latency_ms if m.success else (t_total * 1000)

        await asyncio.sleep(0.5)
        head_cm, tail_cm = await comm_collector.collect(session)
        pm = comm_collector.compute_pipeline(head_cm, tail_cm, e2e_ms, pt, 64)

        r = build_result(name, 1, pt, 64, [m], t_total)
        results.append((r, pm))

        logger.info(
            "  %s | head=%.1fms comm=%.1fms tail=%.1fms | overhead=%.1f%% balance=%.2f",
            "OK" if m.success else m.error_message,
            pm.head_compute_ms, pm.total_comm_overhead_ms, pm.tail_compute_ms,
            pm.pipeline_overhead_pct, pm.stage_balance_ratio,
        )

    # Output scaling
    for ot in config.output_token_sizes:
        name = f"pipeline_p128_o{ot}"
        logger.info("=== Pipeline Breakdown (prompt=128, output=%d) ===", ot)

        await comm_collector.reset(session)
        await asyncio.sleep(0.3)

        prompt = pg.generate_prompt(128, seed=400 + ot)
        payload = {"prompt": prompt, "max_tokens": ot, "temperature": 0}
        t0 = time.perf_counter()
        m = await send_request(session, HEAD_URL, payload, 0, 128, pg)
        t_total = time.perf_counter() - t0
        e2e_ms = m.e2e_latency_ms if m.success else (t_total * 1000)

        await asyncio.sleep(0.5)
        head_cm, tail_cm = await comm_collector.collect(session)
        pm = comm_collector.compute_pipeline(head_cm, tail_cm, e2e_ms, 128, ot)

        r = build_result(name, 1, 128, ot, [m], t_total)
        results.append((r, pm))

        logger.info(
            "  %s | head=%.1fms comm=%.1fms tail=%.1fms | overhead=%.1f%% bw=%.1fMB/s",
            "OK" if m.success else m.error_message,
            pm.head_compute_ms, pm.total_comm_overhead_ms, pm.tail_compute_ms,
            pm.pipeline_overhead_pct, pm.bandwidth_mb_per_s,
        )
    return results


# ── Report Generation ──────────────────────────────────────────────────────


def generate_report(results: list[BenchmarkResult], sys_m: SystemMetrics | None,
                    config: BenchmarkConfig,
                    comm_data: list[tuple[BenchmarkResult, CommMetrics, CommMetrics]] | None = None,
                    pipeline_data: list[tuple[BenchmarkResult, PipelineMetrics]] | None = None,
                    ) -> str:
    L: list[str] = []
    L.append("=" * 80)
    L.append("  MoLink v1 Benchmark Report")
    L.append("=" * 80)
    L.append(f"Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Model:     {config.model_path}")
    L.append(f"Pipeline:  GPU 0 (layers 0-21) -> GPU 1 (layers 21-40)")
    L.append(f"Max len:   {config.max_model_len}")
    L.append("")

    # Functional
    func = [r for r in results if r.test_name == "functional"]
    if func:
        L.append("--- Functional Tests ---")
        for r in func:
            L.append(f"  {r.test_name}: {'PASSED' if r.successful_requests > 0 else 'FAILED'}")
        L.append("")

    # Output scaling
    out_rows = [r for r in results if r.test_name.startswith("output_scaling_")]
    if out_rows:
        L.append("--- Output Token Scaling (single request, prompt=128) ---")
        L.append(f"  {'Output tok':>10} | {'Gen tok':>8} | {'Min(ms)':>8} | {'Avg E2E':>8} | "
                 f"{'P50':>8} | {'P95':>8} | {'P99':>8} | {'Std':>8} | {'tok/s':>8}")
        L.append("  " + "-" * 86)
        for r in out_rows:
            if r.successful_requests > 0:
                std = float(np.std([m.e2e_latency_ms for m in r.metrics if m.success]))
                L.append(
                    f"  {r.max_tokens:>10} | {r.total_output_tokens // r.successful_requests:>8} | "
                    f"{r.min_e2e_ms:>8.0f} | {r.avg_e2e_ms:>7.0f}ms | "
                    f"{r.p50_e2e_ms:>7.0f}ms | {r.p95_e2e_ms:>7.0f}ms | "
                    f"{r.p99_e2e_ms:>7.0f}ms | {std:>7.0f}ms | {r.avg_tokens_per_second:>8.1f}"
                )
            else:
                L.append(f"  {r.max_tokens:>10} | FAILED")
        L.append("")

    # Prompt scaling
    pr_rows = [r for r in results if r.test_name.startswith("prompt_scaling_")]
    if pr_rows:
        L.append("--- Prompt Length Scaling (single request, output=64) ---")
        L.append(f"  {'Prompt tok':>11} | {'Min(ms)':>8} | {'Avg E2E':>8} | "
                 f"{'P50':>8} | {'P95':>8} | {'P99':>8} | {'tok/s':>8}")
        L.append("  " + "-" * 72)
        for r in pr_rows:
            if r.successful_requests > 0:
                L.append(
                    f"  {r.prompt_tokens:>11} | {r.min_e2e_ms:>8.0f} | {r.avg_e2e_ms:>7.0f}ms | "
                    f"{r.p50_e2e_ms:>7.0f}ms | {r.p95_e2e_ms:>7.0f}ms | "
                    f"{r.p99_e2e_ms:>7.0f}ms | {r.avg_tokens_per_second:>8.1f}"
                )
            else:
                L.append(f"  {r.prompt_tokens:>11} | FAILED")
        L.append("")

    # TTFT
    ttft_rows = [r for r in results if r.test_name.startswith("ttft_")]
    if ttft_rows:
        L.append("--- Time To First Token (streaming, output=64) ---")
        L.append(f"  {'Prompt tok':>11} | {'Avg TTFT':>8} | {'P50':>8} | {'P95':>8} | {'E2E avg':>8}")
        L.append("  " + "-" * 52)
        for r in ttft_rows:
            if r.successful_requests > 0 and r.avg_ttft_ms > 0:
                L.append(
                    f"  {r.prompt_tokens:>11} | {r.avg_ttft_ms:>7.1f}ms | "
                    f"{r.p50_ttft_ms:>7.1f}ms | {r.p95_ttft_ms:>7.1f}ms | {r.avg_e2e_ms:>7.0f}ms"
                )
        L.append("")

    # Concurrent
    conc_rows = [r for r in results if r.test_name.startswith("concurrent_")]
    if conc_rows:
        L.append("--- Concurrent Throughput ---")
        L.append(f"  {'Conc':>4} | {'Prompt':>6} | {'Output':>6} | {'RPS':>7} | "
                 f"{'Out tok/s':>9} | {'Avg E2E':>8} | {'P95 E2E':>8} | {'P99 E2E':>8} | {'Ok/Fail':>7}")
        L.append("  " + "-" * 86)
        for r in conc_rows:
            if r.successful_requests > 0:
                L.append(
                    f"  {r.concurrency:>4} | {r.prompt_tokens:>6} | {r.max_tokens:>6} | "
                    f"{r.requests_per_second:>7.2f} | {r.output_tokens_per_second:>9.1f} | "
                    f"{r.avg_e2e_ms:>7.0f}ms | {r.p95_e2e_ms:>7.0f}ms | "
                    f"{r.p99_e2e_ms:>7.0f}ms | {r.successful_requests:>3}/{r.failed_requests:<3}"
                )
            else:
                L.append(
                    f"  {r.concurrency:>4} | {r.prompt_tokens:>6} | {r.max_tokens:>6} | "
                    f"FAILED ({r.failed_requests})"
                )
        L.append("")

    # Communication Layer
    if comm_data:
        L.append("--- Communication Layer Breakdown ---")
        L.append(f"  {'Config':>14} | {'GPU-CPU':>7} | {'Ser':>7} | {'gRPC Push':>9} | "
                 f"{'Deser':>7} | {'gRPC Out':>8} | {'Bytes':>8}")
        L.append("  " + "-" * 80)
        for r, head_cm, tail_cm in comm_data:
            if r.successful_requests > 0:
                cfg = f"p={r.prompt_tokens},o={r.max_tokens}"
                L.append(
                    f"  {cfg:>14} | {head_cm.gpu_to_cpu_ms:>6.1f}ms | "
                    f"{head_cm.serialize_ms:>6.1f}ms | "
                    f"{head_cm.grpc_push_intermediate_ms:>8.1f}ms | "
                    f"{tail_cm.deserialize_ms:>6.1f}ms | "
                    f"{tail_cm.grpc_push_sampler_ms:>7.1f}ms | "
                    f"{head_cm.intermediate_bytes / 1024 / 1024:>7.1f}MB"
                )
            else:
                cfg = f"p={r.prompt_tokens},o={r.max_tokens}"
                L.append(f"  {cfg:>14} | FAILED")
        L.append("")

    # Pipeline Breakdown
    if pipeline_data:
        L.append("--- Pipeline Breakdown ---")
        L.append(f"  {'Config':>14} | {'Head':>8} | {'Comm':>8} | {'Tail':>8} | "
                 f"{'Overhead':>8} | {'Balance':>7} | {'BW(MB/s)':>8}")
        L.append("  " + "-" * 80)
        for r, pm in pipeline_data:
            if r.successful_requests > 0:
                cfg = f"p={pm.prompt_tokens},o={pm.output_tokens}"
                L.append(
                    f"  {cfg:>14} | {pm.head_compute_ms:>7.0f}ms | "
                    f"{pm.total_comm_overhead_ms:>7.1f}ms | "
                    f"{pm.tail_compute_ms:>7.0f}ms | "
                    f"{pm.pipeline_overhead_pct:>7.1f}% | "
                    f"{pm.stage_balance_ratio:>7.2f} | "
                    f"{pm.bandwidth_mb_per_s:>8.1f}"
                )
            else:
                cfg = f"p={pm.prompt_tokens},o={pm.output_tokens}"
                L.append(f"  {cfg:>14} | FAILED")
        L.append("")

    # System
    if sys_m:
        L.append("--- System Metrics ---")
        gs = sys_m.gpu_summary()
        for gid, g in sorted(gs.items()):
            role = "Head (layers 0-21)" if gid == 0 else "Tail (layers 21-40)"
            L.append(f"  GPU {gid} ({role}): util avg={g['avg_util']:.1f}% max={g['max_util']:.1f}% "
                     f"| mem avg={g['avg_mem_pct']:.1f}% max={g['max_mem_pct']:.1f}%")
        cs = sys_m.cpu_summary()
        if cs:
            L.append(f"  CPU: avg={cs['avg']:.1f}% max={cs['max']:.1f}%")
        ns = sys_m.net_summary()
        if ns:
            L.append(f"  Network: avg_sent={ns['avg_sent_MB_s']:.2f} MB/s avg_recv={ns['avg_recv_MB_s']:.2f} MB/s")
        L.append("")

    L.append("=" * 80)
    return "\n".join(L)


# ── JSON Output ────────────────────────────────────────────────────────────


def save_json(results: list[BenchmarkResult], sys_m: SystemMetrics | None,
              config: BenchmarkConfig, timestamp: str,
              comm_data: list[tuple[BenchmarkResult, CommMetrics, CommMetrics]] | None = None,
              pipeline_data: list[tuple[BenchmarkResult, PipelineMetrics]] | None = None,
              ):
    out_file = TESTS_DIR / f"benchmark_results_{timestamp}.json"
    data: dict = {
        "timestamp": timestamp,
        "config": {
            "model_path": config.model_path,
            "head_port": config.head_port,
            "tail_port": config.tail_port,
            "max_model_len": config.max_model_len,
            "output_token_sizes": config.output_token_sizes,
            "prompt_sizes_full": config.prompt_sizes_full,
            "concurrent_tests": config.concurrent_tests,
        },
        "results": [],
        "raw_requests": [],
        "system_metrics": None,
    }

    # Aggregated results
    for r in results:
        data["results"].append({
            "test_name": r.test_name,
            "concurrency": r.concurrency,
            "prompt_tokens": r.prompt_tokens,
            "max_tokens": r.max_tokens,
            "num_requests": r.num_requests,
            "successful_requests": r.successful_requests,
            "failed_requests": r.failed_requests,
            "total_time_s": round(r.total_time_s, 3),
            "requests_per_second": round(r.requests_per_second, 4),
            "min_e2e_ms": round(r.min_e2e_ms, 2),
            "avg_e2e_ms": round(r.avg_e2e_ms, 2),
            "p50_e2e_ms": round(r.p50_e2e_ms, 2),
            "p95_e2e_ms": round(r.p95_e2e_ms, 2),
            "p99_e2e_ms": round(r.p99_e2e_ms, 2),
            "avg_ttft_ms": round(r.avg_ttft_ms, 2),
            "p50_ttft_ms": round(r.p50_ttft_ms, 2),
            "p95_ttft_ms": round(r.p95_ttft_ms, 2),
            "avg_tokens_per_second": round(r.avg_tokens_per_second, 2),
            "total_output_tokens": r.total_output_tokens,
            "output_tokens_per_second": round(r.output_tokens_per_second, 2),
        })

    # Raw per-request data
    for r in results:
        for m in r.metrics:
            data["raw_requests"].append({
                "test_name": r.test_name,
                "concurrency": r.concurrency,
                "prompt_tokens": m.prompt_tokens,
                "max_tokens": m.max_tokens,
                "request_id": m.request_id,
                "timestamp_start": m.timestamp_start,
                "timestamp_end": m.timestamp_end,
                "generated_tokens": m.generated_tokens,
                "generated_text_length": m.generated_text_length,
                "ttft_ms": round(m.ttft_ms, 3),
                "e2e_latency_ms": round(m.e2e_latency_ms, 3),
                "tokens_per_second": round(m.tokens_per_second, 3),
                "success": m.success,
                "error_message": m.error_message,
            })

    # Communication metrics
    if comm_data:
        data["communication_metrics"] = []
        for r, head_cm, tail_cm in comm_data:
            data["communication_metrics"].append({
                "test_name": r.test_name,
                "prompt_tokens": r.prompt_tokens,
                "max_tokens": r.max_tokens,
                "success": r.successful_requests > 0,
                "head": {
                    "gpu_to_cpu_ms": round(head_cm.gpu_to_cpu_ms, 3),
                    "serialize_ms": round(head_cm.serialize_ms, 3),
                    "grpc_push_intermediate_ms": round(head_cm.grpc_push_intermediate_ms, 3),
                    "head_compute_ms": round(head_cm.head_compute_ms, 3),
                    "intermediate_bytes": head_cm.intermediate_bytes,
                },
                "tail": {
                    "deserialize_ms": round(tail_cm.deserialize_ms, 3),
                    "grpc_push_sampler_ms": round(tail_cm.grpc_push_sampler_ms, 3),
                    "tail_compute_ms": round(tail_cm.tail_compute_ms, 3),
                    "sampler_bytes": tail_cm.sampler_bytes,
                },
            })

    # Pipeline metrics
    if pipeline_data:
        data["pipeline_metrics"] = []
        for r, pm in pipeline_data:
            data["pipeline_metrics"].append({
                "test_name": r.test_name,
                "prompt_tokens": pm.prompt_tokens,
                "output_tokens": pm.output_tokens,
                "success": r.successful_requests > 0,
                "head_compute_ms": round(pm.head_compute_ms, 3),
                "tail_compute_ms": round(pm.tail_compute_ms, 3),
                "total_comm_overhead_ms": round(pm.total_comm_overhead_ms, 3),
                "pipeline_overhead_pct": round(pm.pipeline_overhead_pct, 2),
                "stage_balance_ratio": round(pm.stage_balance_ratio, 3),
                "total_bytes": pm.total_bytes,
                "bandwidth_mb_per_s": round(pm.bandwidth_mb_per_s, 2),
                "e2e_latency_ms": round(pm.e2e_latency_ms, 2),
            })

    # Raw system time series
    if sys_m:
        data["system_metrics"] = {
            "duration_s": round(sys_m.duration_s, 2),
            "gpu_summary": sys_m.gpu_summary(),
            "cpu_summary": sys_m.cpu_summary(),
            "network_summary": sys_m.net_summary(),
            "raw_time_series": [
                {
                    "time_s": round(s.relative_time_s, 2),
                    "gpu_utils": s.gpu_utils,
                    "gpu_mem_pct": s.gpu_mem_pct,
                    "gpu_mem_mb": {k: round(v, 1) for k, v in s.gpu_mem_mb.items()},
                    "cpu_pct": s.cpu_pct,
                    "mem_pct": s.mem_pct,
                    "net_sent_bytes_per_s": round(s.net_sent_bytes_per_s, 1),
                    "net_recv_bytes_per_s": round(s.net_recv_bytes_per_s, 1),
                }
                for s in sys_m.samples
            ],
        }

    with open(out_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("JSON saved to %s", out_file)


# ── Chart Generation ───────────────────────────────────────────────────────


class ChartGenerator:
    def __init__(self, results: list[BenchmarkResult], sys_m: SystemMetrics | None,
                 config: BenchmarkConfig, timestamp: str,
                 comm_data: list[tuple[BenchmarkResult, CommMetrics, CommMetrics]] | None = None,
                 pipeline_data: list[tuple[BenchmarkResult, PipelineMetrics]] | None = None,
                 ):
        self._results = results
        self._sys = sys_m
        self._config = config
        self._ts = timestamp
        self._dir = TESTS_DIR
        self._comm_data = comm_data or []
        self._pipeline_data = pipeline_data or []
        self._colors_out = {64: "#1f77b4", 512: "#ff7f0e", 1024: "#2ca02c", 2048: "#d62728"}
        self._colors_gpu = {2: "#1f77b4", 3: "#ff7f0e"}

    def generate_all(self):
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, skipping charts")
            return
        plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
        self._plot_output_scaling()
        self._plot_prompt_scaling()
        self._plot_ttft()
        self._plot_concurrent_rps()
        self._plot_concurrent_latency()
        self._plot_concurrent_throughput()
        self._plot_gpu_utilization()
        self._plot_latency_heatmap()
        self._plot_pipeline_breakdown()
        self._plot_communication_scaling()
        self._plot_pipeline_overhead()
        logger.info("Charts saved to %s", self._dir)

    def _save(self, fig, name: str):
        path = self._dir / f"benchmark_{self._ts}_{name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Chart: %s", path.name)

    def _plot_output_scaling(self):
        rows = [r for r in self._results if r.test_name.startswith("output_scaling_") and r.successful_requests > 0]
        if not rows:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        out_toks = [r.max_tokens for r in rows]
        e2e = [r.avg_e2e_ms for r in rows]
        p95 = [r.p95_e2e_ms for r in rows]
        tps = [r.avg_tokens_per_second for r in rows]

        ax = axes[0]
        ax.plot(out_toks, e2e, "o-", color="#1f77b4", label="Avg E2E")
        ax.plot(out_toks, p95, "s--", color="#ff7f0e", label="P95 E2E")
        ax.set_xlabel("Output Tokens (max_tokens)")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("E2E Latency vs Output Tokens (prompt=128)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax = axes[1]
        ax.plot(out_toks, tps, "o-", color="#2ca02c")
        ax.set_xlabel("Output Tokens (max_tokens)")
        ax.set_ylabel("Decode Speed (tokens/s)")
        ax.set_title("Decode Speed vs Output Tokens (prompt=128)")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        fig.tight_layout()
        self._save(fig, "output_scaling")

    def _plot_prompt_scaling(self):
        rows = [r for r in self._results if r.test_name.startswith("prompt_scaling_") and r.successful_requests > 0]
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        pts = [r.prompt_tokens for r in rows]
        ax.plot(pts, [r.avg_e2e_ms for r in rows], "o-", label="Avg E2E")
        ax.plot(pts, [r.p95_e2e_ms for r in rows], "s--", label="P95 E2E")
        ax.plot(pts, [r.min_e2e_ms for r in rows], "^:", label="Min E2E")
        ax.set_xlabel("Prompt Tokens")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("E2E Latency vs Prompt Length (output=64)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        fig.tight_layout()
        self._save(fig, "prompt_scaling")

    def _plot_ttft(self):
        rows = [r for r in self._results if r.test_name.startswith("ttft_") and r.avg_ttft_ms > 0]
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        pts = [r.prompt_tokens for r in rows]
        ax.bar(range(len(pts)), [r.avg_ttft_ms for r in rows], color="#1f77b4", alpha=0.8, label="Avg TTFT")
        ax.set_xticks(range(len(pts)))
        ax.set_xticklabels([str(p) for p in pts])
        ax.set_xlabel("Prompt Tokens")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title("Time To First Token vs Prompt Length (output=64)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        self._save(fig, "ttft")

    def _plot_concurrent_rps(self):
        rows = [r for r in self._results if r.test_name.startswith("concurrent_") and r.successful_requests > 0]
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        for mt in sorted(set(r.max_tokens for r in rows)):
            subset = sorted([r for r in rows if r.max_tokens == mt], key=lambda r: r.concurrency)
            if not subset:
                continue
            ax.plot([r.concurrency for r in subset], [r.requests_per_second for r in subset],
                    "o-", color=self._colors_out.get(mt, "#333"), label=f"output={mt}")
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Requests / second")
        ax.set_title("Throughput (RPS) vs Concurrency (prompt=128)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        fig.tight_layout()
        self._save(fig, "concurrent_rps")

    def _plot_concurrent_latency(self):
        rows = [r for r in self._results if r.test_name.startswith("concurrent_") and r.successful_requests > 0]
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        for mt in sorted(set(r.max_tokens for r in rows)):
            subset = sorted([r for r in rows if r.max_tokens == mt], key=lambda r: r.concurrency)
            if not subset:
                continue
            ax.plot([r.concurrency for r in subset], [r.avg_e2e_ms for r in subset],
                    "o-", color=self._colors_out.get(mt, "#333"), label=f"Avg E2E (out={mt})")
            ax.plot([r.concurrency for r in subset], [r.p95_e2e_ms for r in subset],
                    "s--", color=self._colors_out.get(mt, "#333"), alpha=0.5, label=f"P95 (out={mt})")
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("E2E Latency vs Concurrency (prompt=128)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        fig.tight_layout()
        self._save(fig, "concurrent_latency")

    def _plot_concurrent_throughput(self):
        rows = [r for r in self._results if r.test_name.startswith("concurrent_") and r.successful_requests > 0]
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        for mt in sorted(set(r.max_tokens for r in rows)):
            subset = sorted([r for r in rows if r.max_tokens == mt], key=lambda r: r.concurrency)
            if not subset:
                continue
            ax.plot([r.concurrency for r in subset], [r.output_tokens_per_second for r in subset],
                    "o-", color=self._colors_out.get(mt, "#333"), label=f"output={mt}")
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Output tokens / second")
        ax.set_title("Output Throughput vs Concurrency (prompt=128)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        fig.tight_layout()
        self._save(fig, "concurrent_throughput")

    def _plot_gpu_utilization(self):
        if not self._sys or not self._sys.samples:
            return
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        times = [s.relative_time_s for s in self._sys.samples]

        # GPU utilization
        ax = axes[0]
        for gid in sorted(self._sys.samples[0].gpu_utils.keys()):
            utils = [s.gpu_utils.get(gid, 0) for s in self._sys.samples]
            ax.plot(times, utils, "-", color=self._colors_gpu.get(gid, "#333"),
                    alpha=0.7, label=f"GPU {gid} ({'Head' if gid == 0 else 'Tail'})")
        ax.set_ylabel("GPU Utilization (%)")
        ax.set_title("GPU Utilization Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # GPU memory
        ax = axes[1]
        for gid in sorted(self._sys.samples[0].gpu_mem_pct.keys()):
            mems = [s.gpu_mem_pct.get(gid, 0) for s in self._sys.samples]
            ax.plot(times, mems, "-", color=self._colors_gpu.get(gid, "#333"),
                    alpha=0.7, label=f"GPU {gid} ({'Head' if gid == 0 else 'Tail'})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("GPU Memory (%)")
        ax.set_title("GPU Memory Usage Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self._save(fig, "gpu_utilization")

    def _plot_latency_heatmap(self):
        rows = [r for r in self._results if r.test_name.startswith("concurrent_") and r.successful_requests > 0]
        if len(rows) < 4:
            return
        concs = sorted(set(r.concurrency for r in rows))
        outs = sorted(set(r.max_tokens for r in rows))
        if len(concs) < 2 or len(outs) < 2:
            return
        matrix = np.full((len(outs), len(concs)), np.nan)
        for r in rows:
            if r.prompt_tokens == 128:
                oi = outs.index(r.max_tokens) if r.max_tokens in outs else -1
                ci = concs.index(r.concurrency) if r.concurrency in concs else -1
                if oi >= 0 and ci >= 0:
                    matrix[oi, ci] = r.avg_e2e_ms

        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(concs)))
        ax.set_xticklabels([str(c) for c in concs])
        ax.set_yticks(range(len(outs)))
        ax.set_yticklabels([str(o) for o in outs])
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Output Tokens")
        ax.set_title("Avg E2E Latency Heatmap (prompt=128, ms)")
        for i in range(len(outs)):
            for j in range(len(concs)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                            color="white" if v > np.nanmax(matrix) * 0.6 else "black")
        fig.colorbar(im, ax=ax, label="ms")
        fig.tight_layout()
        self._save(fig, "latency_heatmap")

    def _plot_pipeline_breakdown(self):
        if not self._pipeline_data:
            return
        rows = [(r, pm) for r, pm in self._pipeline_data if r.successful_requests > 0]
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = [f"p={pm.prompt_tokens}\no={pm.output_tokens}" for _, pm in rows]
        x = np.arange(len(labels))
        width = 0.6

        head_vals = [pm.head_compute_ms for _, pm in rows]
        comm_vals = [pm.total_comm_overhead_ms for _, pm in rows]
        tail_vals = [pm.tail_compute_ms for _, pm in rows]

        ax.bar(x, head_vals, width, label="Head Compute", color="#1f77b4")
        ax.bar(x, comm_vals, width, bottom=head_vals, label="Comm Overhead", color="#ff7f0e")
        ax.bar(x, tail_vals, width,
               bottom=[h + c for h, c in zip(head_vals, comm_vals)],
               label="Tail Compute", color="#2ca02c")

        ax.set_xlabel("Configuration")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Pipeline Breakdown: Compute vs Communication Overhead")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        self._save(fig, "pipeline_breakdown")

    def _plot_communication_scaling(self):
        if not self._comm_data:
            return
        rows = [(r, h, t) for r, h, t in self._comm_data if r.successful_requests > 0]
        if len(rows) < 2:
            return
        fig, ax = plt.subplots(figsize=(9, 5))

        labels = [f"p={r.prompt_tokens},o={r.max_tokens}" for r, _, _ in rows]
        gpu_cpu = [h.gpu_to_cpu_ms for _, h, _ in rows]
        serialize = [h.serialize_ms for _, h, _ in rows]
        grpc_push = [h.grpc_push_intermediate_ms for _, h, _ in rows]
        deserialize = [t.deserialize_ms for _, _, t in rows]

        x = np.arange(len(labels))
        w = 0.2
        ax.bar(x - 1.5 * w, gpu_cpu, w, label="GPU->CPU", color="#1f77b4")
        ax.bar(x - 0.5 * w, serialize, w, label="Serialize", color="#ff7f0e")
        ax.bar(x + 0.5 * w, grpc_push, w, label="gRPC Push", color="#2ca02c")
        ax.bar(x + 1.5 * w, deserialize, w, label="Deserialize", color="#d62728")

        ax.set_xlabel("Configuration")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Communication Layer Breakdown by Configuration")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        self._save(fig, "communication_scaling")

    def _plot_pipeline_overhead(self):
        if not self._pipeline_data:
            return
        rows = [(r, pm) for r, pm in self._pipeline_data if r.successful_requests > 0]
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f"p={pm.prompt_tokens},o={pm.output_tokens}" for _, pm in rows]
        overheads = [pm.pipeline_overhead_pct for _, pm in rows]
        x = np.arange(len(labels))

        colors = ["#d62728" if o > 20 else "#ff7f0e" if o > 10 else "#2ca02c" for o in overheads]
        ax.bar(x, overheads, color=colors, alpha=0.8)
        ax.axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="10% threshold")

        for i, v in enumerate(overheads):
            ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

        ax.set_xlabel("Configuration")
        ax.set_ylabel("Communication Overhead (%)")
        ax.set_title("Pipeline Communication Overhead Percentage")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        self._save(fig, "pipeline_overhead")


# ── Main ────────────────────────────────────────────────────────────────────


async def main():
    config = BenchmarkConfig()
    all_results: list[BenchmarkResult] = []
    comm_data: list[tuple[BenchmarkResult, CommMetrics, CommMetrics]] = []
    pipeline_data: list[tuple[BenchmarkResult, PipelineMetrics]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("  MoLink v1 Benchmark  |  %s", timestamp)
    logger.info("  Model: %s", config.model_path)
    logger.info("  Pipeline: layers 0-21 (GPU 0) / layers 21-40 (GPU 1)")
    logger.info("  Output sizes: %s", config.output_token_sizes)
    logger.info("  Prompt sizes: %s", config.prompt_sizes_full)
    logger.info("  Concurrent tests: %d", len(config.concurrent_tests))
    logger.info("=" * 60)

    pg = PromptGenerator(config.model_path)
    svc = ServiceManager(config)
    comm_collector = CommMetricsCollector(HEAD_URL, TAIL_URL)
    session: Optional[aiohttp.ClientSession] = None
    try:
        if not await svc.start_if_needed():
            logger.error("Cannot start services. Abort.")
            return

        session = aiohttp.ClientSession()

        # Phase 1: Functional
        f_ok = await test_functional(session)
        s_ok = await test_streaming(session)
        all_results.append(BenchmarkResult(
            test_name="functional", concurrency=1, prompt_tokens=4, max_tokens=50,
            num_requests=2, successful_requests=int(f_ok) + int(s_ok),
            failed_requests=int(not f_ok) + int(not s_ok),
        ))
        if not f_ok:
            logger.error("Functional test failed. Abort.")
            return

        # Prompt diversity check
        diverse = pg.generate_diverse_prompts(128, 10)
        unique = len(set(diverse))
        logger.info("Prompt diversity: %d/10 unique prompts of 128 tokens", unique)
        if unique < 8:
            logger.warning("Low prompt diversity (%d/10 unique)!", unique)

        # Phase 2: Warmup
        await warmup(session, config.warmup_requests)

        # Phase 3: Output token scaling
        all_results.extend(await benchmark_output_scaling(session, config, pg))

        # Phase 4: Prompt length scaling
        all_results.extend(await benchmark_prompt_scaling(session, config, pg))

        # Phase 5: TTFT
        all_results.extend(await benchmark_ttft(session, config, pg))

        # Phase 6: Communication layer benchmark
        logger.info("=== Starting Communication Benchmarks ===")
        comm_data = await benchmark_communication(session, pg, comm_collector)
        all_results.extend([r for r, _, _ in comm_data])

        # Phase 7: Pipeline breakdown
        logger.info("=== Starting Pipeline Breakdown ===")
        pipeline_data = await benchmark_pipeline_breakdown(session, config, pg, comm_collector)
        all_results.extend([r for r, _ in pipeline_data])

        # Phase 8: Concurrent (with system monitoring)
        logger.info("=== Starting Concurrent Benchmarks ===")
        monitor = SystemMonitor(gpus_to_monitor=[0, 1])
        monitor.start()
        await asyncio.sleep(1)
        conc_results = await benchmark_concurrent(session, config, pg, svc)
        all_results.extend(conc_results)
        sys_m = monitor.stop()

        # Report
        report_text = generate_report(all_results, sys_m, config,
                                      comm_data, pipeline_data)
        print("\n" + report_text)
        save_json(all_results, sys_m, config, timestamp,
                  comm_data, pipeline_data)

        # Charts
        ChartGenerator(all_results, sys_m, config, timestamp,
                       comm_data, pipeline_data).generate_all()

    except Exception as e:
        logger.error("Benchmark failed: %s", e, exc_info=True)
    finally:
        if session and not session.closed:
            await session.close()
        svc.stop_services()

    logger.info("Benchmark complete. Total results: %d", len(all_results))


if __name__ == "__main__":
    asyncio.run(main())
