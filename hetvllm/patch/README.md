# 补丁说明：hetvllm-vllm-v0.23.0.patch

- **基线**：上游 vLLM `v0.23.0`，commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`
  （https://github.com/vllm-project/vllm）。
- **补丁**：`hetvllm-vllm-v0.23.0.patch`（`git diff --binary`，9 个文件，+1991/−6，
  逐文件统计见 `DIFFSTAT.txt`）。在干净的上游 worktree 上 `git apply --check`
  与 `git apply` 均已验证通过，打完后源码与测试用 fork 逐文件一致。
- 所有改动默认关闭：不设 `HETVLLM_*` 环境变量时，行为与上游 v0.23.0 相同
  （包括基线 native 的跑法）。

## 应用与安装（A、B 两台机器都要装）

```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
git checkout 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
git apply --check /path/to/hetvllm/patch/hetvllm-vllm-v0.23.0.patch
git apply        /path/to/hetvllm/patch/hetvllm-vllm-v0.23.0.patch
uv venv --python 3.12 && . .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install pyzmq aiohttp matplotlib   # pyzmq：链路；后两者：回放与出图
```

补丁只改 Python 和 JSON 文件，所以预编译 wheel（`VLLM_USE_PRECOMPILED=1`）就够用。
测试环境：Python 3.12.13、torch 2.11.0+cu129、ray 2.59.0、pyzmq 27.2.0。

## 逐文件说明

| 文件 | 作用 | 对应技术 |
|---|---|---|
| `vllm/distributed/pp_link.py`（新增，913 行） | 跨机 PP 传输：用 ZMQ 替代 NCCL isend/irecv。消息按 32 KiB 分片；分 hi（解码）/lo（prefill）两档优先级，hi 可以插到在途 prefill 字节前面；每个方向有 ack 窗口做反压；元数据放在消息头里，省掉每步一次 gloo 对象往返；边界张量 int8/int4 分组对称量化；`HETVLLM_PP_MERGE=1` 时只传 s=h+r 一个向量（接收端伪造零残差，下一级融合 add+RMSNorm 精确算出 RMSNorm(s)）。收发全程异步（GPU 上拼好整块、一次非阻塞 D2H、发送线程等 CUDA event；步柄只等上一条消息交给 ZMQ，不等 ack；接收端零拷贝收帧、按 event 回收 pinned 缓冲）、启动时预分配 pinned 池、`HETVLLM_PP_LINK_FROM` 选择跨机那一跳（PP4）；接收端用模块级句柄（修掉每步一个局部类造成的显存泄漏）并分片反量化（README §3.3） | 链路、编解码、边界合并 |
| `vllm/distributed/parallel_state.py` | `isend/irecv_tensor_dict` 在链路开启时改走 pp_link（8 行） | 链路接入 |
| `vllm/v1/worker/gpu_worker.py` | 把调度器给的 hi/lo 类别传给链路（`set_step_hint`），其余是 PP 时间线 trace（读上一步 GPU 时间只查询、不阻塞）和 profiler 窗口（默认关） | 链路优先级 + 观测 |
| `vllm/v1/core/sched/scheduler.py` | TP-C 调度：`pred` 在线拟合完成时间 rt(rows)=base+slope·rows（衰减最小二乘），据此求 chunk 大小；有解码心跳（紧急时拆出 hi 微批先提交），平时合并大块，有防饿死、dip 硬帽、≤256 token 短提示词快车道；`auto`（v2）作热切换基线；TP-D 净感知 chunk 上限；`HETVLLM_SCHED_CTRL` 文件可在线改参数和 mode。`HETVLLM_TOK_TAIL=1` 时，对上一步没被调度的请求不再整份重发 prompt+输出 token 列表（V2 执行器不读，V1 只读输出尾部） | 预测式调度、控制面瘦身 |
| `vllm/v1/core/sched/output.py` | SchedulerOutput 增加 `pp_hi`、`het_t_sched`、`het_rows` 三个字段 | 调度 → 链路/RTT 反馈 |
| `vllm/model_executor/layers/fused_moe/configs/E=128,N=384,…RTX_3090.json`、`…RTX_4090.json` | 两种卡上针对 Qwen3-30B-A3B TP=2 形状调好的 fused-MoE 内核配置（内核级：4090 小 batch 2.3–3.8×、3090 大 batch 1.16–1.37×）。vLLM 按 GPU 型号自动加载，原生基线也会用到，所以不贡献我们对原生的倍率 | MoE 调参 |
| `vllm/entrypoints/openai/completion/serving.py` | 客户端断开后立即 abort 引擎请求（上游 v0.23.0 缺这一步） | SLO 回放正确性 |
| `vllm/v1/engine/core.py` | 启动时打一行实际执行配置（runner/async/batch_queue） | 观测 |

## 环境变量参考（不设 = 上游行为）

| 变量 | 默认 | 终版取值 | 含义 |
|---|---|---|---|
| `HETVLLM_PP_LINK` | 0 | 1 | 启用 pp_link |
| `HETVLLM_PP_LINK_BIND` / `_PEER` | 必填 | A 的 IP / B 的 IP | 链路两端地址（开启链路时必须设置） |
| `HETVLLM_PP_LINK_PORT` | 27100 | — | 起始端口（每条 lane/方向/优先级占一个） |
| `HETVLLM_PP_LINK_CODEC` | bf16 | int4 | 边界编解码：bf16 / int8（质量档）/ int4（吞吐档） |
| `HETVLLM_PP_LINK_INT4_GROUP` | 32 | — | int4 分组通道数 |
| `HETVLLM_PP_LINK_SLICE_KIB` | 32 | 32 | 分片大小（256 KiB 实测更差） |
| `HETVLLM_PP_LINK_WIN_MB` | 8 | 8 | 每方向未确认字节窗口 |
| `HETVLLM_PP_LINK_LANE_MBPS` | 0 | 0 | 每条 lane 限速（0 = 不限） |
| `HETVLLM_PP_MERGE` | 0 | 1 | 边界只传 s=h+r |
| `HETVLLM_PP_LINK_ASYNC` | 1 | 1 | 链路异步收发；0 = 同步路径（对照用，README §5） |
| `HETVLLM_PP_LINK_POOL` | 8 | 8 | 每方向预分配的 pinned 缓冲个数（发送端 ×2） |
| `HETVLLM_PP_LINK_FROM` | 0 | PP2: 0；PP4: 1 | 跨机那一跳的发送级（A 上最后一级） |
| `HETVLLM_TOK_TAIL` | 0 | 1 | 调度器只发 runner 真正读的 token（控制面字节） |
| `VLLM_USE_V2_MODEL_RUNNER`（上游变量） | MoE 默认 V1 | 1 | 强制 Model Runner V2（`--runner v2`）；原生基线也单独测了这一项 |
| `HETVLLM_TPC` | 未设 | 默认不开；`--best-tpc` 为 pred | 调度模式 auto / pred（可选，见 README §3.6） |
| `HETVLLM_TPC_GATE_MS` | 95 | A3B 95；32B 400 | TP-C 每 token 预算（32B 光解码一步就约 100 ms，必须放宽） |
| 其余 `HETVLLM_TPC_*`：`_HI_ROWS` 256、`_SPLIT` 1、`_FASTLANE` 1、`_DIP_MS` 180、`_MIN_ROWS` 64、`_LO_STARVE_MS` 400、`_HI_T_MS` −60、`_ALPHA` 0.6、`_AROW_MS` 0.34、`_BROW_MS` 0.11、`_ROW_MS`（空 = 由带宽估计算）、`_EPS_MS` 0、`_STEPS_PER_TOK` 2 | 如左 | 默认 | TP-C 细项：hi 微批行数上限、紧急 decode 单独成批、短提示快车道、dip 硬帽、最小块、lo 饥饿上限、在线完成时间模型的初值与平滑系数（scheduler.py） |
| `HETVLLM_NET_BW_MBPS` / `_BYTES_PER_TOK` / `_TGT_MS` | 0 / 8192 / 150 | `--best`：0（关）；`--best-tpc`：26 / 1152 / 80 | 带宽和每 token 字节的估计值：pred 模式只用来算每行静态成本 row_ms（进而算 dip 硬帽）；非 pred 模式用于 TP-D chunk 上限 |
| `HETVLLM_SCHED_CTRL` | 未设 | 文件路径 | 在线可调的 JSON 控制文件 |
| `HETVLLM_PP_TRACE`、`HETVLLM_LINK_STATS`、`HETVLLM_PROF_STEPS` | 关 | — | 观测工具 |
| `HETVLLM_PROF_STEPS=ta:b` + `HETVLLM_PROF_CUDA_ONLY=1` | 关 | — | 空泡率测量：首个有效步后第 a–b 秒，只抓 GPU kernel；开窗前若空闲超过 5 秒则重新计时（README §7） |

`kit/het.py start --best` 会把上表“终版取值”一次设齐（`--best-tpc` 再加 TP-C）。
