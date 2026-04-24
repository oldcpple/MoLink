# MoLink v1 自动化性能测试

## 前置条件

- conda 环境：`/opt/conda/envs/molink`
- 模型路径：`/gxq/Qwen3-14B`
- 2 张 GPU（4090 24GB），分别运行 head 和 tail 节点
- 层切分：GPU 0 负责 layers 0-21，GPU 1 负责 layers 21-40

## 运行

```bash
# 如果服务未启动，脚本会自动拉起（需要数分钟加载模型）
# 如果服务已运行，脚本会自动检测并跳过启动
/opt/conda/envs/molink/bin/python /home/MoLink/tests/test_molinkv1_benchmark.py
```

完整运行约 30 分钟（含 2048 token 输出的单请求测试）。

## 测试内容

| 阶段 | 说明 | 指标 |
|------|------|------|
| 功能测试 | 发送普通请求和流式请求，验证服务正常 | 通过/失败 |
| 输出扩展 | 单请求，prompt=128，output=[64, 512, 1024, 2048] | E2E 延迟、decode 速度 |
| Prompt 扩展 | 单请求，output=64，prompt=[32~4096] | E2E 延迟随 prompt 增长的变化 |
| TTFT 测量 | 流式请求，测量首个 token 返回时间 | TTFT vs prompt 长度 |
| 并发吞吐 | 并发=[1,5,10,20,50] x output=[64,512,1024] | RPS、tok/s、延迟分布 |

## 输出文件

每次运行生成以下文件（时间戳区分）：

```
tests/
├── benchmark_results_<时间戳>.json    # 完整原始数据
├── benchmark_<时间戳>_output_scaling.png
├── benchmark_<时间戳>_prompt_scaling.png
├── benchmark_<时间戳>_ttft.png
├── benchmark_<时间戳>_concurrent_rps.png
├── benchmark_<时间戳>_concurrent_latency.png
├── benchmark_<时间戳>_concurrent_throughput.png
├── benchmark_<时间戳>_gpu_utilization.png
└── benchmark_<时间戳>_latency_heatmap.png
```

### JSON 数据结构

```json
{
  "timestamp": "运行时间",
  "config": { "模型路径、端口、层切分等配置" },
  "results": [
    { "test_name": "测试名", "concurrency": 1, "avg_e2e_ms": 3054, "requests_per_second": 0.33, ... }
  ],
  "raw_requests": [
    { "test_name": "...", "request_id": 0, "timestamp_start": "...", "e2e_latency_ms": 3054, "generated_tokens": 64, "success": true, ... }
  ],
  "system_metrics": {
    "gpu_summary": { "0": {"avg_util": 28.9, "avg_mem_pct": 95.6}, "1": {...} },
    "raw_time_series": [ { "time_s": 0, "gpu_utils": {0: 35, 1: 40}, "cpu_pct": 15, ... } ]
  }
}
```

- `results`：每个测试的聚合统计（均值、P50/P95/P99、吞吐量）
- `raw_requests`：每个请求的原始数据（时间戳、延迟、生成 token 数、是否成功）
- `raw_time_series`：每秒采样的 GPU 利用率、显存、CPU、网络流量

## OOM 恢复机制

并发测试阶段，如果某批请求全部失败（OOM 导致服务崩溃），脚本会：

1. 杀掉所有残留进程，清理端口
2. 等待 15 秒释放 GPU 显存
3. 重新启动 head 节点 → 等待健康 → 启动 tail 节点
4. 重试该测试一次
5. 如果重试仍失败，跳过剩余同 output 大小的测试，继续下一个

## 图表说明

| 图表 | 内容 |
|------|------|
| output_scaling | 输出 token 数 vs E2E 延迟和 decode 速度 |
| prompt_scaling | Prompt 长度 vs E2E 延迟 |
| ttft | TTFT vs Prompt 长度（柱状图） |
| concurrent_rps | 并发数 vs 请求吞吐量（按 output 大小分组） |
| concurrent_latency | 并发数 vs E2E 延迟（Avg + P95） |
| concurrent_throughput | 并发数 vs 输出 token 吞吐量 |
| gpu_utilization | GPU 利用率和显存使用的时间序列 |
| latency_heatmap | 并发数 x 输出大小的延迟热力图 |
