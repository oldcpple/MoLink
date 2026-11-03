# 复现指南

一键配对测试（第 4 节）约需 **2 小时**（A3B：原生默认、原生 V2、我们各约 40 分钟；
`NATIVE_V2=0` 跳过原生 V2 时约 80 分钟），不含模型冷加载。所有命令都在 A 机执行；B 机只需装好环境，由脚本经 ssh 远程驱动。

## 0. 硬件与软件（我们的测试环境）

| | A（stage 0） | B（stage 1） |
|---|---|---|
| GPU | 2× RTX 4090 24GB（驱动 570.211.01） | 2× RTX 3090 24GB（驱动 570.195.03） |
| GPU 互联 | 无 P2P（全走 PYNCCL） | 无 P2P（经根复合体） |
| 机间链路 | 1 GbE 直连（约 940 Mbps），用 tc 限速到 100/250/500/1000 Mbps | 同左 |
| 系统 | Ubuntu 22.04.5，Python 3.12.13，torch 2.11.0+cu129，ray 2.59.0，pyzmq 27.2.0 | 同左 |

其他硬件也能跑。倍率依赖两台机器的算力比和链路带宽，第 4 节的数字只对上述环境成立。

## 1. 安装（两台机器都要做）

```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
git checkout 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665          # v0.23.0
git apply /path/to/hetvllm/patch/hetvllm-vllm-v0.23.0.patch
uv venv --python 3.12 && . .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install pyzmq aiohttp matplotlib
```

模型：`Qwen/Qwen3-30B-A3B`（BF16）；可选 `Qwen/Qwen3-32B`（§6.1）、`Qwen/Qwen3-8B`（§6.3、§7.4）。**两台机器上的绝对路径必须相同**
（ray 的 worker 按同一路径加载权重）。

另需：
- A → B 免密 ssh（密钥）。
- 两台机器都能执行 `tc`：A 以 root 运行，或设 `A_SUDO="sudo -n"`；B 设 `B_SUDO`。
- A 上放行端口 26000（API）、6399（ray）、27100–27120（pp_link）。

## 2. 配置

```bash
cd hetvllm/kit
cp cluster.env.example cluster.env      # 填 IP、网卡、GPU、venv、模型路径
set -a; . ./cluster.env; set +a
```

## 3. 冒烟测试（约 5 分钟，冷加载时更久）

```bash
python het.py up                         # ray：A 作 head，B 加入
python het.py start --best && python het.py wait
python het.py tc set 250                 # 必须打印 "verified"
python greedy.py --tag ours              # 写 $HET_RESULTS/ours_gen_fixed.json
python replay.py replay --trace dec_sat --rate 16 --dur 45 \
    --out $HET_RESULTS/runs/smoke.json   # 预期 done=16
python het.py stop && python het.py tc clear
```

`$HET_RESULTS/engine.log` 里应能看到 `hetvllm exec config: runner=V2
async_scheduling=True batch_queue_size=3 pp_size=2 tp_size=2`（`--best` 强制 Model Runner
V2；原生 A3B 默认是 `runner=V1 ... batch_queue_size=2`）。

## 4. 一键配对测试

```bash
bash bench.sh a3b 2>&1 | tee $HET_RESULTS/bench_a3b.log
# 32B：bash bench.sh d32（两边分区都是 32/32；稠密 Qwen3 原生默认就是 V2 执行器）
# 32B 参考：40 项 1.25×、≤250 Mbps 1.42×、100 Mbps 1.67×；过载实验两边都 0/155（超出硬件能力）
```

脚本依次跑原生（vLLM 默认执行器）、原生强制 V2（`--runner v2`，只对 A3B；`NATIVE_V2=0`
跳过）、ours（`--best`），每个系统依次：
1. 清场：确认进程数为 0、GPU 显存已释放。
2. 重建 ray，启动引擎，等健康检查（**如果 20 秒内就“健康”，判为僵尸引擎并中止**：
   旧进程可能经 `SO_REUSEPORT` 仍占着端口）。
3. tc 限速并核对生效。
4. 做 greedy 捕获。
5. 在 100/250/500/1000 Mbps 四档各跑 10 项：
   - 解码 M=1/4/16/32，外加 M=16 配 256 / 4K 输入；
   - prefill 48×2K/4K/8K；
   - 固定混合负载。
6. 在 250 Mbps 下跑 300 秒过载 SLO 实验。

最后打印对照表（`accept_tab.py`），并把图写到 `$HET_RESULTS/figs/`。

**我们的参考结果（A3B，`data/a3b_cells.csv`、`data/a3b_vs_native_v2_cells.csv`）与建议的
复现容差：**

| 指标 | 对原生默认（V1） | 对原生 V2 | 建议判定（对原生默认） |
|---|---|---|---|
| 100 Mbps 档 prefill 48×8K | 6.29× | 5.63× | ≥ 5.0× |
| 100 Mbps 档混合负载 | 4.45× | 4.10× | ≥ 3.5× |
| 100 Mbps 档 10 项几何均值 | 2.93× | 2.81× | ≥ 2.4× |
| ≤250 Mbps 20 项几何均值 | 2.43× | 2.10× | ≥ 2.0× |
| 40 项几何均值 | 1.91× | 1.54× | ≥ 1.6× |
| 1000 Mbps 档几何均值 | 1.42× | 1.08× | ≥ 1.2× |
| 过载 SLO 达标（ours / native） | 153 / 0 | 153 / 6 | ours ≥ 140 且原生远低于 ours |
| 跨机字节/token 比 | 0.14× | | ≤ 0.25× |

32B（`bash bench.sh d32`，`data/d32_cells.csv`）：40 项 1.25×、≤250 Mbps 1.42×、100 Mbps 1.67×；
过载实验两边都是 0/155（超出硬件能力），只比原始产出（1.54×）。

40 项配对每个模型只跑了一次。参考：关闭前缀缓存的 12 项（§4c）是另一时段的独立运行，
它与 40 项在共同测试项上的结论一致（README §2.5）。

## 4b. 量化精度评测（公开数据集，约 1.5 小时）

```bash
HF_ENDPOINT=https://huggingface.co bash get_datasets.sh   # 或换成镜像，如 https://hf-mirror.com
for cfg in "nat:" "m4:--best" "m8:--link --codec int8 --merge --net-bw 0 --runner v2 --tok-tail" \
           "m16:--link --codec bf16 --merge --net-bw 0 --runner v2 --tok-tail" "nat2:"; do
  tag=${cfg%%:*}; flags=${cfg#*:}
  python het.py stop; python het.py up; python het.py start $flags && python het.py wait
  python het.py tc clear                       # 精度与带宽无关，不限速
  python qeval.py --tag $tag                   # 每个配置约 12 分钟
done
python het.py stop
python qeval_cmp.py nat nat2 m16 m8 m4 --csv $HET_RESULTS/qeval/summary.csv
```

`nat2` 是第二次原生运行，用来测批处理顺序带来的不确定性（噪声基线）。参考结果见 README §8
（`data/a3b_quant_accuracy.csv`，我们的 tag 名加了 `q6` 前缀）。建议判定：
- WikiText-2 困惑度：各配置相对原生都在 ±0.3% 以内（我们：−0.15% 到 +0.01%）。
- merged bf16、merged int8：MMLU/ARC/GSM8K 的配对 McNemar p > 0.05，与原生的一致率不低于
  原生重跑减 0.5 个百分点。
- merged int4：MMLU 下降不超过 1.5 个百分点（我们 −0.67）；ARC/GSM8K 变化在 ±1 个百分点以内。

## 4c. 消融、空泡率（README §5、§6、§7，消融约 4 小时）

**消融**：每个配置单独启动引擎，关闭前缀缓存，跑 decode M=4、decode M=32、prefill 48×8K、
混合负载（各在 100/250/1000 Mbps），再跑 250 Mbps 过载 SLO 实验。驱动脚本 `kit/ablate.sh`
读一个“tag + `het.py start` 参数”的清单（脚本开头有 README §5 全部配置的示例），汇总用
`kit/ablate_tab.py`：

```bash
cd hetvllm/kit
cat > cfg.txt <<'X'
natv1
nat2      --runner v2
best      --best
X
bash ablate.sh cfg.txt ab
python ablate_tab.py --pfx ab --base natv1 --tags nat2,best
```

| 配置 | `het.py start` 参数（`ablate.sh` 自动加 `--no-prefix-cache`） |
|---|---|
| 原生默认（V1） | （无） |
| 原生，强制 V2 | `--runner v2` |
| 原生，V2 + tok-tail | `--runner v2 --tok-tail` |
| 原生 V2，去掉 MoE 调参 | `--runner v2`，并在两台机器上临时移走 `fused_moe/configs/E=128,N=384,*.json` |
| + pp_link（bf16，两个向量） | `--link --codec bf16 --net-bw 0 --runner v2 --tok-tail` |
| + int4（不合并） | `--link --codec int4 --net-bw 0 --runner v2 --tok-tail` |
| + 合并（bf16） | `--link --codec bf16 --merge --net-bw 0 --runner v2 --tok-tail` |
| 终版 `--best` | `--best` |
| 终版，同步链路（对照） | `--best --link-sync` |
| 终版 + TP-C | `--best-tpc` |
| PP4：原生 / 原生 V2 / 我们 | `--tp 1 --pp 4` / `--tp 1 --pp 4 --runner v2` / `--tp 1 --pp 4 --best`（`GPUS_A` 给两张卡） |
| Qwen3-8B，1+1 卡 | `GPUS_A` 只给一张卡、`B_NUM_GPUS=1`；`--model q8 --tp 1` 加上面的参数 |

用我们的原始数据复算 §5 的表：`python ablate_tab.py --runs ../data/round6 --base r6:natv1
--tags r6:nat2c,r6:LQM2t`（tag 对应关系见第 5 节）。TP-C dip 上限扫描（README §3.6）是在同一个
引擎上热切参数做的，用的是仓库内部脚本，没有放进 kit；原始结果在 `data/round6/dip_*`。

**空泡率**：`kit/bubble.sh TAG 带宽 [start 参数]`，每个窗口一个新引擎，profiler 只记 GPU kernel，
结束后自动收集两台机器的 trace 并用 `bubble.py --clip 2:32` 分析；按 NCCL 种类拆分用
`nccl_split.py`。脚本开头列出了 README §7.2（TP2×PP2）和 §7.4（PP2）的全部命令，例如：

```bash
bash bubble.sh native_250 250                       # TP2×PP2，原生默认
bash bubble.sh ours_250 250 --best-tpc              # TP2×PP2，我们（§7.2 测的是开 TP-C 的配置）
# PP2：GPUS_A 只给一张 4090，B_NUM_GPUS=1；负载 3 份（216 个请求）
MIX_X=3 bash bubble.sh q8_native_250 250 --model q8 --tp 1
MIX_X=3 bash bubble.sh q8_ours_250 250 --model q8 --tp 1 --best --partition 24,12
```

预期（PP2，250 Mbps，A / B）：原生默认约 89% / 81%；我们 24/12 两次测量分别为 41% / 39%（README §7.4）和 35% / 36%（用本 kit 的 `bubble.sh` 复测）。判定：两段都相对原生默认降低 ≥ 50%。
TP2×PP2，250 Mbps：原生默认约 94% / 90%，我们约 58% / 56%。

## 5. 只看数据、不跑集群

```bash
python hetvllm/make_figs.py                     # 由 data/ 重新生成 figs/ 与 data/*.csv
mkdir -p /tmp/acc/runs && cp hetvllm/data/raw/* /tmp/acc/runs/
HET_RESULTS=/tmp/acc python hetvllm/kit/accept_tab.py fin6_o fin6_n    # A3B 对原生默认
HET_RESULTS=/tmp/acc python hetvllm/kit/accept_tab.py fin6_o fin6_n2   # A3B 对原生 V2
HET_RESULTS=/tmp/acc python hetvllm/kit/accept_tab.py d9_o d9_n        # 32B
QEVAL_DIR=hetvllm/data/qeval python hetvllm/kit/qeval_cmp.py q6nat q6nat2 q6m16 q6m8 q6m4   # 精度
```

原始文件命名为 `{tag}_{cell}_{band}.json`。tag 含义：`fin6_o`/`fin6_n`/`fin6_n2` = A3B
我们/原生默认/原生 V2（2026-10-30 23:27 → 10-31 01:35）；`d9_o`/`d9_n` = 32B 我们/原生
（2026-10-31 09:40 → 13:40）。`data/round6/` 里是关缓存的 12 项：`r6_*`、`r7_*`（A3B
各配置）、`p4_*`（PP4）、`q8_*`（Qwen3-8B）、`dip_*`（dip 扫描）。每项对应的 `.reqs.json` 是
逐请求记录（TTFT、ITL、TPOT、token 数）。`done` 字段统计的是生成长度达到 out_len 的请求数：
在 1 Gbps 下，流式输出偶尔只收到 1023/1024 个 token，此时 `done=0`，但 `good_frac=1.0`，
不影响吞吐口径。

## 6. 常见问题

- **引擎秒级“健康”但请求很慢、结果不对**：旧引擎没杀干净，新旧进程同绑端口。
  先 `het.py stop`，确认 `pgrep -f EngineCore` 为空、GPU 显存已释放，再 `up` + `start`。
- **启动卡在 `Waiting for creating a placement group`**：ray 集群已被杀，但没有
  重建。先 `het.py up`。
- **`tc ... Change operation not supported`**：有流量时 `tc replace` 偶发失败，
  `het.py tc` 会自动重试并核对，只要最终打印 `verified` 即可。
- **冷加载很慢**：权重不在页缓存时，单个 shard 可能要 60–240 s。别在引擎加载
  期间另外读盘预热，两者会抢 IO。
- **p95 类时延对比**：必须同窗背靠背。邻居负载曾让同一系统的 p95 从 88 ms
  涨到 141 ms。
