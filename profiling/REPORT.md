# IPTTCD vs IPTTT prefill 性能调查报告

日期: 2026-07-28 · 硬件: H100 80GB (della pli-c) · 代码: 当前 working tree(含 4/21 的 TTT-at-inference fix)
环境: `.venv` (torch 2.9.1+cu128, transformers 4.57.6, flash-attn varlen)
复现: `sbatch profiling/run_bench_1gpu.sh` / `run_bench_round2.sh` / `run_bench_round3.sh`;汇总 `python profiling/summarize.py`

## TL;DR

**在 8×H100 节点单卡、batch=1、当前代码下,没有复现 "prefill 特别特别慢"。**
实测 IPTTCDv9 prefill 比 IPTTT 慢 **4.6%–13.5%**,比 SWA transformer 慢 ~15%,和理论 FLOP 增量(~10%)一致。两个候选假设都量化过了:

- 假设 1(两次 flash attention):真实存在,但只占 **+6.5%**(64K 时 ~36ms)。
  共享 QKV 把两次 attention fuse 起来的 patch 已写好并验证数值完全一致,只能挽回 ~1-3%。
- 假设 2(HF inference wrapper):**排除**。`model.generate()` 相对裸 forward 只多 1-2ms;
  attention_mask 引起的 unpad/varlen 路径两个模型都走,不构成差异。

## 实测数据(eval_mask_cache = generate() 的 prefill 路径,median ms)

### 64K CT 配置(swa8k, ttt_chunk=4096),Qwen3-0.6B 尺寸

| model | 4K | 16K | 64K | 128K | 256K |
|---|---|---|---|---|---|
| transformer swa8k | 36.4 | 126.6 | 547.0 | 1126.8 | 2276.9 |
| ipttt swa8k | 37.4 | 129.1 | 554.0 | 1154.8 | 2321.0 |
| **ipttcd v9 swa8k** | **44.5** | **147.4** | **629.0** | **1297.8** | **2622.7** |
| v9 消融: 只加 student attn | 39.8 | 135.5 | 585.2 | — | — |
| v9 消融: 关掉全部 TTT | 37.9 | 126.8 | 549.5 | — | — |
| v9 + 优化 patch(见下) | 42.8 | 144.6 | 632.9 | 1275.7 | 2579.7 |

**64K 分解**: +82ms 总开销 = student attention 路径(重复 QKV/RoPE/o_proj + 第二次 flash)+36ms,
TTT MLP scan(student gate/up、双 depthwise conv 18ms、ΔW matmul、cumsum、逐 chunk 输出)+44ms。
GPU 占用率 99%(trace 实测),没有 launch-bound 空转。

### fullattn 配置(teacher 全局 attention)

| model | 16K | 64K |
|---|---|---|
| transformer full | 135.3 | 1538.8 |
| ipttt fullattn | 136.0 | 1544.1 |
| ipttcd v9 fullattn | 154.7 | **1614.9 (+4.6% vs ipttt)** |

teacher attention 越贵,TTT 相对开销越小。

### 512K 配置(window 4096, ttt_chunk=1024 → 256K 时 T=256 个 chunk)

| model | 64K | 128K | 256K |
|---|---|---|---|
| v9 512k-cfg | 448.9 | 886.5 | 1770.7 |
| v9 512k-cfg + batched-scan patch | — | — | 1760.4 |

chunk=1024 的顺序 scan 循环也没有爆炸(GPU 仍然吃满);512K 本身单卡 OOM(transformer baseline 也 OOM,是 activation/unpad 内存问题,与 TTT 无关)。

### 训练模式 forward(对照 "training 只慢 15%")

64K: v9 593.1ms vs ipttt 526.0ms(**+12.8%**)——和 inference prefill 的 +13.5% 基本一样。
即:当前代码里 inference prefill 并没有比 training 相对更慢。

### 真实 checkpoint + 真实 NIAH 64K prompt(step-10000,走 eval_niah.py 同款加载/generate 路径)

| model | prefill (median) | generate(64) | decode/token |
|---|---|---|---|
| transformer swa8k | 516.0 ms | 3301.5 ms | 43.5 ms |
| ipttt swa8k | 527.0 ms | 3384.7 ms | 44.7 ms |
| **ipttcd v9 swa8k** | **599.6 ms (+13.8% vs ipttt)** | **3506.1 ms (+3.6%)** | 45.4 ms |

与合成 benchmark 完全吻合。另一个发现:**NIAH eval 的 wall time 其实被 decode 主导**
(64 token 就要 ~2.9s,128 token 是 prefill 的 ~10 倍),而 decode 两个模型一样慢
(45ms/token,HF eager 逐 token 循环的通病,与 TTT 无关)。如果要加速 eval,
优化 decode(如 CUDA graph / torch.compile decode step)收益远大于优化 prefill。

## 结构性事实(code path)

- 4/21 的 fix(working tree 未提交)给 `modeling_ipttcdv9.py` 加了 `elif self.is_ttt_layer:` inference 分支:
  prefill 时每个 TTT layer(4/28 层)跑 teacher attention(带 KV cache)+ 重新算 QKV/RoPE 的
  student chunk-local flash attention + scanfuse TTT scan。decode(q_len=1)自动跳过 student/TTT,
  和 transformer 一样快(实测 decode ~2.3ms/token,三个模型相同)。
- 理论 FLOP:64K prefill v9 比 transformer 多 ~10%(dup QKV/o 825GF + student attn 2.2TF +
  student gate/up 825GF + scan matmul 825GF + conv 4GF,×4 层)。实测 +15%,基本吻合。
- 5 月那批完整 eval job(已包含 TTT-at-inference)总时长 v9 vs ipttt 也只差 1.5-4%,与本次结论互证。

## 已写好的优化(profiling/patches.py,数值 bit-exact 验证过)

1. `patch_shared_qkv`: prefill 时 QKV/RoPE 只算一次,两次 flash_attn 共享(即同学说的 fuse 方案),
   KV cache 照常维护,decode 不受影响。
2. `patch_batched_scan`: 把 scanfuse 的逐 chunk Python 循环换成一次 cumsum + 一次 batched matmul。

两者合计只能挽回 **~1-3%**(64K: 629→633ms 噪声级;256K: 2623→2580ms;512k-cfg 256K: 1771→1760ms)。
`max|Δlogit| = 0.0`(与现路径完全等价)。结论:**不值得为速度重写 flash attention kernel**;
真正的两次 attention(teacher 窗口 8192 + student 窗口 4096)在数学上就是两个不同的 attention,
除非改模型定义(比如 student 复用 teacher 的 SWA 输出),否则 fuse 的上限就是省掉重复的 projection。

## 多卡复现实验(2026-07-28 下午,回应"多卡时慢 30-50%"的报告)

在 8×H100 整节点(job 11694681)、4×H100(11696767)、1×H100+CPU 饥饿(11697464)上,
把每种多卡形态都测了一遍(64K prefill,median ms):

| 场景 | tf | ipttt | v9 | v9+patch | v9/ipttt gap |
|---|---|---|---|---|---|
| H100 单卡 solo | 516 | 524 | 595 | — | **+13.5%** |
| 8 worker 并发,不绑核(共享 96 核,线程超订) | 517 | 526 | 597 | 590 | **+13.5%** |
| 8 worker 并发,绑核 12 核/worker(=8 个独立 sbatch job) | 519 | 527 | 598 | 591 | **+13.5%** |
| device_map=auto 切 8 卡 | 516 | 522 | 595 | — | +13.9% |
| 4 worker 并发 | 516 | 527 | 596 | 590 | +13.2% |
| 1 卡 + 16 spinner 打满 8 核 cgroup(极端 CPU 饥饿) | 581 | 591 | 667 | 660 | +12.9% |

**结论:多卡/并发/CPU 争抢都不会放大 gap——所有场景下 v9/ipttt 差距稳定在 12-14%,与单卡一致。**
"30-50%" 在当前代码 + H100 上无法复现。1:1 真实 eval 复刻(4 个并发 worker 跑真的
`eval_niah.py` + step-10000 checkpoint + 真实 NIAH 64K 数据 + 128 token 生成):
ipttt ~151s / v9 ~155s / tf ~148s per worker,v9 只慢 **2.6%**(wall time 被 decode 主导)。

### 512K 补测发现的真问题(与"多卡慢"无关但值得修)

- **v9 在 512K 单卡 prefill OOM**(80GB,expandable_segments + 无 mask 也不行);tf/ipttt 勉强放下(peak ~80.16GB)。
  v9 的额外 activation(student swiglu、双 conv、scan 状态)把它推过了线。要在 512K eval 就得做
  chunked TTT MLP / 及时释放中间量。
- **1M 时三个模型全挂**:tf/ipttt 死于 CUBLAS int32 溢出,v9 还多一个 cuDNN depthwise conv
  int32 索引溢出((b·T)·3072·(C+4) > 2³¹)。要上 1M 得分块做 conv/matmul。

## window / chunk 扫描:30-50% 的来源找到了(2026-07-28 晚)

图:`profiling/results/fig_sweep_ablation.png|pdf`(论文 Fig 3(b) 版式)、
`fig4_style_efficiency.png|pdf`(论文 Fig 4 版式)。

**teacher SWA window 扫描(64K prefill,chunk=4096)——gap ∝ 1/window:**

| window | tf | ipttt | v9 | v9 vs ipttt | v9 vs tf |
|---|---|---|---|---|---|
| 2048 | 242ms | 250ms | 321ms | **+28.2%** | **+32.6%** |
| 4096 | 339 | 347 | 415 | +19.3% | +22.4% |
| 8192 | 518 | 531 | 598 | +12.4% | +15.4% |
| 16384 | 851 | 861 | 930 | +8.1% | +9.3% |
| full | 1498 | 1503 | 1577 | +4.9% | +5.2% |

v9 的**绝对**额外开销是常数 ~66-74ms(student swiglu、双 conv、ΔW matmul、scan——全部 ∝ L·d,
与 window 无关);baseline 的 attention ∝ L·W。window 越小 baseline 越快,**相对** gap 就越大。
window=2048 时对 baseline 已是 +33%,window=1024 外推 ~+45%。
**"多卡/某些设置下慢 30-50%" 的最可能解释:那次测量用的是小 window(≤2048-4096)配置,
这是固有的比例关系,不是实现 bug**(GPU 占用率 99%,双 flash fuse 只能省 1-3%)。

**ttt_chunk 扫描(window=8192)——gap 随 chunk 增大而变大(因为 ipttt 变快):**

| chunk | ipttt | v9 | gap |
|---|---|---|---|
| 256 | 569ms | 599ms | +5.4% |
| 1024 | 538 | 586 | +8.9% |
| 4096 | 531 | 601 | +13.3% |
| 8192 | 531 | 625 | +17.7% |

v9 耗时对 chunk 几乎不敏感(586-625ms:student attention ∝ chunk 变贵与 scan 开销变少相抵);
ipttt 随 chunk 增大变快(chunk 一大 T 变小,einsum/cumsum 状态少)。chunk=256 时 T=256
个顺序 scan 迭代也没有把 v9 拖垮(+5.4%),再次证明 scan 循环不是瓶颈。

## 小 window 缓解实验 + 真实 recipe 对照(2026-07-28 深夜)

**缓解实验(64K)**:小 window 下把 chunk(=student 窗口)联动调小 + fuse patch:

| 配置 | w=2048 | w=4096 |
|---|---|---|
| tf / ipttt(c4096) | 236 / 245 | 335 / 343 |
| v9 c4096(默认耦合) | 319 (+30% vs ipttt) | 414 (+21%) |
| v9 c1024 + patch | **295 (+20%)** | **389 (+13%)** |

能挽回约 1/3 的 gap;剩余是 window 无关的 MLP 侧固定开销(双 conv ~18ms、student gate/up ~15ms、scan ~10ms),
要继续压需要 conv kernel 优化或减 TTT 层数。

**真实 recipe 事实(来自 xingyu-experiment-plan.md + 训练/eval 脚本)**:
- 训练三家完全对齐:books_65k、10K steps、seq 64K、lr 1e-4→1e-6、8×H100、batch 1/GPU;eval suite 逐字相同。
- 设计规则:**student 窗口 = teacher 窗口 / 2**(v1: 4096/1024×vis2=2048;v2: 8192/4096×vis1=4096;512K config 沿用 v1 几何)。
- canonical 对比 = window 对齐三元组 {SWA-CT, IPTTT, IPTTCDv9} @ w8k(即本报告的 +12-15% 主数字)+ fullattn 对。
- **v1 → v2 变更包含 "scanfuse always on"**:v1 时代的测速走的是非 scanfuse 的 base MLP 路径(fp32 matmul/cumsum),
  且 v1 window=4096 更小 → "30-50%" 的历史观测很可能来自 v1 代码 + v1 几何(见下节 v1-era 复现)。

## Recipe 对齐 profiling(student = teacher/2 规则,window × seqlen 全矩阵)

图:`profiling/results/fig_recipe_aligned.png|pdf`。teacher W ∈ {2k,4k,8k,16k}(student=W/2),
L ∈ {16K,64K,256K},两种参数化(v1 式 chunk1024×vis、v2 式 chunk=W/2×vis1)。

**IPTTCD vs IPTTT gap(v2 式;v1 式相差 ±1%,更新粒度不影响速度):**

| teacher W | L=16K | L=64K | L=256K |
|---|---|---|---|
| 2048 | +20.5% | +18.2% | +19.1% |
| 4096 | +18.0% | +16.5% | +16.6% |
| 8192(现行) | +16.1% | +13.0% | +13.1% |
| 16384 | +15.5% | +10.7% | +11.2% |

三个规律:
1. **gap 只由 window 驱动,与 seq length 基本无关**(同 W 下三个 L 的 gap 几乎重合;L=16K 略高是固定 per-call 开销);
2. 按 recipe 规则对齐后,小 window 的 gap 比不对齐时温和(W=2048:+18-20% vs 不对齐的 +28%),因为 student 窗口跟着 W 缩;
3. v1 式 / v2 式参数化速度无差——chunk 粒度是纯质量旋钮,速度上可自由选。

fuse patch 在此规则下稳定省 ~7-10ms(2-3%)。

## Speedup optimization(2026-07-28 深夜,已合入 modeling 代码)

在占住的 2×H100 上迭代了 5 类优化,结论:

| 优化 | 结果 | 处置 |
|---|---|---|
| eager 移位乘加替换 depthwise conv | 反而慢 8ms(内存流量>cuDNN) | ❌ 否决 |
| fused gate/up GEMM | 数学等价但改变 cublas 舍入 → 被 TTT 递归放大到 Δlogit=2.5 | ❌ 否决(模型对 kernel 级扰动数值敏感,ttt_lr=0.3 无归一化) |
| torch.compile TTT MLP | ~0 收益(inductor 融合不了大 GEMM 间的算子) | ❌ 否决 |
| batched scan(cumsum+bmm 代替逐 chunk 循环) | 组合中无净收益 | 保留为 patch 不合入 |
| **shared-QKV + student-attention side-stream 重叠** | **bit-exact,全窗口 -1 至 -3pp** | ✅ **已合入 `modeling_ipttcdv9.py`** |

合入后的 inference prefill 分支:QKV/RoPE 只算一次,teacher/student 两次 flash attention 共享,
student 尾部(flash+o_proj+norm)在 side CUDA stream 上与 teacher 重叠;有 padding/cu_seqlens 时
自动回退到原双路径;`IPTTCD_DISABLE_FAST_PREFILL=1` / `IPTTCD_DISABLE_PREFILL_STREAM=1` 可关。

**验证**:真实 step-10000 checkpoint 上 logits 逐 bit 一致(max|Δ|=0.0)、32-token greedy 完全相同。

**合入后的 gap(64K,recipe 对齐 student=W/2;图 `fig_recipe_aligned.*` 已更新,panel d 含前后对比)**:

| teacher W | 合入前 | 合入后 |
|---|---|---|
| 2048 | +18.2% | **+16.2%** |
| 4096 | +16.5% | **+15.8%** |
| 8192(现行) | +13.0% | **+11.9%**(真实 ckpt 实测 589.0ms vs ipttt 527ms = +11.8%) |
| 16384 | +10.7% | +11.7%(轻微回退:大窗口下 stream 同步开销略超收益,~8ms;如在意可按窗口大小 gate) |

追加:`patch_mlp_stream`(student 的 gate/up+swiglu+conv+beta 也搬上 side stream,与 teacher z 路径重叠)
再 -2.5ms,bit-exact,保留在 `profiling/patches.py` 未合入(收益小、代码重复多)。
安全优化累计:595.5 → 586.5ms。

### Fused dual-window attention kernel(Triton,已实现,patch 形式)

设计见 `FUSED_ATTENTION_DESIGN.md`,实现在 `fused_dual_attn.py`:一次 kernel 同时产出
teacher/student 两个输出——K/V 只读一次、QKᵀ 只算一次、重叠区的 P@V 用 online-softmax
行 rescale 恒等式复用。关键调参:BLOCK_M=64(双 accumulator 的寄存器压力下 128 会溢出)。

**Kernel 级**(B=1, T=64K, H=16/KV8, D=128,vs 两次 CUTLASS FA2):

| W_t/W_s | 双 CUTLASS | fused Triton |
|---|---|---|
| 8192/4096 | 21.05ms | **19.66ms** |
| 2048/1024 | 5.72ms | **5.44ms** |
| 16384/8192 | 40.50ms | **35.94ms** |

**端到端**(64K,w8192):583.3ms → gap **+11.5%**(累计 595.5→583.3);与 mlp_stream 叠加反而变差(不叠)。

### v3 kernel:数值对齐 CUTLASS + 性能大突破(2026-07-28 深夜)

按 CUTLASS FA2 逐步对齐数值结构:raw 域 row-max、`p=exp2(s·c−m·c)` 双乘形式、
**从对角线反向遍历 KV 块**(FA2 的顺序)、真 -inf mask + CUTLASS 同款 `-inf→0` guard、
倒数乘 epilogue;性能结构:对角块(causal mask)/共享内部块(零 mask)/student 边界/
teacher 内部/teacher 边界五段循环,大头循环零 mask 开销。

**数值**:vs CUTLASS **99.7-99.9% 逐 bit 相等,99.98% 在 1 ulp 内**,max absΔ≈1e-3
(≈1-2 个 bf16 ulp);5 组边界用例(非对齐 T、fullattn、B=2、无 GQA、T=4097)全过。
模型级 logit 漂移 0.043(fused_gateup 当年 2.5);真实 ckpt NIAH A/B 精度相同、11/12 预测逐字一致。

**Kernel 级**(64K):w8192 两次 CUTLASS 20.1ms → fused **11.3ms**(比单次 CUTLASS teacher
的 13.2ms 还快);w16384 37.7 → 22.5ms;w2048 5.7 → 3.6ms。

**端到端 gap(64K,recipe 对齐,vs 现行 ipttt 实现)**:

| teacher W | 优化前 | stream 合入后 | + fused v3 kernel |
|---|---|---|---|
| 2048 | +18.6% | +16.2% | **+11.3%** (286ms) |
| 4096 | +16.1% | +15.8% | **+8.7%** (379ms) |
| 8192(现行) | +13.9% | +11.9% | **+5.5%** (552ms) |
| 16384 | +10.7% | +11.7% | **+1.6%** (875ms) |

**公平性说明**:fused 的收益 = ① 去重(student 的 K/V 读取和 QKᵀ/PV 免费,v9 独享)
+ ② SWA kernel 本身比 CUTLASS 的 local-attention 路径快 ~15%(FA2 的 local 模式每块都做
mask,我们只在边界块做)。②是通用的——若给 ipttt/tf 的全部 28 层也换上单窗口版
Triton kernel,它们也会各提速 ~50-80ms,gap 会回到 ~+8-9%。真正 v9 专属的净收益是①(~25ms)。

**签核证据链(2026-07-29 凌晨)**:
1. **Kernel 级**:99.7-99.9% 逐 bit 相等,其余 ≤2 ulp;BLOCK_N=64 实证匹配 flash_attn 2.8.3 的分块序列(32/128 只有 ~88-90%)。
2. **分布级(10M token,153 篇 books@64K,真实 ckpt)**:KL p50=7.8e-4 / p99=4.1e-3 / max=1.72 nats;
   TV p50=1.25% / p99=3.8% / max=0.73;top-1 翻转率 1.77%;0.5M/5M/10M 三个规模 quantile 一致(收敛)。
   图:`fig_kl_dist.png|pdf`。校准:比 int8/fp8 量化的漂移小 10-100×,与 bf16-vs-fp32 或跨 GPU 架构同量级。
3. **任务级(4 NIAH 任务 × {32K,64K} × 25 样本)**:**16 格准确率逐格相同**(log: `eval_ab_extended.log`)。

**fused 版最终 efficiency**(图 `fig_efficiency_fused.png|pdf`,gap vs IPTTT):

| teacher W | L=16K | L=64K | L=256K |
|---|---|---|---|
| 2048 | +13.8% | +12.0% | +13.1% |
| 4096 | +11.0% | +9.6% | +10.4% |
| 8192(现行) | +6.8% | **+5.1%** | +5.8% |
| 16384 | +5.2% | +2.7% | +2.7% |

**状态与建议**:`--patch fused_attn` 可用。三级证据(bit 级/分布级/任务级)均通过,可以合入为
inference prefill 默认路径;如组内还想更保守,补一次完整 43-cell RULER+PPL 对分即可。
注意事项:逐字复现旧生成结果需关掉(`IPTTCD_DISABLE_FAST_PREFILL=1` 走 fallback 或不打 patch);
logprob 敏感用途(RL importance weight)单独验证。下一步可选:单窗口版 kernel 给全部层/全部模型
(通用提速 ~15%,同时保持对比公平)。

剩余 gap 基本是模型定义的必需计算(student attention ~29ms + student gate/up ~15ms + conv ~18ms,
均 ∝ L·d);再往下压需要自定义融合 kernel(有数值放大风险,需用 eval 指标而非 logit 对比验收)
或架构选择(减 TTT 层数、缩 student 窗口)。

## Batch size 扫描(2026-07-29 凌晨,图 `fig_bsz_efficiency.*`)

w8192 recipe,eval_nomask_cache,gap vs IPTTT:

| 配置 | v9(默认代码) | v9+fused |
|---|---|---|
| 64K × b∈{1,2,4} | +10.6% / +11.4% / +11.7% | +4.2% / +5.0% / +5.1% |
| 16K × b∈{1,4,8} | +13.5% / +10.9% / +12.8% | +6.6% / +5.1% / +6.0% |

结论:**gap 对 batch size 基本不变**(±1pp 抖动,无系统性放大);总吞吐随 batch 线性扩展
(tf 在 64K 时 b1/b2/b4 都是 ~128 K tok/s——B=1 时 GPU 已饱和,印证单卡 B=1 的测量代表性)。
64K 剩余 overhead 的 kernel 级归因(fused 版,profiler 实测):attention 侧 **−21.6ms(优势)**,
depthwise conv +15.3ms、scan elementwise +14.2ms、student gate/up GEMM +7.5ms。

## Triton depthwise conv + 最终组合:gap 归零(2026-07-29 凌晨)

`profiling/fused_conv.py`:因果 depthwise conv 直接在 (b·T, C, d) 布局上做(fp32 累加 5 tap),
消掉 cuDNN 路径的两次 rearrange 拷贝。**与 cuDNN 逐 bit 相同**(max|Δ|=0.000, bitwise-equal),
单次调用 3.84→0.33ms(11.6×),每 forward 8 次调用共省 ~27ms。模型级验证:
fused_attn+triton_conv 的 Δlogit 与仅 fused_attn 完全相同(4.30e-2)——conv 零额外漂移。

**最终组合(fused dual-window attention + Triton conv,`--patch fused_conv`)@64K:**

| teacher W | 优化前 gap | 最终 gap |
|---|---|---|
| 2048 | +18.6%(不对齐时 +28%) | **+1.4%**(260.5 vs 256.8ms) |
| 8192(现行) | +13.9% | **−0.8%**(526.1 vs 530.5ms)|
| 16384 | +10.7% | **−1.3%**(850.1 vs 861ms) |

batched_scan 在组合中仍无净收益(529.5 vs 526.1),不纳入。

**Attention 部分 batch 扫描**(64K, w8192/4096,kernel 级):所有实现随 B 严格线性;
fused 双输出比 FA2 单次 teacher 快 15-17%(B=1→8 优势稳定微增),比 FA2 两次调用快 ~45%:

| B | FA2 单次 | FA2 两次 | fused | LSE |
|---|---|---|---|---|
| 1 | 13.2 | 20.1 | 11.3 | 16.5 |
| 8 | 110.1 | 168.5 | 91.5 | 136.9 |

## FA3 对比与终版矩阵(2026-07-29 晨)

**终版 3 联图**(`fig_final_window_bsz.*`,FA2 环境,fused_conv 全开,同节点 36 cell):
IPTTCD(final) vs IPTTT 的 gap 全场 **∈ [−1.8%, +2.3%]**(b∈{1,2,4} × w∈{2k,4k,8k,16k})。

**FA3**(forward-only/bf16/hdim128/sm90 裁剪编译,`flash-attention/hopper`,build 脚本见
`fa3_build2.log` 头部 env;`flash_attn_interface.flash_attn_func` 返回 (out, lse) 元组,
接入 fla 需解包 wrapper——repo 预留的 `use_flash_attn_3` 开关直接用会挂):

attention-only @64K w8192/4096(ms):

| B | FA2 1x | FA2 2x | FA3 1x | FA3 2x | 我们的 fused |
|---|---|---|---|---|---|
| 1 | 13.2 | 20.3 | **5.65** | 9.91 | 11.32 |
| 8 | 110.7 | 169.3 | **51.7** | 78.8 | 91.8 |

**FA3 的 local 路径没有 FA2 的逐块 mask 低效,纯 kernel 工艺(TMA/warp-spec)碾压:
我们的 Triton fused 在 FA3 世界不再是最优**(FA3 两次调用 9.91 < fused 11.32)。
FA3 vs FA2 输出 diff 3.9e-3(bf16 级,换用需按惯例过 eval 签核)。

模型级(64K b1 w8192,FA3 装入全部层):

| 配置 | ms | gap vs ipttt-FA3 |
|---|---|---|
| tf + FA3 | 300.5 | — |
| ipttt + FA3 | 322.8 | — |
| **v9 + FA3 双调用 + Triton conv(FA3 世界最优)** | **335.0** | **+3.8%** |
| v9 + Triton fused TTT attn + FA3 其余层 | 350.3 | +8.5% |

**结论与建议**:① 全栈切 FA3 是给所有模型的免费 ~40% 提速(tf 513→300ms),paper 效率数字
可整体刷新;② FA3 世界下 v9 的 TTT 层用 FA3 双调用即可(gap +3.8%),我们的 Triton fused
只在 FA2 环境下是最优解;③ 要在 FA3 世界重新归零 gap,路线是 fork FA3 forward 加第二套
accumulator(设计同 FUSED_ATTENTION_DESIGN.md,est. ~6-7ms 双输出 → gap ~+1-2%),属后续工作。

**FA3 世界完整矩阵**(36 cell,图 `fig_fa3_window_bsz.*` + `fig_fa3_gap_lines.*`):
所有模型提速 40-70%(tf@w2k 达 360-371 K tok/s);IPTTCD(FA3 双调用 + Triton conv)vs
IPTTT 的 gap 稳定在 **+3.5% ~ +6.9%**(均值 ~+5.5%),对 window 和 batch 都基本平坦——
attention 变得足够便宜后,gap 的主导项换成了 MLP 侧的 window 无关计算(student gate/up
+ scan elementwise),1/W 规律因此弱化。

## FA3 baseline 下的进一步优化 + 3-case 终图(2026-07-29 晨)

**FA3-LSE fused attention**(`lse_dual_attn.py: lse_dual_window_attn_fa3_fast`):两次 FA3 调用
(近段 WS + 错位远段 WT−WS,FA3 原生返回 LSE)+ 单遍 Triton merge kernel。
kernel 级 @w8192/64K:**6.72ms vs FA3 两次调用 9.48ms(−29%)**,B=4 33.1 vs 37.6(−12%);
student 输出逐 bit = FA3。模型级收益被 far-band 的 contiguous 拷贝等杂项部分抵消,
实测与 FA3 双调用近似(336.9 vs 335.0ms @w8192/b1),gap 改善主要在 w16k(+0.5%)。

**3-case 终图**(`fig_g3_fa3_final.*`,FA3 全员 baseline,v9 = FA3-LSE fused + Triton conv):

| case | w2k | w4k | w8k | w16k |
|---|---|---|---|---|
| 64K, b=1 | +3.1% | +2.8% | +5.0% | **+0.5%** |
| 64K, b=4 | +4.8% | +4.8% | +3.2% | +2.0% |
| 256K, b=1 | +5.2% | +5.2% | +4.3% | +2.6% |

(本轮为双卡并发测量,±2% 噪声;tf 同轮也比单独测慢 ~2%。)
FA3 世界剩余 gap 的构成 = MLP 侧 window 无关计算(student gate/up + scan elementwise),
attention 已非瓶颈。附:fused kernel 的窄间隔(WT−WS < BLOCK_M+BLOCK_N)双计数隐患已修
(constexpr 分支,recipe 路径 PTX 不变,`test/test_fused_attn_narrow_gap.py` 覆盖);
fused patch 对 cached prefill(Tq≠Tk)加了 fallback 防御。

## 如果你们观测到的差距远大于 15%,下一步需要确认

1. 测的是哪个 GPU / 哪套环境(A100? 别的 venv/flash-attn 版本?)
2. 用什么命令/脚本测的(哪个 config、哪个 checkpoint、batch size、context 长度)
3. 是否 wall-clock 里混入了模型加载 / tokenize(64K prompt 的 CPU tokenize 要几秒)/ 数据生成
4. ipttt 那边的数字是否来自别的 harness(如 hf_TTT_qwen3 的 chunked-prefill 实现)
