# Fused teacher+student FlashAttention kernel — 设计文档

日期: 2026-07-28 · 背景: IPTTCDv9 inference prefill 在每个 TTT 层对同一组 Q/K/V 跑两次
flash attention(teacher 窗口 W、student 窗口 W/2)。实测 64K/w8192 下 student 那次
额外花 7.2ms/层 × 4 层 = 29ms(总 gap 的 ~40%)。本文描述把两次合成一个 kernel 的做法。

## 核心洞察

student 窗口 ⊆ teacher 窗口(recipe 规则 student = W/2 保证),因此对每个 query,
student 需要的所有 attention score 都是 teacher 那次扫描的子集。fuse = 在 teacher 的
一遍 K/V 扫描中顺便维护 student 的 online-softmax 状态,让 student 输出近乎免费。

## FA2 forward 结构回顾(单 head)

- Q 切行块(B_r=128),K/V 切列块(B_c);grid = (Q 块数, batch×heads)。
- 每 CTA 持有一个 Q 块,内循环扫窗口内 K/V 块,维护 (O, m, l):
  `S = Q@Kᵀ`;`m_new = max(m, rowmax(S))`;`O ← O·e^{m−m_new} + e^{S−m_new}·V`;`l` 同步更新。
- SWA:内循环范围裁到 [t−W+1, t],左边界块加列 mask。
- 结束:O /= l,写回。

## Fused kernel

```
每个 CTA(一个 Q 块, 一个 head):
  state_t = (O_t, m_t, l_t)    # teacher, 窗口 W_t
  state_s = (O_s, m_s, l_s)    # student, 窗口 W_s ⊆ W_t
  for kv_block in [q_end − W_t, q_end](旧→新):
      K, V = load(kv_block)              # HBM 读一次(原两次)
      S = Q @ Kᵀ                          # score 算一次(原两次)
      # teacher 更新(FA2 标准流程, 含 teacher 左边界 mask)
      P  = exp(S − m_t_new);  PV = P @ V
      O_t ← O_t·e^{m_t−m_t_new} + PV;  l_t, m_t 更新
      if kv_block ∩ student 窗口 ≠ ∅:
          # 行标量恒等式: P_s = P_t · e^{m_t_new − m_s_new}(逐行)
          # ⇒ P_s@V = diag(e^{m_t_new − m_s_new}) · PV —— 复用 PV
          if kv_block 完全在 student 窗口内:
              O_s ← O_s·e^{m_s−m_s_new} + rowscale(PV)
          else:  # student 左边界块(每 Q 块最多一个)
              对 P 加 student 列 mask 后单独做一次小 PV, 或重算该块
          l_s, m_s 更新(注意 m_s 用 S 在 student 列范围内的 rowmax)
  epilogue: O_t /= l_t; O_s /= l_s; 写两个输出
```

### 节省来源(全部来自重叠区 = student 全窗口)
1. K/V HBM 读取减半(student 范围不再读第二遍)——flash 的主要内存流量;
2. QKᵀ 不重复(attention FLOP 的一半);
3. PV 靠 rescale 恒等式复用(另一半 FLOP)。

### 额外代价
- 每 CTA 多一套 (O,m,l) 寄存器 → 占用率可能小降;
- student 左边界块的特殊处理(每 Q 块 1 个);
- epilogue 双写。

## 收益推算(实测数据)

64K, w8192: teacher 13.6ms/层 + student 7.2ms/层 → fused ≈ 13.6 + 1~3ms。
4 个 TTT 层共省 ~25ms:595 → ~570ms,gap +13.9% → ~+8%。
(Triton 实现基础性能比 CUTLASS 慢 10-20%,净收益估 ~15ms;CUDA fork 可拿全额。)

## 实现路线

| 路线 | 起点 | 预期 | 工作量 |
|---|---|---|---|
| Triton(推荐) | fla `ops/attn/parallel.py` 或 flash-attn Triton fwd;加双 accumulator、STUDENT_WINDOW constexpr、rescale 复用、双输出 | 省 ~15ms | 200-300 行, 2-4 天 |
| CUDA/CUTLASS | fork flash-attn `flash_fwd_kernel.h` | 省 ~25ms | 1 周+, 维护重 |
| FlexAttention | 不可行(单输出) | — | — |

先做 forward(inference prefill);training backward 需要 dS = dS_t + dS_s 在共享
dQ/dK/dV 上求和,结构相同工程翻倍,训练侧可继续用两个独立 kernel。

## 坑

1. **数值验收用 eval 指标**(NIAH/PPL),不能用 logit diff:Triton 与 CUTLASS 归约顺序
   不同不会逐 bit 一致,而本模型对 kernel 级扰动数值混沌(见 REPORT.md 的 fused_gateup
   教训:Δlogit≈2.5)。rescale 复用本身是精确代数。
2. **依赖 student ⊆ teacher**:W/2 规则和 fullattn 配置都满足;若出现 student > teacher
   的配置,内循环取并集,复用率降低但结构不变。
3. 接入点:`custom_models/ipttcd/modeling_ipttcdv9.py` 的 `_use_fast_prefill_path` 分支,
   把两次 `_fla_attn.flash_attn_func` 调用换成 `fused_dual_window_attn(q, k, v, W_t, W_s)`;
   训练分支(shared-QKV fast path)同理可选接入。
