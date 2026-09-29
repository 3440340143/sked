# Apex-CED-1.58

**一个关于「1.58-bit 三值权重 + 全局共享 KV + 滑动窗口注意力」组合的设计探索**

> **Status: 设计规范 + 未验证的参考实现。未经训练，无任何性能实测数据。**
> 本文所有性能与显存数字均为**解析估算**或**设计目标**，已逐项标注。参考实现可运行、掩码逻辑有测试覆盖，但这**不代表模型有效**。

---

## English abstract

A design exploration that combines three existing techniques into one inference-oriented
architecture: (1) 1.58-bit ternary weights, (2) a causal encoder–decoder (CED) layout where a
shallow causal encoder produces a *single* global KV projection shared by all decoder layers,
and (3) sliding-window local attention so local KV stays O(window). The goal is to reduce the
static weight footprint and the long-context KV footprint simultaneously, targeting
memory-constrained inference.

Nothing here has been trained. All figures are analytical estimates or design targets. The
reference implementation is a runnable prototype with mask-correctness tests, not a validated model.

---

## 1. 这个方案想解决什么

长上下文推理同时撞上两堵墙：

1. **静态权重显存**：MoE 的专家池即使每次只激活一小部分，全部专家权重仍必须常驻显存。
2. **动态 KV 显存**：标准 Transformer 每层各存一份 KV，KV 占用随层数与序列长度线性增长，128k 上下文下成为主要开销。

本方案尝试用三个正交手段同时压缩这两项，代价是引入若干尚未验证的假设（见 §7）。

## 2. 三个组成部分

**都不是新发明。** 本方案的贡献主张仅在于**三者的具体组合与参数化**，以及把它落到一份可运行、掩码正确的参考实现上。

| 组件 | 作用 | 来源 |
| :--- | :--- | :--- |
| 1.58-bit 三值权重 | 静态权重压缩约 8×（相对 BF16） | BitNet b1.58 |
| CED 全局共享 KV | KV 从「每层一份」降为「全模型一份」 | Causal Encoder–Decoder 类设计（YOCO / CED 路线） |
| 滑动窗口局部注意力 | 本地 KV 恒定 O(W)，不随序列增长 | Longformer / Mistral / Gemma 2 |

## 3. 架构

```
输入 token 序列
      │
      ▼
┌─────────────────────────────────────────────────────┐
│ 编码器：16 层，Dense SwiGLU + 滑动窗口因果自注意力    │
│ （窗口 W=512，FFN 隐层 5632，全部 1.58-bit）         │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │ 全局 KV 单次投影（全模型仅此一份）   │
        │ global_k_proj / global_v_proj      │
        └────────────────┬───────────────────┘
                         │  共享（跨注意力）
                         ▼
┌─────────────────────────────────────────────────────┐
│ 解码器：16 层                                        │
│   1. 局部滑动窗口因果自注意力                        │
│   2. 跨注意力 → 读取共享全局 KV（带因果掩码）        │
│   3. 三值专家池 MoE（32 专家 / Top-4，隐层 2816）    │
│   4. 输出 Logit Soft-Capping（tanh，cap=30）         │
└────────────────────────┬────────────────────────────┘
                         ▼
                   Next-token logits
```

**关键设计点：** 解码器的跨注意力必须施加因果掩码。编码器本身是因果的，`global_k[j]` 已聚合了
`0..j` 的信息；若解码器第 `i` 位可点乘 `j > i`，就会读到它正要预测的未来 token，训练会退化为
复制任务。这是本类架构最容易出错的地方，参考实现中已显式处理并有测试覆盖。

## 4. 参考实现

```
apex_ced_core.py    # 模型与算子
smoke_test.py       # 冒烟测试（10 项）
```

实现中刻意处理好的两处细节：

- **STE 前向恒为量化值**：`w_ste = w_quant.detach() + mask * (w_scaled - w_scaled.detach())`。
  常见的简写 `w + (w_quant - w).detach()` 会在前向残留浮点项，使前向并非真正三值。
- **跨注意力掩码用绝对索引**：`q_idx = arange(kv_len - q_len, kv_len)`，因此在 prefill（`q_len=L`）
  与增量解码（`q_len=1`）两种状态下都正确。

### 测试结果

```
$ python smoke_test.py
  PASS  bitlinear forward is exactly ternary
  PASS  bitlinear gradients flow
  PASS  forward/backward shapes + loss
  PASS  no NaN/Inf in logits
  PASS  causality: no future leak
  PASS  causality: holds beyond window
  PASS  incremental decode == full forward
  PASS  decode respects window truncation
  PASS  moe balancing bias updates
  PASS  parameter accounting matches spec
all tests passed
```

测试覆盖的是**内部一致性**（掩码是否因果、增量解码是否等价于整段前向、量化是否真三值），
**不涉及任何模型质量**。

## 5. 显存核算（解析估算，非实测）

以下全部是**算术推演**，不是测量值。参数账由 `smoke_test.py` 在 meta device 上复核。

### 5.1 静态权重

| 项目 | 规模 | 精度 | 占用 |
| :--- | ---: | :--- | ---: |
| 三值主干（编码器 + 解码器 + 全局投影） | 10.09 B | 2 bit 打包 | **2.35 GiB** |
| 词表嵌入（与输出头权重绑定） | 262.6 M | BF16 | 0.49 GiB |
| 词表嵌入（同上） | 262.6 M | FP32（原型默认） | 0.98 GiB |
| 归一化 / 缩放 / 偏置 | — | FP32 | ~0.01 GiB |
| **合计** | **10.356 B** | — | **~2.85 GiB（BF16 嵌入）** |

> **注意**：方案早期版本给出的「3.09 GB」前提是嵌入层以 BF16 存储。参考实现按 PyTorch 默认
> 用 FP32 承载嵌入，实测核算为 **3.33 GiB**。两者差 0.49 GiB，全部来自嵌入层精度。以
> `smoke_test.py` 的打印为准。

### 5.2 128k 上下文 KV（单序列）

| 项目 | 计算 | 占用 |
| :--- | :--- | ---: |
| 全局 KV（全模型共享一份，FP8） | 2 × 16 头 × 128 维 × 131072 token × 1 B | **512 MiB** |
| 解码器本地 KV（16 层 × W=512，FP8） | 16 × 2 × 16 × 128 × 512 × 1 B | 32 MiB |
| **合计** | — | **544 MiB ≈ 0.531 GiB** |

前提假设：编码器按块运行一次后其自身缓存即释放（若要逐 token 流式编码，需再加 32 MiB）。
相比逐层存储 KV 的基线，这是把 KV 从「每层一份」压到「全模型一份」带来的收益。

## 6. 设计目标与基线对照

**本方案一列全部是设计目标，未经任何训练验证。** 基线仅采用可溯源的官方报告数字。

| 基准 | R1-Distill-Qwen-7B | Qwen2.5-7B-Instruct | Llama-3.2-3B-Instruct | **Apex-CED-1.58（设计目标）** |
| :--- | ---: | ---: | ---: | ---: |
| AIME 2024 (Pass@1) | 55.5 | — | — | [设计目标] 18–24 |
| MATH-500 (Pass@1) | 92.8 | — | — | [设计目标] 65–72 |
| MATH | — | 75.5 | — | — |
| GPQA Diamond (Pass@1) | 49.1 | 36.4 | 32.8 | [设计目标] 28–34 |
| LiveCodeBench (Pass@1) | 37.6 | 28.7 | — | [设计目标] 22–28 |
| GSM8K | — | 91.6 | — | — |

- `—` 表示官方报告未公布该项，**不作猜测性填充**。
- MATH 与 MATH-500 是不同基准，不可直接比较。
- 基线来源：DeepSeek-R1 技术报告（arXiv:2501.12948）、Qwen2.5 官方发布博客、Meta Llama 3.2 官方数据。

**预期定位**：若各假设成立，一个激活约 2.6 B 的三值模型**不应**期待打平 7B 级浮点模型；
合理的期待是在极小显存预算内、配合规则校验式测试时计算（TTC），逼近 7B 密集模型的**可用下限**。
这一预期本身也未被验证。

## 7. 已知局限与未解问题

1. **128k prefill 的显存问题未解决。** 参考实现构造显式 `[Lq, Lk]` 掩码，128k 下单是掩码就
   O(L²)。要真正跑长上下文，必须换成融合的滑动窗口 kernel（如 FlashAttention-2 的
   `window_size`）。**当前代码只适用于短序列。**
2. **RoPE 长上下文外推缺方案。** 默认 θ=1e4 无法直接外推到 128k，需要 YaRN / NTK 插值 /
   长上下文退火，本方案尚未包含。
3. **三值量化与优化器的兼容性未验证。** Muon 的 Newton–Schulz 正交化在 STE 阶梯截断环境下
   可能放大权重在临界值附近的跳变（chattering），需要消融验证；保守做法是前期退回 AdamW。
4. **硬件层面没有原生三值指令。** 主流 Tensor Core 只原生支持 FP16/BF16/FP8/INT8/INT4。
   在缺少定制内核（如 T-MAC 表驱动查找）的情况下，PyTorch 路径需在线反量化回 FP16 再执行
   GEMM，**不仅没有加速，反而增加搬运开销**。三值的算力红利目前只在 CPU（bitnet.cpp + AVX-512）
   或定制 ASIC 上成立。
5. **知识密集型任务的可能损失。** 三值离散化可能损害长尾事实记忆；本方案预期在规则主导型任务
   （数学、代码）上表现更稳，但这需要实验确认，不能预设。
6. **缺少消融。** 目前无法说明这个组合是否优于三个组件各自单独使用，也无法说明是否优于同等
   参数量的纯密集基线。这是本方案最关键的未验证点。
7. **训练成本。** 10.35B 参数、4.5T token 级别的预训练需要千卡级集群。**「8GB 显卡友好」严格
   仅指推理阶段。**

## 8. 与既有工作的关系

- **BitNet b1.58**（Ma et al., arXiv:2402.17764）—— 三值权重与 BitLinear 算子。
- **YOCO / Causal Encoder–Decoder** 路线 —— 跨层共享 KV 的核心思想；本方案借用其「单份全局 KV」
  结构，但用共享投影而非逐层缓存。
- **DeepSeek-V3 / R1** —— MLA 低秩 KV 压缩、无辅助损失负载均衡、GRPO。
- **Gemma 2**（arXiv:2408.00118）—— Logit soft-capping、滑窗与全局注意力交替。
- **Mistral / Longformer** —— 滑动窗口注意力。

本方案不主张在上述任何单项上有所突破。

## 9. 运行

```bash
pip install -r requirements.txt
python smoke_test.py
```

依赖：Python ≥ 3.10，PyTorch ≥ 2.1（CPU 即可运行测试）。

## 10. 引用

1. Ma et al. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* arXiv:2402.17764
2. DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948
3. Gemma Team. *Gemma 2: Improving Open Language Models at a Practical Size.* arXiv:2408.00118
4. Qwen Team. *Qwen2.5 Technical Report.* arXiv:2412.15115

## License

Apache-2.0