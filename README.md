# SKED

**Shared-KV Encoder–Decoder — 1.58-bit 三值权重 + 全局共享 KV + 滑动窗口注意力的组合设计探索**

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

The architecture is not trained at scale. The figures are analytical estimates, design targets,
or small-scale CPU measurements. Two of those measurements establish concrete properties of the
mechanism: a multi-query associative-recall ablation shows the shared cross-attention path is
what carries long-range information (1.000 with it, 0.388 without), and the same ablation shows
the encoder needs no dense layer, so the prefill path carries no O(L²) term from the encoder.
Two results are unfavourable and are stated rather than omitted: on CPU the ternary path is
1.3–2.1× slower than fp32, and the recall task is solved only at toy scale.

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
│ 全窗口即可，无需稠密层（见关键设计点 1 的消融）       │
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

**关键设计点 1：共享 KV 只需「逐位置可寻址」，不需要「逐位置全前缀」。**

一个很自然、但**错误**的直觉是：既然叫"全局 KV"，产出它的编码器就必须有全局感受野，
否则跨注意力只能读到一份窗口摘要。这个直觉看起来还有证据支持——用探针扰动序列起点
token，末位 `global_k` 的变化是**位级精确的零**：

| `enc_dense_tail` | \|Δ global_k[0]\| | \|Δ global_k[-1]\| |
| ---: | ---: | ---: |
| 0 | 1.687584 | **0.000000** |
| 1 | 1.687584 | 0.016366 |
| 2 | 1.687584 | 0.022712 |

**但这个证据是无关的。** 解码器的跨注意力读取的是整个 `global_k` 集合，而不是单个末位
条目。位置 `p` 的信息会出现在 `global_k[p .. p+W−1]` 这一串条目中；只要 `p ≤ i`，
`global_k[p]` 必然存在。**因此每个位置的信息永远可寻址**，窗口摘要的并集已经覆盖全前缀。

消融直接验证了这一点（`experiments/exp6_ablation.py`，MQAR 6 对，窗口 4，fp32 权重，
2+2 层，1500 步，2 个种子）：

| 变体 | 编码器 | 跨注意力 | 平均准确率 |
| :--- | :--- | :--- | ---: |
| `dense_causal` | 全稠密 | 无 | 1.000 |
| `local_only` | 窗口 | 无 | **0.388** |
| `sked_tail0` | **全窗口** | 有 | **1.000** |
| `sked_tail1` | 窗口+稠密尾层 | 有 | 1.000 |

两条结论：

1. **跨注意力共享 KV 是必要且充分的**：`sked_*` = 1.000，而同样窗口、只去掉跨注意力的
   `local_only` 只有 0.388。
2. **编码器感受野无关紧要**：`sked_tail0`（全窗口编码器）与 `sked_tail1`（稠密尾层）、
   `dense_causal` 同为 1.000。

**因此不需要稠密尾层，prefill 的 O(L²) 项可以完全省掉，编码器可以全窗口化。**
`enc_dense_tail` 保留为消融开关，默认 0。

**关键设计点 2：解码器的跨注意力必须施加因果掩码。** 若解码器第 `i` 位可点乘 `j > i` 的
全局 KV，就会读到它正要预测的未来 token，训练会退化为复制任务。这是本类架构最容易出错的
地方，参考实现中已显式处理并有测试覆盖。

## 4. 参考实现

```
sked_core.py        # 模型与算子
smoke_test.py       # 冒烟测试（10 项）
experiments/        # 前置实验（CPU，小规模）
results/            # 实验输出 JSON
```

实现中刻意处理好的两处细节：

- **STE 前向恒为量化值**：`w_ste = w_quant.detach() + mask * (w_scaled - w_scaled.detach())`。
  常见的简写 `w + (w_quant - w).detach()` 会在前向残留浮点项，使前向并非真正三值。
- **跨注意力掩码用绝对索引**：`q_idx = arange(kv_len - q_len, kv_len)`，因此在 prefill（`q_len=L`）
  与增量解码（`q_len=1`）两种状态下都正确。

另有一个 `ternary` 开关（`make_linear` 工厂），用于在消融实验中绕开量化器：小规模下三值权重会
主导结果，把量化与架构两个轴混在一起测会让消融失去解释力（见 5.3 节）。

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

### 5.3 前置实验（CPU，小规模实测）

四个实验的原始输出在 `results/*.json`，复现命令见 `experiments/`。**这些都是原型级小规模运行，
只检验内部一致性、缩放趋势与一条机制主张，不构成任何模型质量证据。**

| 实验 | 设置 | 结果 | 结论 |
| :--- | :--- | :--- | :--- |
| exp1 KV 缩放 | 4 层原型，fp32，W=256，L=256→4096 | L=4096 时逐层基线 16.0 MiB vs SKED 5.0 MiB（**31%**）；SKED 局部 KV 在 L≥512 后恒定 1.0 MiB | 窗口截断生效，局部 KV 确实不随 L 增长 |
| exp4 三值 matmul | CPU，16 线程，随机未训练权重 | 三值路径比 fp32 **慢 1.32–2.06×**；输出 cos 相似度 0.9376 | **三值在当前硬件上是存储优势，不是算力优势** |
| exp6 召回消融 | MQAR 6 对，seq=14，W=4，fp32，2+2 层，1500 步，2 种子 | `dense_causal` 1.000 / `local_only` **0.388** / `sked_tail0` 1.000 / `sked_tail1` 1.000 | **跨注意力共享 KV 是必要且充分的；编码器感受野无关** |
| exp6 感受野探针 | 扰动 token 0，测末位 `global_k` 变化，无需训练 | `enc_dense_tail=0` 时 **0.000000**（位级零），`=1` 时 0.069 | 该探针测量的是**一个真实但无关的量**，见下 |

**关于 exp6 的三点如实说明：**

- **量化被刻意排除在这个实验之外。** 四个变体全部用 fp32 权重。早期版本在三值权重 + d=64 下
  运行，结果**没有任何变体学会任务**——包括全注意力上界，只有 0.271。那是关于小规模下量化器的
  陈述，不是关于架构的陈述；两个轴混在一起测，消融就没有解释力。因此实验设了一道门控：
  上界学不会就不测下游。上界在 step 1000 达到 1.000。
- **探针结论曾被我们误读。** 探针显示窗口编码器下起点信息无法到达末位条目，看起来证明了
  「编码器必须有全局感受野」。**这个推论是错的**：解码器跨注意力读的是整个 `global_k` 集合，
  位置 p 出现在 `global_k[p .. p+W-1]` 中，只要 `p ≤ i` 就必然可寻址。消融是证明这一点的唯一
  手段——`sked_tail0`（全窗口编码器）与 `sked_tail1`（稠密尾层）、`dense_causal` 同为 1.000。
- **这消掉了一项我们原以为要付的成本**：编码器不需要稠密层，prefill 路径不含来自编码器的
  O(L²) 项。解码器跨注意力对全量 K₀ 的因果掩码仍是 O(L²)，那部分未解决，见第 8 节。

关于 exp4 的两点如实说明：

- **慢是结构性的，不是实现问题。** 主流 Tensor Core 只暴露 FP16/BF16/FP8/INT8/INT4，没有三值
  格式，内核只能在线反量化后再跑标准 GEMM——付了量化的代价，没拿到量化的收益。
- **相对 L2 误差约 0.40，看起来很大。** 但这是在**随机未训练权重**上测的，此时缩放因子 γ 无意义、
  也没有可吸收量化噪声的学习结构，因此绝对误差不代表训练后的三值模型。cos 相似度在这里只作为
  「估计器没有退化」的健全性检查。

## 6. 四个待验证的扩展提案

以下四项**均未实现、未验证**，仅作为下一步值得尝试的方向列出，并标注各自的风险。

1. **Chunked Global KV（CGKV）** —— 把编码器按固定块长 C 分段运行，每块产出一段全局 KV 追加到
   只增缓存；解码器跨注意力按块读取并施加带状掩码。峰值掩码显存从 O(L²) 降到 O(C·W)，且缓存
   可流式处理（块编码完即可释放）。
   *风险*：位置 i 的表征会依赖它落在哪个块边界附近，引入周期性伪影。
2. **共享基座 + 每层低秩 KV 适配器** —— 保留一份共享 K₀,V₀，每层学一个低秩残差
   `K_l = K₀ + A_l B_l K₀`（r ≪ d）。参数量仅 4·N_d·d·r，不增加任何缓存，取「完全共享」（便宜、
   容量低）与「每层独立」（昂贵、容量高）的中间点。
   *风险*：若适配器吸收过多，共享基座会退化为一个初始化技巧，内存论证仍成立但机制名存实亡；
   需按 r ∈ {0,16,64,256} 消融区分。
3. **相位非对称专家激活（PAEA）** —— prefill 是算力受限、decode 是带宽受限，二者本该用不同的
   算力/显存权衡，而多数 MoE 设计对两相用同一套静态配置。提案：编码器用稠密 SwiGLU（只跑 prefill），
   解码器用 MoE，且解码器内**按深度**变化激活专家数——浅层少、深层多。
   *风险*：**这是四项里最弱的一项**，依赖一个未经检验的「层间分工」假设，且深度可变的 top-k 下
   路由器的负载均衡行为并不显然。
4. **双档滑动窗口（DTSW）** —— 解码器偶数层用 W_s、奇数层用 W_l（W_s ≪ W_l）。局部缓存只需一个
   W_l 缓冲加一个 W_s 缓冲，但感受野是层次化的：隔两层可回看 W_s + W_l 个位置。比统一 W_l 便宜，
   比统一 W_s 表达力强。
   *风险*：相对统一取平均窗宽的增益可能很小；且 Gemma 2 已经在交替局部/全局层，本项的区别仅在于
   交替的是两档**局部**窗口，以保证全局路径仍是唯一无界的那条。

## 7. 设计目标与基线对照

**本方案一列全部是设计目标，未经任何训练验证。** 基线仅采用可溯源的官方报告数字。

| 基准 | R1-Distill-Qwen-7B | Qwen2.5-7B-Instruct | Llama-3.2-3B-Instruct | **SKED（设计目标）** |
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

## 8. 已知局限与未解问题

1. **128k prefill 的显存问题未解决。** 参考实现构造显式 `[Lq, Lk]` 掩码。**编码器侧已可避免
   O(L²)**（全窗口化即可，见 5.3 节消融），但**解码器跨注意力对全量 K₀ 的因果掩码仍是 O(L²)**，
   128k 下单是掩码就爆。要真正跑长上下文，必须换成融合的滑动窗口 kernel（如 FlashAttention-2 的
   `window_size`），或按 6 节的 CGKV 分块。**当前代码只适用于短序列。**
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
6. **消融只到玩具规模，不足以判定架构。** exp6 在 6 对 KV、2+2 层、1500 步上让 `sked` 追平了
   稠密模型。但「共享缓存**够用**」比「共享缓存**暂时没有害处**」是强得多的主张，只有前者才能
   支撑显存论证。目前无法说明这个组合是否优于三个组件各自单独使用，也无法说明是否优于同等
   参数量的纯密集基线——**在真实数据上**。这是本方案最关键的未验证点。
7. **训练成本。** 10.35B 参数、4.5T token 级别的预训练需要千卡级集群。**「8GB 显卡友好」严格
   仅指推理阶段。**

## 9. 与既有工作的关系

- **BitNet b1.58**（Ma et al., arXiv:2402.17764）—— 三值权重与 BitLinear 算子。
- **YOCO / Causal Encoder–Decoder** 路线 —— 跨层共享 KV 的核心思想；本方案借用其「单份全局 KV」
  结构，但用共享投影而非逐层缓存。
- **DeepSeek-V3 / R1** —— MLA 低秩 KV 压缩、无辅助损失负载均衡、GRPO。
- **Gemma 2**（arXiv:2408.00118）—— Logit soft-capping、滑窗与全局注意力交替。
- **Mistral / Longformer** —— 滑动窗口注意力。

本方案不主张在上述任何单项上有所突破。

## 10. 论文

`paper/` 下是一份 IEEEtran 双栏短文（`main.tex`），包含架构图、三个实验图表、四个扩展提案，
以及一节「Threats to Validity」逐条列出未验证项。编译：

```bash
python paper/make_figures.py
pdflatex -output-directory=paper paper/main.tex   # 跑两遍以解析交叉引用
```

生成物 `paper/main.pdf`（6 页）。`paper/sked-overleaf.zip` 是可直接上传到 Overleaf 的完整工程包
（`main.tex` + `figures/`），上传后无需任何额外配置即可编译。

## 11. 运行

```bash
pip install -r requirements.txt
python smoke_test.py
```

依赖：Python ≥ 3.10，PyTorch ≥ 2.1（CPU 即可运行测试）。

## 12. 引用

1. Ma et al. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* arXiv:2402.17764
2. DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948
3. Gemma Team. *Gemma 2: Improving Open Language Models at a Practical Size.* arXiv:2408.00118
4. Qwen Team. *Qwen2.5 Technical Report.* arXiv:2412.15115

## License

- **代码**（`sked_core.py`、`smoke_test.py`、`experiments/`、`paper/make_figures.py`）：Apache-2.0，见 `LICENSE`。
- **论文与文档**（`paper/main.tex`、`README.md`、`paper/figures/`）：CC BY 4.0 —— 允许任意转载、
  改编与商用，仅需署名。见 `LICENSE-PAPER`。

引用本工作：

```bibtex
@misc{sked2026,
  title  = {SKED: Sharing a Single Global KV Cache Across All Decoder Layers
            for Memory-Bound Long-Context Inference},
  author = {SKED Project},
  year   = {2026},
  note   = {Design note; reference implementation and preliminary measurements},
  url    = {https://github.com/3440340143/sked}
}
```