# Issue #111 维护者回复草稿

## 回复内容

@SidKC 感谢这份非常详尽、严谨的复现实验报告。这种带着完整测量数据、明确统计问题修复、以及诚实结论（"没有 Sweet Spot"）的 issue，对项目的长期质量非常有价值。

以下是我对报告各部分的回应和后续建议：

---

### 1. 统计问题修复 ✅

你指出的两个问题——路由利用率只保留最近一次快照、recovery 后 epoch 被重复计入调度——确实是会改变实验结论的 bug。感谢你不仅定位了问题，还提供了 accepted-epoch 语义和 recovery-safe trace 的修复方案。

**行动**：请将动态路由统计与 recovery correctness 的 PR 单独提交，我们会优先 review 合并。这是基础设施级别的修复，不应与实验结论 PR 捆绑。

---

### 2. 专家剪枝结果

你的数据非常清晰：
- 0.05–0.20 逻辑阈值在实际物理结构上均为 no-op（3/3/3/3）
- 0.30 产生非平凡结构（2/2/3/3），但 mAP50-95 下降 39.3%，P95 forward 仅改善 1–2%

**结论**：在当前的 soft contribution 信号和固定阈值策略下，VisDrone 上确实不存在满足 "mAP 降幅 ≤ 0.01" 约束的 Sweet Spot。你的判断是正确的——不能把同一 no-op 结构解释为四种不同优化结果。

**建议**：请在 PR 中保留五档逻辑阈值到物理结构的去重执行，这是防止未来实验者误读结果的重要机制。

---

### 3. LoRA10 恢复

"恢复没有达到原始 dense baseline，还增加了推理计算和延迟"——这一点很关键。它说明当前 MoLoRA 的适配范围（adapters + detection head）可能不足以补偿专家剪枝带来的表示能力损失。

两组训练后期都出现质量坍塌（quality collapse），只报告 best checkpoint 会掩盖这个问题。你同时报告 best 和 final 的做法是正确且必要的。

**建议**：下一步可以尝试扩大 LoRA 的 target module 范围（例如纳入 router 或 backbone 特定层），或者增加恢复 epoch 数，但前提是先解决 late-stage collapse 的根因。

---

### 4. Gini 动态调度

三组对照（Fixed Baseline / Gini Dynamic / Fixed-Low Ablation）都出现明显的 late-stage quality collapse，最终 mAP50-95 仅为最佳值的 21-22% 左右。这意味着：
- 动态调度在这个短实验中的微弱优势（best 0.04144 vs 0.03551）不具备统计显著性
- "epoch 1 达到基线最终精度 95%" 这个指标在 collapsed 基线下失去了区分力

你的结论很准确——"只能视为候选现象，不能证明收敛加速或稳定质量收益"。

**建议**：动态调度功能保持默认关闭，但代码和 trace 机制可以保留。后续需要更长的实验周期（≥ 50 epoch）和多 seed 验证，才能判断 Gini 调度是否真正有效。

---

### 5. 后续优先级建议

基于你提供的完整数据，我提议按以下顺序推进：

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P0 | 合并统计修复 PR | 路由累计 + accepted-epoch 语义，这是所有后续实验的基础设施 |
| P0 | 合并 MoLoRA checkpoint 小修复 PR | 阻塞性修复 |
| P1 | 建立稳定长周期基线 | 单 seed 50-epoch dense baseline，确认无 collapse 的稳定训练配置 |
| P1 | 定义不依赖 collapsed final 的收敛目标 | 例如 best mAP 在验证集上的 plateau 检测，或 moving average 收敛 |
| P2 | 多 seed 验证 | 在稳定基线之上，对当前实验结论进行统计验证 |
| P2 | 扩大 LoRA target 范围 | 如果长周期基线能稳定，再尝试补偿剪枝损失 |

---

### 6. 关于 PR 拆分

你提到有以下几个产出：
- 动态路由统计与 recovery correctness PR ✅ 优先合并
- MoLoRA checkpoint 小修复 PR ✅ 优先合并
- Issue #52 实验整合分支

**建议**：将实验整合分支拆分为：
1. **infra PR**：统计修复 + 去重执行 + 双顺序 benchmark 工具（可复用代码）
2. **experiment PR**：实验脚本 + 结果数据 + 本文报告（作为 docs/experiments/issue-111-visdrone.md）

这样可以让 review 更聚焦，也方便后续实验者复用 infra。

---

### 总结

你的实验结论——"当前信号、阈值和恢复配置还不足以形成可部署的 Sweet Spot"——是诚实且数据驱动的。这比强行包装一个"有效"结论更有价值。配套脚本的完整测量记录和明确返回"未观察到 Sweet Spot"的机制，正是工程级标准应有的做法。

请按上述优先级提交 PR，我会安排 review。再次感谢这份高质量的实验报告。

