# YOLO-Master PEFT 消融实验脚本清单

> 生成时间: 2025-07-11 | 验证状态: 全部语法通过

---

## 一、脚本总览

| # | 脚本 | 行数 | 功能 | 验证 |
|---|------|------|------|------|
| 1 | `full_ablation_spec.py` | 309 | 统一数据结构规范 (Dataset/Variant/Result) | 语法OK |
| 2 | `full_ablation.py` | 750 | 全量 COCO 消融 (11 变体) | 语法OK |
| 3 | `full_ablation_scenarios.py` | 749 | 场景数据集扩展 (VisDrone/SKU110K/Cityscapes) | 语法OK |
| 4 | `full_ablation_multiscale.py` | 715 | 多分辨率×多尺度 (320/640/1280) | 语法OK |
| 5 | `ablation_peft_coco128.py` | 688 | COCO128 九方法对比 (Full/LoRA/DoRA/IA3/LoHA/AdaLoRA+3 MoLoRA) | 语法OK |
| 6 | `benchmark_latency.py` | 766 | 推理延迟基准 (PyTorch/ONNX/TensorRT × merged/unmerged) | 语法OK |
| 7 | `ablation_molora_full.py` | 919 | MoLoRA 完整消融 (M1-M4 四大模块) | 语法OK |
| 8 | `ablation_fewshot.py` | 883 | Few-shot K-shot 协议 (K=1,5,10 × 3 seeds) | 语法OK |
| 9 | `ablation_routing_cl.py` | 1242 | 路由诊断 + 持续学习 (Day→Night→Fog) | 语法OK |
| 10 | `ablation_peft_visualize.py` | 1505 | 可视化报告生成 (8 图 + Markdown + HTML) | 语法OK |
| 11 | `run_full_ablation.sh` | 110 | Shell 编排 (E1→E2→E3→EVAL) | 语法OK |

**总计: 8,436 行 Python + 110 行 Shell**

---

## 二、逐个脚本详解

### 1. `full_ablation_spec.py` — 统一规范

- **作用**: 定义所有消融实验共享的数据结构
- **导出**: `DatasetConfig`, `VariantConfig`, `ExperimentResult`, `DATASET_REGISTRY`, `FULL_ABLATION_VARIANTS`, `QUICK_ABLATION_VARIANTS`, `MULTIRES_VARIANTS`
- **数据集注册表**: coco, coco128, visdrone, sku110k, cityscapes, foggy_cityscapes
- **11 个全量变体**: full, lora_r16, dora_r16, loha_r16, ia3, hra_r16, molora_4e2k, molora_4e2k_spatial, molora_4e2k_hybrid, molora_aware, molora_calib

**运行**: 不直接运行，被其他脚本 import

---

### 2. `full_ablation.py` — 全量 COCO 消融

- **作用**: 运行全部 11 个 PEFT 变体在 COCO2017 或 COCO128 上
- **特点**:
  - 支持 `--quick` 快速模式 (COCO128, 3 epochs)
  - 支持 `--measure-latency` 推理延迟测量
  - 支持 `--variants` 过滤变体
  - 多尺度 mAP 分解 (small/medium/large)
  - 实时 JSON 落盘

**运行命令**:
```bash
cd /Users/gatilin/PycharmProjects/YOLO-Master-v0708/scripts/ablation_suite

# 快速验证 (COCO128, 3 epochs)
python full_ablation.py --quick --measure-latency

# 全量实验 (COCO2017, 50 epochs)
python full_ablation.py --dataset coco --seed 42

# 仅跑指定变体
python full_ablation.py --quick --variants full,lora_r16,molora_4e2k
```

**输出**:
- `scripts/ablation_suite/full_ablation_results.json`
- `scripts/ablation_suite/runs_full_ablation/`

---

### 3. `full_ablation_scenarios.py` — 场景数据集扩展

- **作用**: 在 VisDrone / SKU110K / Cityscapes / FoggyCityscapes 上运行核心 7 变体
- **特点**:
  - 支持 domain 标签 (cityscapes="day", foggy_cityscapes="fog")
  - 支持 Cityscapes→FoggyCityscapes 持续学习序列
  - 自动继承上一域模型状态 (state_dict)
  - 覆盖 `--epochs`, `--batch`, `--dataset`, `--variant`

**运行命令**:
```bash
# 全场景实验
python full_ablation_scenarios.py --seed 42 --epochs 10

# 仅持续学习序列
python full_ablation_scenarios.py --cl-only

# 仅 VisDrone
python full_ablation_scenarios.py --dataset visdrone

# 仅 LoRA 变体
python full_ablation_scenarios.py --variant lora_r16
```

**输出**:
- `scripts/ablation_suite/full_ablation_scenarios_results.json`
- `scripts/ablation_suite/runs_full_ablation_scenarios/`

---

### 4. `full_ablation_multiscale.py` — 多分辨率×多尺度

- **作用**: 验证 320/640/1280 分辨率下 LoRA vs MoLoRA 的 scale-wise mAP 差异
- **特点**:
  - 3 分辨率 × 3 placement (full/backbone/neck) = 9 MoLoRA + 3 LoRA = 12 变体
  - Scale-aware 验证: mAP_small, mAP_medium, mAP_large
  - 自动 trade-off 曲线图生成
  - Monkey-patch DetectionValidator.eval_json 捕获 COCOeval stats

**运行命令**:
```bash
python full_ablation_multiscale.py
```

**输出**:
- `scripts/ablation_suite/full_ablation_multiscale_results.json`
- `scripts/ablation_suite/runs_multiscale_ablation/`
- `scripts/ablation_suite/multiscale_tradeoff_curve.png`

---

### 5. `ablation_peft_coco128.py` — 九方法对比 (COCO128)

- **作用**: 在 COCO128 上系统对比 9 种 PEFT 方法
- **变体**: full, lora, dora, ia3, loha, adalora, molora, molora_aware, molora_calib
- **配置**: 3 epochs, batch=8, imgsz=320

**运行命令**:
```bash
python ablation_peft_coco128.py
```

**输出**:
- `scripts/ablation_suite/ablation_peft_coco128_results.json`
- `scripts/ablation_suite/runs_peft_ablation/`

---

### 6. `benchmark_latency.py` — 推理延迟基准

- **作用**: 测量 Baseline / LoRA / MoLoRA 在 merged/unmerged 状态下于多后端的延迟
- **后端**: PyTorch eager (✓), ONNX Runtime (尝试), TensorRT (CUDA only)
- **指标**: mean ± std / median / p95 / min / max / FPS
- **测量方式**: warmup 10 + timed 50, sigma clipping 去异常值

**运行命令**:
```bash
python benchmark_latency.py
```

**输出**:
- `scripts/ablation_suite/benchmark_latency_results.json`

---

### 7. `ablation_molora_full.py` — MoLoRA 完整消融 (M1-M4)

- **作用**: 系统验证 MoLoRA 各组件有效性
- **四大模块**:
  - **M1** (5 变体): vs LoRA 对比 — full, lora, molora, molora_aware, molora_calib
  - **M2** (3 变体): Router 类型消融 — linear, spatial, hybrid
  - **M3** (10 变体): Expert 扫描 — E=2/4/8/16 × K=1/2/4
  - **M4** (1 变体): Merge/Unmerge 精度验证

**运行命令**:
```bash
python ablation_molora_full.py
```

**输出**:
- `scripts/ablation_suite/ablation_molora_full_results.json`
- `scripts/ablation_suite/runs_molora_ablation/`

---

### 8. `ablation_fewshot.py` — Few-shot K-shot 协议

- **作用**: 验证少数据场景下 Full/LoRA/MoLoRA 的性能
- **配置**: K=1,5,10 × 3 seeds (42,123,456) × 3 methods = 27 个实验
- **特点**:
  - 自动从 COCO128 采样 K 张图像构建训练子集
  - 验证集保持完整 128 张
  - MoE 诊断 (ExpertUsageTracker, RoutingCollapseDetector)
  - 聚合报告: mean±std, success rate

**运行命令**:
```bash
python ablation_fewshot.py
```

**输出**:
- `scripts/ablation_suite/ablation_fewshot_results.json` (含 records + summary + config)
- `scripts/ablation_suite/runs_fewshot_ablation/`
- 临时数据集: `scripts/ablation_suite/.temp_kshot/` (退出自动清理)

---

### 9. `ablation_routing_cl.py` — 路由诊断 + 持续学习

- **作用**: (a) 路由健康度监测 (b) 多域持续学习
- **路由诊断**:
  - 每 epoch Gini 系数趋势
  - Router collapse 检测 (threshold=0.8)
  - 主导专家占比追踪
  - 实时可视化图表 (3 张 PNG)
- **持续学习**:
  - Day→Night→Fog 三域顺序训练
  - 域专属专家分配 (allocate_domain_experts)
  - BWT (Backward Transfer) 计算
  - 灾难性遗忘度 (Catastrophic Forgetting)
  - 可视化图表 (4 张 PNG: heatmap, BWT, forgetting, trajectory)

**运行命令**:
```bash
python ablation_routing_cl.py
```

**输出**:
- `scripts/ablation_suite/ablation_routing_cl_results.json`
- `scripts/ablation_suite/runs_routing_cl/`
- `scripts/ablation_suite/runs_routing_cl/routing_diag/*.png`

---

### 10. `ablation_peft_visualize.py` — 可视化报告生成

- **作用**: 聚合所有实验结果，生成多面板图表和 Markdown 报告
- **自动发现数据源**:
  - `e1_*.json` (per-expert rank)
  - `e2_*.json` (router calibration)
  - `e3_viz_outputs/` (expert load)
  - `eval_moe_peft_*.json` (unified eval)
  - `peft_validation/*.json` (standard PEFT)
- **生成 8 张图**:
  1. fig01_parameter_efficiency.png — 参数量对比
  2. fig02_metrics_comparison.png — 精度四面板
  3. fig03_training_efficiency.png — 时间-参数散点
  4. fig04_expert_rank_allocation.png — E1 专家秩分配
  5. fig05_router_calibration.png — E2 校准参数开销
  6. fig06_expert_load_gini.png — E3 负载均衡
  7. fig07_peft_standard_comparison.png — 标准 PEFT 对比
  8. fig08_combined_dashboard.png — 综合仪表盘

**运行命令**:
```bash
python ablation_peft_visualize.py
python ablation_peft_visualize.py --output-dir ./my_reports
```

**输出**:
- `scripts/ablation_suite/ablation_reports/*.png` (8 张)
- `scripts/ablation_suite/ablation_reports/report.md`

---

### 11. `run_full_ablation.sh` — Shell 编排脚本

- **作用**: 顺序执行 E1→E2→E3→EVAL 四个子实验
- **说明**: 这是旧版编排，新版建议使用 `full_ablation.py` 或按需运行独立脚本

**运行命令**:
```bash
cd /Users/gatilin/PycharmProjects/YOLO-Master-v0708/scripts/ablation_suite
bash run_full_ablation.sh
```

---

## 三、实验矩阵汇总

| 实验 | 数据集 | 变体数 | 总实验数 | 预期耗时 | 核心产出 |
|------|--------|--------|----------|----------|----------|
| 全量 COCO | coco/coco128 | 11 | 11 | ~6-48h | mAP, latency, scale-wise mAP |
| 场景扩展 | visdrone/sku110k/cityscapes/foggy | 7×4=28 | 28+ | ~12-24h | 跨场景泛化性 |
| 多分辨率 | coco128 | 12 | 12 | ~2-4h | trade-off 曲线 |
| 九方法对比 | coco128 | 9 | 9 | ~1-2h | 标准 PEFT vs MoLoRA |
| 延迟基准 | coco128 (dummy) | 5×3=15 | 15 | ~30min | FPS, mean±std ms |
| MoLoRA 完整 | coco128 | 19 | 19 | ~3-6h | M1-M4 全部结论 |
| Few-shot | coco128 (K-shot 子集) | 3×3×3=27 | 27 | ~2-4h | K-shot 协议有效性 |
| 路由+CL | coco128 | 1 (含多域) | 1+ | ~1-2h | Gini, BWT, 遗忘度 |
| **总计** | — | **—** | **~112+** | **~27-87h** | **论文全部消融** |

---

## 四、先决条件

### 模型文件
- `YOLO-Master-EsMoE-N.pt` (项目根目录)

### 数据集
- COCO128: 自动下载 (ultralytics 内置)
- COCO2017: 需手动准备 `coco.yaml` 指向数据集路径
- VisDrone / SKU110K / Cityscapes / FoggyCityscapes: 需手动准备对应 YAML

### Python 环境
```bash
# 核心依赖 (已确认存在于当前环境)
ultralytics (当前仓库版本)
torch (MPS/CUDA/CPU)
numpy
matplotlib

# 可选 (用于特定功能)
onnxruntime      # benchmark_latency.py ONNX 后端
faster-coco-eval # full_ablation_multiscale.py scale-aware mAP
```

### 环境变量
```bash
export WANDB_MODE=disabled
export KMP_DUPLICATE_LIB_OK=TRUE
export YOLO_AUTOINSTALL=false
export YOLO_VERBOSE=false
```

---

## 五、验证状态

| 验证项 | 状态 | 说明 |
|--------|------|------|
| 语法检查 (py_compile) | ✅ 10/10 通过 | 全部脚本无语法错误 |
| 导入测试 (import *) | ⚠️ 未批量测试 | 依赖 ultralytics 运行时环境，部分脚本含 assert 检查 |
| MoLoRA 集成测试 | ✅ 通过 | YOLO-Master-EsMoE-N → apply MoLoRA → 3.9M params, 32% trainable |
| LoRA 集成测试 | ✅ 通过 | standard LoRA → 601K trainable (20.6%) |
| 实际训练运行 | ⏳ 未执行 | MPS 上训练太慢，未做端到端验证 |
| 结果 JSON 结构 | ✅ 通过 | full_ablation_spec.ExperimentResult 序列化正常 |

---

## 六、推荐执行顺序

对于论文实验，建议按以下顺序执行：

### Phase 1: 快速验证 (30 min - 2h)
```bash
# 1. COCO128 九方法对比 — 验证所有 PEFT 方法可正常训练
python ablation_peft_coco128.py

# 2. MoLoRA 核心模块 — 验证 M1 (vs LoRA) 可跑通
python ablation_molora_full.py  # 会跑 M1-M4，但可中断
```

### Phase 2: 核心消融 (4-8h)
```bash
# 3. 多分辨率实验
python full_ablation_multiscale.py

# 4. 延迟基准
python benchmark_latency.py

# 5. Few-shot
python ablation_fewshot.py
```

### Phase 3: 扩展实验 (12-24h)
```bash
# 6. 路由诊断 + 持续学习
python ablation_routing_cl.py

# 7. 场景数据集
python full_ablation_scenarios.py --epochs 20
```

### Phase 4: 全量实验 (24-48h, 建议 GPU 集群)
```bash
# 8. COCO2017 全量
python full_ablation.py --dataset coco --measure-latency
```

### Phase 5: 报告生成
```bash
# 9. 可视化
python ablation_peft_visualize.py
```

---

## 七、待确认事项

请确认以下内容：

1. **数据集可用性**: COCO2017 / VisDrone / SKU110K / Cityscapes 是否已准备？YAML 路径是否正确？
2. **运行环境**: 是否有 CUDA GPU 可用？当前仅 MPS (Apple Silicon) 已配置。
3. **实验范围**: 是否需要裁剪变体数量？全量 112+ 实验可能需要数天。
4. **epoch 数**: 当前默认 COCO128 用 3 epochs, COCO 用 50 epochs，是否需要调整？
5. **输出位置**: 所有结果默认写入 `scripts/` 子目录，是否需要集中到 `runs/` 或 `experiments/`？
6. **持续学习**: Cityscapes→FoggyCityscapes 需要 foggy 数据集，是否已准备？

---

## 八、文件路径汇总

```
/Users/gatilin/PycharmProjects/YOLO-Master-v0708/
├── scripts/
│   ├── full_ablation_spec.py          # 规范定义
│   ├── full_ablation.py               # 全量消融主脚本
│   ├── full_ablation_scenarios.py     # 场景扩展
│   ├── full_ablation_multiscale.py    # 多分辨率
│   ├── ablation_peft_coco128.py       # 九方法对比
│   ├── benchmark_latency.py           # 延迟基准
│   ├── ablation_molora_full.py        # MoLoRA M1-M4
│   ├── ablation_fewshot.py            # Few-shot
│   ├── ablation_routing_cl.py         # 路由+CL
│   ├── ablation_peft_visualize.py     # 可视化
│   └── run_full_ablation.sh           # Shell 编排
│
│   # 运行时生成 (执行后才会出现)
│   ├── full_ablation_results.json
│   ├── full_ablation_scenarios_results.json
│   ├── full_ablation_multiscale_results.json
│   ├── ablation_peft_coco128_results.json
│   ├── benchmark_latency_results.json
│   ├── ablation_molora_full_results.json
│   ├── ablation_fewshot_results.json
│   ├── ablation_routing_cl_results.json
│   ├── multiscale_tradeoff_curve.png
│   ├── ablation_reports/              # 可视化输出
│   │   ├── fig01_parameter_efficiency.png
│   │   ├── fig02_metrics_comparison.png
│   │   ├── ... (共 8 张)
│   │   └── report.md
│   └── runs_*/                        # 训练日志
│
├── YOLO-Master-EsMoE-N.pt             # 预训练权重
└── coco128.yaml / coco.yaml / ...     # 数据集配置
```

---

*本清单由 Orchestrator 自动生成，供用户确认实验计划。*
