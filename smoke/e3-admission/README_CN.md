# E3 路由 Smoke 测试

> 一句话说明：本目录记录对 YOLO-Master 三种路由模块（MoE / MoT / Latent Mixture）的路由行为
> smoke 验证——在未训练（随机初始化）模型上运行前向/验证，采集路由统计并归档证据。
> 本验证只说明“代码可以运行、数据可以采集”，不评估模型效果，也不代表训练后行为。

## 当前状态

已完成：

- 已完成三类模块（MoT / MoE / Latent）的 Smoke 测试：三条命令均可运行并采集到路由统计数据。
  Smoke 只证明代码可运行、数据可采集，不代表模型性能或训练后效果已验证
- 本次 smoke 使用的路由诊断/采集代码（均位于锁定上游 commit 内）：
  `scripts/diagnose_mot_routing.py`、`ultralytics/nn/modules/moe/analysis.py`
  （`ExpertUsageTracker`）、`ultralytics/nn/modules/latent_mixture.py`
  （`last_routing_snapshot`）
- 输出证据已归档到本目录：终端日志、3 个 CSV、1 张热力图 PNG
- hook 开销测量脚本已入库：`scripts/measure_routing_hook_overhead.py`
  （本次测量数值未单独归档日志，见“当前限制”）

未完成：

- 统一路由字段 schema：只有设计草案，未实现（见 `E3_准入检查_技术产出.md`）
- 统一采集层 adapter：未实现
- 训练后 checkpoint 上的路由坍塌复测：未做（当前只覆盖随机初始化模型）
- MoT 在 MOT 跟踪任务上的评估：未做

## 实际做了什么

仓库基线是 ultralytics 8.4.101 的 YOLO-Master fork（本地分支 `baseline/2026-08-22`）。
E3 阶段对三类路由模块做了可复现的路由行为验证：

- MoT（`ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml`）：
  用 `scripts/diagnose_mot_routing.py --synthetic` 生成 4 类合成场景
  （`dense_small` / `large_regular` / `irregular_occluded` / `sparse_small`），无需数据集
- MoE（`ultralytics/cfg/models/master/v0_9/det/yolo-master-n.yaml`）：
  用 `ExpertUsageTracker` hook 在 coco8 验证集（4 张图，自动下载）上采集
  各 router 的专家命中（hits）与加权和（weighted_sum）
- Latent（`ultralytics/cfg/models/26/yolo26-master-latent-n.yaml`）：
  随机 640×640 单张前向，读取 `last_routing_snapshot`

观察到的现象（未训练基线的真实特征，不是脚本 bug）：

- MoT 在全部 4 类场景下 `LocalConvTransformer` 专家的 `top1_share` 恒为 1.00，
  另外两个专家为 0——随机初始化阶段即出现专家坍塌
- MoE 的 3 个 router（`model.5.routing` / `model.8.routing` / `model.11.routing`）
  均成功挂 hook，`usage_stats` 有正常数值
- Latent 的 3 个模块（`model.23` / `model.24` / `model.25`）的 `last_routing_snapshot`
  字段数均为 36，见 `e3_完整终端日志.txt` 的 Latent 段落

## 代码现状与已验证的修复（位于锁定上游 commit 内）

当前代码与锁定上游 commit `3eb6cd9` 一致；以下修复在复现/验证时被确认生效，
但属于上游已包含的内容，本仓库未新增：

- `top_k == num_experts` 时验证不再误走稀疏裁剪：稀疏分支仅在
  `top_k < num_experts` 时生效（`ultralytics/nn/modules/moe/modules.py:616`）
- `_blend_experts` dtype 不匹配（Half×Float）修复：见
  `experiments/issue54_mot_ablation/REPORT.md`（`block.py` 中 `.to(out.dtype)`）
- MoT 稀疏调度遥测与 warmup 计数器修复：上游提交 `e183ea4`
- 验证器已包含 `LOCAL_RANK` / `torch_distributed_zero_first` 导入，并仅对 `.ndjson`
  数据调用现有转换器（`ultralytics/engine/validator.py`，见 `README_reproduce.md`）

## 如何复现

环境：Windows 本机，Python 3.11.9，torch 2.13.0+cpu（CPU-only）。
复跑分支 `baseline/2026-08-22`；仓库锁定上游 commit
`3eb6cd914b651a06e2cd08ea87d12c28cab95502`（上游 main，2026-08-23）。

MoT 路由 smoke：

```bash
python scripts/diagnose_mot_routing.py --synthetic --device cpu --nc 80
```

MoE 路由 smoke（coco8 自动下载）：

```python
from ultralytics import YOLO
from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker

model = YOLO('ultralytics/cfg/models/master/v0_9/det/yolo-master-n.yaml')
with ExpertUsageTracker(model.model) as tracker:
    model.val(data='coco8.yaml', split='val', batch=1, device='cpu', verbose=False)
    print(tracker.usage_stats)
```

Latent 路由快照：

```python
from ultralytics import YOLO
import torch

model = YOLO('ultralytics/cfg/models/26/yolo26-master-latent-n.yaml')
m = model.model.eval()
with torch.no_grad():
    m(torch.randn(1, 3, 640, 640))

for name, mod in m.named_modules():
    snap = getattr(mod, 'last_routing_snapshot', None)
    if snap:
        print(name, list(snap.keys()))
```

运行原始输出写入 `runs/`（已被 `.gitignore` 忽略，不入库），归档副本见本目录。

## Smoke 测试结果与证据文件

| 文件 | 内容 | 结果 |
| --- | --- | --- |
| `e3_完整终端日志.txt` | 三条 smoke 的完整终端输出 | 三个段落齐全，日志以“日志生成完毕”结尾 |
| `mot_routing_scenarios.csv` | 4 场景 × 3 专家的 `top1_share` 均值 | 全部场景 `LocalConvTransformer=1.00`，其余专家为 0 |
| `mot_routing_detailed.csv` | 逐层逐专家 `top1_share` / `mean_weight` / token 统计 | 与场景表一致 |
| `mot_deformable_activation_check.csv` | Deformable 专家激活显著性检查（baseline vs irregular） | 未训练模型下无显著差异 |
| `mot_expert_heatmap_top1_share.png` | 专家 `top1_share` 热力图 | 图片有效可打开 |

MoE：`model.5.routing` / `model.8.routing` / `model.11.routing` 均显示 `✅ Hooked`；
coco8 val 4/4 跑完；`usage_stats` 输出 hits / weighted_sum 正常数值。
Latent：`model.23/24/25` 的 `last_routing_snapshot` 字段数均为 36，见 `e3_完整终端日志.txt` 的 Latent 段落。

完整原始输出见 `e3_完整终端日志.txt`。

## 其他真实实验

本仓库中与 E3 相关的其他实验（文件均位于锁定上游 commit 内，非本仓库新增；均有仓库证据）：

- 犀牛鸟复现（VisDrone / SKU-110K，120 epoch）：`README_reproduce.md`、`scripts/reproduce/`。
  注意：120 轮原始 `results.csv` 未入库（`artifacts/` 被 gitignore），报告中的表格数字
  无法在仓库内直接核对。
- issue49 垂直数据集基线训练（VisDrone / GlobalWheat2020）：
  `scripts/issue49/`、`scripts/reproduce/results/issue49_*.csv`
- issue52 coco128 专家剪枝与动态调度：
  `reports/moe-pruning/*.csv`、`reports/issues-52-moe-pruning-dynamic-scheduling.md`
  （质量门槛未通过，见下）
- issue53 MoA vs MoE VisDrone 训练验证（50 epoch）：
  `reports/issue-53-training-validation.md`
- issue54 VisDrone MoT/MoA 消融（4 变体，50 epoch，A40）：
  `experiments/issue54_mot_ablation/`（`REPORT.md` 与 4 个 `results.csv` 一致）
- CI 状态：`.github/workflows/ci.yml` 中的 `MixtureDDP` / `MixtureP0Regression` 回归作业
  来自上游（fork 未新增这些作业）。远端 `main` 最近一次 CI 通过（2026-08-25）；
  此前 E3 相关提交曾出现 CI 失败，由后续 `fix(ci)` 提交修复。
  本地 `baseline/2026-08-22` 分支未单独触发 CI。

## 失败/未完成实验

- issue52 剪枝：coco128 上 threshold=0.05 直接剪枝后 mAP50-95 由 0.540 降至 0.113，
  LoRA10 恢复至 0.465；threshold≥0.1 直接归零。所有档位 `quality_gate_pass=false`，
  **质量门槛未通过**。该实验如实保留，不当作成功结论。
- issue54：MoT / MoA 在单帧 VisDrone 检测上均低于 MoE 基线（v10 mAP50 0.20768 最高），
  其价值需在 MOT 跟踪任务上评估，该项未完成。
- 统一 schema / 采集层：未实现，仅有字段对照草案。

## 当前限制

- 当前测试使用未训练（随机初始化）权重，Smoke 只证明代码可运行、数据可采集，不代表训练后行为
- 统一 routing schema 尚未实现，目前只有字段对照草案（见 `E3_准入检查_技术产出.md`）
- 训练后 checkpoint 上的路由行为复测尚未完成
- MoT 在 MOT 跟踪任务上的效果尚未验证
- issue52 剪枝实验未通过质量门槛（`quality_gate_pass=false`，如实保留，不作成功结论）
- hook 开销测量脚本已入库（`scripts/measure_routing_hook_overhead.py`），测量方法见
  `E3_准入检查_技术产出.md`；本次测量数值未单独归档日志，可自行复跑验证
- 运行原始输出在 `runs/`（不入库），仓库内只有本目录的归档副本
- 本仓库不包含预训练权重（`*.pt` 不入库）

## 与上游 YOLO-Master 的关系

本仓库是 [Tencent/YOLO-Master](https://github.com/Tencent/YOLO-Master) 的 fork
（远端 `Aliferous-spec/YOLO-Master`）。

本仓库以锁定上游 commit `3eb6cd9` 为界：

- 上游内容（锁定 commit 中已存在，非本仓库新增）：MoE / MoT / MoA / Latent 模型实现
  与配置、路由诊断脚本 `scripts/diagnose_mot_routing.py`、复现脚本与报告
  （`scripts/reproduce/`、`README_reproduce.md`）、issue49/52/53/54 实验与报告、
  `MixtureDDP` / `MixtureP0Regression` CI 回归作业、ultralytics 8.4.101 迁移基线
  （`reports/migration/v8.4.101-native-baseline.json`）
- 本仓库独有（锁定 commit 中不存在）：本目录的 smoke 证据与文档（终端日志、3 个 CSV、
  热力图 PNG、本 README、`E3_准入检查_技术产出.md`）、hook 开销测量脚本
  （`scripts/measure_routing_hook_overhead.py`）
- 上游 README 中的性能数字（如 42.4% AP @ 1.62ms、+0.8%、17.8% faster）在本仓库
  未独立复现

## 项目结构与版本信息

- 锁定上游 commit：`3eb6cd914b651a06e2cd08ea87d12c28cab95502`（上游 main，2026-08-23）
- 本地复跑分支：`baseline/2026-08-22`（工作区无代码改动，仅文档修订未提交）
- 环境：Python 3.11.9 + torch 2.13.0+cpu
- 字段说明、设计草案与实验细节：见 `E3_准入检查_技术产出.md`
