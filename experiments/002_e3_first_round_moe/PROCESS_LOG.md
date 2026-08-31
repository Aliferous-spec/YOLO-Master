# E3 首轮 MoE 实验过程记录

2026-08-31 晚上，第一次在这个仓库跑 MoE 路由观测实验。

## 做了什么

- 模型：`ultralytics/cfg/models/master/v0_9/det/yolo-master-n.yaml`（MoE v0_9，backbone 里有 3 个 `DetailAwareLowRankHybridAdaptiveGateMoE`）
- 数据：`ultralytics/cfg/datasets/coco8.yaml` 的 `val` split，只有 4 张图
- 跑了两遍验证：baseline（纯验证，不挂 hook）和 E3（验证 + `ExpertUsageTracker` 挂 hook 观测路由）
- 两遍用同一份权重：模型只从 YAML 随机初始化一次，存下 state_dict，baseline 跑之前 load 回去；E3 复用同一个 model 对象（代码注释里说明两遍在同一份权重上跑）
- 参数：`imgsz 640 / batch 1 / device cpu / seed 0 / conf 0.001 / iou 0.7 / max_det 300`

## 踩过的坑（按时间顺序）

1. **模型路径少了分隔符**：一开始命令行里写的是 `...v0_9\detyolo-master-n.yaml`，少了个反斜杠。仓库里实际文件是 `det\yolo-master-n.yaml`，`detyolo-master-n.yaml` 这个路径根本不存在，对着文件列表才发现。
2. **venv 里的 ultralytics 不是仓库本体**：`python -c "import ultralytics"` 解析到的是 `D:\Claude_Workspace\projects\YOLO-Master-review`，不是 `D:\YOLO-Master`。不处理的话跑的就是另一份代码。脚本里加了 import gate：import 后检查 `ultralytics.__file__` 是否在仓库根下，不在就抛错。
3. **第一次跑（22:56）生成的 `comparison.json` 少了 `value: null`**：`routing_fields_availability` 里只写了 `available: false`，和约定格式不一致。改脚本补上后 23:08 重跑。
4. **`stdout.log` 进不了提交**：`.gitignore` 第 64 行是 `*.log`，完整运行日志被忽略，只留在本地；提交里没有它。
5. **给 `analysis.md` 加过程记录引用时重跑了好几趟**：为了让骨架里能引用这份 `PROCESS_LOG.md`，23:33 跑了一趟（当时文件还没放进去，没引用上），之后带 `--force` 又跑了两趟。`--force` 会把整个实验目录删掉重建，头一回把 `PROCESS_LOG.md` 也删了，后来给脚本加了"重跑时保留 PROCESS_LOG.md"的逻辑。最后一趟（23:35:07）是提交进仓库的版本，命令里带 `--force`（见 `config.yaml` 的 command 字段）。

## 结果（都能对上数据）

- 两个 arm 的 `mAP50 / mAP50-95 / precision / recall` 全是 0。模型是随机初始化、完全没训练，所以这不是"模型不行"，是本来就还没学。数值见 `baseline.json` 和 `e3.json`。
- 3 个路由层都被 hook 到了：`model.5.routing`、`model.8.routing`、`model.11.routing`，模块类型都是 `DualStreamGateRouter`。
- 整个验证只统计到 15 个 token（`e3.json` 的 `aggregate.total_tokens = 15`）。样本这么小，`usage_gini`、`dominant_share` 这些只能算"代码能跑、hook 能采数"的证据，不能拿去做结论。
- 概率类字段拿不到：`DualStreamGateRouter` 在 eval 模式只返回 top-k 的 weights/indices，内部算出来的全量 softmax（`probs`）不对外暴露。所以 `routing_probability_mean` 和 `routing_entropy` 按约定记成 `available: false, value: null`，没有编数据。

## 一个还没查的怪现象

- 同样的命令、同样的 seed、同样的随机初始化权重（state_dict sha256 每次都一样），不同趟跑出来的 `mean gini` 不完全一样：23:08 和最后一趟是 `0.6625`，23:33 和中间一趟是 `0.6167`。路由统计并不是每次都完全一样。原因还没查，先记录在这里（可能是数据加载顺序或 torch 随机状态的问题）。

## 没做的 / 下不了结论的

- 没有训练、没有 checkpoint，所以没有任何关于模型质量或 MoE 效果的结论。
- 概率口径的 entropy、训练后会不会 collapse，都还没有数据。
- 4 张图、15 个 token 的统计不说明任何模型质量问题。