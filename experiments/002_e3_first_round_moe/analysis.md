# 实验记录骨架

- 实验名称: 002_e3_first_round_moe
- 实验时间: 2026-08-31T23:08:08+08:00
- git commit: 1c4669f9ee44c1c5e5759d93b49e68ce2531d3b3
- git dirty: True
- checkpoint: random_init_from_yaml (sha256: a90ea997903241cf6e837520e6fcaf0ba9159e96ae894b67624b556dd5574267)
- dataset: ultralytics\cfg\datasets\coco8.yaml
- split: val
- imgsz: 640
- batch: 1
- device: cpu
- seed: 0
- dtype: torch.float32
- warmup: 0
- repeat: 1

## 当前实验目标

建立 MoE v0_9 在 coco8 val 上的 baseline 与 E3（仅增加路由观测）首轮可比运行记录。

## 当前已知限制

- MoE v0_9 的 `DualStreamGateRouter` 在 eval 路径只返回 top-k，不暴露全量 softmax 概率；`routing_probability_mean` 与概率口径 `routing_entropy` 记为 `available: false, value: null`。
- 模型为随机初始化权重，mAP 不代表模型质量。

## 结果

- baseline: `baseline.json`
- e3: `e3.json`
- comparison: `comparison.json`
- 指标数值以 JSON 为准，本文件不重复。

## 备注

（留空：脚本不自动生成结论。）
