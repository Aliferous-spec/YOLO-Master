# LoRA Quickstart Guide — YOLO-Master

> **P2-3**: Minimal-effort fine-tuning with LoRA/MoLoRA on YOLO-Master.

## 1. 30-Second Start

```python
from ultralytics import YOLO
from ultralytics.utils.lora.api import apply_lora

# 1. Load pretrained model
model = YOLO("yolo-master-v0_stable.pt")

# 2. Apply LoRA (r=16 is a good default for detection)
model = apply_lora(model, r=16, alpha=32)

# 3. Train — only LoRA params update, base is frozen
model.train(data="coco128.yaml", epochs=50, imgsz=640, batch=16)
```

That's it. Only ~2-5% of parameters are trainable, saving memory and time.

## 2. MoLoRA (MoE-Aware LoRA)

For MoE models, use MoLoRA to selectively adapt expert layers:

```python
from ultralytics import YOLO
from ultralytics.nn.peft.molora.moe_aware import MoLoRAMoEAwareConfig

model = YOLO("yolo-master-v0_stable.pt")

# MoLoRA: applies LoRA to MoE expert layers specifically
config = MoLoRAMoEAwareConfig(
    r=16,
    alpha=32,
    include_moe=True,        # Target MoE expert layers
    include_attention=False,  # Skip attention (optional)
    lr_mult=3.0,              # Higher LR for LoRA params
)

from ultralytics.utils.lora.api import apply_lora
model = apply_lora(model, config)
model.train(data="coco128.yaml", epochs=50, imgsz=640)
```

## 3. Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r` | 16 | LoRA rank — higher = more capacity, more params |
| `alpha` | 32 | Scaling factor (typically 2× r) |
| `dropout` | 0.0 | Dropout on LoRA layers |
| `include_moe` | True | Apply LoRA to MoE expert layers |
| `include_head` | False | Apply LoRA to detection head |
| `freeze_bn` | True | Freeze BatchNorm stats |
| `lr_mult` | 3.0 | LR multiplier for LoRA params |
| `few_shot_mode` | False | Auto-adjust for small datasets (<100 images) |
| `only_backbone` | False | Only apply to backbone (skip neck/head) |

## 4. Few-Shot Mode

For datasets with <100 images:

```python
model = apply_lora(
    model,
    r=16,
    few_shot_mode=True,       # Auto-increases rank, reduces dropout
    few_shot_adaptive_rank=True,
)
```

## 5. Save & Load LoRA Adapters

```python
# Save — only LoRA weights (small, ~1-5MB)
model.save_lora("my_lora_adapter/")

# Load base + LoRA
model = YOLO("yolo-master-v0_stable.pt")
model.load_lora("my_lora_adapter/")

# Merge LoRA into base for deployment (no runtime overhead)
model.merge_lora()
model.export(format="onnx")
```

## 6. Profile Recommendations

| Scenario | r | alpha | include_moe | lr_mult |
|----------|---|-------|-------------|---------|
| General fine-tune | 16 | 32 | True | 3.0 |
| Few-shot (<100 imgs) | 32 | 64 | True | 5.0 |
| Domain adaptation | 8 | 16 | True | 2.0 |
| MoE expert tuning | 16 | 32 | True | 3.0 |
| Backbone only | 16 | 32 | False | 3.0 |

## 7. Verify LoRA is Working

```python
# Check trainable params
trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.model.parameters())
print(f"Trainable: {trainable}/{total} ({100*trainable/total:.1f}%)")

# Use MoE diagnostics to verify routing is healthy during training
from ultralytics.nn.modules.moe import collect_all_moe_info
info = collect_all_moe_info(model.model)
for name, i in info.items():
    print(f"{name}: {i.class_name}, experts={i.num_experts}, aux={i.aux_loss_value:.4f}")
```
