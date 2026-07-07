# Baseline Training for Vertical Datasets (Issue #49)

Reproducible training workflow for [Tencent/YOLO-Master issue #49](https://github.com/Tencent/YOLO-Master/issues/49), focused on the dense-scene vertical datasets `VisDrone` and `SKU-110K`, with per-epoch logging of the required metrics (`mAP50`, `mAP50-95`, `box_loss`, `cls_loss`, `moe_loss`).

Files in this directory:

- [`yolo_master_issue_49.py`](./yolo_master_issue_49.py): command-line training entry used by the shell commands in this README.
- [`yolo_master_issue_49.ipynb`](./yolo_master_issue_49.ipynb): notebook version of the same workflow and functionality; after completing the environment setup in [Prerequisites](#prerequisites), open it in Jupyter or an IDE notebook runner and execute the cells directly, without using the command-line commands below.

## Prerequisites

This reproduction was run with the following setup:

| Item | Value                                   |
| --- |-----------------------------------------|
| OS | `Windows 10 10.0.19045`                 |
| CPU | `6` physical cores / `12` logical cores |
| GPU | `NVIDIA GeForce RTX 2070 (8GB)`         |
| Python | `CPython 3.11.15`                       |
| PyTorch | `2.5.1`                                 |
| CUDA | `12.1`                                  |
| W&B CLI | `0.28.0`                                |

For environment setup, follow the official repository installation flow and install the current repo in editable mode:

```bash
python -m pip install -e .
```

Optionally, enable Weights & Biases tracking:

```bash
python -m pip install wandb
wandb login
```

Notes:

- If you enable W&B, `wandb login` will prompt for your API key. If needed, create or copy it from your W&B account settings page.
- If you train on GPU, make sure your `torch` build matches the local CUDA environment.

## Dataset Download

The training command already includes dataset download and validation.

```bash
python scripts/issue49/yolo_master_issue_49.py --dataset VisDrone --model YOLO-Master-v0.1-N
```

The script inherits the repository dataset preparation flow and calls `check_det_dataset(..., autodownload=True)` internally, so dataset download and path checks are completed before training starts.

If you want to inspect the built-in datasets or prepare them separately, use the commands below.

List the built-in datasets and models:

```bash
python scripts/issue49/yolo_master_issue_49.py --list-datasets
python scripts/issue49/yolo_master_issue_49.py --list-models
```

Download and prepare `VisDrone` only:

```bash
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('ultralytics/cfg/datasets/VisDrone.yaml', autodownload=True)"
```

Download and prepare `SKU-110K` only:

```bash
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('ultralytics/cfg/datasets/SKU-110K.yaml', autodownload=True)"
```

Notes:

- `VisDrone` is prepared under `datasets/VisDrone/` by default.
- `SKU-110K` is prepared under `datasets/SKU-110K/` by default.
- `VisDrone` requires about `2.3 GB` of disk space.
- `SKU-110K` requires about `13.6 GB` of disk space.


## Training Commands

Base training command:

```bash
python scripts/issue49/yolo_master_issue_49.py --dataset VisDrone --model YOLO-Master-v0.1-N
```

Common optional arguments:

| Flag | Default | Explanation |
| --- | --- | --- |
| `--dataset {VisDrone,SKU-110K}` | `VisDrone` | Select a built-in dataset. |
| `--dataset path/to/data.yaml --dataset-name MyDataset` | off | Use a custom dataset YAML. |
| `--model {YOLO-Master-v0.1-N,YOLO-Master-EsMoE-N}` | `YOLO-Master-v0.1-N` | Select a built-in model. |
| `--model path/to/model.yaml --model-name MyModel` | off | Use a custom model YAML. |
| `--uses-esmoe` | off | Enable ES-MoE-specific handling for a custom ES-MoE model. |
| `--dense-eval-for-esmoe` | off | Enable dense evaluation during validation for ES-MoE models. |
| `--epochs / --imgsz / --batch` | `100 / 640 / 16` | Training epochs, input image size, and batch size. |
| `--device / --workers` | `0 if CUDA else cpu / 4` | Device and DataLoader worker count. On Windows, `0` or `2` is often a good starting point. |
| `--run-tag <tag>` | auto | Custom run tag; defaults to auto-generated names such as `run001`, `run002`, and so on. |
| `--wandb-group <group>` | `visdrone` | W&B group name. |
| `--wandb-mode {online,offline,disabled}` | `online` | W&B mode. `offline` is often useful to reduce online syncing overhead. |
| `--no-wandb` | off | Disable W&B logging. |


## Expected Results

W&B online：[https://wandb.ai/zheliang-/yolo_master_issue49](https://wandb.ai/zheliang-/yolo_master_issue49)

| Dataset | Model | Key Hparams | 	mAP50 | mAP50-95 | W&B | Runtime |
| --- | --- | --- | --- | --- | --- |---------|


