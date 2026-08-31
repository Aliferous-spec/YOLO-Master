#!/usr/bin/env python3
"""E3 first-round experiment entry.

Runs two strictly comparable validation arms for one YOLO-Master model on a
dataset val split (default coco8.yaml):

- baseline arm: plain Ultralytics validation, no hooks.
- E3 arm: identical validation plus routing observation only
  (``ExpertUsageTracker`` on MoE routers).

Both arms use the exact same weights: the random-init model is built once, its
state_dict is captured, and it is restored before each arm. Outputs go to
``experiments/<name>/``: config.yaml, baseline.json, e3.json, comparison.json,
analysis.md (record skeleton only), stdout.log.

No YOLO-Master core code is modified. No experiment condition is adjusted to
improve results. Probability-based routing fields the current code does not
expose are recorded as ``{"available": false, "value": null}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


class Tee:
    """Duplicate writes to the console and a log file."""

    def __init__(self, *streams):
        self.streams = list(streams)

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except ValueError:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except ValueError:
                pass

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="002_e3_first_round_moe", help="experiment directory name under experiments/")
    parser.add_argument("--model", default="ultralytics/cfg/models/master/v0_9/det/yolo-master-n.yaml")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/coco8.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--force", action="store_true", help="overwrite an existing experiment directory")
    return parser.parse_args()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def git_info() -> dict:
    """Return git commit + dirty state for the repo root (read-only)."""

    def run(*cmd):
        return subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
        )

    head = run("git", "rev-parse", "HEAD")
    if head.returncode != 0:
        raise RuntimeError(f"git rev-parse failed: {head.stderr.strip()}")
    status = run("git", "status", "--porcelain")
    if status.returncode != 0:
        raise RuntimeError(f"git status failed: {status.stderr.strip()}")
    dirty_files = [line for line in status.stdout.splitlines() if line.strip()]
    return {"commit": head.stdout.strip(), "dirty": bool(dirty_files), "dirty_files": dirty_files[:50]}


def state_dict_sha256(state_dict) -> str:
    import numpy as np  # noqa: F401  (numpy ships with the torch runtime)

    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        digest.update(key.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(state_dict[key].detach().cpu().contiguous().numpy().tobytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def _entropy(share: list[float]) -> float:
    """Shannon entropy (natural log) over a normalized usage share vector."""
    total = float(sum(share))
    if total <= 0.0:
        return 0.0
    eps = 1e-12
    value = 0.0
    for p in share:
        q = p / total
        if q > eps:
            value -= q * math.log(q)
    return float(value)


def run_val(model, val_args: dict) -> dict:
    """Run one official Ultralytics validation pass and collect metrics + speed."""
    args = {**getattr(model, "overrides", {}), "rect": True, **val_args, "mode": "val"}
    validator = model._smart_load("validator")(args=args, _callbacks=model.callbacks)
    validator(model=model.model)
    metrics = validator.metrics
    if metrics is None:
        raise RuntimeError("validation returned no metrics (validator.metrics is None)")
    box = metrics.box
    speed = dict(validator.speed or {})
    n_images = None
    try:
        n_images = len(validator.dataloader.dataset)
    except Exception:  # noqa: BLE001
        n_images = None
    return {
        "metrics": {
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
        },
        "latency_ms_per_image": {key: float(value) for key, value in speed.items()},
        "latency_total_ms_per_image": float(sum(speed.values())),
        "validation_images": n_images,
        "save_dir": str(validator.save_dir),
    }


def moe_routing_fields(model, tracker) -> dict:
    """Build per-layer routing fields from ExpertUsageTracker usage stats.

    Usage-derived fields come from hits / weighted top-k sums. Probability-based
    fields (full router softmax) are not exposed by the current eval path for
    MoE v0_9 routers, so they are recorded as unavailable (no fabricated data).
    """
    import torch  # noqa: F401  (torch available in the experiment runtime)

    from ultralytics.nn.modules.moe.protocol import usage_gini

    modules = {name: module for name, module in model.model.named_modules()}
    layers = []
    for name in sorted(tracker.usage_stats.keys()):
        module = modules.get(name)
        num_experts = int(getattr(module, "num_experts", 0))
        top_k = int(getattr(module, "top_k", 0))
        counts = [0.0] * num_experts
        weighted = [0.0] * num_experts
        for expert_id, stats in tracker.usage_stats[name].items():
            idx = int(expert_id)
            if 0 <= idx < num_experts:
                counts[idx] += float(stats.hits)
                weighted[idx] += float(stats.weighted_sum)
        total_hits = float(sum(counts))
        share = [c / total_hits if total_hits > 0 else 0.0 for c in counts]
        dominant = max(range(num_experts), key=lambda i: counts[i]) if num_experts else -1
        mean_topk = [w / c if c > 0 else 0.0 for w, c in zip(weighted, counts)]
        layers.append(
            {
                "routing_layer": name,
                "module_type": type(module).__name__ if module is not None else None,
                "num_experts": num_experts,
                "top_k": top_k,
                "expert_usage_counts": counts,
                "expert_usage_share": share,
                "mean_topk_weight": mean_topk,
                "usage_gini": usage_gini(counts),
                "usage_entropy": _entropy(share),
                "dominant_expert": dominant,
                "dominant_share": share[dominant] if dominant >= 0 else 0.0,
                "routing_probability_mean": {
                    "available": False,
                    "value": None,
                    "note": "MoE v0_9 router eval path returns top-k only; full softmax probabilities are not exposed",
                },
                "routing_entropy": {
                    "available": False,
                    "value": None,
                    "note": "not derivable without full routing probabilities",
                },
                "basis": "usage_counts",
            }
        )
    mean_gini = sum(float(layer["usage_gini"]) for layer in layers) / len(layers) if layers else 0.0
    mean_dominant_share = sum(float(layer["dominant_share"]) for layer in layers) / len(layers) if layers else 0.0
    return {
        "observation_source": "ExpertUsageTracker",
        "entropy_definition": "Shannon entropy (natural log) over expert_usage_share",
        "gini_function": "usage_gini (ultralytics/nn/modules/moe/protocol.py)",
        "layers": layers,
        "aggregate": {
            "routed_layers": len(layers),
            "collapsed_layers": sum(1 for layer in layers if float(layer["dominant_share"]) >= 0.8),
            "mean_gini": float(mean_gini),
            "mean_dominant_share": float(mean_dominant_share),
            "total_tokens": int(getattr(tracker, "total_tokens", 0)),
        },
    }


def _delta(baseline_value, e3_value):
    delta_abs = e3_value - baseline_value
    delta_rel = delta_abs / baseline_value if baseline_value not in (None, 0) else None
    return {
        "baseline": baseline_value,
        "e3": e3_value,
        "delta_absolute": delta_abs,
        "delta_relative": delta_rel,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_analysis_skeleton(args, exp_dir: Path, run_meta: dict, git: dict, checkpoint: dict, failure=None) -> None:
    lines = [
        "# 实验记录骨架",
        "",
        f"- 实验名称: {args.name}",
        f"- 实验时间: {run_meta['timestamp']['start']}",
        f"- git commit: {git['commit']}",
        f"- git dirty: {git['dirty']}",
        f"- checkpoint: {checkpoint['source']} (sha256: {checkpoint['sha256']})",
        f"- dataset: {args.data}",
        f"- split: {args.split}",
        f"- imgsz: {args.imgsz}",
        f"- batch: {args.batch}",
        f"- device: {args.device}",
        f"- seed: {args.seed}",
        f"- dtype: {run_meta.get('dtype', 'TBD')}",
        f"- warmup: {run_meta.get('warmup', 'TBD')}",
        f"- repeat: {run_meta.get('repeat', 'TBD')}",
        "",
        "## 当前实验目标",
        "",
        "建立 MoE v0_9 在 coco8 val 上的 baseline 与 E3（仅增加路由观测）首轮可比运行记录。",
        "",
        "## 当前已知限制",
        "",
        "- MoE v0_9 的 `DualStreamGateRouter` 在 eval 路径只返回 top-k，不暴露全量 softmax 概率；`routing_probability_mean` 与概率口径 `routing_entropy` 记为 `available: false, value: null`。",
        "- 模型为随机初始化权重，mAP 不代表模型质量。",
    ]
    if failure is not None:
        lines += [
            "",
            "## 运行状态",
            "",
            "- 状态: failed",
            f"- 失败原因: {failure}",
            "- 详细堆栈见 `stdout.log`。",
            "",
            "## 结果",
            "",
            "- 本轮未产出有效数据；指标与路由字段均为 TBD，详见 `stdout.log`。",
        ]
    else:
        lines += [
            "",
            "## 结果",
            "",
            "- baseline: `baseline.json`",
            "- e3: `e3.json`",
            "- comparison: `comparison.json`",
            "- 指标数值以 JSON 为准，本文件不重复。",
        ]
    lines += [
        "",
        "## 备注",
        "",
        "（留空：脚本不自动生成结论。）",
        "",
    ]
    (exp_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(args: argparse.Namespace, exp_dir: Path) -> None:
    # Repo-root first on sys.path: running `python scripts/...` sets sys.path[0] to scripts/.
    sys.path.insert(0, str(REPO_ROOT))
    # Import ultralytics after stdout/stderr redirection so its LOGGER is captured.
    import torch

    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker

    if str(REPO_ROOT).lower() not in str(Path(ultralytics.__file__).resolve()).lower():
        raise RuntimeError(
            f"import gate failed: ultralytics resolved to {ultralytics.__file__}, "
            f"expected under {REPO_ROOT}. Run from the repo root or with the repo on sys.path."
        )

    git = git_info()
    model_cfg = _resolve(REPO_ROOT, args.model)
    data_cfg = _resolve(REPO_ROOT, args.data)
    for path, label in ((model_cfg, "model"), (data_cfg, "data")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} YAML not found: {path}")

    ts_start = _now()
    torch.manual_seed(args.seed)
    model = YOLO(str(model_cfg))
    model.model.eval()
    dtype = str(next(model.model.parameters()).dtype)

    state = {key: value.detach().cpu().clone() for key, value in model.model.state_dict().items()}
    state_sha = state_dict_sha256(state)
    checkpoint = {
        "source": "random_init_from_yaml",
        "path": None,
        "sha256": state_sha,
        "arms_run_on": "same_fused_model_object",
    }

    val_args = {
        "data": str(data_cfg),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "seed": args.seed,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "plots": False,
        "verbose": True,
        "project": str(REPO_ROOT / "runs" / "baseline"),
        "name": args.name,
    }

    # --- baseline arm: plain validation, no hooks ---
    model.model.load_state_dict(state, strict=True)
    torch.manual_seed(args.seed)
    print("\n=== baseline arm: plain validation (no hooks) ===")
    t0 = time.perf_counter()
    baseline = run_val(model, val_args)
    baseline_duration = time.perf_counter() - t0
    print(f"baseline arm done in {baseline_duration:.1f}s")

    # --- E3 arm: identical validation + routing observation only ---
    # The baseline validator fuses Conv+BN in place (ultralytics AutoBackend fuse=True).
    # Reuse the SAME fused model object: both arms execute on byte-identical weights,
    # so no state reload is performed here.
    torch.manual_seed(args.seed)
    print("\n=== E3 arm: validation with ExpertUsageTracker (observation only) ===")
    t0 = time.perf_counter()
    with ExpertUsageTracker(model.model) as tracker:
        e3 = run_val(model, val_args)
        routing = moe_routing_fields(model, tracker)
    e3_duration = time.perf_counter() - t0
    print(f"e3 arm done in {e3_duration:.1f}s")

    ts_end = _now()
    run_meta = {
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "git_dirty_files": git["dirty_files"],
        "checkpoint": checkpoint,
        "model_config": str(model_cfg),
        "dataset": str(data_cfg),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "dtype": dtype,
        "seed": args.seed,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "warmup": 0,
        "repeat": 1,
        "observer": "ExpertUsageTracker (MoE routers)",
        "timestamp": {"start": ts_start, "end": ts_end},
        "command": " ".join(sys.argv),
    }

    baseline_payload = {
        "status": "ok",
        "arm": "baseline",
        **run_meta,
        "metrics": baseline["metrics"],
        "latency_ms_per_image": baseline["latency_ms_per_image"],
        "latency_total_ms_per_image": baseline["latency_total_ms_per_image"],
        "validation_images": baseline["validation_images"],
        "duration_s": round(baseline_duration, 3),
    }
    e3_payload = {
        "status": "ok",
        "arm": "e3",
        **run_meta,
        "metrics": e3["metrics"],
        "latency_ms_per_image": e3["latency_ms_per_image"],
        "latency_total_ms_per_image": e3["latency_total_ms_per_image"],
        "validation_images": e3["validation_images"],
        "duration_s": round(e3_duration, 3),
        "routing": routing,
    }

    metric_keys = ["mAP50", "mAP50-95", "precision", "recall"]
    comparison_payload = {
        "status": "ok",
        "experiment": {
            key: run_meta[key]
            for key in (
                "git_commit",
                "git_dirty",
                "checkpoint",
                "model_config",
                "dataset",
                "split",
                "imgsz",
                "batch",
                "device",
                "dtype",
                "seed",
                "conf",
                "iou",
                "max_det",
                "warmup",
                "repeat",
                "observer",
                "timestamp",
                "command",
            )
        },
        "metrics": {key: _delta(baseline["metrics"][key], e3["metrics"][key]) for key in metric_keys},
        "latency": {
            "inference_ms_per_image": _delta(
                baseline["latency_ms_per_image"].get("inference", 0.0),
                e3["latency_ms_per_image"].get("inference", 0.0),
            ),
            "total_ms_per_image": _delta(baseline["latency_total_ms_per_image"], e3["latency_total_ms_per_image"]),
        },
        "routing_fields_availability": {
            "routing_probability_mean": {
                "available": False,
                "value": None,
                "note": "not exposed by current MoE v0_9 eval path",
            },
            "routing_entropy": {
                "available": False,
                "value": None,
                "note": "not derivable without full routing probabilities",
            },
        },
    }

    config_payload = {
        "experiment": {
            "name": args.name,
            "created_at": ts_start,
            "status": "ok",
            "command": " ".join(sys.argv),
        },
        "run_parameters": {
            "model": str(model_cfg),
            "data": str(data_cfg),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "dtype": dtype,
            "seed": args.seed,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "warmup": 0,
            "repeat": 1,
        },
        "git": git,
        "checkpoint": checkpoint,
        "observer": "ExpertUsageTracker (MoE routers)",
        "outputs": [
            "config.yaml",
            "baseline.json",
            "e3.json",
            "comparison.json",
            "analysis.md",
            "stdout.log",
        ],
    }

    write_json(exp_dir / "baseline.json", baseline_payload)
    write_json(exp_dir / "e3.json", e3_payload)
    write_json(exp_dir / "comparison.json", comparison_payload)
    (exp_dir / "config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_analysis_skeleton(args, exp_dir, run_meta, git, checkpoint)

    print("\n=== experiment complete ===")
    print(f"outputs: {exp_dir}")
    print(f"  baseline metrics: {baseline['metrics']}")
    print(f"  e3 metrics:       {e3['metrics']}")
    print(f"  routed layers:    {routing['aggregate']['routed_layers']}")
    print(f"  mean gini:        {routing['aggregate']['mean_gini']:.4f}")


def write_failure_artifacts(args: argparse.Namespace, exp_dir: Path, exc: Exception) -> None:
    error = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    try:
        git = git_info()
    except Exception:  # noqa: BLE001
        git = {"commit": "unknown", "dirty": None, "dirty_files": []}
    run_meta = {
        "timestamp": {"start": _now(), "end": _now()},
        "dtype": "TBD",
        "warmup": 0,
        "repeat": 1,
    }
    checkpoint = {"source": "random_init_from_yaml", "path": None, "sha256": "TBD"}
    (exp_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {"experiment": {"name": args.name, "status": "failed", "command": " ".join(sys.argv)}, "git": git},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    for name in ("baseline.json", "e3.json", "comparison.json"):
        path = exp_dir / name
        if not path.exists():
            write_json(path, dict(error))
    write_analysis_skeleton(args, exp_dir, run_meta, git, checkpoint, failure=str(exc))


def main() -> int:
    args = parse_args()
    exp_dir = EXPERIMENTS_DIR / args.name
    exp_dir = exp_dir.resolve()
    if not exp_dir.is_relative_to(EXPERIMENTS_DIR.resolve()):
        print(f"ERROR: experiment path escapes experiments/: {exp_dir}", file=sys.stderr)
        return 2
    if exp_dir.exists():
        if not args.force:
            print(
                f"ERROR: experiment directory already exists: {exp_dir}\nUse --force to overwrite it explicitly.",
                file=sys.stderr,
            )
            return 2
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = exp_dir / "stdout.log"
    log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 - kept open for the Tee stream
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = Tee(real_out, log_file)
    sys.stderr = Tee(real_err, log_file)
    try:
        run_experiment(args, exp_dir)
        return 0
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - deliberate top-level failure boundary
        print("\n=== EXPERIMENT FAILED ===")
        traceback.print_exc()
        print(f"failure: {exc}")
        write_failure_artifacts(args, exp_dir, exc)
        return 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
