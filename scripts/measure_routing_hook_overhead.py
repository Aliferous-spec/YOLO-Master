#!/usr/bin/env python3
"""Measure the runtime overhead of the MoE ExpertUsageTracker hook.

Runs ``N`` forward passes of a YOLO-Master MoE model with and without the
``ExpertUsageTracker`` hook active and reports the relative overhead. Only
forward + hook callback time is measured (data loading and postprocessing are
excluded).

Example:
    python scripts/measure_routing_hook_overhead.py --iterations 50
"""

from __future__ import annotations

import argparse
import time

import torch

from ultralytics import YOLO
from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="ultralytics/cfg/models/master/v0_9/det/yolo-master-n.yaml",
        help="MoE model YAML to measure",
    )
    parser.add_argument("--iterations", type=int, default=50, help="forward passes per arm")
    parser.add_argument("--warmup", type=int, default=3, help="warmup forwards before timing")
    parser.add_argument("--size", type=int, default=640, help="input resolution (square)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    m = model.model.eval()
    x = torch.randn(1, 3, args.size, args.size)

    with torch.no_grad():
        for _ in range(args.warmup):
            m(x)

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(args.iterations):
            m(x)
        t_off = time.perf_counter() - t0

    with ExpertUsageTracker(m):
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(args.iterations):
                m(x)
            t_on = time.perf_counter() - t0

    overhead = (t_on - t_off) / t_off * 100.0
    print(f"without hook: {t_off:.3f}s ({t_off / args.iterations * 1000:.1f} ms/forward)")
    print(f"with hook:    {t_on:.3f}s ({t_on / args.iterations * 1000:.1f} ms/forward)")
    print(f"overhead:     {overhead:.2f}% (target < 10%)")


if __name__ == "__main__":
    main()
