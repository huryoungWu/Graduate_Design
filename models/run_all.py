# -*- coding: utf-8 -*-
"""按类别批量运行所有模型。

用法：
  python -m models.run_all --stage train --group forecasting
  python -m models.run_all --stage train --group reconstruction
  python -m models.run_all --stage detect --group all
  python -m models.run_all --group reconstruction --models ae vae --stage train
  python -m models.run_all --smoke --stage train --group forecasting
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "forecasting": [
        "lstm", "rnn", "transformer", "itransformer", "tcn",
        "timesnet", "micn", "moderntcn", "graph_wavenet",
        "seq2seq_lstm",
    ],
    "reconstruction": [
        "ae", "vae", "usad", "lstm_vae",
        "anomaly_transformer", "dcdetector", "sbt",
    ],
    "classical": [
        "isolation_forest", "dagmm", "adt",
    ],
}


def main() -> int:
    p = argparse.ArgumentParser(description="批量运行预测类/重构类模型")
    p.add_argument("--group",
                   choices=["forecasting", "reconstruction", "classical", "all"],
                   default="all")
    p.add_argument("--models", nargs="*", default=None,
                   help="只运行指定模型名（如 lstm tcn ae）")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--stage", choices=["all", "train", "detect"],
                   default="all",
                   help="all=训练+检测；train=只训练；detect=只检测")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    groups = MODELS.keys() if args.group == "all" else [args.group]
    for group in groups:
        for name in MODELS[group]:
            if args.models and name not in args.models:
                continue
            cmd = [sys.executable, "-m", f"models.{group}.{name}",
                   "--stage", args.stage]
            if args.smoke:
                cmd.append("--smoke")
            if args.no_plot:
                cmd.append("--no-plot")
            print("\n" + "#" * 80)
            print(">>>", " ".join(cmd))
            code = subprocess.run(cmd, cwd=ROOT).returncode
            if code != 0 and not args.continue_on_error:
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
