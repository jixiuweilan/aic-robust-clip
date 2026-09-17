"""Thin commands over the shared preflight/workflow library."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .contracts import write_json


def _execute(action):
    try:
        result = action()
    except (ValueError, RuntimeError, OSError, KeyError, ImportError, TypeError) as exc:
        print(f"failed (no automatic retry): {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


def _parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path)
    return parser


def doctor_main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect environment/inputs without training or implicit setup")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bind-training", type=Path,
                        help="explicitly enroll a separate CUDA-visible RTX 4060 or Tesla T4 training host")
    parser.add_argument("--record-lock", type=Path, help="record installed dependencies, not GPU-validation success")
    args = parser.parse_args(argv)
    def action():
        from .environment import bind_training, environment_report
        report = bind_training(args.bind_training) if args.bind_training else environment_report()
        if args.config:
            from .configuration import prepare
            report["configuration"] = prepare(args.config).summary()
        for path in (args.output, args.record_lock):
            if path is not None and path.exists():
                raise FileExistsError(path)
        if args.output:
            write_json(args.output, report)
        if args.record_lock:
            freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
            args.record_lock.parent.mkdir(parents=True, exist_ok=True)
            args.record_lock.write_text(f"# Installed environment on {report['platform']}\n# Python {sys.version.split()[0]}\n" + freeze.stdout, encoding="utf-8")
        return report
    return _execute(action)


def provision_main(argv=None):
    parser = argparse.ArgumentParser(description="Explicitly provision the pinned official OpenAI CLIP snapshot")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from .models.provision import provision_weights
    return _execute(lambda: provision_weights(args.output, args.revision))


def model_smoke_main(argv=None):
    parser = argparse.ArgumentParser(description="Two official-CLIP updates on generated images only; no downloads")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--recipe", choices=("B01", "B04", "B03", "R01", "F100", "F010", "F001"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    def action():
        if args.output.exists():
            raise FileExistsError(args.output)
        from .training.model_smoke import check_official_model
        result = check_official_model(args.weights, args.revision, args.recipe, device=args.device)
        write_json(args.output, result)
        return result
    return _execute(action)


def cache_main(argv=None):
    parser = _parser("Generate fixed-view train/dev features; no implicit audit or training")
    parser.add_argument("--partition", choices=("train", "dev"), required=True)
    args = parser.parse_args(argv)
    from .workflow import cache_command
    return _execute(lambda: cache_command(args.config, args.partition))


def init_head_main(argv=None):
    args = _parser("Explicit shared-head initialization; bounded smoke or enrolled experiment machine").parse_args(argv)
    from .workflow import init_head_command
    return _execute(lambda: init_head_command(args.config))


def train_main(argv=None):
    parser = _parser("Train a resolved recipe; local execution is bounded smoke only")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-updates", type=int)
    args = parser.parse_args(argv)
    from .workflow import train_command
    return _execute(lambda: train_command(args.config, resume=args.resume, stop_after_updates=args.stop_after_updates))


def evaluate_main(argv=None):
    parser = _parser("Read-only dev or explicitly locked confirm evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition", choices=("dev", "confirm"), required=True)
    parser.add_argument("--selection-record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from .workflow import evaluate_command
    return _execute(lambda: evaluate_command(args.config, args.checkpoint, args.partition, args.output, selection=args.selection_record))


def lock_selection_main(argv=None):
    parser = _parser("Explicitly freeze one checkpoint/recipe decision before confirm or formal prediction")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from .workflow import lock_selection_command
    return _execute(lambda: lock_selection_command(args.config, args.checkpoint, args.output))


def predict_main(argv=None):
    parser = _parser("Single-checkpoint prediction and validated packaging; no test-time fitting")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--selection-record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from .workflow import predict_command
    return _execute(lambda: predict_command(args.config, args.checkpoint, args.test_manifest, args.output, selection=args.selection_record))
