"""Prepare isolated T4 candidates from the operator's actual path configuration."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .configuration import load_config
from .contracts import write_json
from .performance import PerformanceConfig


def prepare_candidates(config_path, output_dir):
    source = load_config(config_path)  # resolves paths relative to the source, not the new directory
    PerformanceConfig.from_config(source)
    if (source.get("execution_mode") != "formal" or source.get("device") != "cuda"
            or source["recipe"] not in {"B03", "B04"} or source.get("effective_batch_size", 128) != 128):
        raise ValueError("candidates require a formal CUDA B03/B04 configuration with effective batch 128")
    parameters = source.get("parameters", {})
    if (parameters.get("epochs", parameters.get("formal_epochs", 10)) != 10
            or parameters.get("objective", "ce") != "ce"
            or any(parameters.get(key, 0) for key in ("weighting", "lambda_preserve", "prior_tau"))):
        raise ValueError("candidates preserve the ten-epoch plain CE paired controls only")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    variants = [("current", None, None), ("m16-w0", 16, 0), ("m16-w4", 16, 4),
                ("m32-w0", 32, 0), ("m32-w4", 32, 4), ("m16-w2", 16, 2), ("m32-w2", 32, 2)]
    paths = []
    for name, micro, workers in variants:
        for recipe in ("B03", "B04"):
            config = copy.deepcopy(source)
            config["recipe"] = recipe
            config["output"] = str(root / "runs" / f"{recipe}-{name}")
            if micro is not None:
                config["batch_size"] = micro
                config.setdefault("parameters", {}).pop("accumulation_steps", None)
                config["performance"] = {"cache_batch_size": 64, "eval_batch_size": 64,
                    "head_batch_size": 128, "num_workers": workers, "eval_num_workers": 0,
                    "prefetch_factor": 2, "pin_memory": True}
            PerformanceConfig.from_config(config)
            path = root / f"{recipe}-{name}.json"
            write_json(path, config)
            paths.append(str(path))
    write_json(root / "candidates.json", {"source_config": str(Path(config_path).resolve()),
        "candidate_configs": paths, "status": "unmeasured_candidates_not_selected",
        "note": "Cache/head paths are reused, not regenerated. Formal run outputs are isolated. No jobs were started."})
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write T4 candidate configurations only; no reads of images, downloads or jobs")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in prepare_candidates(args.config, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
