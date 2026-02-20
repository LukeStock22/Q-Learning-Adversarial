#!/usr/bin/env python3
"""Run default or named experiments from base + overrides."""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
from pathlib import Path

import yaml

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base and return merged dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def run_main(config_path: Path) -> None:
    cmd = [
        "python",
        "-m",
        "qlearning_adversarial.main",
        "--config",
        str(config_path),
    ]
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": "src"})


def run_one_experiment(name: str, merged_cfg: dict) -> None:
    """Run one merged experiment config."""
    merged_cfg = copy.deepcopy(merged_cfg)
    merged_cfg.setdefault("project", {})

    experiment_root = Path("outputs") / name
    merged_cfg["project"]["output_dir"] = str(experiment_root)
    config_path = experiment_root / "config_used.yaml"
    save_yaml(config_path, merged_cfg)
    print(f"\n[RUN] {name}")
    run_main(config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--base", default="configs/default.yaml", help="Base config path")
    parser.add_argument(
        "--manifest",
        default="configs/experiments/manifest.yaml",
        help="Manifest listing experiments",
    )
    parser.add_argument(
        "--experiments-dir",
        default="configs/experiments",
        help="Directory containing override YAML files",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=[],
        help="Run only these experiment names (without .yaml), e.g. --only baseline lower_forklift_move",
    )
    parser.add_argument(
        "--use-manifest",
        action="store_true",
        help="Run experiments listed in manifest.yaml",
    )
    args = parser.parse_args()

    base_path = Path(args.base)
    manifest_path = Path(args.manifest)
    experiments_dir = Path(args.experiments_dir)
    base_cfg = load_yaml(base_path)

    if args.only:
        names = args.only
    elif args.use_manifest:
        manifest = load_yaml(manifest_path)
        names = manifest.get("experiments", [])
    else:
        names = []

    if not names:
        run_one_experiment(name="default", merged_cfg=base_cfg)
        return

    for name in names:
        override_path = experiments_dir / f"{name}.yaml"
        override_cfg = load_yaml(override_path)
        merged_cfg = deep_merge(base_cfg, override_cfg)
        run_one_experiment(name=name, merged_cfg=merged_cfg)


if __name__ == "__main__":
    main()
