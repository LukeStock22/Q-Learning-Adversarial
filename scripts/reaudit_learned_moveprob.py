#!/usr/bin/env python3
"""Rerun learned-adversary experiments under matched move-probability settings.

This audit script runs a curated learned-adversary suite in two variants:

- current: current config semantics, with old invalid context expansion disabled
- mp05_skip: adversary move probability forced to 0.5, and no learned-adversary
  Q-update on turns where the adversary is not allowed to move

Outputs are written under outputs/reaudit_learned_moveprob/{current,mp05_skip}/.
The script also generates a markdown and CSV comparison report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
EXPERIMENT_DIR = REPO_ROOT / "configs" / "experiments"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "reaudit_learned_moveprob"
REPORT_DIR = OUTPUT_ROOT / "reports"
SEED_SUFFIX_RE = re.compile(r"_s(\d+)$")

SUITE_BASE_STEMS = [
    "learningadversary",
    "la_ic_ds",
    "la_zs_ic_ds",
    "la_zs_active",
    "la_heuristic_active",
    "la_zs_fast_alpha",
    "la_zs_active_7x7",
    "la_zs_freeze_30k",
    "la_zs_no_rdest",
    "la_zs_prox",
    "la_zs_coarse",
    "la_zs_multi",
    "la_zs_coarse_multi",
    "la_zs_freeze10k",
    "la_zs_multi_freeze20k",
]

EXPERIMENT_LABELS = {
    "learningadversary": "Exploratory learned rerun",
    "la_ic_ds": "Exploratory learned heuristic + context",
    "la_zs_ic_ds": "Exploratory learned zero-sum + context",
    "la_zs_active": "Exp 4 learned active",
    "la_heuristic_active": "Exp 5 learned heuristic-active",
    "la_zs_fast_alpha": "Exp 6 learned fast-alpha",
    "la_zs_active_7x7": "Exp 7 learned 7x7 active",
    "la_zs_freeze_30k": "Exp 8 learned freeze@30k",
    "la_zs_no_rdest": "Exp 9 learned valid baseline",
    "la_zs_prox": "Exp 11 learned proximity",
    "la_zs_coarse": "Exp 13 learned coarse direction",
    "la_zs_multi": "Exp 15 learned multi-layout",
    "la_zs_coarse_multi": "Exp 17 learned coarse+multi",
    "la_zs_freeze10k": "Exp 18 learned freeze@10k",
    "la_zs_multi_freeze20k": "Exp 19 learned multi+freeze@20k",
}

EXPLORATORY_SINGLE_NAMES = {"learningadversary", "la_ic_ds", "la_zs_ic_ds"}

KEY_METRICS = [
    "ood_layout_nature_on_nature_delivered",
    "ood_layout_adversary_on_nature_delivered",
    "ood_layout_nature_on_nature_collisions",
    "ood_layout_adversary_on_nature_collisions",
    "ood_layout_nature_on_nature_avg_steps",
    "ood_layout_adversary_on_nature_avg_steps",
    "ood_layout_nature_on_adversary_delivered",
    "ood_layout_adversary_on_adversary_delivered",
    "id_a_nature_on_nature_delivered",
    "id_a_adversary_on_nature_delivered",
]


@dataclass(frozen=True)
class RunSpec:
    config_name: str
    config_path: Path
    run_name: str
    family: str
    seed: int


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def deep_merge(base: dict, override: dict) -> dict:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merged_config(config_path: Path) -> dict:
    cfg = load_yaml(DEFAULT_CONFIG)
    if config_path.resolve() == DEFAULT_CONFIG.resolve():
        return cfg
    return deep_merge(cfg, load_yaml(config_path))


def discover_runs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for stem in SUITE_BASE_STEMS:
        base_path = EXPERIMENT_DIR / f"{stem}.yaml"
        if base_path.exists():
            merged = merged_config(base_path)
            seed = int(merged.get("project", {}).get("seed", 77))
            run_name = stem if stem in EXPLORATORY_SINGLE_NAMES else f"{stem}_s{seed}"
            specs.append(
                RunSpec(
                    config_name=stem,
                    config_path=base_path,
                    run_name=run_name,
                    family=stem,
                    seed=seed,
                )
            )

        for seed_path in sorted(EXPERIMENT_DIR.glob(f"{stem}_s*.yaml")):
            merged = merged_config(seed_path)
            seed = int(merged.get("project", {}).get("seed", 77))
            specs.append(
                RunSpec(
                    config_name=seed_path.stem,
                    config_path=seed_path,
                    run_name=seed_path.stem,
                    family=stem,
                    seed=seed,
                )
            )

    unique: dict[str, RunSpec] = {spec.run_name: spec for spec in specs}
    return [unique[name] for name in sorted(unique)]


def build_variant_config(spec: RunSpec, variant: str) -> dict:
    cfg = merged_config(spec.config_path)
    project = cfg.setdefault("project", {})
    project["output_dir"] = str((OUTPUT_ROOT / variant / spec.run_name).resolve())

    state_features = cfg.setdefault("state_features", {})
    # Keep all reruns on the corrected, valid state-space side of the codebase.
    state_features["include_relative_package_destination"] = False

    scenarios = cfg.setdefault("scenarios", {})
    adversary = scenarios.setdefault("adversary", {})
    adversary["adversary_learning_update_on_skip"] = variant != "mp05_skip"
    if variant == "mp05_skip":
        adversary["adversary_move_prob"] = 0.5

    return cfg


def write_config(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def run_one(spec: RunSpec, variant: str) -> None:
    cfg = build_variant_config(spec, variant)
    run_root = Path(cfg["project"]["output_dir"])
    metrics_path = run_root / "txt" / "metrics.txt"
    if metrics_path.exists():
        print(f"[skip] {variant}/{spec.run_name}")
        return
    config_path = run_root / "config_used.yaml"
    write_config(config_path, cfg)

    cache_root = (OUTPUT_ROOT / ".cache" / variant).resolve()
    mpl_cache = cache_root / "mpl"
    xdg_cache = cache_root / "xdg"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "src.qlearning_adversarial.main",
        "--config",
        str(config_path),
    ]
    env = {
        **os.environ,
        "MPLCONFIGDIR": str(mpl_cache),
        "XDG_CACHE_HOME": str(xdg_cache),
    }
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def parse_metrics(path: Path) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key in {"experiment_mode", "tier"}:
                metrics[key] = value
                continue
            try:
                if "." in value:
                    metrics[key] = float(value)
                else:
                    metrics[key] = int(value)
            except ValueError:
                metrics[key] = value
    return metrics


def compute_ratio(metrics: dict[str, float | str], numerator: str, denominator: str) -> float | None:
    num = metrics.get(numerator)
    den = metrics.get(denominator)
    if not isinstance(num, (int, float)) or not isinstance(den, (int, float)) or den == 0:
        return None
    return float(num) / float(den)


def add_derived_metrics(metrics: dict[str, float | str]) -> dict[str, float | str]:
    augmented = dict(metrics)
    augmented["ood_ratio_an_over_nn"] = compute_ratio(
        metrics,
        "ood_layout_adversary_on_nature_delivered",
        "ood_layout_nature_on_nature_delivered",
    )
    augmented["ood_ratio_aa_over_na"] = compute_ratio(
        metrics,
        "ood_layout_adversary_on_adversary_delivered",
        "ood_layout_nature_on_adversary_delivered",
    )
    augmented["id_ratio_an_over_nn"] = compute_ratio(
        metrics,
        "id_a_adversary_on_nature_delivered",
        "id_a_nature_on_nature_delivered",
    )
    augmented["id_ratio_aa_over_na"] = compute_ratio(
        metrics,
        "id_a_adversary_on_adversary_delivered",
        "id_a_nature_on_adversary_delivered",
    )
    return augmented


def load_variant_metrics(specs: Iterable[RunSpec], variant: str) -> dict[str, dict[str, float | str]]:
    out: dict[str, dict[str, float | str]] = {}
    for spec in specs:
        path = OUTPUT_ROOT / variant / spec.run_name / "txt" / "metrics.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics for {variant}/{spec.run_name}: {path}")
        out[spec.run_name] = add_derived_metrics(parse_metrics(path))
    return out


def mean_numeric(values: Iterable[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return mean(nums)


def format_num(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def write_csv_report(
    specs: list[RunSpec],
    current_metrics: dict[str, dict[str, float | str]],
    mp05_metrics: dict[str, dict[str, float | str]],
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "per_run_metric_diff.csv"
    metric_keys = sorted(
        set().union(*(metrics.keys() for metrics in current_metrics.values()), *(metrics.keys() for metrics in mp05_metrics.values()))
    )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["family", "run_name", "seed"]
        for key in metric_keys:
            header.extend([f"current:{key}", f"mp05_skip:{key}", f"delta:{key}"])
        writer.writerow(header)

        for spec in specs:
            current = current_metrics[spec.run_name]
            mp05 = mp05_metrics[spec.run_name]
            row = [spec.family, spec.run_name, spec.seed]
            for key in metric_keys:
                cur_val = current.get(key)
                new_val = mp05.get(key)
                delta = None
                if isinstance(cur_val, (int, float)) and isinstance(new_val, (int, float)):
                    delta = float(new_val) - float(cur_val)
                row.extend([cur_val, new_val, delta])
            writer.writerow(row)
    return csv_path


def write_markdown_report(
    specs: list[RunSpec],
    current_metrics: dict[str, dict[str, float | str]],
    mp05_metrics: dict[str, dict[str, float | str]],
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "summary.md"
    families = sorted({spec.family for spec in specs})

    lines: list[str] = []
    lines.append("# Learned-Adversary Move-Probability Reaudit")
    lines.append("")
    lines.append("This report compares two rerun variants for the learned-adversary suite:")
    lines.append("")
    lines.append("- `current`: current config semantics, with invalid relative-package/destination context disabled")
    lines.append("- `mp05_skip`: adversary move probability forced to `0.5`, and skipped learned-adversary turns do not update the adversary Q-table")
    lines.append("")
    lines.append("All runs were executed from the current codebase without editing the paper.")
    lines.append("")

    lines.append("## Suite Overview")
    lines.append("")
    lines.append(f"- Run count: {len(specs)}")
    lines.append(f"- Family count: {len(families)}")
    lines.append("- Primary comparison metrics: OOD `A→N`, OOD `N→N`, their ratio, collisions, and average steps")
    lines.append("")

    lines.append("## Family Summary")
    lines.append("")
    lines.append("| Family | Label | Runs | Current OOD N→N Del | New OOD N→N Del | Current OOD A→N Del | New OOD A→N Del | Current OOD Ratio | New OOD Ratio | Δ OOD Ratio |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for family in families:
        family_specs = [spec for spec in specs if spec.family == family]
        current_vals = [current_metrics[spec.run_name] for spec in family_specs]
        mp05_vals = [mp05_metrics[spec.run_name] for spec in family_specs]

        cur_nn = mean_numeric(
            m.get("ood_layout_nature_on_nature_delivered")
            for m in current_vals
            if isinstance(m.get("ood_layout_nature_on_nature_delivered"), (int, float))
        )
        new_nn = mean_numeric(
            m.get("ood_layout_nature_on_nature_delivered")
            for m in mp05_vals
            if isinstance(m.get("ood_layout_nature_on_nature_delivered"), (int, float))
        )
        cur_an = mean_numeric(
            m.get("ood_layout_adversary_on_nature_delivered")
            for m in current_vals
            if isinstance(m.get("ood_layout_adversary_on_nature_delivered"), (int, float))
        )
        new_an = mean_numeric(
            m.get("ood_layout_adversary_on_nature_delivered")
            for m in mp05_vals
            if isinstance(m.get("ood_layout_adversary_on_nature_delivered"), (int, float))
        )
        cur_ratio = mean_numeric(m.get("ood_ratio_an_over_nn") for m in current_vals)
        new_ratio = mean_numeric(m.get("ood_ratio_an_over_nn") for m in mp05_vals)
        delta_ratio = None if cur_ratio is None or new_ratio is None else new_ratio - cur_ratio

        lines.append(
            f"| `{family}` | {EXPERIMENT_LABELS.get(family, family)} | {len(family_specs)} | "
            f"{format_num(cur_nn, 2)} | {format_num(new_nn, 2)} | {format_num(cur_an, 2)} | "
            f"{format_num(new_an, 2)} | {format_num(cur_ratio, 3)} | {format_num(new_ratio, 3)} | {format_num(delta_ratio, 3)} |"
        )

    lines.append("")
    lines.append("## Detailed Metric Changes")
    lines.append("")
    for family in families:
        family_specs = [spec for spec in specs if spec.family == family]
        lines.append(f"### {EXPERIMENT_LABELS.get(family, family)}")
        lines.append("")
        lines.append("| Run | Seed | Current OOD N→N | New OOD N→N | Current OOD A→N | New OOD A→N | Current Ratio | New Ratio | Δ Ratio | Current OOD A→A | New OOD A→A |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for spec in family_specs:
            current = current_metrics[spec.run_name]
            mp05 = mp05_metrics[spec.run_name]
            cur_ratio = current.get("ood_ratio_an_over_nn")
            new_ratio = mp05.get("ood_ratio_an_over_nn")
            delta_ratio = None
            if isinstance(cur_ratio, (int, float)) and isinstance(new_ratio, (int, float)):
                delta_ratio = float(new_ratio) - float(cur_ratio)
            lines.append(
                f"| `{spec.run_name}` | {spec.seed} | "
                f"{format_num(current.get('ood_layout_nature_on_nature_delivered'), 2)} | "
                f"{format_num(mp05.get('ood_layout_nature_on_nature_delivered'), 2)} | "
                f"{format_num(current.get('ood_layout_adversary_on_nature_delivered'), 2)} | "
                f"{format_num(mp05.get('ood_layout_adversary_on_nature_delivered'), 2)} | "
                f"{format_num(cur_ratio, 3)} | {format_num(new_ratio, 3)} | {format_num(delta_ratio, 3)} | "
                f"{format_num(current.get('ood_layout_adversary_on_adversary_delivered'), 2)} | "
                f"{format_num(mp05.get('ood_layout_adversary_on_adversary_delivered'), 2)} |"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `la_ic_ds`, `la_zs_ic_ds`, and the original Exp 4–8 family were rerun with `include_relative_package_destination=false` to avoid reintroducing the previously invalidated state-space explosion bug.")
    lines.append("- The per-run full metric diff is available in `per_run_metric_diff.csv`.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun learned-adversary experiments under matched move probability.")
    parser.add_argument(
        "--mode",
        choices=["run", "report", "all"],
        default="all",
        help="Run experiments, generate report, or both.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["current", "mp05_skip"],
        default=["current", "mp05_skip"],
        help="Variants to run when mode includes execution.",
    )
    parser.add_argument(
        "--only-family",
        nargs="*",
        default=[],
        help="Optional subset of family stems to run/report.",
    )
    args = parser.parse_args()

    specs = discover_runs()
    if args.only_family:
        allow = set(args.only_family)
        specs = [spec for spec in specs if spec.family in allow]

    if args.mode in {"run", "all"}:
        for variant in args.variants:
            for spec in specs:
                print(f"[{variant}] {spec.run_name}")
                run_one(spec, variant)

    if args.mode in {"report", "all"}:
        current_metrics = load_variant_metrics(specs, "current")
        mp05_metrics = load_variant_metrics(specs, "mp05_skip")
        csv_path = write_csv_report(specs, current_metrics, mp05_metrics)
        md_path = write_markdown_report(specs, current_metrics, mp05_metrics)
        print(f"Wrote report: {md_path}")
        print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
