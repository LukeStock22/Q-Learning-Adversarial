# Experiment Structure Update Report

## Goal
Set up a clean, explicit comparison between two training strategies:
- `Nature`: Shelves + Random Forklift(s)
- `Adversary`: Shelves + Pursuit Adversary

This now exists as a first-class experiment mode in the codebase.

## What Changed

### 1) Scenario definitions are now explicit and consistent
Updated `src/qlearning_adversarial/env.py` to support named scenarios:
- `nature` scenario:
  - Fixed shelves
  - Random forklift hazard(s) (`forklift_count`)
- `adversary` scenario:
  - Fixed shelves
  - Strategic pursuit adversary (minimizes Manhattan distance)

Added clear constants and render layers for:
- forklift hazard
- adversary hazard

### 2) Comparison experiment pipeline added
Updated `src/qlearning_adversarial/main.py`:
- Added `comparison` experiment mode.
- Trains two separate policies:
  - policy trained in `nature`
  - policy trained in `adversary`
- Evaluates both policies on both scenarios.
- Writes matrix output: `strategy_comparison.csv`

### 3) Training artifact handling improved
Updated `src/qlearning_adversarial/train.py`:
- Added `artifact_prefix` so each strategy writes separate artifacts:
  - `nature_q_table.pkl`, `adversary_q_table.pkl`
  - `nature_env_config.pkl`, `adversary_env_config.pkl`
- Persisted scenario metadata in env config:
  - `scenario`, `forklift_count`, `adversary_enabled`

### 4) Visualization supports strategy-specific artifacts
Updated `scripts/visualize_result.py`:
- Added `--prefix` argument for selecting which trained policy to replay.
- Examples:
  - `--prefix nature`
  - `--prefix adversary`

### 5) Documentation aligned to new terminology
Updated `README.md`:
- Removed ambiguous wording.
- Added exact scenario definitions.
- Documented comparison workflow and new outputs.
- Documented how to replay each trained policy.

### 6) Dependencies
Updated `requirements.txt`:
- Added `imageio` for GIF generation.

## How to Run the New Experiments

1. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

2. Run comparison experiment:
```bash
PYTHONPATH=src python -m qlearning_adversarial.main
```

3. Inspect outputs:
- `outputs/strategy_comparison.csv`
- `outputs/metrics.txt`
- strategy-specific reward curves and layouts

4. Visualize learned behavior:
```bash
PYTHONPATH=src python scripts/visualize_result.py --prefix nature
PYTHONPATH=src python scripts/visualize_result.py --prefix adversary
```

## Config Profile Used
Current defaults in `configs/default.yaml`:
- `experiment.mode: comparison`
- `run.tier: debug`
  - `train_episodes: 5000`
  - `eval_episodes: 50`
- `training.agent_count: 1`
- `training.num_packages: 1`

Scenario defaults:
- `nature`: `forklift_count: 1`, `forklift_move_prob: 0.5`, `adversary_enabled: false`, `spill_count: 0`
- `adversary`: `forklift_count: 0`, `adversary_enabled: true`, `adversary_random_tiebreak: true`, `spill_count: 0`

## Notes
- Current reward values in `env.py` were retained to stay consistent with recent good behavior:
  - step `-1`, obstacle `-3`, pickup `+10`, delivery `+50`
  - forklift collision `-25`
  - adversary catch `-50`
- Forklift movement model is now explicit:
  - with probability `p = forklift_move_prob`, move to a random valid adjacent cell
  - with probability `1-p`, stay put
- Adversary remains non-learning by design for this comparison baseline, with random tie-break among equally good Manhattan-distance moves.
- `single` mode still exists for quick local checks, but comparison mode is now the default research workflow.
