# Q-Learning-Adversarial

Tabular Q-learning in a grid warehouse with two training regimes:
- `nature` training: static shelves + stochastic forklifts
- `adversary` training: static shelves + active adversary

The project compares which training regime transfers better under ID-A and OOD-layout evaluation.

## Terminology

- `Policy`: the trained Q-table for the delivery agent.
- `Nature policy`: agent policy trained in the `nature` scenario.
- `Adversary policy`: agent policy trained in the `adversary` scenario.
- `Scenario`: environment disturbance model used during rollout.
  - `nature`: shelves + forklift hazards
  - `adversary`: shelves + adversary hazard
- `Shelf`: static obstacle (2x1 block) fixed for a run.
- `Forklift`: stochastic hazard; each step it moves to a random valid neighbor with probability `forklift_move_prob`, else stays.
- `Adversary`:
  - `deterministic`: greedy pursuit toward the agent (Manhattan distance), random tie-breaks
  - `learning`: separate tabular learner with configurable objective (`heuristic` or `zero_sum`)
- `ID-A`: evaluation on the same base layout used for training in that run.
- `OOD-layout`: evaluation on unseen layouts sampled from the same generation rules (default: 5 layouts).
- `Cross-scenario evaluation`: each trained policy is evaluated on both scenarios (`nature`, `adversary`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Experiments

### 1) Run baseline only

```bash
python scripts/run_experiments.py
```

This runs `configs/default.yaml` and writes results to:
- `outputs/default/`

### 2) Create a new experiment override

1. Copy template:
```bash
cp configs/experiments/EXPERIMENT_TEMPLATE.yaml configs/experiments/my_experiment.yaml
```
2. Edit only keys you want to override (everything else comes from `configs/default.yaml`).

Example minimal override:
```yaml
rewards:
  distance_shaping_enabled: true
  distance_shaping_scale: 1.0
```

### 3) Run that experiment

```bash
python scripts/run_experiments.py --only my_experiment
```

Important:
- Use experiment name **without** `.yaml`.
- This writes to `outputs/my_experiment/`.

### 4) Run multiple or manifest-based experiments

```bash
python scripts/run_experiments.py --only exp_a exp_b
python scripts/run_experiments.py --use-manifest
```

## Evaluation and Design Assumptions

- Comparison mode trains **two separate agent policies** per run:
  - one in `nature`
  - one in `adversary`
- Within a run, both policies share the same sampled base layout primitives during training:
  - shelf placement
  - agent start
  - package and destination
  - matched hazard start anchor
- ID-A uses that training layout.
- OOD-layout uses new unseen layouts (`eval.ood_layout_count`, default `5`).
- OOD reported metrics are aggregated across all OOD layouts and eval episodes.
- Evaluation uses greedy agent actions with random tie-breaks.
- Disturbance motion remains stochastic according to scenario settings.

## Repository Structure

- `configs/`
  - `default.yaml`: baseline config and default assumptions
  - `experiments/*.yaml`: per-experiment overrides
- `src/qlearning_adversarial/`
  - `main.py`: orchestrates train + eval + output writing
  - `env.py`: gridworld dynamics, rewards, hazards, rendering
  - `agent.py`: tabular Q-learning agent(s)
  - `train.py`: training loop and artifact persistence
  - `eval.py`: evaluation loops and aggregate stats
  - `utils.py`: misc utilities
- `scripts/`
  - `run_experiments.py`: merges config and executes baseline/overrides
  - `visualize_result.py`: replay utility for saved artifacts
- `outputs/`
  - one folder per run (`default`, `la_zs_ic_ds`, etc.)
- `docs/`
  - `TUNING_BOARD.md`: parameter catalog for controlled sweeps

## Output Layout (per experiment)

For an experiment named `my_experiment`, outputs are written to `outputs/my_experiment/`:

- `config_used.yaml`: exact merged config used
- `csv/`
  - `nature_rewards.csv`, `adversary_rewards.csv`
  - `strategy_comparison_ida.csv`
  - `strategy_comparison_ood_layout.csv`
- `learningcurves/`
  - `nature_learning_curve.png`
  - `adversary_learning_curve.png`
- `png/`
  - static layout snapshots
- `gif/`
  - ID-A and OOD rollout animations for cross-scenario pairs
- `pkl/`
  - trained tables and serialized env configs
- `txt/`
  - `metrics.txt` summary

## Visualization

Primary visualization is generated automatically in each run:
- `outputs/<experiment_name>/gif/`

Recommended usage:
- Open the run-specific GIFs produced by `qlearning_adversarial.main` (ID-A and OOD cross-scenario rollouts).

Optional replay script:
```bash
PYTHONPATH=src python scripts/visualize_result.py --prefix nature
PYTHONPATH=src python scripts/visualize_result.py --prefix adversary
```
- This replays from artifacts in the output directory pointed to by `project.output_dir` in `configs/default.yaml`.
- For experiment-specific artifacts, use the auto-generated GIFs in `outputs/<experiment_name>/gif/` unless you intentionally repoint `project.output_dir`.

## Key Config Areas

- `run.*`: train/eval episode budgets
- `training.*`: agent learning hyperparameters
- `scenarios.nature.*`: forklift-driven disturbance behavior
- `scenarios.adversary.*`: adversary behavior and learning objective
- `rewards.*`: reward/cost model (step, pickup/delivery, penalties, shaping)
- `state_features.*`: state representation toggles (for example relative package/destination context)
- `eval.*`: evaluation controls (including OOD layout count)

