# Tuning Board

Use this as the central list of parameters to tune. Override values in files under
`configs/experiments/*.yaml` and run the batch runner.

## Core Experiment Controls
- `experiment.mode`: `comparison` or `single`
- `run.tier`: `debug` or `report`
- `run.debug.train_episodes`, `run.debug.eval_episodes`
- `run.report.train_episodes`, `run.report.eval_episodes`
- `eval.max_steps`
- `eval.ood_layout_count`

## Environment Geometry
- `project.grid_size`
- `training.agent_count`
- `training.num_packages`
- `training.max_steps`

## Learning Hyperparameters
- `training.alpha`
- `training.gamma`
- `training.epsilon_start`
- `training.epsilon_end`
- `training.shared_q`

## Nature Scenario (Static + Stochastic)
- `scenarios.nature.forklift_count`
- `scenarios.nature.forklift_move_prob`
- `scenarios.nature.spill_count`

## Adversary Scenario (Static + Strategic)
- `scenarios.adversary.adversary_policy` (`deterministic` | `learning`)
- `scenarios.adversary.adversary_move_prob`
- `scenarios.adversary.adversary_max_moves` (episode-level move budget; one grid step per move)
- `scenarios.adversary.adversary_enabled`
- `scenarios.adversary.adversary_random_tiebreak`
- `scenarios.adversary.spill_count`
- `scenarios.adversary.adversary_learning_alpha`
- `scenarios.adversary.adversary_learning_gamma`
- `scenarios.adversary.adversary_learning_epsilon_start`
- `scenarios.adversary.adversary_learning_epsilon_end`
- `scenarios.adversary.adversary_learning_epsilon_decay_episodes`
- `scenarios.adversary.adversary_learning_progress_reward_scale`
- `scenarios.adversary.adversary_learning_catch_reward`
- `scenarios.adversary.adversary_learning_objective` (`heuristic` | `zero_sum`)

## Reward Shaping
- `rewards.step_penalty`
- `rewards.obstacle_penalty`
- `rewards.collision_penalty`
- `rewards.forklift_penalty`
- `rewards.adversary_penalty`
- `rewards.pickup_reward`
- `rewards.delivery_reward`
- `rewards.distance_shaping_enabled`
- `rewards.distance_shaping_scale`

## State Representation
- `state_features.include_relative_package_destination`

## Reproducibility / Storage
- `project.seed`
- `project.output_dir`

## Recommended Tuning Procedure
1. Keep one baseline experiment.
2. Change one parameter group at a time.
3. Run a small sweep via `scripts/run_experiments.py`.
4. Compare `outputs/<name>/csv/strategy_comparison_ida.csv` and
   `outputs/<name>/csv/strategy_comparison_ood_layout.csv`.
