# Project Progress Log

## Research Question (Current Framing)

Primary question:
- Does training the delivery agent against `nature` disturbances (forklifts) vs `adversary` disturbances produce better robustness and transfer?

Secondary question:
- Does a learned adversary (including zero-sum variants) create training pressure that improves or harms downstream generalization?

---

## What Has Been Built So Far

Core system capabilities now in place:
- Config-driven experiment system with `default.yaml` + per-experiment overrides in `configs/experiments/`.
- Standard run command:
  - `python scripts/run_experiments.py --only <experiment_name>`
- Comparison mode that trains two separate policies per run:
  - nature-trained policy
  - adversary-trained policy
- Cross-scenario evaluation matrices for:
  - ID-A (training layout)
  - OOD-layout (unseen layouts; now aggregated across `eval.ood_layout_count`)
- Organized output structure per run under `outputs/<experiment_name>/`.
- Visual diagnostics:
  - static layout images
  - rollout GIFs for ID-A and OOD cross-scenario pairs
  - learning curves
- Extended environment features added incrementally:
  - package pickup + delivery task logic
  - shelves + hazards
  - learning adversary option
  - zero-sum adversary objective option
  - optional relative package/destination context in state encoding
  - distance shaping reward option
- Tie-break handling improved:
  - random tie-breaks for agent greedy/action selection and adversary greedy choices

---

## Key Implementation Lessons Learned

2. **OOD behavior was highly sensitive to tie-breaking**
- Deterministic argmax tie-breaking in unseen states caused “frozen” behavior.
- Random tie-breaks removed some artifacts, but did not solve generalization by itself.

3. **State representation was a major bottleneck**
- Early state encoding did not include package/destination location context.
- Adding relative package/destination context improved task awareness but introduced a much larger tabular state space.

4. **Bigger state space + tabular Q-learning created data sparsity**
- Relative-context experiments still struggle OOD because many OOD states remain under-visited.

5. **Reward totals can be misleading without step metrics**
- High delivery + low collisions can still produce lower return if path length is high due to strong step penalties.
- Adding `avg_steps` to evaluation output improved interpretability.

6. **Learned adversary quality is still unresolved**
- Learned adversary has often underperformed deterministic pursuit in practical pressure generation.
- Zero-sum objective support is implemented, but behavior quality remains inconsistent.

---

## Experimental Story (From Outputs)

Data source folders reviewed:
- `outputs/default`
- `outputs/learningadversary`
- `outputs/distance_shaping`
- `outputs/inc_context`
- `outputs/inc_context_distance_shaping`
- `outputs/la_ic_ds`
- `outputs/la_zs_ic_ds`

### Snapshot Table (same-scenario metrics)

Notes:
- Most runs used `ood_layout_count=5`.
- `la_zs_ic_ds` used `ood_layout_count=10` and `train_episodes=20000`, so OOD values are not directly apples-to-apples vs earlier runs.

| Experiment | Key Change | ID-A Nature->Nature | ID-A Adv->Adv | OOD Nature->Nature | OOD Adv->Adv | Read |
|---|---|---:|---:|---:|---:|---|
| `default` | Baseline deterministic adversary setup | 86.80 | 94.40 | -433.67 | -101.51 | Strong ID-A, poor OOD |
| `learningadversary` | Learning adversary in adversary scenario | 86.80 | 33.84 | -433.67 | -119.32 | Learned adversary policy weak |
| `distance_shaping` | Distance shaping enabled | 96.18 | 108.66 | -390.11 | -79.49 | Better ID-A, OOD still poor |
| `inc_context` | Relative package/destination context | 86.80 | 94.40 | -400.00 | -66.78 | Mixed; still brittle OOD |
| `inc_context_distance_shaping` | Relative context + shaping | 101.74 | 105.06 | -91.95 | -61.91 | Best OOD trend among 5-layout runs |
| `la_ic_ds` | Learning adversary + context + shaping | 101.74 | 91.42 | -91.95 | -63.61 | Similar OOD; adversary learning not clearly better |
| `la_zs_ic_ds` | Zero-sum learning adversary + context + shaping | 95.78 | 66.38 | -105.16* | -103.32* | OOD still weak; adversary pressure quality unresolved |

\* from 10 OOD layouts and different training budget.

### What this suggests
- The biggest practical OOD lift so far came from:
  - adding relative task context + distance shaping
- Switching to learning adversary has **not yet** shown consistent OOD benefit.
- Zero-sum adversary support is implemented, but resulting adversary behavior is still not reliably “agent-opposing” in an effective way.

---

## Where We Are Still Struggling

1. **Core unresolved question**
- Whether adversary-trained policies generalize better than nature-trained policies remains inconclusive due to high variance and mixed results.

2. **OOD generalization remains poor overall**
- Even best observed OOD runs remain negative reward with high collision rates.

3. **Learned adversary quality**
- The learned adversary often appears passive or not strategically punishing enough in rollouts.
- Deterministic adversary often provides cleaner pressure than learned adversary so far.

4. **Tabular scalability limits**
- Relative-context state expansion worsens sample inefficiency.
- OOD layout changes expose under-covered state regions.

5. **Evaluation stochasticity confounds comparisons**
- Hazard stochasticity + tie-break randomness add variance; single-run comparisons can be misleading.

---

## Current Assumptions in the System

- Two separate agent policies are always trained in comparison mode (nature-trained and adversary-trained).
- ID-A uses the same base layout as training for that run.
- OOD-layout uses newly sampled layouts under same generation rules.
- OOD metrics are aggregate over `eval.ood_layout_count`.
- Agent evaluation is greedy with random tie-breaks.
- Hazards remain stochastic per configured move probabilities.
- Adversary movement probability default is `0.5`.

---

## Recommended Next Moves (Prioritized)

### 1) Improve experimental reliability before adding complexity
- Run repeated seeds per experiment (fixed protocol) and report mean + std.
- Keep one canonical comparison suite:
  - `default`
  - `inc_context_distance_shaping`
  - `la_ic_ds`
  - `la_zs_ic_ds`

### 2) Make adversary-learning diagnostics explicit
- Add per-episode adversary diagnostics:
  - adversary moves taken
  - catches
  - distance-to-agent trend
  - zero-sum adversary reward statistics
- This will verify whether adversary is actually learning or just drifting.

### 3) Stabilize zero-sum adversary training
- Tune adversary learner hyperparameters independently:
  - higher epsilon start and slower decay
  - learning rate sweep
  - progress/catch terms for heuristic baseline as control
- Compare directly against deterministic adversary with matched move probability.

### 4) Address tabular OOD limitations
- Short term:
  - additional context features only if they materially improve OOD under repeated seeds
  - prune unnecessary state dimensions where possible
- Medium term:
  - migrate to function approximation (e.g., DQN) if OOD remains poor despite tuning.

### 5) Tighten evaluation protocol
- Keep ID-A and OOD clearly separated in reporting.
- Continue including `avg_steps` alongside reward/collision/delivery.
- Add a concise per-experiment summary table in report outputs for easier side-by-side comparisons.

---

## Bottom Line

Progress is substantial on infrastructure and experimental control.  
The pipeline now supports meaningful scenario-vs-scenario comparison, richer state/reward options, and better diagnostics.  

The main technical blocker is unchanged: **robust OOD generalization is still weak, and learned adversary behavior is not yet reliably useful.**  
The next phase should focus on repeatable multi-seed evidence and adversary-learning diagnostics before introducing further complexity.

