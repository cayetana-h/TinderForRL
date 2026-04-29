# TinderForRL

A reinforcement learning project about matching the right algorithm to the right environment.

The project compares tabular RL and deep RL on Mountain Car variants, focusing on:
- performance,
- stability,
- reward structure,
- and interpretability.

---

## Project overview

The core idea is simple: the same problem can behave very differently depending on how you define the reward and the action space.

This repo studies:
- **MountainCar-v0** with discrete Q-learning,
- **MountainCar-v0** with an action-cost variant,
- **MountainCarContinuous-v0** with TD3,
- **MountainCarContinuous-v0** with SAC.

The goal is to compare which approach fits each formulation best.

---

## Repository structure

```text
TinderForRL/
├── agents/
│   └── agent_qtable.py
├── analysis/
│   ├── compare_all_approaches.py
│   ├── qtable_discrete_analysis.ipynb
│   └── qtable_continuous_analysis.ipynb
├── config/
│   ├── deeprl.yaml
│   ├── qtable_continuous.yaml
│   └── qtable_discrete.yaml
├── evaluation/
│   ├── evaluate_qtable.py
│   └── evaluate_sb3.py
├── results/
│   ├── comparison/
│   ├── metrics/
│   └── models/
├── training/
│   ├── train_continuous_deeprl.py
│   ├── train_continuous_qtable.py
│   ├── train_qtable_discrete.py
│   ├── train_sac_continuous.py
│   └── train_td3_continuous.py
├── utils/
│   ├── io.py
│   ├── metrics.py
│   └── wrappers.py
├── generate_continuous_plots.py
├── run_all_training.py
├── requirements.txt
└── README.md
```

---

## Implemented approaches

### 1) Discrete Q-learning
File: `training/train_qtable_discrete.py`

A tabular Q-learning agent trained on `MountainCar-v0` with:
- state discretization,
- velocity-based reward shaping,
- metrics logging,
- greedy evaluation after training.

This version is the most interpretable and the easiest to visualize.

### 2) Q-learning with action cost
File: `training/train_continuous_qtable.py`

A discrete-action variant on `MountainCar-v0` where non-neutral actions incur an extra cost.

This setup is useful to study how reward shaping changes the learned policy even when the algorithm stays the same.

### 3) TD3 on continuous Mountain Car
File: `training/train_td3_continuous.py`

TD3 is trained on `MountainCarContinuous-v0` for the continuous-control version of the task.

### 4) SAC on continuous Mountain Car
File: `training/train_sac_continuous.py`

SAC is trained on the same continuous-control task and compared with TD3.

### 5) Combined deep RL launcher
File: `training/train_continuous_deeprl.py`

This script runs the TD3 and SAC training sequence.

---

## Evaluation and comparison

The repo includes dedicated scripts for evaluation and comparison:

- `evaluation/evaluate_qtable.py`
- `evaluation/evaluate_sb3.py`
- `analysis/compare_all_approaches.py`
- `generate_continuous_plots.py`

These scripts are used to:
- evaluate trained agents,
- save summary metrics,
- generate comparison plots,
- and aggregate results into a common format.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run everything

```bash
python run_all_training.py
```

This will:
- train the discrete Q-learning agent,
- train the action-cost Q-learning variant,
- train TD3 and SAC,
- evaluate the trained models,
- generate comparison plots,
- and save summary metrics.

---

## Run individual scripts

### Discrete Q-learning
```bash
python training/train_qtable_discrete.py
```

### Q-learning with action cost
```bash
python training/train_continuous_qtable.py
```

### Deep RL launcher
```bash
python training/train_continuous_deeprl.py
```

### Comparison script
```bash
python analysis/compare_all_approaches.py
```

### Plot generation
```bash
python generate_continuous_plots.py
```

---

## Outputs

The main outputs are saved under `results/`:

- `results/models/` for trained models
- `results/metrics/` for per-episode metrics and summaries
- `results/comparison/` for plots and comparison tables

---

## What the analysis should compare

For each approach, the most useful metrics are:

- success rate,
- average reward,
- average steps to solve,
- action cost,
- training stability,
- and interpretability.

The main question is not just “does it solve the task?”, but:
- how efficiently it solves it,
- how stable learning is,
- and how the learned policy changes with reward structure.

---

## Design notes

This project intentionally compares:
- tabular RL vs deep RL,
- discrete vs continuous control,
- shaped reward vs standard reward,
- interpretability vs performance.

That is why different algorithms are used for different formulations instead of forcing one method everywhere.

---

## Dependencies

Main libraries used in the project:

- `gymnasium`
- `numpy`
- `matplotlib`
- `stable-baselines3`
- `torch`
- `pyyaml`

---

## Notes

The continuous-control scripts use the standard `MountainCarContinuous-v0` environment.

The discrete action-cost variant remains a discrete-action setup on `MountainCar-v0`, but with an extra penalty for non-neutral actions.

---

## Next steps

- Add multi-seed evaluation.
- Add confidence intervals.
- Extend the comparison tables.
- Prepare the final written report.