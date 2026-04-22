from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def first_existing(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these paths exist: {', '.join(str(p) for p in candidates)}")


def main():
    py = sys.executable

    # Training
    run([py, str(ROOT / "training" / "train_qtable_discrete.py")])
    run([py, str(ROOT / "training" / "train_continuous_qtable.py")])
    run([py, str(ROOT / "training" / "train_td3_continuous.py")])

    # Evaluation of tabular agents
    run(
        [
            py,
            str(ROOT / "evaluation" / "evaluate_qtable.py"),
            "--model",
            str(ROOT / "results" / "models" / "qtable_discrete.npy"),
            "--config",
            str(ROOT / "config" / "qtable_discrete.yaml"),
            "--name",
            "qtable_discrete",
        ]
    )

    run(
        [
            py,
            str(ROOT / "evaluation" / "evaluate_qtable.py"),
            "--model",
            str(ROOT / "results" / "models" / "qtable_action_cost.npy"),
            "--config",
            str(ROOT / "config" / "qtable_continuous.yaml"),
            "--name",
            "qtable_action_cost",
        ]
    )

    # Evaluation of deep RL agents
    td3_model = first_existing(
        ROOT / "results" / "models" / "td3_continuous.zip",
        ROOT / "results" / "models" / "td3_continuous" / "td3_continuous.zip",
    )
    run(
        [
            py,
            str(ROOT / "evaluation" / "evaluate_sb3.py"),
            "--model",
            str(td3_model),
            "--algo",
            "td3",
            "--config",
            str(ROOT / "config" / "deeprl.yaml"),
            "--name",
            "td3_continuous",
        ]
    )

    sac_model = first_existing(
        ROOT / "results" / "models" / "sac_continuous.zip",
        ROOT / "results" / "models" / "sac_continuous" / "sac_continuous.zip",
    )
    run(
        [
            py,
            str(ROOT / "evaluation" / "evaluate_sb3.py"),
            "--model",
            str(sac_model),
            "--algo",
            "sac",
            "--config",
            str(ROOT / "config" / "deeprl.yaml"),
            "--name",
            "sac_continuous",
        ]
    )

    # Comparison / plots
    run([py, str(ROOT / "analysis" / "compare_all_approaches.py")])
    run([py, str(ROOT / "generate_continuous_plots.py")])


if __name__ == "__main__":
    main()