from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: Path, *args: str):
    cmd = [sys.executable, str(script), *args]
    subprocess.run(cmd, check=True)


def main():
    run(ROOT / "training" / "train_qtable_discrete.py")
    run(ROOT / "training" / "train_continuous_qtable.py")
    run(ROOT / "training" / "train_continuous_deeprl.py")

    run(
        ROOT / "evaluation" / "evaluate_qtable.py",
        "--model",
        str(ROOT / "results" / "models" / "qtable_discrete.npy"),
        "--config",
        str(ROOT / "config" / "qtable_discrete.yaml"),
        "--name",
        "qtable_discrete",
    )

    run(
        ROOT / "evaluation" / "evaluate_qtable.py",
        "--model",
        str(ROOT / "results" / "models" / "qtable_action_cost.npy"),
        "--config",
        str(ROOT / "config" / "qtable_continuous.yaml"),
        "--name",
        "qtable_action_cost",
    )

    run(
        ROOT / "evaluation" / "evaluate_sb3.py",
        "--model",
        str(ROOT / "results" / "models" / "td3_continuous" / "td3_continuous.zip"),
        "--algo",
        "td3",
        "--config",
        str(ROOT / "config" / "deeprl.yaml"),
        "--name",
        "td3_continuous",
    )

    run(
        ROOT / "evaluation" / "evaluate_sb3.py",
        "--model",
        str(ROOT / "results" / "models" / "sac_continuous" / "sac_continuous.zip"),
        "--algo",
        "sac",
        "--config",
        str(ROOT / "config" / "deeprl.yaml"),
        "--name",
        "sac_continuous",
    )

    run(ROOT / "analysis" / "compare_all_approaches.py")
    run(ROOT / "generate_continuous_plots.py")


if __name__ == "__main__":
    main()