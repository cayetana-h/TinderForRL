from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: Path, config: Path):
    cmd = [sys.executable, str(script), "--config", str(config)]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "deeprl.yaml"))
    args = parser.parse_args()

    config = Path(args.config)
    run(ROOT / "training" / "train_td3_continuous.py", config)
    run(ROOT / "training" / "train_sac_continuous.py", config)


if __name__ == "__main__":
    main()