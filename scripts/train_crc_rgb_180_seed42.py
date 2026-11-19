"""Train and score the seed-42 crc model using the public configuration."""
import argparse
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config.local.json")
    args = parser.parse_args()
    subprocess.run(
        [sys.executable, "jobs.py", "--config", str(args.config.resolve()),
         "run", "threshold", "crc", "180", "42"],
        cwd=root, check=True,
    )
