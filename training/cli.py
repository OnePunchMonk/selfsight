"""Run one SFT iteration from a config file.

    python -m training.cli --config configs/sft_qwen2vl.yaml
"""

from __future__ import annotations

import argparse
import logging

from training.config import SFTConfig
from training.sft import run_sft


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a training YAML config")
    args = parser.parse_args()

    config = SFTConfig.from_yaml(args.config)
    output_dir = run_sft(config)
    print(f"candidate checkpoint: {output_dir}")


if __name__ == "__main__":
    main()
