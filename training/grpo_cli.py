"""Run one GRPO iteration from a config file.

    python -m training.grpo_cli --config configs/grpo_internvl.yaml
"""

from __future__ import annotations

import argparse
import logging

from training.config import GRPOConfig
from training.grpo import run_grpo


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a GRPO YAML config")
    args = parser.parse_args()

    config = GRPOConfig.from_yaml(args.config)
    output_dir = run_grpo(config)
    print(f"candidate checkpoint: {output_dir}")


if __name__ == "__main__":
    main()
