"""Advance the self-improvement loop by exactly one iteration.

    python -m training.loop_cli --config configs/loop_internvl.yaml

Refuses to run past the RSI iteration cap or on a rollout set that shows
the reward-hacking collapse signature (see training/loop.py) unless you
pass the corresponding override flag -- both exist to force a human to
actually look, not to be left on.
"""

from __future__ import annotations

import argparse
import logging

from training.loop import (
    CapReachedError,
    CollapseDetectedError,
    LoopConfig,
    default_evaluate_and_gate,
    default_generate_and_train,
    load_state,
    run_iteration,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a loop YAML config")
    parser.add_argument(
        "--override-cap",
        action="store_true",
        help="proceed even if the iteration cap has been reached -- use only after reviewing "
        "the eval trajectory, sample rollouts, and diversity metrics",
    )
    parser.add_argument(
        "--override-collapse",
        action="store_true",
        help="proceed even if this iteration's rollouts show the reward-hacking collapse "
        "signature -- use only after reviewing the actual rollouts",
    )
    args = parser.parse_args()

    config = LoopConfig.from_yaml(args.config)
    state = load_state(config.state_path, default_model=config.current_model)

    try:
        state = run_iteration(
            config,
            state,
            default_generate_and_train,
            default_evaluate_and_gate,
            override_cap=args.override_cap,
            override_collapse=args.override_collapse,
        )
    except (CapReachedError, CollapseDetectedError) as exc:
        print(f"STOPPED: {exc}")
        raise SystemExit(1) from exc

    last = state.history[-1]
    verdict = "PROMOTED" if last.promoted else "REJECTED"
    print(
        f"iteration {last.iteration}: {verdict} -- {last.reason}\n"
        f"  curated examples: {last.n_curated_examples}, "
        f"mean agreement: {last.mean_agreement:.3f}, mean entropy: {last.mean_entropy:.3f}\n"
        f"  current model is now: {state.current_model}"
    )


if __name__ == "__main__":
    main()
