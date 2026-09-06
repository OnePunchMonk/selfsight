"""Rollout -> score -> filter, end to end, fully offline by default.

    python -m data.cli --model mock:demo-v1 --benchmark demo_mc --k 5 \\
        --min-agreement 0.6 --output curated.jsonl

Pass --scored-output to also dump every rollout (not just the SFT-filtered
majority ones) for GRPO (training/grpo.py), which needs a group's
disagreeing rollouts too to compute a nonzero group-relative advantage.

Pass --metrics-output to append this run's mean agreement/diversity to a
history file and check it against the previous run on the same
(model, benchmark) pair for the reward-hacking signature: rising agreement
with falling diversity (see data/diversity.py).
"""

from __future__ import annotations

import argparse

from data.diversity import append_run_metrics, detect_collapse, load_run_history, summarize
from data.filtering import filter_rollouts, write_jsonl
from data.rollouts import RolloutGenerator
from data.scoring import SelfConsistencyScorer, write_scored_groups_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mock:demo-v1", help="provider:model_id")
    parser.add_argument("--benchmark", default="demo_mc")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--k", type=int, default=5, help="rollouts sampled per prompt")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--min-agreement", type=float, default=0.6)
    parser.add_argument("--output", default="curated.jsonl")
    parser.add_argument(
        "--scored-output", default=None, help="also write all scored rollouts here (for GRPO)"
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="append mean agreement/diversity to this history file and check for collapse",
    )
    args = parser.parse_args()

    generator = RolloutGenerator(args.model, k=args.k, temperature=args.temperature)
    groups = list(
        generator.generate_for_benchmark(
            args.benchmark, split=args.split, max_samples=args.max_samples
        )
    )

    scorer = SelfConsistencyScorer()
    scored = scorer.score_all(groups)

    curated = filter_rollouts(
        scored, min_agreement=args.min_agreement, benchmark=args.benchmark, split=args.split
    )
    write_jsonl(curated, args.output)

    kept_prompts = sum(1 for g in scored if g.agreement >= args.min_agreement)
    print(
        f"{len(scored)} prompts sampled, {kept_prompts} passed agreement >= "
        f"{args.min_agreement}, {len(curated)} curated examples written to {args.output}"
    )

    if args.scored_output:
        write_scored_groups_jsonl(
            scored, args.scored_output, benchmark=args.benchmark, split=args.split
        )
        print(f"all scored rollouts written to {args.scored_output}")

    if args.metrics_output:
        history = load_run_history(args.metrics_output, model=args.model, benchmark=args.benchmark)
        current = summarize(scored, model=args.model, benchmark=args.benchmark)
        print(
            f"mean agreement={current.mean_agreement:.3f} mean entropy={current.mean_entropy:.3f} "
            f"({len(history)} prior run(s) on record for this model/benchmark)"
        )
        warning = detect_collapse(current, history)
        if warning:
            print(f"WARNING: {warning.summary()}")
        append_run_metrics(current, args.metrics_output)


if __name__ == "__main__":
    main()
