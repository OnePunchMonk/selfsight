"""Rollout -> score -> filter, end to end, fully offline by default.

    python -m data.cli --model mock:demo-v1 --benchmark demo_mc --k 5 \\
        --min-agreement 0.6 --output curated.jsonl
"""

from __future__ import annotations

import argparse

from data.filtering import filter_rollouts, write_jsonl
from data.rollouts import RolloutGenerator
from data.scoring import SelfConsistencyScorer


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
    args = parser.parse_args()

    generator = RolloutGenerator(args.model, k=args.k, temperature=args.temperature)
    groups = list(
        generator.generate_for_benchmark(
            args.benchmark, split=args.split, max_samples=args.max_samples
        )
    )

    scorer = SelfConsistencyScorer()
    scored = scorer.score_all(groups)

    curated = filter_rollouts(scored, min_agreement=args.min_agreement)
    write_jsonl(curated, args.output)

    kept_prompts = sum(1 for g in scored if g.agreement >= args.min_agreement)
    print(
        f"{len(scored)} prompts sampled, {kept_prompts} passed agreement >= "
        f"{args.min_agreement}, {len(curated)} curated examples written to {args.output}"
    )


if __name__ == "__main__":
    main()
