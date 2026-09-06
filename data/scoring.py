"""Self-consistency scoring: turn k rollouts per prompt into a confidence signal.

The idea (Wang et al., 2022): sample a prompt multiple times and treat
agreement across samples as a proxy for correctness, with no ground truth and
no separate judge model required. A prompt where 5/5 samples land on the same
extracted answer is far more likely to be one the base VLM actually has right
than one where the samples split 2/2/1.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from data.rollouts import Rollout, RolloutGroup


@dataclass
class ScoredGroup:
    """A rollout group plus the self-consistency verdict over it."""

    sample_id: str
    prompt: str
    majority: str  # the extracted answer with the most votes
    agreement: float  # majority_votes / n, in [0, 1]
    rollouts: list[Rollout]

    def majority_rollouts(self) -> list[Rollout]:
        """The rollouts whose extracted answer matches the majority vote."""
        return [r for r in self.rollouts if r.extracted == self.majority]


class SelfConsistencyScorer:
    """Scores rollout groups by extracted-answer agreement."""

    def score(self, group: RolloutGroup) -> ScoredGroup:
        if not group.rollouts:
            raise ValueError(f"rollout group {group.sample_id!r} has no rollouts to score")
        votes = Counter(r.extracted for r in group.rollouts)
        majority, count = votes.most_common(1)[0]
        agreement = count / len(group.rollouts)
        return ScoredGroup(
            sample_id=group.sample_id,
            prompt=group.prompt,
            majority=majority,
            agreement=agreement,
            rollouts=group.rollouts,
        )

    def score_all(self, groups: list[RolloutGroup]) -> list[ScoredGroup]:
        return [self.score(g) for g in groups]


def write_scored_groups_jsonl(
    groups: list[ScoredGroup], path: str | Path, benchmark: str = "", split: str = ""
) -> None:
    """Dump every rollout in every group, unfiltered -- for GRPO (training/grpo.py).

    Unlike data/filtering.py's write_jsonl (which keeps only the majority
    rollouts from groups clearing an agreement threshold, for SFT),
    group-relative advantage estimation needs *all* rollouts per prompt,
    including the disagreeing ones -- they're the reward=0 half of the
    contrast that gives a group a nonzero advantage signal at all.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for group in groups:
            for rollout in group.rollouts:
                f.write(
                    json.dumps(
                        {
                            "sample_id": rollout.sample_id,
                            "prompt": rollout.prompt,
                            "response": rollout.raw,
                            "extracted": rollout.extracted,
                            "majority": group.majority,
                            "agreement": group.agreement,
                            "model_id": rollout.model_id,
                            "benchmark": benchmark,
                            "split": split,
                        }
                    )
                    + "\n"
                )
