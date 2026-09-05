"""Filtering: turn scored rollout groups into a curated SFT training set.

Keeps every rollout that agrees with its prompt's majority vote, on prompts
where that majority clears an agreement threshold — following the
rejection-sampling recipe (e.g. STaR, RFT): keep all correct-looking
completions per prompt, not just one, since duplicate correct phrasings are
still useful SFT signal. Prompts the model is unsure about (low agreement)
are dropped rather than trained on.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from data.scoring import ScoredGroup


@dataclass
class CuratedExample:
    """One (prompt, response) pair selected for post-training."""

    sample_id: str
    prompt: str
    response: str
    extracted: str
    agreement: float
    model_id: str
    # Which benchmark/split this sample came from. Rollouts don't carry
    # images (see rollouts.py), so training/dataset.py re-fetches them from
    # the benchmark loader by (benchmark, split, sample_id) at train time.
    # Optional/blank for callers that curate outside a benchmark context.
    benchmark: str = ""
    split: str = ""


def filter_rollouts(
    groups: list[ScoredGroup],
    min_agreement: float = 0.6,
    benchmark: str = "",
    split: str = "",
) -> list[CuratedExample]:
    """Select training examples from scored rollout groups.

    A group is dropped entirely if its majority agreement is below
    `min_agreement`; otherwise every rollout matching that majority is kept.
    """
    curated: list[CuratedExample] = []
    for group in groups:
        if group.agreement < min_agreement:
            continue
        for rollout in group.majority_rollouts():
            curated.append(
                CuratedExample(
                    sample_id=rollout.sample_id,
                    prompt=rollout.prompt,
                    response=rollout.raw,
                    extracted=rollout.extracted,
                    agreement=group.agreement,
                    model_id=rollout.model_id,
                    benchmark=benchmark,
                    split=split,
                )
            )
    return curated


def write_jsonl(examples: list[CuratedExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for example in examples:
            f.write(json.dumps(asdict(example)) + "\n")
