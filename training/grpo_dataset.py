"""GRPO dataset: every rollout in a self-consistency group, weighted by its
group-relative advantage, instead of SFT's hard majority-only filter.

Reward is the same signal data/scoring.py already computes -- 1.0 if a
rollout's extracted answer matches its group's majority vote, 0.0
otherwise. GRPO's contribution over SFT is using the *disagreeing*
rollouts too: their reward=0 pulls the group mean down, which is what
gives the majority rollouts a nonzero (rather than saturated) advantage,
and gives the model gradient signal on what *not* to do, not just what to
imitate.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.dataset import _ensure_min_size, _index_images


@dataclass
class GRPOExample:
    prompt: str
    response: str
    images: list  # list[PIL.Image.Image]
    advantage: float


def _advantages_for_group(rewards: list[float]) -> list[float]:
    """Group-relative advantage: (reward - mean) / std, GRPO-style.

    A single-member or zero-variance group gets an advantage of 0 for every
    rollout in it -- there's no relative signal to extract (everyone agreed,
    or there's nothing to compare against).
    """
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    if std == 0.0:
        return [0.0] * len(rewards)
    return [(r - mean) / std for r in rewards]


def load_scored_groups(
    jsonl_path: str | Path,
    benchmark_override: str | None = None,
    split_override: str | None = None,
    min_reward_std: float = 1e-4,
) -> list[GRPOExample]:
    """Load a scored.jsonl (data/scoring.py:write_scored_groups_jsonl),
    compute per-rollout advantages within each (sample_id) group, and
    attach images. Groups whose reward std falls below `min_reward_std`
    (every rollout agreed, or none did) are dropped -- zero advantage
    either way, so skipping them saves a wasted forward pass.
    """
    rows: list[dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return []

    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)

    kept_rows: list[dict[str, Any]] = []
    advantages: list[float] = []
    for sample_id, group_rows in by_sample.items():
        rewards = [1.0 if r["extracted"] == r["majority"] else 0.0 for r in group_rows]
        if statistics.pstdev(rewards) < min_reward_std:
            continue
        group_advantages = _advantages_for_group(rewards)
        kept_rows.extend(group_rows)
        advantages.extend(group_advantages)

    if not kept_rows:
        return []

    by_benchmark: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(kept_rows):
        benchmark = row.get("benchmark") or benchmark_override
        split = row.get("split") or split_override
        if not benchmark or not split:
            raise ValueError(
                f"scored rollout {row.get('sample_id')!r} has no benchmark/split recorded "
                "and no override was given -- can't re-fetch its image"
            )
        by_benchmark[(benchmark, split)].append(i)

    examples: list[GRPOExample | None] = [None] * len(kept_rows)
    for (benchmark, split), indices in by_benchmark.items():
        ids = {kept_rows[i]["sample_id"] for i in indices}
        images_by_id = _index_images(benchmark, split, ids)
        for i in indices:
            row = kept_rows[i]
            examples[i] = GRPOExample(
                prompt=row["prompt"],
                response=row["response"],
                images=images_by_id.get(row["sample_id"], []),
                advantage=advantages[i],
            )
    return examples  # type: ignore[return-value]


class GRPODataset:
    """Tokenizes GRPOExamples the same way SFTDataset does (prompt span
    masked to -100 in `labels`) but also carries each example's scalar
    `advantage`, which GRPOTrainer.compute_loss multiplies the response
    tokens' log-probability by.
    """

    def __init__(self, examples: list[GRPOExample], processor, max_length: int = 2048):
        self.examples = examples
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        import torch

        ex = self.examples[idx]
        images = [_ensure_min_size(img) for img in ex.images]
        prompt_messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in ex.images]
                + [{"type": "text", "text": ex.prompt}],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + ex.response

        prompt_ids = self.processor(text=prompt_text, images=images or None, return_tensors="pt")
        full = self.processor(
            text=full_text,
            images=images or None,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = full["input_ids"][0]
        labels = input_ids.clone()
        prompt_len = min(prompt_ids["input_ids"].shape[-1], labels.shape[0])
        labels[:prompt_len] = -100

        item = {k: v[0] for k, v in full.items()}
        item["labels"] = labels
        item["advantage"] = torch.tensor(ex.advantage, dtype=torch.float32)
        return item
