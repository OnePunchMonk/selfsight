"""Multimodal SFT dataset: curated (prompt, response) pairs + their image(s).

data/filtering.py's CuratedExample never carries images -- data/rollouts.py
deliberately doesn't persist them either, since the benchmark loader in
vlm-evaluation-harness is the single source of truth for a sample's image.
This module re-fetches each curated sample's image(s) by scanning its source
benchmark split once (per benchmark/split pair, not per row) and indexing by
sample_id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Qwen2-VL's vision tower requires each image's pixel grid to divide evenly
# by patch_size * merge_size (28 for the 2B checkpoint); tiny images (e.g.
# the 64x64 demo_mc fixture squares) don't satisfy that and crash deep
# inside attention with a split_with_sizes mismatch. Upscaling to a safe
# multiple of 28 before tokenizing sidesteps it.
_MIN_IMAGE_SIDE = 336


def _ensure_min_size(image):
    width, height = image.size
    if width >= _MIN_IMAGE_SIDE and height >= _MIN_IMAGE_SIDE:
        return image
    scale = max(_MIN_IMAGE_SIDE / width, _MIN_IMAGE_SIDE / height)
    return image.resize((round(width * scale), round(height * scale)))


@dataclass
class SFTExample:
    prompt: str
    response: str
    images: list  # list[PIL.Image.Image]


def _index_images(benchmark: str, split: str, sample_ids: set[str]) -> dict[str, list]:
    """One pass over a benchmark split, keeping images only for wanted ids."""
    from vlm_evaluation_harness.benchmarks.loader import BenchmarkLoader
    from vlm_evaluation_harness.benchmarks.registry import get_registry

    manifest = get_registry().get(benchmark)
    loader = BenchmarkLoader()
    found: dict[str, list] = {}
    for sample in loader.load(manifest, split=split):
        if sample.sample_id in sample_ids:
            found[sample.sample_id] = sample.images
            if len(found) == len(sample_ids):
                break
    return found


def load_curated_examples(
    jsonl_path: str | Path,
    benchmark_override: str | None = None,
    split_override: str | None = None,
) -> list[SFTExample]:
    """Load a curated.jsonl (see data/filtering.py) and attach images.

    `benchmark_override`/`split_override` are only used for rows that predate
    the `benchmark`/`split` fields on CuratedExample; a row that already
    carries them always wins.
    """
    rows: list[dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return []

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        benchmark = row.get("benchmark") or benchmark_override
        split = row.get("split") or split_override
        if not benchmark or not split:
            raise ValueError(
                f"curated example {row.get('sample_id')!r} has no benchmark/split recorded "
                "and no override was given -- can't re-fetch its image"
            )
        groups.setdefault((benchmark, split), []).append(row)

    examples: list[SFTExample] = []
    for (benchmark, split), group_rows in groups.items():
        ids = {r["sample_id"] for r in group_rows}
        images_by_id = _index_images(benchmark, split, ids)
        for row in group_rows:
            examples.append(
                SFTExample(
                    prompt=row["prompt"],
                    response=row["response"],
                    images=images_by_id.get(row["sample_id"], []),
                )
            )
    return examples


class SFTDataset:
    """torch Dataset tokenizing SFTExamples with the prompt span masked out
    of the loss -- only the model's own (self-consistency-filtered) response
    tokens contribute to the SFT gradient.
    """

    def __init__(self, examples: list[SFTExample], processor, max_length: int = 2048):
        self.examples = examples
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
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
        return item


def collate_single(batch: list[dict]) -> dict:
    """Adds the batch dimension back for batch_size=1 training.

    Batching >1 sample needs a model-specific collator -- VLM processors emit
    a variable pixel_values shape per sample (patch count depends on each
    image's resolution under dynamic-resolution vision towers like
    Qwen2-VL's), so naive padding/stacking is wrong. Out of scope for this
    phase; per_device_train_batch_size stays 1 and gradient accumulation
    provides the effective batch size instead.
    """
    if len(batch) != 1:
        raise ValueError("collate_single only supports per_device_train_batch_size=1")
    return {k: v.unsqueeze(0) for k, v in batch[0].items()}
