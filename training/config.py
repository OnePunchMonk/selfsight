"""One iteration's training recipe, loaded from a YAML config.

Kept as a flat dataclass rather than an ad-hoc dict so a config file is the
single reproducible record of "which checkpoint, which data, which LoRA
rank, which schedule" for a given loop iteration -- the thing configs/ is
scoped to own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoRAConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # Default target modules for Qwen2-VL / LLaVA-family decoder attention
    # blocks. Wrong for an architecture with different module names -- override
    # per-model in the config file rather than guessing here.
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class SFTConfig:
    # Base checkpoint this iteration starts from (the "current checkpoint" in
    # the loop diagram) and where the candidate gets written -- promotion to
    # "current" is the eval harness's job, not this script's.
    base_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    output_dir: str = "checkpoints/candidate"

    # Curated data produced by `python -m data.cli` (data/filtering.py).
    curated_jsonl: str = "curated.jsonl"
    # Only needed when curated_jsonl rows don't carry benchmark/split
    # (pre-dating that field, or curated outside a benchmark context).
    # Required, otherwise, to re-fetch each sample's image at train time.
    benchmark_override: str | None = None
    split_override: str | None = None

    max_seq_length: int = 2048
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    bf16: bool = True
    seed: int = 42

    lora: LoRAConfig = field(default_factory=LoRAConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTConfig:
        import yaml

        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        lora_raw = raw.pop("lora", {}) or {}
        return cls(**raw, lora=LoRAConfig(**lora_raw))


@dataclass
class GRPOConfig:
    """One GRPO iteration's recipe -- phase 3, replacing/complementing SFT.

    Reuses the same self-consistency rollouts as phase 2, but instead of
    filtering to a hard majority-only training set, every rollout in a
    group contributes a policy-gradient update weighted by its
    group-relative advantage (reward minus the group's mean reward,
    normalized by the group's reward std). No separate reward model or
    judge call -- the reward is still just "did this rollout agree with its
    group's majority answer", the same signal SFT used, just not
    thresholded away.
    """

    base_model: str = "OpenGVLab/InternVL3-2B-hf"
    output_dir: str = "checkpoints/candidate"

    # All-rollouts data from `python -m data.cli --scored-output ...`
    # (data/scoring.py:write_scored_groups_jsonl), not the SFT-filtered set.
    scored_jsonl: str = "scored.jsonl"
    benchmark_override: str | None = None
    split_override: str | None = None

    # Groups with zero reward variance (all rollouts agree, or none do)
    # produce a zero advantage for every rollout in them -- no gradient
    # signal, just wasted forward passes. Skipped below this floor.
    min_reward_std: float = 1e-4

    # KL penalty coefficient against a frozen copy of the base model.
    # Default 0 (pure REINFORCE/GRPO, no reference model) to keep memory
    # footprint on a single modest GPU -- loading a second full copy of the
    # policy for a KL term roughly doubles VRAM use. Set > 0 once the loop
    # has compute headroom to spare it.
    kl_coef: float = 0.0

    max_seq_length: int = 2048
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    bf16: bool = True
    seed: int = 42

    lora: LoRAConfig = field(default_factory=LoRAConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GRPOConfig:
        import yaml

        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        lora_raw = raw.pop("lora", {}) or {}
        return cls(**raw, lora=LoRAConfig(**lora_raw))
