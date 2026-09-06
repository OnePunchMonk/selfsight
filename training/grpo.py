"""GRPO over self-consistency-scored rollouts -- phase 3 of the loop.

Where phase 2 (training/sft.py) does supervised fine-tuning on the hard
majority-only filter, this trains on every rollout in a group, weighted by
its group-relative advantage (GRPO: Shao et al., 2024) -- reward minus the
group's mean reward, normalized by the group's reward std, with the reward
itself still just self-consistency agreement (data/scoring.py), no separate
reward model. This is meant to run *instead of or after* an SFT pass, not
replace the eval-harness gate: a GRPO candidate still has to clear the same
paired-significance regression test before promotion.
"""

from __future__ import annotations

import logging
from pathlib import Path

from training.checkpoint import save_candidate_checkpoint
from training.config import GRPOConfig
from training.dataset import collate_single
from training.grpo_dataset import GRPODataset, load_scored_groups

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """Wraps transformers.Trainer with an advantage-weighted policy-gradient
    loss in place of plain cross-entropy.

    `model(**inputs)` with `labels` set already gives the mean per-token
    negative log-likelihood of the response span as `outputs.loss` (the
    prompt span is masked to -100 by GRPODataset); multiplying that by the
    example's advantage and keeping the sign convention (loss = advantage *
    NLL, since NLL = -logprob) is exactly the REINFORCE/GRPO gradient: push
    logprob up where advantage > 0, down where advantage < 0.
    """

    def __init__(self, kl_coef: float = 0.0):
        self.kl_coef = kl_coef

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        import torch

        advantage = inputs.pop("advantage")
        outputs = model(**inputs)
        policy_loss = advantage.mean() * outputs.loss

        loss = policy_loss
        if self.kl_coef > 0:
            if not hasattr(model, "disable_adapter"):
                raise ValueError("kl_coef > 0 requires lora.enabled=True (needs a PeftModel)")
            with torch.no_grad(), model.disable_adapter():
                ref_outputs = model(**inputs)
            # k1 estimator: E[logp_policy - logp_ref] approximates KL(policy || ref).
            # outputs.loss is mean NLL = -mean_logprob, so this is mean_logprob_policy
            # - mean_logprob_ref. Noisy per-example, but unbiased in expectation --
            # adequate for a soft penalty term, not a substitute for a real KL if this
            # loop later needs one computed from full token distributions.
            kl_estimate = ref_outputs.loss - outputs.loss
            loss = loss + self.kl_coef * kl_estimate

        return (loss, outputs) if return_outputs else loss


def run_grpo(config: GRPOConfig) -> str:
    """Runs one GRPO pass and returns the output checkpoint directory.

    Like run_sft, only produces a candidate -- the eval harness decides
    promotion, this function never touches a "current" slot.
    """
    import torch
    from transformers import AutoProcessor, Trainer, TrainingArguments

    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelForVLM  # type: ignore[assignment]

    examples = load_scored_groups(
        config.scored_jsonl,
        benchmark_override=config.benchmark_override,
        split_override=config.split_override,
        min_reward_std=config.min_reward_std,
    )
    if not examples:
        raise ValueError(
            f"no GRPO examples survived loading {config.scored_jsonl} -- either it's empty "
            "or every group had zero reward variance (all rollouts agreed, or none did)"
        )
    logger.info("loaded %d GRPO examples (nonzero-variance groups only)", len(examples))

    processor = AutoProcessor.from_pretrained(config.base_model, trust_remote_code=True)
    model = AutoModelForVLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        trust_remote_code=True,
    )

    vision_config = getattr(model.config, "vision_config", None)
    if getattr(processor, "patch_size", None) is None and vision_config is not None:
        processor.patch_size = vision_config.patch_size
        processor.vision_feature_select_strategy = getattr(
            model.config, "vision_feature_select_strategy", "default"
        )

    if config.lora.enabled:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif config.kl_coef > 0:
        raise ValueError("kl_coef > 0 requires lora.enabled=True (needs a PeftModel)")

    dataset = GRPODataset(examples, processor, max_length=config.max_seq_length)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        bf16=config.bf16,
        seed=config.seed,
        report_to=[],
        remove_unused_columns=False,
    )

    grpo = GRPOTrainer(kl_coef=config.kl_coef)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_single,
    )
    # A closure, not a bound-method assignment: transformers.Trainer.train()
    # calls `self.compute_loss(...)`, which looks up the *instance*
    # attribute below rather than the class method -- binding it to `trainer`
    # via __get__ would make `self` inside GRPOTrainer.compute_loss the
    # Trainer instance, not the GRPOTrainer holding kl_coef.
    trainer.compute_loss = (
        lambda model, inputs, return_outputs=False, num_items_in_batch=None: grpo.compute_loss(
            model, inputs, return_outputs, num_items_in_batch
        )
    )

    trainer.train()

    # See training/checkpoint.py for why this isn't a plain
    # trainer.save_model() + processor.save_pretrained().
    save_candidate_checkpoint(model, processor, config.base_model, str(output_dir), config.lora.enabled)

    logger.info("candidate checkpoint written to %s -- run eval harness before promoting", output_dir)
    return str(output_dir)
