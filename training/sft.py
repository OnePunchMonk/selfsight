"""LoRA SFT over data/'s curated rollouts -- phase 2 of the loop.

Deliberately not RL/self-refine (phase 3): this is the loop's first full
trip, meant to confirm the mechanism end-to-end (rollout -> score -> filter
-> train -> eval-harness gate) before adding a harder training objective.
"""

from __future__ import annotations

import logging
from pathlib import Path

from training.config import SFTConfig
from training.dataset import SFTDataset, collate_single, load_curated_examples

logger = logging.getLogger(__name__)


def run_sft(config: SFTConfig) -> str:
    """Runs one LoRA SFT pass and returns the output checkpoint directory.

    Only produces a *candidate* checkpoint -- promoting it over the current
    one is the eval harness's job (paired-significance regression test), not
    this function's. Nothing here writes to a "current"/production slot.
    """
    import torch
    from transformers import AutoProcessor, Trainer, TrainingArguments

    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelForVLM  # type: ignore[assignment]

    examples = load_curated_examples(
        config.curated_jsonl,
        benchmark_override=config.benchmark_override,
        split_override=config.split_override,
    )
    if not examples:
        raise ValueError(f"no curated examples found in {config.curated_jsonl}")
    logger.info("loaded %d curated SFT examples", len(examples))

    processor = AutoProcessor.from_pretrained(config.base_model, trust_remote_code=True)
    model = AutoModelForVLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        trust_remote_code=True,
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

    dataset = SFTDataset(examples, processor, max_length=config.max_seq_length)

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

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_single,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    logger.info("candidate checkpoint written to %s -- run eval harness before promoting", output_dir)
    return str(output_dir)
