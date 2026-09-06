"""Shared candidate-checkpoint saving for training/sft.py and training/grpo.py.

Merges LoRA into the base weights when used, then writes the base model's
*original* tokenizer/processor files byte-for-byte instead of re-serializing
them through `processor.save_pretrained()`.

Why: github.com/OnePunchMonk/selfsight/issues/4. A `processor.save_pretrained()`
round trip on OpenGVLab/InternVL3-2B-hf produces a tokenizer/processor that is
JSON-identical to the hub original in every field compared (config.json,
tokenizer_config.json, preprocessor_config.json), and even the raw tokenizer
round-trips a repeated special token correctly -- yet loading the *directory*
back through the eval harness fails with an image-token accounting mismatch
that never reproduces against the hub repo. Whatever `save_pretrained` loses
isn't visible in any file we diffed, so rather than keep chasing it inside
transformers' InternVL processing code, this sidesteps it: copy the hub's
processor files unchanged, and only write our own fine-tuned model weights.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# Every file a HF multimodal processor/tokenizer might ship, beyond the model
# weights/config themselves. Not all of these exist for every model; missing
# ones are just skipped.
_PROCESSOR_FILENAMES = [
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
]


def save_candidate_checkpoint(
    model, processor, base_model: str, output_dir: str, lora_enabled: bool
) -> None:
    """Writes a standalone candidate checkpoint to `output_dir`.

    `model` is the (possibly LoRA-wrapped) trained model; `processor` is
    only used as a fallback if `base_model` isn't a resolvable hub id (e.g.
    it's already a local path -- an iteration training from a previous
    iteration's candidate).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if lora_enabled:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(output_path))
    else:
        model.save_pretrained(str(output_path))

    try:
        from huggingface_hub import snapshot_download

        snapshot_dir = Path(snapshot_download(base_model, allow_patterns=_PROCESSOR_FILENAMES))
        copied_any = False
        for filename in _PROCESSOR_FILENAMES:
            src = snapshot_dir / filename
            if src.exists():
                shutil.copy2(src, output_path / filename)
                copied_any = True
        if not copied_any:
            processor.save_pretrained(str(output_path))
    except Exception:
        # base_model isn't a hub id we can snapshot (e.g. it's already a
        # local checkpoint path) -- fall back to the normal save, which is
        # what hit #4 for a from-hub base model but may be fine here.
        processor.save_pretrained(str(output_path))
