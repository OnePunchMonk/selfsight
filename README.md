# SelfSight

Self-improving post-training and evaluation for vision-language models (VLMs).

## Idea

Close the loop between a VLM's outputs and its own training signal: generate,
score/verify, and filter model rollouts on multimodal tasks, then use that
curated data for further post-training (SFT / RL). Evaluation harnesses track
whether each iteration actually improves the model, not just fits the loop.

Promotion of a candidate checkpoint back into the "current" slot is gated by
[vlm-evaluation-harness](https://github.com/OnePunchMonk/vlm-evaluation-harness)
on a held-out benchmark set, via a paired-significance regression test rather
than a threshold on aggregate score deltas — this is what keeps the loop
honest as a recursive self-improvement (RSI) system: the eval gate never
shares code with the in-loop self-consistency scorer it's checking.

## Layout

```
selfsight/
  data/        # rollout generation, self-consistency scoring, filtering — built
  training/    # LoRA SFT over curated rollouts — built; RL/self-refinement (phase 3) not yet
  eval/        # delegates to vlm-evaluation-harness (separate repo, the promotion gate)
  configs/     # per-iteration training recipes
  infra/       # Modal app for rollout generation + SFT training
```

## Quickstart

```bash
pip install -e ".[dev,training]"
pytest tests/ -q

# phase 1: rollout -> self-consistency score -> filter (offline, mock adapter)
python -m data.cli --model mock:demo-v1 --benchmark demo_mc --k 5 \
    --min-agreement 0.6 --output curated.jsonl

# phase 2: LoRA SFT over the curated set
python -m training.cli --config configs/sft_qwen2vl.yaml
```

Both steps also run on [Modal](https://modal.com) (`infra/modal_app.py`),
pinned to an L4 GPU — no A100/H100 for this phase's model size:

```bash
modal run infra/modal_app.py::generate_rollouts --model mock:demo-v1 --benchmark demo_mc
modal run infra/modal_app.py::train_sft --config configs/sft_qwen2vl.yaml
```

## Status

Phase 0 (eval harness) and phase 1 (rollout generation + scoring + filtering)
are done. Phase 2 (LoRA SFT, one full loop iteration) is built and tested
offline; running it against the eval-harness gate on a real checkpoint is the
next step. Phase 3 (RL / self-refinement) is not started.

## License

MIT
