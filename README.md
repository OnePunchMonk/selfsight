# SelfSight

Self-improving post-training and evaluation for vision-language models (VLMs).

## Idea

Close the loop between a VLM's outputs and its own training signal: generate,
score/verify, and filter model rollouts on multimodal tasks, then use that
curated data for further post-training (SFT / RL). Evaluation harnesses track
whether each iteration actually improves the model, not just fits the loop.

## Layout

```
selfsight/
  training/    # post-training loops (SFT, RL/self-refinement)
  eval/        # benchmark + task evaluation harnesses
  data/        # rollout generation, scoring, filtering utilities
  configs/     # experiment configs
```

## Status

Early scaffold. No runnable code yet.

## License

MIT
