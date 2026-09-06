"""Phase 4: automate rollout -> train -> eval -> gate, bounded by the RSI
hard iteration cap and human-checkpoint rule from the architecture scope.

This is deliberately *not* a while-True daemon. `run_iteration` advances
the loop by exactly one iteration and refuses to run once `iteration_cap`
is reached unless explicitly overridden -- automating up to the cap, not
past it, per the RSI scope's "no unattended run past N iterations without a
human reviewing the eval trajectory, sample rollouts, and diversity
metrics" rule. A cron/wrapper script calling this repeatedly still stops
there; the override is a conscious, separate action, not a config default.

The training and eval/gate steps are injected (`generate_and_train_fn`,
`evaluate_and_gate_fn`) so the cap/collapse/promote/reject orchestration
logic here is unit-testable without a GPU or real model calls -- see
`default_generate_and_train` and `default_evaluate_and_gate` for what
actually runs in production, and tests/test_loop.py for the orchestration
tests using stubs instead.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from data.diversity import CollapseWarning, RunMetrics, detect_collapse, load_run_history, summarize


class CapReachedError(RuntimeError):
    """Raised instead of silently running past the RSI iteration cap."""


class CollapseDetectedError(RuntimeError):
    """Raised instead of silently training on a rollout set that shows the
    reward-hacking collapse signature (rising agreement, falling diversity).
    """


@dataclass
class LoopConfig:
    # Starting point: a hub id or a previous iteration's candidate path.
    current_model: str
    train_benchmark: str
    eval_benchmark: str
    split: str = "validation"
    k: int = 5
    temperature: float = 0.9
    min_agreement: float = 0.6

    training_mode: str = "sft"  # "sft" | "grpo"
    # Path to an existing SFTConfig/GRPOConfig YAML -- base_model,
    # curated_jsonl/scored_jsonl, and output_dir are overridden per
    # iteration; everything else (LoRA rank, LR, etc.) comes from this file.
    training_config_path: str = "configs/sft_internvl_real_demo.yaml"

    work_dir: str = "loop_work"
    state_path: str = "loop_state.json"
    metrics_path: str = "loop_metrics.jsonl"

    iteration_cap: int = 5
    regression_threshold: float = 0.03
    min_reward_std: float = 1e-4  # only used when training_mode == "grpo"

    @classmethod
    def from_yaml(cls, path: str | Path) -> LoopConfig:
        import yaml

        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**raw)


@dataclass
class IterationRecord:
    iteration: int
    timestamp: str
    train_benchmark: str
    eval_benchmark: str
    n_curated_examples: int
    mean_agreement: float
    mean_entropy: float
    promoted: bool
    candidate_model: str
    reason: str


@dataclass
class LoopState:
    current_model: str
    iteration: int = 0
    history: list[IterationRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> LoopState:
        history = [IterationRecord(**r) for r in data.get("history", [])]
        return cls(current_model=data["current_model"], iteration=data.get("iteration", 0), history=history)


def load_state(path: str | Path, default_model: str) -> LoopState:
    path = Path(path)
    if not path.exists():
        return LoopState(current_model=default_model)
    return LoopState.from_dict(json.loads(path.read_text()))


def save_state(state: LoopState, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2))


# (n_curated_examples, mean_agreement, mean_entropy, candidate_model_path)
GenerateAndTrainFn = Callable[["LoopConfig", "LoopState", int], tuple[int, float, float, str]]
# True if the candidate should be promoted (no flagged regression).
EvaluateAndGateFn = Callable[["LoopConfig", str], bool]


def run_iteration(
    config: LoopConfig,
    state: LoopState,
    generate_and_train_fn: GenerateAndTrainFn,
    evaluate_and_gate_fn: EvaluateAndGateFn,
    override_cap: bool = False,
    override_collapse: bool = False,
) -> LoopState:
    """Advances the loop by exactly one iteration and returns the updated
    (and already-persisted) state.

    Raises CapReachedError / CollapseDetectedError rather than proceeding
    silently -- both require an explicit override from a human, not a
    config flag left on by default.
    """
    if state.iteration >= config.iteration_cap and not override_cap:
        raise CapReachedError(
            f"iteration {state.iteration} has reached the cap ({config.iteration_cap}) -- "
            "a human needs to review the eval trajectory, sample rollouts, and diversity "
            "metrics before continuing (pass override_cap=True to proceed)"
        )

    n_examples, mean_agreement, mean_entropy, candidate_model = generate_and_train_fn(
        config, state, state.iteration
    )

    current_run = RunMetrics(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=state.current_model,
        benchmark=config.train_benchmark,
        n_groups=0,  # filled in by the caller's own metrics recording; not needed for the check
        mean_agreement=mean_agreement,
        mean_entropy=mean_entropy,
    )
    prior_runs = load_run_history(config.metrics_path, model=state.current_model, benchmark=config.train_benchmark)
    collapse: CollapseWarning | None = detect_collapse(current_run, prior_runs)
    if collapse and not override_collapse:
        raise CollapseDetectedError(
            f"{collapse.summary()} (pass override_collapse=True to train on this rollout "
            "set anyway, after reviewing it)"
        )

    promoted = evaluate_and_gate_fn(config, candidate_model)

    record = IterationRecord(
        iteration=state.iteration,
        timestamp=datetime.now(timezone.utc).isoformat(),
        train_benchmark=config.train_benchmark,
        eval_benchmark=config.eval_benchmark,
        n_curated_examples=n_examples,
        mean_agreement=mean_agreement,
        mean_entropy=mean_entropy,
        promoted=promoted,
        candidate_model=candidate_model,
        reason="no flagged regression" if promoted else "flagged regression on eval_benchmark",
    )
    state.history.append(record)
    state.iteration += 1
    if promoted:
        state.current_model = candidate_model

    save_state(state, config.state_path)
    return state


def default_generate_and_train(
    config: LoopConfig, state: LoopState, iteration: int
) -> tuple[int, float, float, str]:
    """The real (GPU-requiring) implementation of one iteration's rollout
    generation + self-consistency scoring/filtering + training. Not covered
    by fast unit tests -- see training/sft.py / training/grpo.py, which are
    exercised on real Modal GPU runs instead.
    """
    from data.filtering import filter_rollouts, write_jsonl
    from data.rollouts import RolloutGenerator
    from data.scoring import SelfConsistencyScorer, write_scored_groups_jsonl

    work_dir = Path(config.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    curated_path = work_dir / f"curated-{iteration}.jsonl"
    scored_path = work_dir / f"scored-{iteration}.jsonl"

    generator = RolloutGenerator(state.current_model, k=config.k, temperature=config.temperature)
    groups = list(generator.generate_for_benchmark(config.train_benchmark, split=config.split))
    scored = SelfConsistencyScorer().score_all(groups)

    write_scored_groups_jsonl(scored, scored_path, benchmark=config.train_benchmark, split=config.split)
    curated = filter_rollouts(
        scored,
        min_agreement=config.min_agreement,
        benchmark=config.train_benchmark,
        split=config.split,
    )
    write_jsonl(curated, curated_path)

    run_metrics = summarize(scored, model=state.current_model, benchmark=config.train_benchmark)
    from data.diversity import append_run_metrics

    append_run_metrics(run_metrics, config.metrics_path)

    output_dir = str(work_dir / f"candidate-{iteration}")
    if config.training_mode == "grpo":
        from training.config import GRPOConfig
        from training.grpo import run_grpo

        cfg = GRPOConfig.from_yaml(config.training_config_path)
        cfg.base_model = state.current_model
        cfg.scored_jsonl = str(scored_path)
        cfg.output_dir = output_dir
        cfg.min_reward_std = config.min_reward_std
        run_grpo(cfg)
    else:
        from training.config import SFTConfig
        from training.sft import run_sft

        cfg = SFTConfig.from_yaml(config.training_config_path)
        cfg.base_model = state.current_model
        cfg.curated_jsonl = str(curated_path)
        cfg.output_dir = output_dir
        run_sft(cfg)

    return len(curated), run_metrics.mean_agreement, run_metrics.mean_entropy, output_dir


def default_evaluate_and_gate(config: LoopConfig, candidate_model: str) -> bool:
    """The real (GPU-requiring) implementation of the promotion check:
    tracks a real eval run for the candidate, then diffs it against the
    current model's latest tracked run on the same eval_benchmark via
    vlm-evaluation-harness's own paired-significance regression check.
    """
    from vlm_evaluation_harness.adapters.registry import get_adapter
    from vlm_evaluation_harness.engine.runner import EvalConfig, EvalRunner
    from vlm_evaluation_harness.tracking import HistoryStore, compare_models

    baseline_spec = f"hf:{config.current_model}" if ":" not in config.current_model else config.current_model
    candidate_spec = f"hf:{candidate_model}"

    history = HistoryStore()
    if history.latest(baseline_spec, config.eval_benchmark) is None:
        runner = EvalRunner(get_adapter(baseline_spec))
        result = runner.run(EvalConfig(model_spec=baseline_spec, benchmark=config.eval_benchmark, split=config.split))
        history.record_result(result)

    runner = EvalRunner(get_adapter(candidate_spec))
    result = runner.run(EvalConfig(model_spec=candidate_spec, benchmark=config.eval_benchmark, split=config.split))
    history.record_result(result)

    deltas = compare_models(
        history, baseline_spec, candidate_spec, benchmarks=[config.eval_benchmark], threshold=config.regression_threshold
    )
    return len(deltas) > 0 and not any(d.flagged for d in deltas)
