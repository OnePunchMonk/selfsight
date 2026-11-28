from data.diversity import append_run_metrics, summarize
from data.rollouts import Rollout
from data.scoring import ScoredGroup
from training.loop import (
    CapReachedError,
    CollapseDetectedError,
    LoopConfig,
    LoopState,
    load_state,
    run_iteration,
)


def _config(tmp_path, **overrides) -> LoopConfig:
    defaults = dict(
        current_model="org/base-model",
        train_benchmark="demo_mc",
        eval_benchmark="SpatialCount",
        work_dir=str(tmp_path / "work"),
        state_path=str(tmp_path / "state.json"),
        metrics_path=str(tmp_path / "metrics.jsonl"),
        iteration_cap=3,
    )
    defaults.update(overrides)
    return LoopConfig(**defaults)


def _stub_generate_and_train(n_examples=10, agreement=0.8, entropy=0.4, model="candidate-dir"):
    def fn(config, state, iteration):
        return n_examples, agreement, entropy, model

    return fn


def _group(agreement, entropy):
    return ScoredGroup(
        sample_id="s0",
        prompt="p",
        majority="A",
        agreement=agreement,
        rollouts=[Rollout(sample_id="s0", draw=0, prompt="p", raw="A", extracted="A", model_id="m")],
        entropy=entropy,
    )


def test_promoted_candidate_becomes_current_model(tmp_path):
    config = _config(tmp_path)
    state = LoopState(current_model=config.current_model)

    state = run_iteration(
        config, state, _stub_generate_and_train(model="candidate-0"), lambda c, b, m: True
    )

    assert state.current_model == "candidate-0"
    assert state.iteration == 1
    assert state.history[0].promoted is True


def test_rejected_candidate_keeps_current_model(tmp_path):
    config = _config(tmp_path)
    state = LoopState(current_model=config.current_model)

    state = run_iteration(
        config, state, _stub_generate_and_train(model="candidate-0"), lambda c, b, m: False
    )

    assert state.current_model == config.current_model
    assert state.iteration == 1
    assert state.history[0].promoted is False
    assert "regression" in state.history[0].reason


def test_state_persists_across_calls(tmp_path):
    config = _config(tmp_path)
    state = load_state(config.state_path, default_model=config.current_model)
    state = run_iteration(config, state, _stub_generate_and_train(), lambda c, b, m: True)

    reloaded = load_state(config.state_path, default_model="should-not-be-used")
    assert reloaded.current_model == state.current_model
    assert reloaded.iteration == 1
    assert len(reloaded.history) == 1


def test_cap_reached_raises_without_override(tmp_path):
    config = _config(tmp_path, iteration_cap=1)
    state = LoopState(current_model=config.current_model, iteration=1)

    try:
        run_iteration(config, state, _stub_generate_and_train(), lambda c, b, m: True)
        assert False, "expected CapReachedError"
    except CapReachedError:
        pass


def test_cap_reached_proceeds_with_override(tmp_path):
    config = _config(tmp_path, iteration_cap=1)
    state = LoopState(current_model=config.current_model, iteration=1)

    state = run_iteration(
        config, state, _stub_generate_and_train(), lambda c, b, m: True, override_cap=True
    )
    assert state.iteration == 2


def test_collapse_detected_raises_without_override(tmp_path):
    config = _config(tmp_path)
    state = LoopState(current_model=config.current_model)

    # Prior run: low agreement, high diversity.
    prior = summarize([_group(0.5, 0.8)], model=config.current_model, benchmark=config.train_benchmark)
    append_run_metrics(prior, config.metrics_path)

    # This iteration's rollouts: agreement way up, diversity way down --
    # the reward-hacking collapse signature.
    generate_and_train = _stub_generate_and_train(agreement=0.95, entropy=0.1)

    try:
        run_iteration(config, state, generate_and_train, lambda c, b, m: True)
        assert False, "expected CollapseDetectedError"
    except CollapseDetectedError:
        pass

    # State must not have advanced -- the whole point is refusing silently.
    assert state.iteration == 0


def test_collapse_detected_proceeds_with_override(tmp_path):
    config = _config(tmp_path)
    state = LoopState(current_model=config.current_model)
    prior = summarize([_group(0.5, 0.8)], model=config.current_model, benchmark=config.train_benchmark)
    append_run_metrics(prior, config.metrics_path)

    generate_and_train = _stub_generate_and_train(agreement=0.95, entropy=0.1)
    state = run_iteration(
        config, state, generate_and_train, lambda c, b, m: True, override_collapse=True
    )
    assert state.iteration == 1


def test_no_collapse_when_both_agreement_and_diversity_improve(tmp_path):
    config = _config(tmp_path)
    state = LoopState(current_model=config.current_model)
    prior = summarize([_group(0.5, 0.3)], model=config.current_model, benchmark=config.train_benchmark)
    append_run_metrics(prior, config.metrics_path)

    generate_and_train = _stub_generate_and_train(agreement=0.7, entropy=0.5)
    state = run_iteration(config, state, generate_and_train, lambda c, b, m: True)
    assert state.iteration == 1


def test_gate_baseline_is_currently_promoted_model_not_original_config_model(tmp_path):
    """Iteration 0 promotes B from A. Iteration 1 must then gate C against B
    (the currently active checkpoint), not against A (config.current_model,
    which goes stale after the first promotion)."""
    config = _config(tmp_path)
    state = LoopState(current_model=config.current_model)  # "org/base-model" == A

    seen_baselines = []

    def gate(cfg, baseline_model, candidate_model):
        seen_baselines.append(baseline_model)
        return True

    state = run_iteration(config, state, _stub_generate_and_train(model="B"), gate)
    assert state.current_model == "B"

    state = run_iteration(config, state, _stub_generate_and_train(model="C"), gate)
    assert state.current_model == "C"

    assert seen_baselines == ["org/base-model", "B"]


def test_config_rejects_identical_train_and_eval_benchmark(tmp_path):
    try:
        _config(tmp_path, eval_benchmark="demo_mc")  # same as train_benchmark
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_collapse_uses_history_before_generator_appends_current_run(tmp_path):
    import pytest

    config = _config(tmp_path)
    state = LoopState(current_model=config.current_model)
    append_run_metrics(summarize([_group(0.5, 0.8)], state.current_model,
                                 config.train_benchmark), config.metrics_path)

    def generate(cfg, state, iteration):
        append_run_metrics(summarize([_group(0.95, 0.1)], state.current_model,
                                     cfg.train_benchmark), cfg.metrics_path)
        return 10, 0.95, 0.1, "candidate"

    with pytest.raises(CollapseDetectedError):
        run_iteration(config, state, generate, lambda *args: True)
    assert state.iteration == 0


def test_default_path_checks_collapse_before_training(tmp_path, monkeypatch):
    import pytest
    import data.rollouts
    import training.sft
    from training.loop import default_generate_and_train

    config = _config(tmp_path)
    state = LoopState(current_model=config.current_model)
    append_run_metrics(summarize([_group(0.5, 0.8)], state.current_model,
                                 config.train_benchmark), config.metrics_path)
    from data.rollouts import RolloutGroup
    group = RolloutGroup(sample_id="s0", prompt="p", rollouts=[
        Rollout(sample_id="s0", draw=i, prompt="p", raw="A", extracted="A", model_id="m")
        for i in range(5)
    ])
    specs = []
    monkeypatch.setattr(data.rollouts.RolloutGenerator, "__init__",
                        lambda self, model_spec, **kw: specs.append(model_spec))
    monkeypatch.setattr(data.rollouts.RolloutGenerator, "generate_for_benchmark", lambda *a, **kw: iter([group]))
    trained = []
    monkeypatch.setattr(training.sft, "run_sft", lambda cfg: trained.append(cfg))
    with pytest.raises(CollapseDetectedError):
        run_iteration(config, state, default_generate_and_train, lambda *args: True)
    assert trained == []
    assert specs == ["hf:org/base-model"]
    assert state.iteration == 0
    run_iteration(config, state, default_generate_and_train, lambda *args: True,
                  override_collapse=True)
    assert len(trained) == 1
    assert state.iteration == 1


def test_staged_collapse_survives_promotion_retry_and_override(tmp_path):
    import pytest
    from training.loop import RolloutArtifact
    from data.diversity import RunMetrics, load_run_history

    for mode in ('sft', 'grpo'):
        root = tmp_path / mode
        config = _config(root, training_mode=mode)
        state = LoopState(current_model=config.current_model)
        trained = []
        values = iter([(0.6, 0.8), (0.9, 0.2), (0.9, 0.2), (0.9, 0.2)])

        def generate(cfg, st, iteration):
            agreement, entropy = next(values)
            return RolloutArtifact(10, RunMetrics('now', st.current_model,
                cfg.train_benchmark, 5, agreement, entropy), 'curated', 'scored')

        def train(cfg, st, iteration, artifact):
            trained.append(iteration)
            return f'candidate-{iteration}'

        kwargs = dict(generate_fn=generate, train_fn=train,
                      evaluate_and_gate_fn=lambda *args: True)
        run_iteration(config, state, **kwargs)
        assert state.current_model == 'candidate-0'
        state = load_state(config.state_path, config.current_model)
        for _ in range(2):
            with pytest.raises(CollapseDetectedError):
                run_iteration(config, state, **kwargs)
        assert trained == [0]
        rows = load_run_history(config.metrics_path, None, config.train_benchmark)
        assert len(rows) == 2
        assert rows[-1].status == 'blocked'
        run_iteration(config, state, override_collapse=True, **kwargs)
        assert trained == [0, 1]
        rows = load_run_history(config.metrics_path, None, config.train_benchmark)
        assert len(rows) == 2
        assert rows[-1].override_collapse
        assert state.history[-1].override_collapse
        assert state.history[-1].run_id == rows[-1].run_id
