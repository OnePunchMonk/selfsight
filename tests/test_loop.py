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
        config, state, _stub_generate_and_train(model="candidate-0"), lambda c, m: True
    )

    assert state.current_model == "candidate-0"
    assert state.iteration == 1
    assert state.history[0].promoted is True


def test_rejected_candidate_keeps_current_model(tmp_path):
    config = _config(tmp_path)
    state = LoopState(current_model=config.current_model)

    state = run_iteration(
        config, state, _stub_generate_and_train(model="candidate-0"), lambda c, m: False
    )

    assert state.current_model == config.current_model
    assert state.iteration == 1
    assert state.history[0].promoted is False
    assert "regression" in state.history[0].reason


def test_state_persists_across_calls(tmp_path):
    config = _config(tmp_path)
    state = load_state(config.state_path, default_model=config.current_model)
    state = run_iteration(config, state, _stub_generate_and_train(), lambda c, m: True)

    reloaded = load_state(config.state_path, default_model="should-not-be-used")
    assert reloaded.current_model == state.current_model
    assert reloaded.iteration == 1
    assert len(reloaded.history) == 1


def test_cap_reached_raises_without_override(tmp_path):
    config = _config(tmp_path, iteration_cap=1)
    state = LoopState(current_model=config.current_model, iteration=1)

    try:
        run_iteration(config, state, _stub_generate_and_train(), lambda c, m: True)
        assert False, "expected CapReachedError"
    except CapReachedError:
        pass


def test_cap_reached_proceeds_with_override(tmp_path):
    config = _config(tmp_path, iteration_cap=1)
    state = LoopState(current_model=config.current_model, iteration=1)

    state = run_iteration(
        config, state, _stub_generate_and_train(), lambda c, m: True, override_cap=True
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
        run_iteration(config, state, generate_and_train, lambda c, m: True)
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
        config, state, generate_and_train, lambda c, m: True, override_collapse=True
    )
    assert state.iteration == 1


def test_no_collapse_when_both_agreement_and_diversity_improve(tmp_path):
    config = _config(tmp_path)
    state = LoopState(current_model=config.current_model)
    prior = summarize([_group(0.5, 0.3)], model=config.current_model, benchmark=config.train_benchmark)
    append_run_metrics(prior, config.metrics_path)

    generate_and_train = _stub_generate_and_train(agreement=0.7, entropy=0.5)
    state = run_iteration(config, state, generate_and_train, lambda c, m: True)
    assert state.iteration == 1
