from data.diversity import (
    RunMetrics,
    append_run_metrics,
    detect_collapse,
    load_run_history,
    summarize,
)
from data.rollouts import Rollout
from data.scoring import ScoredGroup


def _group(agreement: float, entropy: float) -> ScoredGroup:
    return ScoredGroup(
        sample_id="s0",
        prompt="p",
        majority="A",
        agreement=agreement,
        rollouts=[Rollout(sample_id="s0", draw=0, prompt="p", raw="A", extracted="A", model_id="m")],
        entropy=entropy,
    )


def test_summarize_averages_agreement_and_entropy():
    groups = [_group(1.0, 0.0), _group(0.6, 0.8), _group(0.8, 0.4)]
    metrics = summarize(groups, model="m", benchmark="b")
    assert metrics.n_groups == 3
    assert round(metrics.mean_agreement, 4) == round((1.0 + 0.6 + 0.8) / 3, 4)
    assert round(metrics.mean_entropy, 4) == round((0.0 + 0.8 + 0.4) / 3, 4)


def test_summarize_empty_raises():
    import pytest

    with pytest.raises(ValueError):
        summarize([], model="m", benchmark="b")


def test_append_and_load_run_history_filters_by_model_and_benchmark(tmp_path):
    path = tmp_path / "metrics.jsonl"
    m1 = summarize([_group(0.8, 0.5)], model="model-a", benchmark="bench-1")
    m2 = summarize([_group(0.9, 0.3)], model="model-a", benchmark="bench-2")
    m3 = summarize([_group(0.7, 0.6)], model="model-b", benchmark="bench-1")
    append_run_metrics(m1, path)
    append_run_metrics(m2, path)
    append_run_metrics(m3, path)

    history = load_run_history(path, model="model-a", benchmark="bench-1")
    assert len(history) == 1
    assert history[0].mean_agreement == 0.8


def test_load_run_history_missing_file_returns_empty(tmp_path):
    assert load_run_history(tmp_path / "does-not-exist.jsonl", model="m", benchmark="b") == []


def test_detect_collapse_flags_rising_agreement_falling_diversity():
    previous = RunMetrics(
        timestamp="t0", model="m", benchmark="b", n_groups=10, mean_agreement=0.6, mean_entropy=0.5
    )
    current = RunMetrics(
        timestamp="t1", model="m", benchmark="b", n_groups=10, mean_agreement=0.75, mean_entropy=0.3
    )
    warning = detect_collapse(current, [previous])
    assert warning is not None
    assert warning.agreement_delta > 0
    assert warning.entropy_delta < 0


def test_detect_collapse_ok_when_agreement_and_diversity_both_rise():
    previous = RunMetrics(
        timestamp="t0", model="m", benchmark="b", n_groups=10, mean_agreement=0.6, mean_entropy=0.3
    )
    current = RunMetrics(
        timestamp="t1", model="m", benchmark="b", n_groups=10, mean_agreement=0.75, mean_entropy=0.5
    )
    assert detect_collapse(current, [previous]) is None


def test_detect_collapse_no_history_returns_none():
    current = RunMetrics(
        timestamp="t1", model="m", benchmark="b", n_groups=10, mean_agreement=0.75, mean_entropy=0.3
    )
    assert detect_collapse(current, []) is None


def test_detect_collapse_ignores_small_deltas_below_threshold():
    previous = RunMetrics(
        timestamp="t0", model="m", benchmark="b", n_groups=10, mean_agreement=0.60, mean_entropy=0.50
    )
    current = RunMetrics(
        timestamp="t1", model="m", benchmark="b", n_groups=10, mean_agreement=0.61, mean_entropy=0.49
    )
    assert detect_collapse(current, [previous]) is None
