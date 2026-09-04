"""Integration test: real vlm-evaluation-harness mock adapter + demo_mc benchmark.

Proves the pipeline is actually wired into the harness's manifest/loader/
formatter/extractor, not reimplementing them, and runs fully offline.
"""

from data.filtering import filter_rollouts
from data.rollouts import RolloutGenerator
from data.scoring import SelfConsistencyScorer


def test_rollout_generation_against_demo_mc():
    generator = RolloutGenerator("mock:demo-v1", k=5, temperature=0.9)
    groups = list(generator.generate_for_benchmark("demo_mc", split="validation"))

    assert len(groups) == 12  # demo_mc's fixture size
    for group in groups:
        assert len(group.rollouts) == 5
        # first_letter/uppercase extraction on a multiple-choice prompt
        assert all(r.extracted in {"A", "B", "C", "D", "E", ""} for r in group.rollouts)


def test_seed_suffixed_mock_draws_actually_vary():
    """Without the per-draw model_id suffix, MockAdapter would return the
    identical answer 5/5 times and self-consistency would be untestable."""
    generator = RolloutGenerator("mock:demo-v1", k=5, temperature=0.9)
    groups = list(generator.generate_for_benchmark("demo_mc", split="validation", max_samples=3))
    extracted_per_group = [{r.extracted for r in g.rollouts} for g in groups]
    assert any(len(s) > 1 for s in extracted_per_group)


def test_end_to_end_produces_curated_examples():
    generator = RolloutGenerator("mock:demo-v1", k=5, temperature=0.9)
    groups = list(generator.generate_for_benchmark("demo_mc", split="validation"))
    scored = SelfConsistencyScorer().score_all(groups)

    # demo_mc's mock output happens to split 2/2/1 on this seeding (agreement
    # 0.4): a threshold below that keeps everything, one above keeps nothing.
    # Exercise both sides of the gate rather than assume a specific split.
    permissive = filter_rollouts(scored, min_agreement=0.4)
    strict = filter_rollouts(scored, min_agreement=0.8)

    assert len(strict) == 0
    assert len(permissive) == sum(len(g.majority_rollouts()) for g in scored)
    assert len(permissive) > 0
    for example in permissive:
        assert example.agreement >= 0.4
