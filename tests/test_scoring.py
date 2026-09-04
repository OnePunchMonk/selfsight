from data.rollouts import Rollout, RolloutGroup
from data.scoring import SelfConsistencyScorer


def _rollout(draw: int, extracted: str) -> Rollout:
    return Rollout(
        sample_id="s0",
        draw=draw,
        prompt="p",
        raw=extracted,
        extracted=extracted,
        model_id="mock:demo",
    )


def test_unanimous_group_has_agreement_one():
    group = RolloutGroup(sample_id="s0", prompt="p", rollouts=[_rollout(i, "A") for i in range(5)])
    scored = SelfConsistencyScorer().score(group)
    assert scored.majority == "A"
    assert scored.agreement == 1.0
    assert len(scored.majority_rollouts()) == 5


def test_split_vote_picks_the_larger_side():
    rollouts = [_rollout(0, "A"), _rollout(1, "A"), _rollout(2, "A"), _rollout(3, "B"), _rollout(4, "B")]
    group = RolloutGroup(sample_id="s0", prompt="p", rollouts=rollouts)
    scored = SelfConsistencyScorer().score(group)
    assert scored.majority == "A"
    assert scored.agreement == 3 / 5
    assert [r.draw for r in scored.majority_rollouts()] == [0, 1, 2]


def test_empty_group_raises():
    import pytest

    group = RolloutGroup(sample_id="s0", prompt="p", rollouts=[])
    with pytest.raises(ValueError):
        SelfConsistencyScorer().score(group)
