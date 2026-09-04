from data.filtering import filter_rollouts, write_jsonl
from data.rollouts import Rollout
from data.scoring import ScoredGroup


def _group(sample_id: str, agreement: float, votes: list[str]) -> ScoredGroup:
    rollouts = [
        Rollout(sample_id=sample_id, draw=i, prompt="p", raw=v, extracted=v, model_id="mock:demo")
        for i, v in enumerate(votes)
    ]
    majority = max(set(votes), key=votes.count)
    return ScoredGroup(
        sample_id=sample_id, prompt="p", majority=majority, agreement=agreement, rollouts=rollouts
    )


def test_low_agreement_group_is_dropped():
    groups = [_group("s0", agreement=0.4, votes=["A", "A", "B", "B", "C"])]
    curated = filter_rollouts(groups, min_agreement=0.6)
    assert curated == []


def test_high_agreement_keeps_only_majority_rollouts():
    groups = [_group("s1", agreement=0.8, votes=["A", "A", "A", "A", "B"])]
    curated = filter_rollouts(groups, min_agreement=0.6)
    assert len(curated) == 4
    assert all(example.extracted == "A" for example in curated)


def test_write_jsonl_round_trips(tmp_path):
    import json

    groups = [_group("s2", agreement=1.0, votes=["A", "A"])]
    curated = filter_rollouts(groups, min_agreement=0.6)
    out = tmp_path / "curated.jsonl"
    write_jsonl(curated, out)

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["sample_id"] == "s2"
    assert row["extracted"] == "A"
