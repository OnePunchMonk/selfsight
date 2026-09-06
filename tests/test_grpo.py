import json

import torch

from training.grpo import GRPOTrainer
from training.grpo_dataset import GRPODataset, GRPOExample, _advantages_for_group, load_scored_groups


def _write_scored(tmp_path, rows):
    path = tmp_path / "scored.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def test_advantages_for_group_zero_variance_is_zero():
    assert _advantages_for_group([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    assert _advantages_for_group([0.0, 0.0]) == [0.0, 0.0]
    assert _advantages_for_group([1.0]) == [0.0]


def test_advantages_for_group_mixed_rewards():
    advantages = _advantages_for_group([1.0, 1.0, 0.0])
    # two majority (reward=1) rollouts get the same positive advantage,
    # the disagreeing one gets a negative advantage.
    assert advantages[0] == advantages[1] > 0
    assert advantages[2] < 0


def test_load_scored_groups_drops_zero_variance_groups(tmp_path):
    rows = [
        # s0: everyone agrees -- zero reward variance, should be dropped.
        {"sample_id": "demo_mc_00", "prompt": "p0", "response": "A", "extracted": "A",
         "majority": "A", "agreement": 1.0, "model_id": "m", "benchmark": "demo_mc", "split": "validation"},
        {"sample_id": "demo_mc_00", "prompt": "p0", "response": "A", "extracted": "A",
         "majority": "A", "agreement": 1.0, "model_id": "m", "benchmark": "demo_mc", "split": "validation"},
        # s1: split vote -- nonzero variance, should be kept.
        {"sample_id": "demo_mc_01", "prompt": "p1", "response": "B", "extracted": "B",
         "majority": "B", "agreement": 0.5, "model_id": "m", "benchmark": "demo_mc", "split": "validation"},
        {"sample_id": "demo_mc_01", "prompt": "p1", "response": "C", "extracted": "C",
         "majority": "B", "agreement": 0.5, "model_id": "m", "benchmark": "demo_mc", "split": "validation"},
    ]
    path = _write_scored(tmp_path, rows)
    examples = load_scored_groups(path)
    assert len(examples) == 2
    assert {e.prompt for e in examples} == {"p1"}
    advantages = sorted(e.advantage for e in examples)
    assert advantages[0] < 0 < advantages[1]


def test_load_scored_groups_requires_benchmark_and_split(tmp_path):
    rows = [
        {"sample_id": "s0", "prompt": "p", "response": "A", "extracted": "A", "majority": "A"},
        {"sample_id": "s0", "prompt": "p", "response": "B", "extracted": "B", "majority": "A"},
    ]
    path = _write_scored(tmp_path, rows)
    try:
        load_scored_groups(path)
        assert False, "expected ValueError"
    except ValueError:
        pass


class _FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = messages[0]["content"][-1]["text"]
        return f"PROMPT:{text}|"

    def __call__(self, text, images=None, return_tensors="pt", truncation=False, max_length=None):
        ids = [ord(c) % 97 for c in text]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }


def test_grpo_dataset_carries_advantage_and_masks_prompt():
    ex = GRPOExample(prompt="hello", response=" world", images=[], advantage=1.5)
    ds = GRPODataset([ex], _FakeProcessor(), max_length=64)
    item = ds[0]

    prompt_len = len("PROMPT:hello|")
    assert torch.all(item["labels"][:prompt_len] == -100)
    assert torch.all(item["labels"][prompt_len:] != -100)
    assert item["advantage"].item() == 1.5


def test_grpo_trainer_loss_sign_matches_advantage():
    class _FakeOutputs:
        def __init__(self, loss):
            self.loss = loss

    class _FakeModel:
        def __init__(self, loss):
            self._loss = loss

        def __call__(self, **kwargs):
            return _FakeOutputs(torch.tensor(self._loss))

    grpo = GRPOTrainer(kl_coef=0.0)

    # Positive advantage: loss = advantage * NLL should be positive and
    # scale with advantage (gradient descent on this pushes NLL down, i.e.
    # logprob up, for a rollout the model should imitate more).
    inputs = {"labels": torch.tensor([1, 2, 3]), "advantage": torch.tensor([2.0])}
    loss = grpo.compute_loss(_FakeModel(0.5), dict(inputs))
    assert loss.item() == 1.0

    # Negative advantage flips the sign -- descending this loss pushes NLL
    # up (logprob down) for a rollout the model should avoid.
    inputs = {"labels": torch.tensor([1, 2, 3]), "advantage": torch.tensor([-2.0])}
    loss = grpo.compute_loss(_FakeModel(0.5), dict(inputs))
    assert loss.item() == -1.0


def test_grpo_trainer_survives_the_actual_trainer_assignment_pattern():
    """Regression test for the self-binding bug found in Modal validation
    (github.com/OnePunchMonk/selfsight/issues/3): `run_sft` doesn't call
    GRPOTrainer.compute_loss directly, it assigns a closure over it as a
    `transformers.Trainer` *instance* attribute (see training/grpo.py's
    `run_grpo`), because that's the shape `Trainer.train()` actually calls
    through. Binding via `grpo.compute_loss.__get__(trainer, Trainer)`
    silently made `self` inside the method the Trainer, not the GRPOTrainer,
    so `self.kl_coef` raised AttributeError on the first real training step
    -- a failure mode the tests above never exercised, since they call the
    method directly on a real GRPOTrainer instance.
    """

    class _FakeOutputs:
        def __init__(self, loss):
            self.loss = loss

    class _FakeModel:
        def __call__(self, **kwargs):
            return _FakeOutputs(torch.tensor(0.5))

    class _FakeTrainer:
        pass

    grpo = GRPOTrainer(kl_coef=0.0)
    trainer = _FakeTrainer()
    # The exact assignment pattern training/grpo.py:run_grpo uses.
    trainer.compute_loss = (
        lambda model, inputs, return_outputs=False, num_items_in_batch=None: grpo.compute_loss(
            model, inputs, return_outputs, num_items_in_batch
        )
    )

    inputs = {"labels": torch.tensor([1, 2, 3]), "advantage": torch.tensor([2.0])}
    loss = trainer.compute_loss(_FakeModel(), inputs, num_items_in_batch=3)
    assert loss.item() == 1.0
