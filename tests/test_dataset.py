import json

import torch

from training.dataset import SFTDataset, SFTExample, collate_single, load_curated_examples


def _write_curated(tmp_path, rows):
    path = tmp_path / "curated.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def test_load_curated_examples_fetches_images_from_source_benchmark(tmp_path):
    rows = [
        {
            "sample_id": "demo_mc_00",
            "prompt": "p0",
            "response": "D",
            "extracted": "D",
            "agreement": 1.0,
            "model_id": "mock:demo-v1",
            "benchmark": "demo_mc",
            "split": "validation",
        },
        {
            "sample_id": "demo_mc_01",
            "prompt": "p1",
            "response": "B",
            "extracted": "B",
            "agreement": 1.0,
            "model_id": "mock:demo-v1",
            "benchmark": "demo_mc",
            "split": "validation",
        },
    ]
    path = _write_curated(tmp_path, rows)
    examples = load_curated_examples(path)
    assert len(examples) == 2
    assert examples[0].prompt == "p0"
    assert examples[0].response == "D"
    assert len(examples[0].images) == 1


def test_load_curated_examples_requires_benchmark_and_split(tmp_path):
    rows = [{"sample_id": "s0", "prompt": "p", "response": "r"}]
    path = _write_curated(tmp_path, rows)
    try:
        load_curated_examples(path)
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


def test_sft_dataset_masks_prompt_tokens():
    ex = SFTExample(prompt="hello", response=" world", images=[])
    ds = SFTDataset([ex], _FakeProcessor(), max_length=64)
    item = ds[0]

    prompt_len = len("PROMPT:hello|")
    assert torch.all(item["labels"][:prompt_len] == -100)
    assert torch.all(item["labels"][prompt_len:] != -100)
    assert torch.equal(item["labels"][prompt_len:], item["input_ids"][prompt_len:])


def test_collate_single_adds_batch_dim():
    item = {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 2, 3])}
    batch = collate_single([item])
    assert batch["input_ids"].shape == (1, 3)


def test_collate_single_rejects_batches_larger_than_one():
    item = {"input_ids": torch.tensor([1])}
    try:
        collate_single([item, item])
        assert False, "expected ValueError"
    except ValueError:
        pass
