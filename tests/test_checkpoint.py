import json
from unittest.mock import patch

from training.checkpoint import save_candidate_checkpoint


class _FakeModel:
    def __init__(self, merged=False):
        self.merged = merged
        self.saved_to = None

    def merge_and_unload(self):
        return _FakeModel(merged=True)

    def save_pretrained(self, path):
        self.saved_to = path
        with open(f"{path}/config.json", "w") as f:
            json.dump({"merged": self.merged}, f)


class _FakeProcessor:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, path):
        self.saved_to = path


def test_save_candidate_checkpoint_copies_hub_processor_files(tmp_path):
    hub_snapshot = tmp_path / "hub_snapshot"
    hub_snapshot.mkdir()
    (hub_snapshot / "tokenizer_config.json").write_text('{"model_max_length": 8192}')
    (hub_snapshot / "chat_template.jinja").write_text("{{ messages }}")

    output_dir = tmp_path / "candidate"
    model = _FakeModel()
    processor = _FakeProcessor()

    with patch("huggingface_hub.snapshot_download", return_value=str(hub_snapshot)):
        save_candidate_checkpoint(model, processor, "org/some-model", str(output_dir), lora_enabled=True)

    assert (output_dir / "tokenizer_config.json").read_text() == '{"model_max_length": 8192}'
    assert (output_dir / "chat_template.jinja").read_text() == "{{ messages }}"
    # The model saved was the merged one, not the LoRA-wrapped original.
    saved_config = json.loads((output_dir / "config.json").read_text())
    assert saved_config == {"merged": True}
    # Processor.save_pretrained was never called -- files were copied instead.
    assert processor.saved_to is None


def test_save_candidate_checkpoint_falls_back_when_snapshot_download_fails(tmp_path):
    output_dir = tmp_path / "candidate"
    model = _FakeModel()
    processor = _FakeProcessor()

    with patch("huggingface_hub.snapshot_download", side_effect=OSError("not a hub id")):
        save_candidate_checkpoint(
            model, processor, "/already/local/path", str(output_dir), lora_enabled=False
        )

    assert processor.saved_to == str(output_dir)
    assert model.saved_to == str(output_dir)


def test_save_candidate_checkpoint_without_lora_saves_model_directly(tmp_path):
    hub_snapshot = tmp_path / "hub_snapshot"
    hub_snapshot.mkdir()
    (hub_snapshot / "tokenizer_config.json").write_text("{}")

    output_dir = tmp_path / "candidate"
    model = _FakeModel()
    processor = _FakeProcessor()

    with patch("huggingface_hub.snapshot_download", return_value=str(hub_snapshot)):
        save_candidate_checkpoint(model, processor, "org/some-model", str(output_dir), lora_enabled=False)

    saved_config = json.loads((output_dir / "config.json").read_text())
    assert saved_config == {"merged": False}
