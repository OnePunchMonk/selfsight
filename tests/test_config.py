from training.config import SFTConfig


def test_from_yaml_loads_repo_default(tmp_path):
    config = SFTConfig.from_yaml("configs/sft_qwen2vl.yaml")
    assert config.base_model == "Qwen/Qwen2-VL-2B-Instruct"
    assert config.lora.enabled is True
    assert config.lora.r == 16


def test_from_yaml_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "base_model: foo/bar\n"
        "output_dir: out\n"
        "curated_jsonl: curated.jsonl\n"
        "lora:\n"
        "  enabled: false\n"
    )
    config = SFTConfig.from_yaml(path)
    assert config.base_model == "foo/bar"
    assert config.lora.enabled is False
    # untouched lora fields keep their dataclass defaults
    assert config.lora.r == 16
