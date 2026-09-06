"""Modal app for phase 1 (rollout generation) and phase 2 (LoRA SFT).

Deliberately pinned to L4 (24GB, ~$0.80/hr on Modal as of writing) -- no
A100/H100. Qwen2-VL-2B-Instruct + LoRA + gradient accumulation fits
comfortably; bump GPU_TYPE below only if you outgrow it, and know the cost
step you're taking when you do.

Usage:
    modal run infra/modal_app.py::generate_rollouts --model mock:demo-v1 \\
        --benchmark demo_mc --k 5 --output curated.jsonl
    modal run infra/modal_app.py::train_sft --config configs/sft_qwen2vl.yaml

Both write into a persistent Modal Volume ("selfsight-data") so a curated
set produced by one run is visible to a later training run without
re-downloading it.
"""

from __future__ import annotations

from pathlib import Path

import modal

GPU_TYPE = "L4"  # explicitly not A100/H100 -- keep this modest.
REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("selfsight")

volume = modal.Volume.from_name("selfsight-data", create_if_missing=True)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.55,<5",
        "accelerate",
        "peft",
        "datasets",
        "pillow",
        "pyyaml",
        "vlm-evaluation-harness @ git+https://github.com/OnePunchMonk/vlm-evaluation-harness",
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/selfsight",
        copy=True,
        ignore=["**/.venv/**", "**/.git/**", "**/__pycache__/**", "**/*.egg-info/**"],
    )
)


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=3600)
def generate_rollouts(
    model: str = "mock:demo-v1",
    benchmark: str = "demo_mc",
    split: str = "validation",
    max_samples: int | None = None,
    k: int = 5,
    temperature: float = 0.9,
    min_agreement: float = 0.6,
    output: str = "curated.jsonl",
) -> str:
    """Phase 1: rollout -> self-consistency score -> filter, on Modal."""
    import subprocess
    import sys

    sys.path.insert(0, "/root/selfsight")

    out_path = f"{VOLUME_PATH}/{output}"
    cmd = [
        sys.executable,
        "-m",
        "data.cli",
        "--model",
        model,
        "--benchmark",
        benchmark,
        "--split",
        split,
        "--k",
        str(k),
        "--temperature",
        str(temperature),
        "--min-agreement",
        str(min_agreement),
        "--output",
        out_path,
    ]
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]

    subprocess.run(cmd, cwd="/root/selfsight", check=True)
    volume.commit()
    return out_path


@app.function(image=image, gpu=GPU_TYPE, volumes={VOLUME_PATH: volume}, timeout=14400)
def train_sft(config: str = "configs/sft_qwen2vl_modal_demo.yaml") -> str:
    """Phase 2: LoRA SFT over a curated set already sitting in the volume.

    The config's `curated_jsonl` and `output_dir` are resolved relative to
    the Modal volume, not the local filesystem -- point them at paths under
    /data (e.g. curated_jsonl: /data/curated.jsonl) when running here.
    """
    import sys

    sys.path.insert(0, "/root/selfsight")

    from training.cli import main as train_main

    sys.argv = ["training.cli", "--config", f"/root/selfsight/{config}"]
    train_main()
    volume.commit()
    return "done"
