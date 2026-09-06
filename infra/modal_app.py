"""Modal app for phase 1 (rollout generation), phase 2 (LoRA SFT), and
phase 3 (GRPO).

Deliberately pinned to L4 (24GB, ~$0.80/hr on Modal as of writing) -- no
A100/H100. Qwen2-VL-2B-Instruct + LoRA + gradient accumulation fits
comfortably; bump GPU_TYPE below only if you outgrow it, and know the cost
step you're taking when you do.

Usage:
    modal run infra/modal_app.py::generate_rollouts --model mock:demo-v1 \\
        --benchmark demo_mc --k 5 --output curated.jsonl --scored-output scored.jsonl
    modal run infra/modal_app.py::train_sft --config configs/sft_qwen2vl.yaml
    modal run infra/modal_app.py::train_grpo --config configs/grpo_internvl.yaml
    modal run infra/modal_app.py::eval_model --model hf:OpenGVLab/InternVL3-2B-hf \\
        --bench spatial_count --max-samples 8
    modal run infra/modal_app.py::check_regression \\
        --baseline hf:OpenGVLab/InternVL3-2B-hf --current hf:/data/checkpoints/candidate-demo \\
        --bench spatial_count

All of these write into a persistent Modal Volume ("selfsight-data") so a
curated set (or a run's tracked eval history, at HOME=/data) produced by one
call is visible to a later call without redoing the work.
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


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=60)
def cat_file(path: str = "/data/checkpoints/candidate-demo/tokenizer_config.json") -> str:
    """Debug helper: print a small file from the volume. CPU-only, no GPU."""
    with open(path) as f:
        content = f.read()
    print(content)
    return content


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=600, cpu=4)
def diagnose_processor(
    hub_model: str = "OpenGVLab/InternVL3-2B-hf",
    local_model: str = "/data/checkpoints/candidate-demo",
    bench: str = "spatial_count",
    split: str = "validation",
    sample_index: int = 0,
) -> str:
    """No-GPU diagnostic: run the *same* image+prompt through the hub
    processor and the local (merged-checkpoint) processor, and compare the
    resulting input_ids length and image-token count -- isolates whether
    the eval-harness failure on the local checkpoint is a processor/config
    difference (this would show it) or something else entirely (this
    would come back identical, pointing at generation-time behavior
    instead).
    """
    import sys

    sys.path.insert(0, "/root/selfsight")

    from transformers import AutoProcessor
    from vlm_evaluation_harness.benchmarks.loader import BenchmarkLoader
    from vlm_evaluation_harness.benchmarks.registry import get_registry
    from vlm_evaluation_harness.prompt.formatter import PromptFormatter

    manifest = get_registry().get(bench)
    loader = BenchmarkLoader()
    sample = next(
        s
        for i, s in enumerate(loader.load(manifest, split=split, max_samples=sample_index + 1))
        if i == sample_index
    )
    formatted = PromptFormatter().format(manifest, sample.images, sample.text_fields)

    lines = []
    for label, model_id in [("hub", hub_model), ("local", local_model), ("local+fix", local_model)]:
        kwargs = {"fix_mistral_regex": True} if label == "local+fix" else {}
        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **kwargs)
            image_token_id = getattr(processor, "image_token_id", None)
            # Mirror HuggingFaceAdapter._render_prompt's chat-template path --
            # InternVLProcessor requires the rendered text to already carry an
            # <image> placeholder per image, which only apply_chat_template
            # inserts; raw formatted.text (no placeholder) raises instead.
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in formatted.images]
                    + [{"type": "text", "text": formatted.text}],
                }
            ]
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=prompt_text, images=formatted.images, return_tensors="pt")
            input_ids = inputs["input_ids"][0]
            n_image_tokens = (
                int((input_ids == image_token_id).sum()) if image_token_id is not None else -1
            )
            pixel_shape = tuple(inputs["pixel_values"].shape) if "pixel_values" in inputs else None
            line = (
                f"{label}: model={model_id} image_token_id={image_token_id} "
                f"input_ids_len={input_ids.shape[0]} n_image_tokens={n_image_tokens} "
                f"pixel_values_shape={pixel_shape}"
            )
        except Exception as exc:  # noqa: BLE001 -- diagnostic, want every variant's outcome
            line = f"{label}: model={model_id} FAILED: {exc!r}"
        print(line)
        lines.append(line)

    # Isolate the tokenizer itself, independent of apply_chat_template /
    # image processing: does encoding a raw string of repeated
    # <IMG_CONTEXT> tokens round-trip to the same count for both?
    from transformers import AutoTokenizer

    for label, model_id in [("hub", hub_model), ("local", local_model)]:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        raw = "<IMG_CONTEXT>" * 1024
        ids = tok(raw, add_special_tokens=False)["input_ids"]
        image_token_id = getattr(tok, "image_token_id", None) or 151667
        n = sum(1 for i in ids if i == image_token_id)
        line = f"{label} raw-tokenizer: len(ids)={len(ids)} n_image_tokens={n}"
        print(line)
        lines.append(line)

    return "\n".join(lines)


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
    scored_output: str | None = "scored.jsonl",
) -> str:
    """Phase 1: rollout -> self-consistency score -> filter, on Modal.

    `scored_output`, when given, also dumps every rollout (not just the
    SFT-filtered majority ones) for phase 3's GRPO to consume.
    """
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
    if scored_output is not None:
        cmd += ["--scored-output", f"{VOLUME_PATH}/{scored_output}"]

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


@app.function(image=image, gpu=GPU_TYPE, volumes={VOLUME_PATH: volume}, timeout=14400)
def train_grpo(config: str = "configs/grpo_internvl_modal_demo.yaml") -> str:
    """Phase 3: GRPO over a scored-rollouts set already sitting in the volume.

    Like train_sft, the config's `scored_jsonl`/`output_dir` should point
    under /data when running here (e.g. scored_jsonl: /data/scored.jsonl).
    """
    import sys

    sys.path.insert(0, "/root/selfsight")

    from training.grpo_cli import main as grpo_main

    sys.argv = ["training.grpo_cli", "--config", f"/root/selfsight/{config}"]
    grpo_main()
    volume.commit()
    return "done"


# HOME=/data makes vlm-evaluation-harness's HistoryStore (~/.vlm-evaluation-harness/)
# land on the persistent volume, so a baseline eval recorded by one `modal run`
# is still there for check_regression's `modal run` to compare against later --
# each invocation otherwise gets a fresh, empty container filesystem.
_EVAL_ENV = {"HOME": VOLUME_PATH}


@app.function(
    image=image, gpu=GPU_TYPE, volumes={VOLUME_PATH: volume}, timeout=7200, env=_EVAL_ENV
)
def eval_model(
    model: str = "hf:OpenGVLab/InternVL3-2B-hf",
    bench: str = "spatial_count",
    split: str = "validation",
    max_samples: int | None = 8,
) -> str:
    """Run a real (non-mock) eval-harness pass and record it to tracked
    history, for either the baseline checkpoint or a trained candidate.

    Candidate model specs are local paths under the volume, e.g.
    `hf:/data/checkpoints/candidate-demo` -- the harness's HuggingFaceAdapter
    does a plain `AutoModelForImageTextToText.from_pretrained(path)`, which
    works the same for a local directory as for a hub id.
    """
    import subprocess

    cmd = [
        "vlm-evaluation-harness",
        "eval",
        "--model",
        model,
        "--bench",
        bench,
        "--split",
        split,
        "--track",
    ]
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        result.check_returncode()
    volume.commit()
    return result.stdout


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=300, env=_EVAL_ENV)
def check_regression(
    baseline: str = "hf:OpenGVLab/InternVL3-2B-hf",
    current: str = "hf:/data/checkpoints/candidate-demo",
    bench: str = "spatial_count",
    threshold: float = 0.03,
) -> str:
    """The promotion gate: paired-significance regression check between two
    tracked runs (eval_model, above, with --track, for both). CPU-only --
    this only reads the local history file, no model inference.

    Exit code is nonzero if vlm-evaluation-harness's own `regression`
    command finds a comparable pair of runs; the printed report says
    whether any benchmark/metric is flagged as a real (not just noisy)
    regression -- that's the promote/reject decision, not this function's.
    """
    import subprocess

    cmd = [
        "vlm-evaluation-harness",
        "regression",
        "--baseline",
        baseline,
        "--current",
        current,
        "--bench",
        bench,
        "--threshold",
        str(threshold),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.stdout
