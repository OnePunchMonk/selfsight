"""Rollout generation: sample k free-form responses per prompt from a base VLM.

No ground truth is consulted here — that is the eval harness's job, on a
held-out benchmark the training loop never sees. This module only produces
candidate rollouts for the self-consistency scorer (scoring.py) to judge.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from vlm_evaluation_harness.adapters.base import VLMAdapter
from vlm_evaluation_harness.adapters.registry import get_adapter
from vlm_evaluation_harness.benchmarks.loader import BenchmarkLoader, BenchmarkSample
from vlm_evaluation_harness.benchmarks.registry import get_registry
from vlm_evaluation_harness.benchmarks.schema import BenchmarkManifest
from vlm_evaluation_harness.parsing.extractor import AnswerExtractor
from vlm_evaluation_harness.prompt.formatter import PromptFormatter


@dataclass
class Rollout:
    """One sampled response to one prompt."""

    sample_id: str
    draw: int  # 0..k-1, which of the k independent samples this is
    prompt: str
    raw: str  # model's raw text
    extracted: str  # AnswerExtractor(...).normalized, the value voted on
    model_id: str


@dataclass
class RolloutGroup:
    """All k rollouts sampled for a single benchmark prompt."""

    sample_id: str
    prompt: str
    rollouts: list[Rollout] = field(default_factory=list)


class RolloutGenerator:
    """Samples k rollouts per benchmark prompt from a base VLM.

    MockAdapter is a pure function of (model_id, prompt): it has no sampling
    noise even at temperature > 0, since it exists to make the *rest* of the
    harness deterministic and testable. To still exercise self-consistency
    fully offline, when the resolved provider is `mock` and k > 1, each draw
    gets its own MockAdapter instance with a seed-suffixed model_id — standing
    in for k independent decodes. Any other provider is called k times on one
    adapter and relies on its own temperature-driven sampling, exactly as it
    would in the real loop.
    """

    def __init__(self, model_spec: str, k: int = 5, temperature: float = 0.9):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.model_spec = model_spec
        self.k = k
        self.temperature = temperature
        self._provider, self._model_id = model_spec.split(":", 1)
        self._base_adapter = get_adapter(model_spec)
        self._formatter = PromptFormatter()
        self._extractor = AnswerExtractor()

    def _adapter_for_draw(self, draw: int) -> VLMAdapter:
        if self._provider == "mock" and self.k > 1:
            return get_adapter(f"mock:{self._model_id}#s{draw}")
        return self._base_adapter

    def generate_for_benchmark(
        self,
        benchmark: str,
        split: str = "validation",
        max_samples: int | None = None,
    ) -> Iterator[RolloutGroup]:
        manifest = get_registry().get(benchmark)
        loader = BenchmarkLoader()
        for sample in loader.load(manifest, split=split, max_samples=max_samples):
            yield self.generate_for_sample(manifest, sample)

    def generate_for_sample(
        self, manifest: BenchmarkManifest, sample: BenchmarkSample
    ) -> RolloutGroup:
        formatted = self._formatter.format(manifest, sample.images, sample.text_fields)
        rollouts = []
        for draw in range(self.k):
            adapter = self._adapter_for_draw(draw)
            response = adapter.generate(
                images=formatted.images,
                prompt=formatted.text,
                system=formatted.system,
                history=formatted.history or None,
                temperature=self.temperature,
                parts=formatted.parts or None,
            )
            extraction = self._extractor.extract(response.text, manifest.answer_extraction)
            rollouts.append(
                Rollout(
                    sample_id=sample.sample_id,
                    draw=draw,
                    prompt=formatted.text,
                    raw=response.text,
                    extracted=extraction.normalized,
                    model_id=response.model_id,
                )
            )
        return RolloutGroup(sample_id=sample.sample_id, prompt=formatted.text, rollouts=rollouts)
