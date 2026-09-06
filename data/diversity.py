"""Run-level diversity tracking and reward-hacking/collapse detection.

The RSI scope this project runs under (see the architecture-scope doc) calls
out a specific failure mode: self-consistency filtering can quietly select
for confident-and-wrong answers, since agreement is a proxy for correctness
that a model can satisfy by being confidently biased instead of actually
right. The signature of that failure is rising self-consistency agreement
paired with *falling* rollout diversity across iterations -- the model
isn't converging on correct answers, it's converging on a narrower set of
answers regardless of correctness. Tracking both numbers side by side, over
a run history, is what makes that visible instead of hidden behind a single
improving-looking agreement metric.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.scoring import ScoredGroup


@dataclass
class RunMetrics:
    """Aggregate self-consistency signal for one rollout-generation run."""

    timestamp: str
    model: str
    benchmark: str
    n_groups: int
    mean_agreement: float
    mean_entropy: float


def summarize(groups: list[ScoredGroup], model: str, benchmark: str) -> RunMetrics:
    """Aggregates a run's scored groups into the two numbers the collapse
    check compares run over run.
    """
    if not groups:
        raise ValueError("no scored groups to summarize")
    return RunMetrics(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
        benchmark=benchmark,
        n_groups=len(groups),
        mean_agreement=statistics.fmean(g.agreement for g in groups),
        mean_entropy=statistics.fmean(g.entropy for g in groups),
    )


def append_run_metrics(metrics: RunMetrics, path: str | Path) -> None:
    """Appends one run's metrics to a JSON-lines history file -- same
    append-only, diffable convention as vlm-evaluation-harness's own
    HistoryStore, kept separate since this tracks the in-loop scorer's
    signal, not the eval harness's.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(metrics)) + "\n")


def load_run_history(path: str | Path, model: str, benchmark: str) -> list[RunMetrics]:
    """Loads prior runs for the same (model, benchmark) pair, oldest first."""
    path = Path(path)
    if not path.exists():
        return []
    history = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            if row.get("model") == model and row.get("benchmark") == benchmark:
                history.append(RunMetrics(**row))
    return history


@dataclass
class CollapseWarning:
    agreement_delta: float
    entropy_delta: float

    def summary(self) -> str:
        return (
            f"possible self-consistency collapse: agreement {self.agreement_delta:+.3f}, "
            f"diversity (entropy) {self.entropy_delta:+.3f} vs. the previous run on this "
            "benchmark -- rising agreement with falling diversity means the model may be "
            "converging on confident-wrong answers, not correct ones"
        )


def detect_collapse(
    current: RunMetrics,
    history: list[RunMetrics],
    agreement_rise_threshold: float = 0.05,
    entropy_drop_threshold: float = 0.05,
) -> CollapseWarning | None:
    """Flags the reward-hacking signature: agreement up, diversity down,
    both past a noise-floor threshold, versus the immediately preceding run
    on the same (model, benchmark) pair. Returns None when there's no prior
    run to compare against, or the pattern doesn't match.
    """
    if not history:
        return None
    previous = history[-1]
    agreement_delta = current.mean_agreement - previous.mean_agreement
    entropy_delta = current.mean_entropy - previous.mean_entropy
    if agreement_delta >= agreement_rise_threshold and entropy_delta <= -entropy_drop_threshold:
        return CollapseWarning(agreement_delta=agreement_delta, entropy_delta=entropy_delta)
    return None
