from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microtensor.core.protocol import ArtifactFormat, LoadManifest
from microtensor.harness.contract import Request
from microtensor.harness.engines.gguf import GgufEngine
from microtensor.training.arena import (
    BASE_MODEL,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_QUANT,
    MAX_OUTPUT_TOKENS,
)
from microtensor.training.dataset import TrainError, load_sft, resolve_sft

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class EvalResult:
    tasks: int
    completed: int
    json_valid: int
    exact_tasks: int
    true_positive: int
    false_positive: int
    false_negative: int
    tool_call_f1: float
    ttft_p95_ms: int
    total_p95_ms: int


def _payload(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    blocks = _FENCE.findall(candidate)
    if blocks:
        candidate = max(blocks, key=len).strip()
    try:
        value = json.loads(candidate)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("tool_calls"), list):
        return None
    return value


def _calls(value: dict[str, Any] | None) -> set[str]:
    if value is None:
        return set()
    calls: set[str] = set()
    for call in value["tool_calls"]:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            continue
        # Canonical JSON makes argument ordering irrelevant while preserving
        # JSON types: 5, 5.0, "5", and true remain distinct calls.
        calls.add(
            json.dumps(
                {"name": name, "arguments": arguments},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return calls


def _p95(values: list[float]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return int(math.ceil(ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]))


def evaluate_gguf(
    model: Path,
    data: Path,
    *,
    limit: int = 0,
    heldout_fraction: float = 0.08,
    context_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    show_failures: int = 5,
) -> EvalResult:
    """Run validator-style greedy chat generation on the local held-out split."""
    if not model.is_file():
        raise TrainError(f"GGUF model {model} is missing; run `mt train export` first")
    examples = load_sft(resolve_sft(data))
    heldout = max(1, int(len(examples) * heldout_fraction))
    selected = examples[:heldout]
    if limit > 0:
        selected = selected[:limit]

    manifest = LoadManifest(
        format=ArtifactFormat.GGUF,
        quantization=DEFAULT_QUANT,
        entrypoint=model.name,
        max_input={"tokens": context_tokens},
        base_model=BASE_MODEL,
    )
    engine = GgufEngine()
    try:
        engine.load(model.parent, manifest)
    except Exception as exc:
        raise TrainError(f"validator GGUF engine could not load {model}: {exc}") from exc

    completed = valid = exact = tp = fp = fn = 0
    ttft: list[float] = []
    totals: list[float] = []
    failures = 0
    try:
        for index, example in enumerate(selected, start=1):
            response = engine.generate(
                Request(
                    task_ref=example.ref,
                    prompt=example.prompt,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    chat=True,
                )
            )
            gold = _payload(example.completion)
            predicted = _payload(str(response.output)) if response.ok else None
            gold_calls = _calls(gold)
            predicted_calls = _calls(predicted)

            if response.ok:
                completed += 1
                ttft.append(response.ttft_ms)
                totals.append(response.total_ms)
            if predicted is not None:
                valid += 1
            if predicted_calls == gold_calls:
                exact += 1
            tp += len(predicted_calls & gold_calls)
            fp += len(predicted_calls - gold_calls)
            fn += len(gold_calls - predicted_calls)

            if predicted_calls != gold_calls and failures < show_failures:
                failures += 1
                detail = response.error if not response.ok else str(response.output)
                print(f"\nFAIL {example.ref}\n  gold: {example.completion}\n  pred: {detail}")
            print(
                f"\r[{index:>3}/{len(selected)}] "
                f"json {valid / index:.1%} exact {exact / index:.1%}",
                end="",
                flush=True,
            )
    finally:
        engine.unload()
    print()

    denominator = 2 * tp + fp + fn
    f1 = (2 * tp / denominator) if denominator else 1.0
    return EvalResult(
        tasks=len(selected),
        completed=completed,
        json_valid=valid,
        exact_tasks=exact,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        tool_call_f1=f1,
        ttft_p95_ms=_p95(ttft),
        total_p95_ms=_p95(totals),
    )
