#!/usr/bin/env python3
"""CPU latency harness matching the validator GGUF pin.

THREADS=1, GPU_LAYERS=0, greedy decode, real corpus prompts (not synthetic
prefill lengths). Reports (exact-match, p95_ms) as a pair plus TTFT / total
latency percentiles and decision/copy splits.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from microtensor.core.protocol import ArtifactFormat, LoadManifest
from microtensor.harness.contract import Request
from microtensor.harness.engines.gguf import GPU_LAYERS, THREADS, GgufEngine
from microtensor.harness.limits import pin_threads
from microtensor.scoring.metrics import span_accuracy
from microtensor.training.arena import (
    BASE_MODEL,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_QUANT,
    MAX_P95_MS,
    REFERENCE_COST_MS,
)
from microtensor.training.dataset import TrainError, load_sft, resolve_sft


@dataclass(frozen=True, slots=True)
class LatencyReport:
    tasks: int
    exact_match: float
    decision_acc: float
    copy_given_flag: float
    mean_fbeta: float
    input_tokens_p50: int
    input_tokens_p95: int
    input_tokens_p99: int
    input_tokens_max: int
    ttft_p50_ms: int
    ttft_p95_ms: int
    total_p50_ms: int
    total_p95_ms: int
    total_p99_ms: int
    mean_output_tokens: float
    max_output_tokens_cap: int
    threads: int
    gpu_layers: int
    reference_cost_ms: float
    arena_max_p95_ms: int
    cost_ok: bool
    size_bytes: int


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def _parse_spans(text: str) -> list[str] | None:
    try:
        payload = json.loads(text.strip())
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or "unsupported" not in payload:
        return None
    spans = payload.get("unsupported")
    if not isinstance(spans, list) or not all(isinstance(s, str) for s in spans):
        return None
    return spans


def load_holdout_examples(data: Path, holdout: Path) -> list[Any]:
    refs = {line.strip() for line in holdout.read_text(encoding="utf-8").splitlines() if line.strip()}
    examples = [ex for ex in load_sft(resolve_sft(data)) if ex.ref in refs]
    if not examples:
        raise TrainError(f"no examples matched holdout refs in {holdout}")
    missing = refs - {ex.ref for ex in examples}
    if missing:
        raise TrainError(f"holdout refs missing from data: {sorted(missing)[:5]}")
    return examples


def run_latency_harness(
    model: Path,
    examples: list[Any],
    *,
    max_output_tokens: int = 40,
    context_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    warmup: int = 2,
    limit: int = 0,
) -> LatencyReport:
    """Validator-style CPU bench: one thread, no GPU, real prompts."""
    pin_threads()
    if THREADS != 1 or GPU_LAYERS != 0:
        raise TrainError(f"engine pin drifted: THREADS={THREADS} GPU_LAYERS={GPU_LAYERS}")
    if not model.is_file():
        raise TrainError(f"GGUF missing: {model}")

    selected = list(examples[:limit] if limit else examples)
    artifact_dir = model.parent
    manifest = LoadManifest(
        format=ArtifactFormat.GGUF,
        quantization=DEFAULT_QUANT,
        entrypoint=model.name,
        max_input={"tokens": context_tokens},
        base_model=BASE_MODEL,
    )
    engine = GgufEngine()
    engine.load(artifact_dir, manifest)

    # Warmup on the shortest prompt so first measured call is steady-state.
    warm_prompt = min(selected, key=lambda ex: len(ex.prompt)).prompt
    for _ in range(max(0, warmup)):
        engine.generate(
            Request(
                task_ref="warmup",
                prompt=warm_prompt,
                max_output_tokens=min(8, max_output_tokens),
                chat=True,
            )
        )

    exact = decisions = copy_ok = flagged_correct = 0
    fbetas: list[float] = []
    ttfts: list[float] = []
    totals: list[float] = []
    input_tokens: list[float] = []
    output_tokens = 0

    try:
        assert engine._model is not None
        for index, example in enumerate(selected, start=1):
            # Count tokens on the same chat-rendered string the engine prefills.
            messages = [{"role": "user", "content": example.prompt}]
            rendered = engine._render_chat(messages) or example.prompt
            try:
                n_in = len(
                    engine._model.tokenize(rendered.encode("utf-8"), add_bos=False)
                )
            except Exception:
                n_in = max(1, len(example.prompt) // 4)
            input_tokens.append(float(n_in))

            response = engine.generate(
                Request(
                    task_ref=example.ref,
                    prompt=example.prompt,
                    max_output_tokens=max_output_tokens,
                    chat=True,
                )
            )
            if not response.ok:
                raise TrainError(f"generation failed on {example.ref}: {response.error}")

            ttfts.append(float(response.ttft_ms))
            totals.append(float(response.total_ms))
            text = str(response.output).strip()
            output_tokens += int(response.output_tokens or 0)

            score = float(span_accuracy(text, example.completion))
            fbetas.append(score)
            if score >= 1.0 - 1e-9:
                exact += 1

            pred_spans = _parse_spans(text)
            gold_spans = _parse_spans(example.completion) or []
            gold_pos = bool(gold_spans)
            pred_pos = bool(pred_spans) if pred_spans is not None else False
            if pred_spans is not None and pred_pos == gold_pos:
                decisions += 1
            if gold_pos and pred_pos:
                flagged_correct += 1
                if score >= 1.0 - 1e-9:
                    copy_ok += 1

            print(
                f"\r[{index:>3}/{len(selected)}] "
                f"exact {exact / index:.3f}  "
                f"total_p95~{_quantile(totals, 0.95):.0f}ms  "
                f"in_tok={n_in}",
                end="",
                flush=True,
            )
    finally:
        engine.unload()
        print()

    n = len(selected)
    total_p95 = int(round(_quantile(totals, 0.95)))
    return LatencyReport(
        tasks=n,
        exact_match=exact / n,
        decision_acc=decisions / n,
        copy_given_flag=(copy_ok / flagged_correct) if flagged_correct else 0.0,
        mean_fbeta=sum(fbetas) / n,
        input_tokens_p50=int(round(_quantile(input_tokens, 0.50))),
        input_tokens_p95=int(round(_quantile(input_tokens, 0.95))),
        input_tokens_p99=int(round(_quantile(input_tokens, 0.99))),
        input_tokens_max=int(max(input_tokens) if input_tokens else 0),
        ttft_p50_ms=int(round(_quantile(ttfts, 0.50))),
        ttft_p95_ms=int(round(_quantile(ttfts, 0.95))),
        total_p50_ms=int(round(_quantile(totals, 0.50))),
        total_p95_ms=total_p95,
        total_p99_ms=int(round(_quantile(totals, 0.99))),
        mean_output_tokens=output_tokens / n,
        max_output_tokens_cap=max_output_tokens,
        threads=THREADS,
        gpu_layers=GPU_LAYERS,
        reference_cost_ms=REFERENCE_COST_MS,
        arena_max_p95_ms=MAX_P95_MS,
        cost_ok=total_p95 < REFERENCE_COST_MS,
        size_bytes=model.stat().st_size,
    )


def pair(exact_match: float, p95_ms: float) -> tuple[float, float]:
    """Canonical (exact-match, ms) pair for sweep tables."""
    return (float(exact_match), float(p95_ms))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="path to model.gguf")
    parser.add_argument("--data", type=Path, required=True, help="sft.jsonl or data dir")
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=40)
    parser.add_argument("--context-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="write JSON report")
    args = parser.parse_args()

    try:
        examples = load_holdout_examples(args.data, args.holdout)
        report = run_latency_harness(
            args.model,
            examples,
            max_output_tokens=args.max_output_tokens,
            context_tokens=args.context_tokens,
            warmup=args.warmup,
            limit=args.limit,
        )
    except TrainError as exc:
        raise SystemExit(f"error: {exc}") from exc

    print("=== latency harness (THREADS=1, GPU_LAYERS=0) ===")
    print(f"pair (exact, p95_ms):     ({report.exact_match:.4f}, {report.total_p95_ms})")
    print(f"exact_match:              {report.exact_match:.6f}")
    print(f"P(correct decision):      {report.decision_acc:.6f}")
    print(f"P(exact copy|flag):       {report.copy_given_flag:.6f}")
    print(f"mean Fβ=2:                {report.mean_fbeta:.6f}")
    print(
        f"input tokens p50/p95/p99/max: "
        f"{report.input_tokens_p50}/{report.input_tokens_p95}/"
        f"{report.input_tokens_p99}/{report.input_tokens_max}"
    )
    print(f"ttft p50/p95:             {report.ttft_p50_ms}/{report.ttft_p95_ms} ms")
    print(
        f"total p50/p95/p99:        "
        f"{report.total_p50_ms}/{report.total_p95_ms}/{report.total_p99_ms} ms"
    )
    print(f"mean output tokens:       {report.mean_output_tokens:.2f} (cap {report.max_output_tokens_cap})")
    print(
        f"vs reference_cost_ms:     {report.total_p95_ms} / {report.reference_cost_ms:.0f} "
        f"({'OK' if report.cost_ok else 'OVER — ci clamps'})"
    )
    print(f"vs arena max_p95_ms:      {report.ttft_p95_ms} / {report.arena_max_p95_ms} (TTFT gate)")
    print(f"size:                     {report.size_bytes / 1024**3:.3f} GiB")

    payload = asdict(report)
    payload["pair"] = list(pair(report.exact_match, report.total_p95_ms))
    out = args.out or (args.model.parent / "latency_report.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
