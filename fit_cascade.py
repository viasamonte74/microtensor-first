#!/usr/bin/env python3
"""Fit a declarative threshold router for the guard cascade.

The live GGUF engine does not populate logprobs/entropies, and cascade.py
computes input_tokens as whitespace words and schema_valid as always-true.
A router that reads seq_logprob_* will see zeros on the validator. This
fitter therefore treats output_tokens (stream piece count) as the cost dial,
optionally with input_tokens, using the same features_from() the validator
runs.

Specialist quality is supplied as a JSONL of {ref, output} (HF or GGUF).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from microtensor.core.hashing import digest_file
from microtensor.core.protocol import ArtifactFormat, LoadManifest
from microtensor.harness.contract import Request, Response
from microtensor.harness.engines.gguf import GPU_LAYERS, THREADS, GgufEngine
from microtensor.harness.engines.router import Decision, features_from, load_router
from microtensor.harness.limits import pin_threads
from microtensor.scoring.metrics import span_accuracy
from microtensor.training.arena import (
    BASE_MODEL,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_QUANT,
    HARDWARE_CLASS,
    MAX_OUTPUT_TOKENS,
    REFERENCE_COST_MS,
    SPECIALIST_MODEL,
    TRACK,
)
from microtensor.training.dataset import TrainError, load_sft, resolve_sft


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


def _legal(text: str) -> bool:
    return _parse_spans(text) is not None


def _metrics(preds: list[str], golds: list[str]) -> dict[str, float]:
    n = len(preds)
    exact = decisions = copy_ok = flagged = 0
    tp = fp = tn = fn = 0
    fbetas: list[float] = []
    for pred, gold in zip(preds, golds, strict=True):
        score = float(span_accuracy(pred, gold))
        fbetas.append(score)
        if score >= 1.0 - 1e-9:
            exact += 1
        pred_spans = _parse_spans(pred)
        gold_spans = _parse_spans(gold) or []
        gold_pos = bool(gold_spans)
        pred_pos = bool(pred_spans) if pred_spans is not None else False
        if pred_spans is not None and pred_pos == gold_pos:
            decisions += 1
        if gold_pos and pred_pos:
            tp += 1
            flagged += 1
            if score >= 1.0 - 1e-9:
                copy_ok += 1
        elif gold_pos and not pred_pos:
            fn += 1
        elif not gold_pos and pred_pos:
            fp += 1
        else:
            tn += 1
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "exact_match": exact / n if n else 0.0,
        "decision_acc": decisions / n if n else 0.0,
        "copy_given_flag": (copy_ok / flagged) if flagged else 0.0,
        "specificity": spec,
        "sensitivity": sens,
        "mean_fbeta": sum(fbetas) / n if n else 0.0,
        "legal": sum(1 for p in preds if _legal(p)) / n if n else 0.0,
        "n": float(n),
    }


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def collect_front(
    model: Path,
    examples: list[Any],
    *,
    max_output_tokens: int,
    context_tokens: int,
    warmup: int,
) -> list[dict[str, Any]]:
    pin_threads()
    if THREADS != 1 or GPU_LAYERS != 0:
        raise TrainError(f"engine pin drifted: THREADS={THREADS} GPU_LAYERS={GPU_LAYERS}")
    engine = GgufEngine()
    engine.load(
        model.parent,
        LoadManifest(
            format=ArtifactFormat.GGUF,
            quantization=DEFAULT_QUANT,
            entrypoint=model.name,
            max_input={"tokens": context_tokens},
            base_model=BASE_MODEL,
        ),
    )
    warm = min(examples, key=lambda ex: len(ex.prompt)).prompt
    for _ in range(max(0, warmup)):
        engine.generate(
            Request(
                task_ref="warmup",
                prompt=warm,
                max_output_tokens=min(8, max_output_tokens),
                chat=True,
            )
        )
    traces: list[dict[str, Any]] = []
    try:
        for index, example in enumerate(examples, start=1):
            request = Request(
                task_ref=example.ref,
                prompt=example.prompt,
                max_output_tokens=max_output_tokens,
                chat=True,
            )
            response = engine.generate(request)
            if not response.ok:
                raise TrainError(f"front failed on {example.ref}: {response.error}")
            # Match cascade.py: word-split prompt, default schema_valid=True.
            features = features_from(
                response,
                prompt_tokens=len(request.prompt.split()),
                schema_valid=True,
            )
            pred = str(response.output).strip()
            score = float(span_accuracy(pred, example.completion))
            traces.append(
                {
                    "ref": example.ref,
                    "gold": example.completion,
                    "front_output": pred,
                    "front_exact": score >= 1.0 - 1e-9,
                    "front_fbeta": score,
                    "front_legal": _legal(pred),
                    "front_ms": float(response.total_ms),
                    "front_ttft_ms": float(response.ttft_ms),
                    "features": features,
                }
            )
            print(
                f"\rfront [{index:>3}/{len(examples)}] "
                f"exact {sum(t['front_exact'] for t in traces) / index:.3f}  "
                f"out_tok={features['output_tokens']:.0f}",
                end="",
                flush=True,
            )
    finally:
        engine.unload()
        print()
    return traces


def load_specialist_outputs(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        mapping[str(row["ref"])] = str(row["output"]).strip()
    return mapping


def apply_policy(
    traces: list[dict[str, Any]],
    specialist: dict[str, str],
    *,
    escalate: list[bool],
    specialist_ms: float,
) -> dict[str, Any]:
    preds: list[str] = []
    golds: list[str] = []
    used_ms: list[float] = []
    n_esc = 0
    for row, flag in zip(traces, escalate, strict=True):
        golds.append(row["gold"])
        if flag:
            n_esc += 1
            spec = specialist.get(row["ref"])
            if spec is None:
                raise TrainError(f"specialist missing output for {row['ref']}")
            preds.append(spec)
            used_ms.append(float(row["front_ms"]) + specialist_ms)
        else:
            preds.append(row["front_output"])
            used_ms.append(float(row["front_ms"]))
    metrics = _metrics(preds, golds)
    rho = 1.0 - (n_esc / len(traces) if traces else 0.0)
    front_ms = sum(float(r["front_ms"]) for r in traces) / len(traces)
    expected_ms = front_ms + (1.0 - rho) * specialist_ms
    return {
        **metrics,
        "resolve_rate": rho,
        "escalate_rate": 1.0 - rho,
        "front_ms": front_ms,
        "specialist_ms": specialist_ms,
        "expected_ms": expected_ms,
        "p95_ms": _quantile(used_ms, 0.95),
        "cost_ok": expected_ms < REFERENCE_COST_MS,
        "n_escalate": n_esc,
    }


def policy_escalate_ge(traces: list[dict[str, Any]], feature: str, threshold: float) -> list[bool]:
    return [float(row["features"][feature]) >= threshold for row in traces]


def policy_escalate_outside(
    traces: list[dict[str, Any]], lo: float, hi: float
) -> list[bool]:
    return [
        not (lo <= float(row["features"]["output_tokens"]) <= hi) for row in traces
    ]


def write_router(path: Path, clauses: list[dict[str, Any]], default: str = "resolve") -> None:
    payload = {"form": "threshold", "default": default, "clauses": clauses}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Round-trip through the validator loader.
    features = tuple(dict.fromkeys(c["feature"] for c in clauses))
    load_router(path, features)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--specialist-outputs", type=Path, default=None)
    parser.add_argument("--traces", type=Path, default=None, help="reuse collected front traces")
    parser.add_argument("--specialist-ms", type=float, default=30_000.0)
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--context-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    refs = {
        line.strip()
        for line in args.holdout.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    examples = [ex for ex in load_sft(resolve_sft(args.data)) if ex.ref in refs]
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit("holdout produced no examples")

    args.out.mkdir(parents=True, exist_ok=True)
    traces_path = args.traces or (args.out / "front_traces.jsonl")
    if traces_path.is_file():
        traces = [json.loads(line) for line in traces_path.read_text().splitlines() if line.strip()]
        print(f"loaded {len(traces)} front traces from {traces_path}")
    else:
        traces = collect_front(
            args.model,
            examples,
            max_output_tokens=args.max_output_tokens,
            context_tokens=args.context_tokens,
            warmup=args.warmup,
        )
        traces_path.write_text(
            "".join(json.dumps(row) + "\n" for row in traces), encoding="utf-8"
        )
        print(f"wrote {traces_path}")

    specialist: dict[str, str] = {}
    if args.specialist_outputs and args.specialist_outputs.is_file():
        specialist = load_specialist_outputs(args.specialist_outputs)
    else:
        # Oracle upper bound until specialist outputs land.
        specialist = {row["ref"]: row["gold"] for row in traces}
        print("WARNING: no specialist outputs; scoring an ORACLE specialist (upper bound)")

    golds = [row["gold"] for row in traces]
    front_metrics = _metrics([row["front_output"] for row in traces], golds)
    spec_preds = [specialist[row["ref"]] for row in traces]
    spec_metrics = _metrics(spec_preds, golds)

    candidates: list[dict[str, Any]] = []
    out_toks = sorted({int(row["features"]["output_tokens"]) for row in traces})
    for threshold in out_toks:
        flags = policy_escalate_ge(traces, "output_tokens", threshold)
        result = apply_policy(
            traces, specialist, escalate=flags, specialist_ms=args.specialist_ms
        )
        candidates.append(
            {
                "name": f"output_tokens>= {threshold}",
                "clauses": [
                    {
                        "feature": "output_tokens",
                        "op": "ge",
                        "value": float(threshold),
                        "decision": "escalate",
                    }
                ],
                **result,
            }
        )
    # Escalate anything that is not a short abstain-shaped completion.
    for lo, hi in ((8, 12), (9, 13), (7, 14), (6, 15)):
        flags = policy_escalate_outside(traces, lo, hi)
        result = apply_policy(
            traces, specialist, escalate=flags, specialist_ms=args.specialist_ms
        )
        candidates.append(
            {
                "name": f"output_tokens not in [{lo},{hi}]",
                "clauses": [
                    {
                        "feature": "output_tokens",
                        "op": "lt",
                        "value": float(lo),
                        "decision": "escalate",
                    },
                    {
                        "feature": "output_tokens",
                        "op": "gt",
                        "value": float(hi),
                        "decision": "escalate",
                    },
                ],
                **result,
            }
        )

    front_only = apply_policy(
        traces, specialist, escalate=[False] * len(traces), specialist_ms=args.specialist_ms
    )
    always = apply_policy(
        traces, specialist, escalate=[True] * len(traces), specialist_ms=args.specialist_ms
    )

    # Prefer cost-feasible points, then quality, then lower escalate rate.
    feasible = [c for c in candidates if c["cost_ok"] and c["exact_match"] >= 0.65]
    pool = feasible or [c for c in candidates if c["exact_match"] >= 0.65] or candidates
    chosen = max(
        pool,
        key=lambda c: (c["cost_ok"], c["exact_match"], -c["expected_ms"], -c["escalate_rate"]),
    )

    router_path = args.out / "router.json"
    write_router(router_path, chosen["clauses"], default="resolve")

    # Verify the saved router reproduces the chosen policy.
    router = load_router(router_path, tuple(dict.fromkeys(c["feature"] for c in chosen["clauses"])))
    replay = []
    for row in traces:
        replay.append(router.decide(row["features"]) is Decision.ESCALATE)
    replay_metrics = apply_policy(
        traces, specialist, escalate=replay, specialist_ms=args.specialist_ms
    )

    report = {
        "track": TRACK,
        "hardware_class": HARDWARE_CLASS,
        "specialist_model": SPECIALIST_MODEL,
        "oracle_specialist": not (args.specialist_outputs and args.specialist_outputs.is_file()),
        "reference_cost_ms": REFERENCE_COST_MS,
        "front": front_metrics,
        "specialist_alone": spec_metrics,
        "front_only": front_only,
        "always_escalate": always,
        "chosen": {"name": chosen["name"], "clauses": chosen["clauses"], **{k: chosen[k] for k in chosen if k not in {"name", "clauses"}}},
        "replay": replay_metrics,
        "candidates": [
            {
                "name": c["name"],
                "exact_match": c["exact_match"],
                "escalate_rate": c["escalate_rate"],
                "expected_ms": c["expected_ms"],
                "cost_ok": c["cost_ok"],
                "decision_acc": c["decision_acc"],
                "copy_given_flag": c["copy_given_flag"],
            }
            for c in candidates
        ],
        "router_digest": digest_file(router_path),
        "note": (
            "Validator GGUF engine currently emits empty logprobs/entropies and "
            "cascade.py sets schema_valid=True always. Shipped router uses "
            "output_tokens only. specialist_ms is an estimate until a host-profile "
            "GGUF latency sample exists."
        ),
    }
    (args.out / "fit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("=== front ===")
    print(
        f"exact={front_metrics['exact_match']:.4f}  "
        f"decision={front_metrics['decision_acc']:.4f}  "
        f"copy|flag={front_metrics['copy_given_flag']:.4f}"
    )
    print("=== specialist alone ===")
    print(
        f"exact={spec_metrics['exact_match']:.4f}  "
        f"decision={spec_metrics['decision_acc']:.4f}  "
        f"copy|flag={spec_metrics['copy_given_flag']:.4f}"
        f"{'  [ORACLE]' if report['oracle_specialist'] else ''}"
    )
    print("=== chosen router ===")
    print(chosen["name"])
    print(json.dumps(chosen["clauses"], indent=2))
    print(
        f"pair (exact, expected_ms)=({chosen['exact_match']:.4f}, {chosen['expected_ms']:.0f})  "
        f"escalate={chosen['escalate_rate']:.2%}  "
        f"cost_ok={chosen['cost_ok']}"
    )
    print(f"router {router_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
