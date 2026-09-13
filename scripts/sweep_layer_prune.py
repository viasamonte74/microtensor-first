#!/usr/bin/env python3
"""STEP 4 — per-layer prune sweep on the guard teacher.

Computes block influence once on the guard SFT set, then materialises
students at --keep 6/8/10/12 (force-keeping 0,1,27) and reports
(exact-match, p95_ms) for each. Does not heal — that is STEP 6.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from prune_layers import block_influence, select_layers
from microtensor.training.distill_common import (
    DistillError,
    evaluate_guard,
    load_rows,
    read_holdout,
)
from latency_harness import pair


def _load_split(data: Path, holdout: Path) -> tuple[list, list]:
    rows = load_rows(data)
    holdout_refs = read_holdout(holdout)
    training = [row for row in rows if row.ref not in holdout_refs]
    held = [row for row in rows if row.ref in holdout_refs]
    if not training or not held:
        raise DistillError("training or holdout split is empty")
    return training, held


def _apply_keep(model, kept: list[int], original_layers: int) -> None:
    model.model.layers = nn.ModuleList([model.model.layers[index] for index in kept])
    for new_index, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_index
    config = model.config
    for name, value in vars(config).items():
        if isinstance(value, list) and len(value) == original_layers:
            setattr(config, name, [value[index] for index in kept])
    config.num_hidden_layers = len(kept)
    if getattr(config, "max_window_layers", None) is not None:
        config.max_window_layers = min(int(config.max_window_layers), len(kept))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="28-layer teacher checkpoint")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--keeps", default="6,8,10,12")
    parser.add_argument("--force-keep", default="0,1,27")
    parser.add_argument("--strategy", default="per-layer")
    parser.add_argument("--max-seqs", type=int, default=600)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--eval-limit", type=int, default=0, help="0 = full holdout")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument(
        "--influence-cache",
        type=Path,
        default=None,
        help="reuse a previously written influence json",
    )
    args = parser.parse_args()

    keeps = [int(value) for value in args.keeps.split(",") if value.strip()]
    forced = {int(value) for value in args.force_keep.split(",") if value.strip()}
    training, held = _load_split(args.data, args.holdout)

    args.out_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    )
    original_layers = len(model.model.layers)
    model.to(device)

    influence_path = args.influence_cache or (args.out_root / "block_influence.json")
    if influence_path.is_file():
        influence = json.loads(influence_path.read_text(encoding="utf-8"))["influence"]
        print(f"loaded influence from {influence_path}")
    else:
        influence = block_influence(model, tokenizer, training, args.max_seqs, args.max_len)
        influence_path.write_text(
            json.dumps(
                {
                    "influence": influence,
                    "max_seqs": args.max_seqs,
                    "force_keep": sorted(forced),
                    "strategy": args.strategy,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote influence to {influence_path}")

    ranked = sorted(
        range(original_layers), key=lambda index: (-influence[index], index)
    )
    print("layer influence rank (high → low):", ranked)
    print("forced:", sorted(forced))

    # Teacher baseline before any prune.
    print("evaluating teacher baseline…")
    baseline = evaluate_guard(
        model, tokenizer, held, limit=args.eval_limit, max_new_tokens=args.max_new_tokens
    )
    print(
        f"teacher  exact={baseline['exact_match']:.4f}  "
        f"p95_ms={baseline['p95_ms']:.0f}  "
        f"decision={baseline['decision_acc']:.4f}"
    )

    results = [{"keep": original_layers, "kept_layers": list(range(original_layers)), **baseline}]
    teacher_state = {k: v.cpu() for k, v in model.state_dict().items()}
    del model
    torch.cuda.empty_cache()

    for keep in keeps:
        out = args.out_root / f"student-{keep}l"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)

        student = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, trust_remote_code=True
        )
        # Reload teacher weights in case from_pretrained re-read disk differently
        student.load_state_dict(teacher_state, strict=True)
        kept = select_layers(influence, keep, forced, args.strategy)
        before = sum(parameter.numel() for parameter in student.parameters())
        _apply_keep(student, kept, original_layers)
        after = sum(parameter.numel() for parameter in student.parameters())
        student.to(device)

        metrics = evaluate_guard(
            student,
            tokenizer,
            held,
            limit=args.eval_limit,
            max_new_tokens=args.max_new_tokens,
        )
        student.cpu()
        student.save_pretrained(out, safe_serialization=True)
        tokenizer.save_pretrained(out)
        report = {
            "keep": keep,
            "kept_layers": kept,
            "parameters_before": before,
            "parameters_after": after,
            "pair": list(pair(metrics["exact_match"], metrics["p95_ms"])),
            **metrics,
        }
        (out / "prune_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        results.append(report)
        print(
            f"keep={keep:2d} layers={kept}  "
            f"pair=({metrics['exact_match']:.4f}, {metrics['p95_ms']:.0f}ms)  "
            f"decision={metrics['decision_acc']:.4f}  "
            f"spec={metrics['specificity']:.4f}  "
            f"sens={metrics['sensitivity']:.4f}  "
            f"copy|flag={metrics['copy_given_flag']:.4f}  "
            f"params={after:,}"
        )
        del student
        torch.cuda.empty_cache()

    summary_path = args.out_root / "sweep_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "force_keep": sorted(forced),
                "strategy": args.strategy,
                "note": (
                    "pair=(exact_match, p95_ms). p95_ms here is wall time on this "
                    "eval device (often GPU). Validator-faithful numbers: "
                    "`python latency_harness.py --model …/model.gguf` "
                    "(THREADS=1, GPU_LAYERS=0, real prompt lengths, max_out=40)."
                ),
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"summary: {summary_path}")
    print()
    print(f"{'keep':>6}  {'pair (exact, ms)':>22}  {'decision':>8}  {'spec':>6}  {'sens':>6}")
    for row in results:
        print(
            f"{row['keep']:>6}  ({row['exact_match']:.4f}, {row['p95_ms']:.0f}ms)"
            f"{'':>4}  {row['decision_acc']:>8.4f}  {row['specificity']:>6.4f}  "
            f"{row['sensitivity']:>6.4f}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DistillError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
