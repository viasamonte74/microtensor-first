#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from microtensor.training.distill_common import (
    DistillError,
    load_rows,
    read_holdout,
)


def select_layers(
    influence: list[float],
    keep: int,
    forced: set[int],
    strategy: str,
) -> list[int]:
    total = len(influence)
    if not forced <= set(range(total)):
        raise DistillError("force-keep contains an invalid layer")
    if len(forced) > keep:
        raise DistillError("more forced layers than --keep")
    if strategy == "per-layer":
        ranked = sorted(
            (index for index in range(total) if index not in forced),
            key=lambda index: (-influence[index], index),
        )
        return sorted(forced | set(ranked[: keep - len(forced)]))

    remove = total - keep
    candidates: list[tuple[float, int]] = []
    for start in range(total - remove + 1):
        removed = set(range(start, start + remove))
        if removed & forced:
            continue
        candidates.append((sum(influence[index] for index in removed), start))
    if not candidates:
        raise DistillError("no contiguous removal block respects --force-keep")
    _, start = min(candidates)
    removed = set(range(start, start + remove))
    return [index for index in range(total) if index not in removed]


def block_influence(
    model: Any,
    tokenizer: Any,
    rows: list[Any],
    max_seqs: int,
    max_len: int,
) -> list[float]:
    layers = model.model.layers
    sums = [0.0] * len(layers)
    counts = [0] * len(layers)
    entering: list[torch.Tensor | None] = [None] * len(layers)
    current_mask: torch.Tensor | None = None
    handles = []

    def pre_hook(index: int):
        def hook(module: nn.Module, args: tuple[Any, ...]) -> None:
            entering[index] = args[0].detach()

        return hook

    def post_hook(index: int):
        def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            before = entering[index]
            after = output[0] if isinstance(output, tuple) else output
            if before is None or current_mask is None:
                raise DistillError(f"layer {index} hook did not capture its residual")
            similarity = F.cosine_similarity(before.float(), after.detach().float(), dim=-1)
            mask = current_mask.bool()
            sums[index] += float((1.0 - similarity[mask]).sum().item())
            counts[index] += int(mask.sum().item())
            entering[index] = None

        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(pre_hook(index)))
        handles.append(layer.register_forward_hook(post_hook(index)))

    device = next(model.parameters()).device
    model.eval()
    try:
        with torch.no_grad():
            for number, row in enumerate(rows[:max_seqs], 1):
                rendered = tokenizer.apply_chat_template(
                    row.messages, tokenize=False, add_generation_prompt=False
                )
                inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
                if inputs["input_ids"].shape[1] > max_len:
                    raise DistillError(
                        f"{row.ref} has {inputs['input_ids'].shape[1]} tokens, over {max_len}"
                    )
                inputs = inputs.to(device)
                current_mask = inputs["attention_mask"]
                model(**inputs, use_cache=False)
                if number % 25 == 0:
                    print(f"\rblock influence {number}/{min(max_seqs, len(rows))}", end="", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    print()
    return [sums[index] / counts[index] for index in range(len(layers))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune a Qwen2 teacher by block influence")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, default=Path("holdout_ids.txt"))
    parser.add_argument("--keep", type=int, default=10)
    parser.add_argument("--force-keep", default="0,1,27")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strategy", choices=("per-layer", "contiguous"), default="per-layer")
    parser.add_argument("--max-seqs", type=int, default=600)
    parser.add_argument("--max-len", type=int, default=1536)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out}")
    rows = load_rows(args.data)
    holdout_refs = read_holdout(args.holdout)
    training = [row for row in rows if row.ref not in holdout_refs]
    holdout = [row for row in rows if row.ref in holdout_refs]
    forced = {int(value) for value in args.force_keep.split(",") if value.strip()}

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    )
    original_layers = len(model.model.layers)
    if args.keep <= 0 or args.keep >= original_layers:
        raise SystemExit(f"--keep must be between 1 and {original_layers - 1}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    influence = block_influence(model, tokenizer, training, args.max_seqs, args.max_len)
    kept = select_layers(influence, args.keep, forced, args.strategy)
    before_parameters = sum(parameter.numel() for parameter in model.parameters())

    model.model.layers = nn.ModuleList([model.model.layers[index] for index in kept])
    for new_index, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = new_index
    config = model.config
    for name, value in vars(config).items():
        if isinstance(value, list) and len(value) == original_layers:
            setattr(config, name, [value[index] for index in kept])
    config.num_hidden_layers = args.keep
    if getattr(config, "max_window_layers", None) is not None:
        config.max_window_layers = min(int(config.max_window_layers), args.keep)
    after_parameters = sum(parameter.numel() for parameter in model.parameters())

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    report = {
        "strategy": args.strategy,
        "original_layers": original_layers,
        "kept_layers": kept,
        "block_influence": {str(index): value for index, value in enumerate(influence)},
        "parameters_before": before_parameters,
        "parameters_after": after_parameters,
    }
    (args.out / "prune_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    del model
    reloaded = AutoModelForCausalLM.from_pretrained(
        args.out, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    assert len(reloaded.model.layers) == args.keep
    assert reloaded.config.num_hidden_layers == args.keep
    print(f"kept original layers: {kept}")
    print(f"parameters: {before_parameters:,} -> {after_parameters:,}")
    reloaded.eval()
    with torch.inference_mode():
        for row in holdout[:5]:
            prompt = tokenizer.apply_chat_template(
                row.messages[:-1], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            generated = reloaded.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            continuation = generated[0, inputs["input_ids"].shape[1] :]
            print(f"{row.ref}: {tokenizer.decode(continuation, skip_special_tokens=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
