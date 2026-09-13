#!/usr/bin/env python3
"""Width-prune Llama MLP and GQA attention (STEP 5).

Runs *before* KD heal. Targets Config A by default:
  intermediate 8192 → 2816
  heads 24/8 → 12/4  (preserve 3 q heads per kv head, head_dim 128)

FFN channels and KV groups are ranked by mean absolute activation over the
guard SFT prompts so the kept subspace reflects entailment traffic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from microtensor.training.distill_common import DistillError, evaluate_guard, load_rows, read_holdout


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect_ffn_scores(
    model: Any,
    tokenizer: Any,
    rows: list[Any],
    *,
    max_seqs: int,
    max_len: int,
) -> torch.Tensor:
    """Mean |SiLU(gate) * up| over tokens → score per intermediate channel."""
    device = next(model.parameters()).device
    n_layers = len(model.model.layers)
    intermediate = int(model.config.intermediate_size)
    scores = torch.zeros(n_layers, intermediate, dtype=torch.float64)
    counts = torch.zeros(n_layers, dtype=torch.float64)
    handles = []

    def make_hook(index: int):
        def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            hidden = args[0]
            # gate/up: (batch, seq, intermediate) after linear — recompute from weight
            # cheaper: use forward of gate_proj/up_proj on hidden
            gate = module.gate_proj(hidden)
            up = module.up_proj(hidden)
            activated = F.silu(gate) * up
            # (B, T, I) → sum over B,T of abs
            scores[index] += activated.detach().float().abs().sum(dim=(0, 1)).double().cpu()
            counts[index] += float(hidden.shape[0] * hidden.shape[1])

        return hook

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.register_forward_hook(make_hook(index)))

    model.eval()
    try:
        with torch.no_grad():
            for number, row in enumerate(rows[:max_seqs], 1):
                rendered = tokenizer.apply_chat_template(
                    row.messages, tokenize=False, add_generation_prompt=False
                )
                inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
                if inputs["input_ids"].shape[1] > max_len:
                    continue
                model(**inputs.to(device), use_cache=False)
                if number % 50 == 0:
                    print(f"\rffn scores {number}/{min(max_seqs, len(rows))}", end="", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    print()
    counts = counts.clamp_min(1.0).unsqueeze(1)
    return (scores / counts).float()


def _collect_kv_group_scores(
    model: Any,
    tokenizer: Any,
    rows: list[Any],
    *,
    max_seqs: int,
    max_len: int,
) -> torch.Tensor:
    """Mean |q| magnitude aggregated per KV group (GQA)."""
    device = next(model.parameters()).device
    n_layers = len(model.model.layers)
    n_heads = int(model.config.num_attention_heads)
    n_kv = int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", None) or (model.config.hidden_size // n_heads))
    if n_heads % n_kv != 0:
        raise DistillError("GQA requires num_attention_heads % num_key_value_heads == 0")
    group = n_heads // n_kv

    scores = torch.zeros(n_layers, n_kv, dtype=torch.float64)
    counts = torch.zeros(n_layers, dtype=torch.float64)
    entering: list[torch.Tensor | None] = [None] * n_layers
    handles = []

    def pre_hook(index: int):
        def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if args:
                entering[index] = args[0]
            elif "hidden_states" in kwargs:
                entering[index] = kwargs["hidden_states"]

        return hook

    def post_hook(index: int):
        def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            hidden = entering[index]
            entering[index] = None
            if hidden is None:
                return
            q = module.q_proj(hidden)
            batch, seq, _ = q.shape
            q = q.view(batch, seq, n_kv, group, head_dim)
            group_score = q.detach().float().abs().mean(dim=(0, 1, 3, 4))
            scores[index] += group_score.double().cpu() * float(batch * seq)
            counts[index] += float(batch * seq)

        return hook

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(pre_hook(index), with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(post_hook(index)))

    model.eval()
    try:
        with torch.no_grad():
            for number, row in enumerate(rows[:max_seqs], 1):
                rendered = tokenizer.apply_chat_template(
                    row.messages, tokenize=False, add_generation_prompt=False
                )
                inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
                if inputs["input_ids"].shape[1] > max_len:
                    continue
                model(**inputs.to(device), use_cache=False)
                if number % 50 == 0:
                    print(f"\rattn scores {number}/{min(max_seqs, len(rows))}", end="", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    print()
    counts = counts.clamp_min(1.0).unsqueeze(1)
    return (scores / counts).float()


def _slice_linear(weight: torch.Tensor, *, rows: torch.Tensor | None = None, cols: torch.Tensor | None = None) -> nn.Linear:
    data = weight.detach()
    if rows is not None:
        data = data.index_select(0, rows.to(data.device))
    if cols is not None:
        data = data.index_select(1, cols.to(data.device))
    out = nn.Linear(data.shape[1], data.shape[0], bias=False)
    out.weight = nn.Parameter(data.clone())
    return out


def prune_ffn_layer(layer: Any, keep_idx: torch.Tensor) -> None:
    gate = layer.mlp.gate_proj.weight
    up = layer.mlp.up_proj.weight
    down = layer.mlp.down_proj.weight
    layer.mlp.gate_proj = _slice_linear(gate, rows=keep_idx)
    layer.mlp.up_proj = _slice_linear(up, rows=keep_idx)
    layer.mlp.down_proj = _slice_linear(down, cols=keep_idx)


def prune_attn_layer(
    layer: Any,
    *,
    keep_kv: torch.Tensor,
    n_heads: int,
    n_kv: int,
    head_dim: int,
    new_n_heads: int,
    new_n_kv: int,
) -> None:
    group = n_heads // n_kv
    # Expand KV keep → Q head indices
    q_indices: list[int] = []
    for kv in keep_kv.tolist():
        start = int(kv) * group
        q_indices.extend(range(start, start + group))
    if len(q_indices) != new_n_heads:
        raise DistillError(f"expected {new_n_heads} q heads, got {len(q_indices)}")
    q_idx = torch.tensor(q_indices, dtype=torch.long)
    kv_idx = keep_kv.long()

    hidden = layer.self_attn.q_proj.weight.shape[1]

    def slice_out_heads(weight: torch.Tensor, head_index: torch.Tensor, n_src_heads: int) -> torch.Tensor:
        # weight: (n_heads * head_dim, hidden)
        reshaped = weight.view(n_src_heads, head_dim, hidden)
        kept = reshaped.index_select(0, head_index.to(weight.device))
        return kept.reshape(-1, hidden)

    def slice_in_heads(weight: torch.Tensor, head_index: torch.Tensor, n_src_heads: int) -> torch.Tensor:
        # o_proj weight: (hidden, n_heads * head_dim)
        reshaped = weight.view(hidden, n_src_heads, head_dim)
        kept = reshaped.index_select(1, head_index.to(weight.device))
        return kept.reshape(hidden, -1)

    q_w = slice_out_heads(layer.self_attn.q_proj.weight.detach(), q_idx, n_heads)
    k_w = slice_out_heads(layer.self_attn.k_proj.weight.detach(), kv_idx, n_kv)
    v_w = slice_out_heads(layer.self_attn.v_proj.weight.detach(), kv_idx, n_kv)
    o_w = slice_in_heads(layer.self_attn.o_proj.weight.detach(), q_idx, n_heads)

    layer.self_attn.q_proj = nn.Linear(hidden, new_n_heads * head_dim, bias=False)
    layer.self_attn.q_proj.weight = nn.Parameter(q_w.clone())
    layer.self_attn.k_proj = nn.Linear(hidden, new_n_kv * head_dim, bias=False)
    layer.self_attn.k_proj.weight = nn.Parameter(k_w.clone())
    layer.self_attn.v_proj = nn.Linear(hidden, new_n_kv * head_dim, bias=False)
    layer.self_attn.v_proj.weight = nn.Parameter(v_w.clone())
    layer.self_attn.o_proj = nn.Linear(new_n_heads * head_dim, hidden, bias=False)
    layer.self_attn.o_proj.weight = nn.Parameter(o_w.clone())

    # Keep module attributes in sync for RoPE / eager attention.
    if hasattr(layer.self_attn, "num_heads"):
        layer.self_attn.num_heads = new_n_heads
    if hasattr(layer.self_attn, "num_key_value_heads"):
        layer.self_attn.num_key_value_heads = new_n_kv
    if hasattr(layer.self_attn, "num_key_value_groups"):
        layer.self_attn.num_key_value_groups = new_n_heads // new_n_kv
    if hasattr(layer.self_attn, "hidden_size"):
        # Some Llama builds store attn hidden as n_heads * head_dim.
        layer.self_attn.hidden_size = new_n_heads * head_dim


def apply_width_prune(
    model: Any,
    ffn_scores: torch.Tensor,
    kv_scores: torch.Tensor,
    *,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> dict[str, Any]:
    n_heads = int(model.config.num_attention_heads)
    n_kv = int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", None) or (model.config.hidden_size // n_heads))
    if num_attention_heads % num_key_value_heads != 0:
        raise DistillError("target heads must preserve GQA grouping")
    if (n_heads // n_kv) != (num_attention_heads // num_key_value_heads):
        raise DistillError(
            f"GQA ratio must stay {n_heads // n_kv}:1 "
            f"(got {num_attention_heads}/{num_key_value_heads})"
        )

    kept_ffn: dict[str, list[int]] = {}
    kept_kv: dict[str, list[int]] = {}
    for index, layer in enumerate(model.model.layers):
        ffn_idx = torch.topk(ffn_scores[index], k=intermediate_size, largest=True).indices.sort().values
        kv_idx = torch.topk(kv_scores[index], k=num_key_value_heads, largest=True).indices.sort().values
        prune_ffn_layer(layer, ffn_idx)
        prune_attn_layer(
            layer,
            keep_kv=kv_idx,
            n_heads=n_heads,
            n_kv=n_kv,
            head_dim=head_dim,
            new_n_heads=num_attention_heads,
            new_n_kv=num_key_value_heads,
        )
        kept_ffn[str(index)] = ffn_idx.tolist()
        kept_kv[str(index)] = kv_idx.tolist()

    model.config.intermediate_size = intermediate_size
    model.config.num_attention_heads = num_attention_heads
    model.config.num_key_value_heads = num_key_value_heads
    if hasattr(model.config, "head_dim"):
        model.config.head_dim = head_dim
    return {"ffn_indices": kept_ffn, "kv_group_indices": kept_kv, "head_dim": head_dim}


def _smoke_forward(model: Any, tokenizer: Any) -> None:
    """Ensure RoPE + GQA still run after the slice."""
    device = next(model.parameters()).device
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hi."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    model.eval()
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    if not torch.isfinite(out.logits).all():
        raise DistillError("width-pruned forward produced non-finite logits")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="layer-pruned student checkpoint")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--intermediate-size", type=int, default=2816)
    parser.add_argument("--num-attention-heads", type=int, default=12)
    parser.add_argument("--num-key-value-heads", type=int, default=4)
    parser.add_argument("--max-seqs", type=int, default=600)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out}")

    rows = load_rows(args.data)
    holdout_refs = read_holdout(args.holdout)
    training = [row for row in rows if row.ref not in holdout_refs]
    held = [row for row in rows if row.ref in holdout_refs]

    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    before = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"input: layers={model.config.num_hidden_layers} "
        f"ffn={model.config.intermediate_size} "
        f"heads={model.config.num_attention_heads}/{model.config.num_key_value_heads} "
        f"params={before:,}"
    )

    ffn_scores = _collect_ffn_scores(
        model, tokenizer, training, max_seqs=args.max_seqs, max_len=args.max_len
    )
    kv_scores = _collect_kv_group_scores(
        model, tokenizer, training, max_seqs=args.max_seqs, max_len=args.max_len
    )
    kept = apply_width_prune(
        model,
        ffn_scores,
        kv_scores,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
    )
    _smoke_forward(model, tokenizer)
    after = sum(parameter.numel() for parameter in model.parameters())

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)

    metrics: dict[str, float] = {}
    if not args.skip_eval and held:
        metrics = evaluate_guard(
            model, tokenizer, held, limit=args.eval_limit, max_new_tokens=args.max_new_tokens
        )
        print(
            f"pre-heal pair=({metrics['exact_match']:.4f}, {metrics['p95_ms']:.0f}ms)  "
            f"decision={metrics['decision_acc']:.4f}"
        )

    report = {
        "source": args.model,
        "intermediate_size": args.intermediate_size,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "head_dim": kept["head_dim"],
        "parameters_before": before,
        "parameters_after": after,
        "ffn_indices": kept["ffn_indices"],
        "pair": (
            [metrics["exact_match"], metrics["p95_ms"]] if metrics else None
        ),
        "kv_group_indices": kept["kv_group_indices"],
        **metrics,
    }
    (args.out / "width_prune_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"params: {before:,} -> {after:,}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DistillError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
