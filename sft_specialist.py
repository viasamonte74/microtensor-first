#!/usr/bin/env python3
"""Full-parameter SFT of the Phi-4-mini cascade specialist on guard JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, Phi3ForCausalLM, Trainer

from microtensor.training.distill_common import (
    DistillError,
    collator,
    encode_rows,
    evaluate_guard,
    load_rows,
    read_holdout,
    rows_fitting_max_len,
    training_arguments,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out}")

    try:
        rows = load_rows(args.data)
        holdout_refs = read_holdout(args.holdout)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        holdout = rows_fitting_max_len(
            tokenizer, [row for row in rows if row.ref in holdout_refs], args.max_len
        )
        train_rows = rows_fitting_max_len(
            tokenizer, [row for row in rows if row.ref not in holdout_refs], args.max_len
        )
        dataset = encode_rows(tokenizer, train_rows, args.max_len)
        # Vendor modeling_phi3.py imports LossKwargs removed in this
        # transformers; the in-tree Phi3ForCausalLM loads the same weights.
        model = Phi3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    except (DistillError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    print(
        f"specialist SFT rows={len(train_rows)} holdout={len(holdout)} "
        f"layers={model.config.num_hidden_layers} vocab={model.config.vocab_size}"
    )
    trainer = Trainer(
        model=model,
        args=training_arguments(
            args.out,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
        ),
        train_dataset=dataset,
        data_collator=collator(tokenizer),
    )
    trainer.train()
    model.config.use_cache = True
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    metrics = evaluate_guard(model, tokenizer, holdout, max_new_tokens=40)
    print("=== specialist holdout ===")
    for key in (
        "exact_match",
        "decision_acc",
        "copy_given_flag",
        "specificity",
        "sensitivity",
        "mean_output_tokens",
        "p95_ms",
    ):
        print(f"{key}: {metrics[key]}")

    lines: list[str] = []
    model.eval()
    with torch.inference_mode():
        for row in holdout:
            prompt = tokenizer.apply_chat_template(
                row.messages[:-1], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=40,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            text = tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            lines.append(json.dumps({"ref": row.ref, "output": text}, ensure_ascii=False))
    (args.out / "holdout_outputs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "sft_report.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out / 'holdout_outputs.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
