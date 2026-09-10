#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

from microtensor.training.distill_common import (
    DistillError,
    augment_rows,
    choose_holdout,
    collator,
    encode_rows,
    evaluate_model,
    load_rows,
    make_probes,
    training_arguments,
    write_holdout,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-parameter SFT of the 28-layer teacher")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--holdout-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--augment", type=int, default=8)
    parser.add_argument("--probe-rows", type=int, default=300)
    parser.add_argument("--max-len", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out}")

    try:
        rows = load_rows(args.data)
        holdout = choose_holdout(rows)
        write_holdout(args.holdout_out, holdout)
        holdout_refs = {row.ref for row in holdout}
        base_train = [row for row in rows if row.ref not in holdout_refs]
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_rows = augment_rows(
            base_train,
            args.augment,
            tokenizer=tokenizer,
            max_len=args.max_len,
        ) + make_probes(args.probe_rows)
        dataset = encode_rows(tokenizer, train_rows, args.max_len)
    except (DistillError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    )
    if model.config.num_hidden_layers != 28:
        raise SystemExit(f"teacher input has {model.config.num_hidden_layers} layers, expected 28")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    train_args = training_arguments(
        args.out,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator(tokenizer),
    )
    print(
        f"rows: {len(rows)} total, {len(holdout)} holdout, "
        f"{len(train_rows)} optimized ({args.augment} augmentations/original)"
    )
    trainer.train()
    model.config.use_cache = True
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    mean_f1, mean_tokens = evaluate_model(model, tokenizer, holdout)
    print(f"teacher holdout F1: {mean_f1:.6f}")
    print(f"teacher mean output tokens: {mean_tokens:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
