#!/usr/bin/env python3
"""Heal a width/layer-pruned student with full-parameter KD against the teacher."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

from microtensor.training.distill_common import (
    SEED,
    DistillError,
    augment_rows,
    collator,
    encode_rows,
    evaluate_guard,
    load_rows,
    make_probes,
    probe_words,
    read_holdout,
    rows_fitting_max_len,
    training_arguments,
)


class KDTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        teacher: Any,
        alpha: float,
        temperature: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        labels = inputs["labels"]
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

        student_logits = outputs.logits[:, :-1].float()
        teacher_logits = teacher_outputs.logits[:, :-1].float()
        targets = labels[:, 1:]
        mask = targets.ne(-100)
        temperature = self.temperature
        token_kl = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        kl_loss = token_kl[mask].mean() * temperature**2
        ce_loss = F.cross_entropy(
            student_logits.reshape(-1, student_logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
        )
        loss = self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss
        return (loss, outputs) if return_outputs else loss


def cpu_latency_proxy(model: Any, tokenizer: Any) -> float:
    rng = random.Random(SEED + 2)
    words = probe_words()
    text = ""
    ids: list[int] = []
    while len(ids) < 284:
        text += " " + " ".join(rng.choice(words) for _ in range(400))
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([ids[:284]], dtype=torch.long)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = model.to("cpu").eval()
    started = time.perf_counter()
    with torch.inference_mode():
        model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    elapsed_ms = (time.perf_counter() - started) * 1000
    torch.set_num_threads(old_threads)
    return elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Heal a pruned student with full-parameter KD")
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument(
        "--augment",
        type=int,
        default=0,
        help="tool-call distractor augmentations; keep 0 for guard",
    )
    parser.add_argument("--probe-rows", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not 0 <= args.alpha <= 1:
        raise SystemExit("--alpha must be in [0, 1]")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out}")

    try:
        rows = load_rows(args.data)
        holdout_refs = read_holdout(args.holdout)
        if len(holdout_refs) != 150:
            raise SystemExit(f"holdout has {len(holdout_refs)} refs, expected 150")
        holdout = [row for row in rows if row.ref in holdout_refs]
        base_train = [row for row in rows if row.ref not in holdout_refs]

        tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
        holdout = rows_fitting_max_len(tokenizer, holdout, args.max_len)
        base_train = rows_fitting_max_len(tokenizer, base_train, args.max_len)
        train_rows = list(base_train)
        if args.augment:
            train_rows = augment_rows(
                base_train,
                args.augment,
                tokenizer=tokenizer,
                max_len=args.max_len,
            )
        if args.probe_rows:
            train_rows = train_rows + make_probes(args.probe_rows)
        student = AutoModelForCausalLM.from_pretrained(
            args.student, dtype=torch.bfloat16, trust_remote_code=True
        )
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, dtype=torch.bfloat16, trust_remote_code=True
        )
        if student.config.vocab_size != teacher.config.vocab_size:
            raise SystemExit(
                f"vocabulary mismatch: student={student.config.vocab_size}, "
                f"teacher={teacher.config.vocab_size}"
            )
        if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
            raise SystemExit("student and teacher tokenizers differ")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dataset = encode_rows(tokenizer, train_rows, args.max_len)
    except (DistillError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    for parameter in student.parameters():
        parameter.requires_grad_(True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student.config.use_cache = False
    teacher.config.use_cache = False
    student.gradient_checkpointing_enable()
    teacher.eval()
    teacher.to("cuda" if torch.cuda.is_available() else "cpu")

    train_args = training_arguments(
        args.out,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )
    trainer = KDTrainer(
        model=student,
        teacher=teacher,
        alpha=args.alpha,
        temperature=args.temperature,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator(tokenizer),
    )
    print(
        f"KD rows: {len(train_rows)}; holdout: {len(holdout)}; "
        f"student layers={student.config.num_hidden_layers} "
        f"ffn={student.config.intermediate_size} "
        f"heads={student.config.num_attention_heads}/{student.config.num_key_value_heads}"
    )
    trainer.train()
    student.config.use_cache = True
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    teacher.config.use_cache = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student.to(device)
    teacher.to(device)

    teacher_metrics = evaluate_guard(teacher, tokenizer, holdout, max_new_tokens=40)
    student_metrics = evaluate_guard(student, tokenizer, holdout, max_new_tokens=40)

    print("=== teacher holdout ===")
    print(f"exact_match:          {teacher_metrics['exact_match']:.6f}")
    print(f"P(correct decision):  {teacher_metrics['decision_acc']:.6f}")
    print(f"P(exact copy|flag):   {teacher_metrics['copy_given_flag']:.6f}")
    print(f"specificity:          {teacher_metrics['specificity']:.6f}")
    print(f"sensitivity:          {teacher_metrics['sensitivity']:.6f}")

    print("=== healed student holdout ===")
    print(f"exact_match:          {student_metrics['exact_match']:.6f}")
    print(f"P(correct decision):  {student_metrics['decision_acc']:.6f}")
    print(f"P(exact copy|flag):   {student_metrics['copy_given_flag']:.6f}")
    print(f"specificity:          {student_metrics['specificity']:.6f}")
    print(f"sensitivity:          {student_metrics['sensitivity']:.6f}")
    print(f"mean output tokens:   {student_metrics['mean_output_tokens']:.2f}")
    print(f"p95_ms (eval device): {student_metrics['p95_ms']:.1f}")

    report = {
        "source_student": args.student,
        "source_teacher": args.teacher,
        "epochs": args.epochs,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "teacher_holdout": teacher_metrics,
        "healed_student_holdout": student_metrics,
    }
    (args.out / "heal_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"student CPU prefill, 284 tokens, 1 thread: "
        f"{cpu_latency_proxy(student, tokenizer):.2f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
