from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from microtensor.training.arena import (
    BASE_REPO,
    BASE_REVISION,
    DEFAULT_MAX_SEQ_LEN,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGETS,
)
from microtensor.training.dataset import TrainError, SftExample, hf_token, load_sft, resolve_sft

log = logging.getLogger("microtensor.training.lora")

TRAIN_HINT = 'source /venv/main/bin/activate && uv pip install -e ".[train]"'


def _require_train_stack() -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        raise TrainError(
            f"the LoRA stack is not installed ({exc}). {TRAIN_HINT}"
        ) from exc
    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
    }


def _auth() -> dict[str, str]:
    token = hf_token()
    return {"token": token} if token else {}


class _CompletionDataset:
    """Prompt tokens masked; loss only on the gold JSON."""

    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def _encode(
    tokenizer: Any,
    example: SftExample,
    max_seq_len: int,
) -> dict[str, list[int]] | None:
    user = [{"role": "user", "content": example.prompt}]
    both = [
        {"role": "user", "content": example.prompt},
        {"role": "assistant", "content": example.completion},
    ]
    prompt_text = tokenizer.apply_chat_template(
        user, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        both, tokenize=False, add_generation_prompt=False
    )
    if not isinstance(prompt_text, str) or not isinstance(full_text, str):
        return None
    if not full_text.startswith(prompt_text):
        # Template inserted a suffix we cannot split on; train the whole
        # sequence rather than silently skipping the example.
        prompt_text = ""

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"] if prompt_text else []
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if not full_ids:
        return None

    if len(full_ids) > max_seq_len:
        overflow = len(full_ids) - max_seq_len
        # Keep the completion; drop prompt prefix. A truncated gold JSON is
        # worse than a truncated tool list.
        prompt_ids = prompt_ids[overflow:] if overflow < len(prompt_ids) else []
        full_ids = full_ids[overflow:]

    labels = list(full_ids)
    cut = min(len(prompt_ids), len(full_ids))
    for i in range(cut):
        labels[i] = -100
    if all(v == -100 for v in labels):
        return None
    return {"input_ids": list(full_ids), "labels": labels, "attention_mask": [1] * len(full_ids)}


def _collator(tokenizer: Any) -> Callable[[Sequence[dict[str, list[int]]]], dict[str, Any]]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise TrainError("tokenizer has no pad_token_id")

    def collate(features: Sequence[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(f["input_ids"]) for f in features)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        mask: list[list[int]] = []
        for feat in features:
            pad = width - len(feat["input_ids"])
            input_ids.append(feat["input_ids"] + [pad_id] * pad)
            labels.append(feat["labels"] + [-100] * pad)
            mask.append(feat["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    return collate


def _hook_callback(stack: dict[str, Any], hook: Any) -> Any:
    if hook is None:
        return None

    class _Cb(stack["TrainerCallback"]):  # type: ignore[misc, valid-type]
        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
            logs = logs or {}
            if "loss" not in logs:
                return control
            metrics = {"loss": float(logs["loss"]), "step": float(state.global_step)}
            if "learning_rate" in logs:
                metrics["learning_rate"] = float(logs["learning_rate"])
            hook.on_epoch_end(int(state.epoch or 0), metrics)
            return control

        def on_epoch_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            hook.on_epoch_end(int(state.epoch or 0), {"step": float(state.global_step)})
            return control

    return _Cb()


def train_lora(
    data: Path,
    output: Path,
    *,
    model_name: str = BASE_REPO,
    revision: str | None = BASE_REVISION,
    epochs: float = 3.0,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 2e-4,
    lora_r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    load_4bit: bool = True,
    seed: int = 42,
    val_fraction: float = 0.08,
    hook: Any = None,
) -> Path:
    """QLoRA SFT of the pinned xLAM checkpoint on the public train split."""
    stack = _require_train_stack()
    torch = stack["torch"]
    if not torch.cuda.is_available():
        raise TrainError("LoRA training needs a CUDA GPU; this host has none")

    token_kwargs = _auth()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    examples = load_sft(resolve_sft(data))
    log.info("loaded %d sft examples from %s", len(examples), data)

    model_path = Path(model_name).expanduser()
    model_revision = None if model_path.is_dir() else revision
    tokenizer = stack["AutoTokenizer"].from_pretrained(
        model_name, revision=model_revision, trust_remote_code=True, **token_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    encoded: list[dict[str, list[int]]] = []
    dropped = 0
    for ex in examples:
        row = _encode(tokenizer, ex, max_seq_len)
        if row is None:
            dropped += 1
            continue
        encoded.append(row)
    if dropped:
        log.warning("dropped %d examples that would not encode", dropped)
    if len(encoded) < 10:
        raise TrainError(f"only {len(encoded)} usable examples; check the sft jsonl")

    n_val = max(1, int(len(encoded) * val_fraction)) if val_fraction > 0 else 0
    train_rows = encoded[n_val:]
    val_rows = encoded[:n_val]
    log.info("train %d / val %d, max_seq_len %d", len(train_rows), len(val_rows), max_seq_len)

    compute = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs: dict[str, Any] = {
        "revision": model_revision,
        "trust_remote_code": True,
        "torch_dtype": compute,
        **token_kwargs,
    }
    if load_4bit:
        model_kwargs["quantization_config"] = stack["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute,
        )
        model_kwargs["device_map"] = "auto"

    model = stack["AutoModelForCausalLM"].from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    if load_4bit:
        model = stack["prepare_model_for_kbit_training"](model)
    model.gradient_checkpointing_enable()

    lora = stack["LoraConfig"](
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=stack["TaskType"].CAUSAL_LM,
        target_modules=list(LORA_TARGETS),
    )
    model = stack["get_peft_model"](model, lora)
    model.print_trainable_parameters()

    output.mkdir(parents=True, exist_ok=True)
    ta_kwargs: dict[str, Any] = {
        "output_dir": str(output),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 10,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "eval_strategy": "epoch" if val_rows else "no",
        "bf16": compute == torch.bfloat16,
        "fp16": compute != torch.bfloat16,
        "gradient_checkpointing": True,
        "optim": "adamw_torch",
        "seed": seed,
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": 2,
        "save_total_limit": 2,
    }
    try:
        args = stack["TrainingArguments"](**ta_kwargs)
    except TypeError:
        ta_kwargs["evaluation_strategy"] = ta_kwargs.pop("eval_strategy")
        args = stack["TrainingArguments"](**ta_kwargs)

    callbacks = []
    cb = _hook_callback(stack, hook)
    if cb is not None:
        callbacks.append(cb)

    trainer = stack["Trainer"](
        model=model,
        args=args,
        train_dataset=_CompletionDataset(train_rows),
        eval_dataset=_CompletionDataset(val_rows) if val_rows else None,
        data_collator=_collator(tokenizer),
        callbacks=callbacks or None,
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))

    card = {
        "base_model": f"{BASE_REPO}@{BASE_REVISION}",
        "training_checkpoint": str(model_path.resolve()) if model_path.is_dir() else model_name,
        "training_revision": model_revision or "",
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "max_seq_len": max_seq_len,
        "epochs": epochs,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "load_4bit": load_4bit,
    }
    (output / "train_card.json").write_text(
        json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log.info("adapter written to %s", output)
    return output
