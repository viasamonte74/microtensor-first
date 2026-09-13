from __future__ import annotations

import ast
import json
import random
import re
import subprocess
<<<<<<< HEAD
import time
=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SEED = 1240
EMPTY_CALLS = '{"tool_calls": []}'
<<<<<<< HEAD
EMPTY_UNSUPPORTED = '{"unsupported": []}'
=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
_FUNCTION_BLOCK = re.compile(r"(Functions:\n)(.*?)(\n\nRequest:)", re.DOTALL)
_FUNCTION_LINE = re.compile(r"^- ([^(]+)\((.*?)\)(.*)$")
_CALL_NOISE = re.compile(r"```(?:json)?|</?tool_call>|</?function_call>", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


class DistillError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Row:
    ref: str
    prompt: str
    completion: str

    @property
    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": self.prompt},
            {"role": "assistant", "content": self.completion},
        ]


class TokenDataset:
    def __init__(self, rows: Sequence[dict[str, list[int]]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                messages = value["messages"]
                prompt = str(messages[0]["content"])
                completion = str(messages[-1]["content"])
                rows.append(Row(str(value["ref"]), prompt, completion))
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                raise DistillError(f"{path}:{number} is malformed: {exc}") from exc
    if len({row.ref for row in rows}) != len(rows):
        raise DistillError("data contains duplicate refs")
    if len(rows) < 151:
        raise DistillError("at least 151 rows are required")
    return rows


def choose_holdout(rows: Sequence[Row], count: int = 150) -> list[Row]:
    return random.Random(SEED).sample(list(rows), count)


def write_holdout(path: Path, rows: Sequence[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row.ref}\n" for row in rows), encoding="utf-8")


def read_holdout(path: Path) -> set[str]:
    if not path.is_file():
        raise DistillError(f"holdout file is missing: {path}")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _function_lines(prompt: str) -> list[str]:
    match = _FUNCTION_BLOCK.search(prompt)
    if not match:
        return []
    return [line for line in match.group(2).splitlines() if line.startswith("- ")]


def _line_parts(line: str) -> tuple[str, list[str], str]:
    match = _FUNCTION_LINE.match(line)
    if not match:
        raise DistillError(f"function line is malformed: {line}")
    params = []
    if match.group(2).strip():
        params = [part.split(":", 1)[0].strip() for part in match.group(2).split(",")]
    return match.group(1).strip(), params, match.group(3)


def _called_names(completion: str) -> set[str]:
    try:
        value = json.loads(completion)
    except json.JSONDecodeError:
        return set()
    return {
        str(call.get("name"))
        for call in value.get("tool_calls", [])
        if isinstance(call, dict) and call.get("name")
    }


def _suffix_name(name: str, suffix: str) -> str:
    def rename_part(part: str) -> str:
        pieces = part.split("_")
        if len(pieces) > 1:
            return pieces[0] + "".join(piece[:1].upper() + piece[1:] for piece in pieces[1:])
        return f"{part}{suffix.upper()}"

    return ".".join(rename_part(part) for part in name.split("."))


def _rename(
    lines: Sequence[str], completion: str, suffix: str
) -> tuple[list[str], str]:
    function_map: dict[str, str] = {}
    parameter_map: dict[str, dict[str, str]] = {}
    rewritten: list[str] = []
    for line in lines:
        name, params, _ = _line_parts(line)
        new_name = _suffix_name(name, suffix)
        function_map[name] = new_name
        parameter_map[name] = {param: _suffix_name(param, suffix) for param in params}
        new_line = line.replace(f"- {name}(", f"- {new_name}(", 1)
        for old, new in parameter_map[name].items():
            new_line = re.sub(rf"(?<=[(,] ){re.escape(old)}(?=:)", new, new_line)
            if new_line.startswith(f"- {new_name}({old}:"):
                new_line = new_line.replace(f"({old}:", f"({new}:", 1)
        rewritten.append(new_line)

    value = json.loads(completion)
    for call in value.get("tool_calls", []):
        if not isinstance(call, dict):
            continue
        old_name = str(call.get("name", ""))
        if old_name in function_map:
            call["name"] = function_map[old_name]
        arguments = call.get("arguments")
        if isinstance(arguments, dict):
            names = parameter_map.get(old_name, {})
            call["arguments"] = {names.get(key, key): val for key, val in arguments.items()}
    return rewritten, json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def augment_rows(
    rows: Sequence[Row],
    variants: int,
    *,
    tokenizer: Any | None = None,
    max_len: int | None = None,
) -> list[Row]:
    if variants <= 0:
        return list(rows)
    rng = random.Random(SEED)
    pool: list[tuple[str, str, str]] = []
    for row in rows:
        for line in _function_lines(row.prompt):
            try:
                pool.append((row.ref, _line_parts(line)[0], line))
            except DistillError:
                continue

    result = list(rows)
    for row_index, row in enumerate(rows):
        own = _function_lines(row.prompt)
        called = _called_names(row.completion)
        candidates = [
            (name, line)
            for source_ref, name, line in pool
            if source_ref != row.ref and name not in called
        ]
        for variant in range(variants):
            for attempt in range(100):
                selected = list(own)
                wanted = rng.randint(1, 3)
                candidate_order = list(candidates)
                rng.shuffle(candidate_order)
                # Bias toward shorter copied signatures so long originals still
                # satisfy the strict no-truncation sequence limit.
                candidate_order.sort(key=lambda item: len(item[1]) + rng.randint(0, 300))
                present = {_line_parts(line)[0] for line in selected}
                for name, line in candidate_order:
                    if name in present:
                        continue
                    selected.append(line)
                    present.add(name)
                    if len(selected) >= len(own) + wanted:
                        break
                if len(selected) != len(own) + wanted:
                    raise DistillError(f"not enough eligible distractors for {row.ref}")
                renamed, completion = _rename(
                    selected,
                    row.completion,
                    chr(ord("a") + (row_index + variant) % 26),
                )
                rng.shuffle(renamed)
                match = _FUNCTION_BLOCK.search(row.prompt)
                if not match:
                    raise DistillError(f"{row.ref} has no Functions/Request block")
                prompt = (
                    row.prompt[: match.start(2)]
                    + "\n".join(renamed)
                    + row.prompt[match.end(2) :]
                )
                augmented = Row(f"{row.ref}-aug{variant}", prompt, completion)
                if tokenizer is not None and max_len is not None:
                    rendered = tokenizer.apply_chat_template(
                        augmented.messages, tokenize=False, add_generation_prompt=False
                    )
                    length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
                    if length > max_len:
                        continue
                result.append(augmented)
                break
            else:
                raise DistillError(
                    f"could not make a <= {max_len}-token augmentation for {row.ref}"
                )
    return result


def probe_words() -> tuple[str, ...]:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(root), "show", "origin/main:microtensor/envelope/probe.py"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise DistillError(f"could not read probe words: {result.stderr.strip()}")
    module = ast.parse(result.stdout)
    for node in module.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "_WORDS" and node.value is not None:
                value = ast.literal_eval(node.value)
                return tuple(str(word) for word in value)
    raise DistillError("origin/main probe.py has no _WORDS tuple")


def make_probes(count: int) -> list[Row]:
    rng = random.Random(SEED + 1)
    words = probe_words()
    return [
        Row(
            f"probe-{index:05d}",
            " ".join(rng.choice(words) for _ in range(rng.randint(400, 700))),
            EMPTY_CALLS,
        )
        for index in range(count)
    ]


def encode_row(tokenizer: Any, row: Row, max_len: int) -> dict[str, list[int]]:
    prefix = tokenizer.apply_chat_template(
        row.messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(row.messages, tokenize=False, add_generation_prompt=False)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_len:
        raise DistillError(f"{row.ref} has {len(full_ids)} tokens, over --max-len {max_len}")
    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids) :]
    if len(labels) != len(full_ids) or all(label == -100 for label in labels):
        raise DistillError(f"could not identify assistant tokens for {row.ref}")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


<<<<<<< HEAD
def packed_token_count(tokenizer: Any, row: Row) -> int:
    full = tokenizer.apply_chat_template(
        row.messages, tokenize=False, add_generation_prompt=False
    )
    return len(tokenizer(full, add_special_tokens=False)["input_ids"])


def rows_fitting_max_len(
    tokenizer: Any, rows: Sequence[Row], max_len: int
) -> list[Row]:
    """Drop examples that cannot train/eval inside the declared context.

    Raising max_len to fit RAGTruth would train long prefills and inflate
    validator p95. Keep max_len at the arena input budget and skip the tail.
    """
    kept: list[Row] = []
    dropped = 0
    longest = 0
    for row in rows:
        n = packed_token_count(tokenizer, row)
        longest = max(longest, n)
        if n <= max_len:
            kept.append(row)
        else:
            dropped += 1
    print(
        f"max_len={max_len}: kept {len(kept)} / {len(rows)} "
        f"(dropped {dropped} over-length, longest={longest})"
    )
    if not kept:
        raise DistillError(f"no rows fit --max-len {max_len} (longest={longest})")
    return kept


=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
def encode_rows(tokenizer: Any, rows: Sequence[Row], max_len: int) -> TokenDataset:
    return TokenDataset([encode_row(tokenizer, row, max_len) for row in rows])


def collator(tokenizer: Any):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate(features: Sequence[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        output: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = width - len(feature["input_ids"])
            output["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
            output["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            output["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in output.items()}

    return collate


def training_arguments(
    out: Path,
    *,
    epochs: float,
    lr: float,
    batch_size: int,
    grad_accum: int,
) -> Any:
    from transformers import TrainingArguments

    values = dict(
        output_dir=str(out),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
<<<<<<< HEAD
        save_total_limit=1,
        # Full-parameter 3B AdamW state is ~12 GiB; a second epoch checkpoint
        # OOM'd the disk (basic_ios / unexpected pos) after train had finished.
        save_only_model=True,
=======
        save_total_limit=2,
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    try:
        return TrainingArguments(**values)
    except TypeError:
        values.pop("warmup_ratio")
        values["warmup_steps"] = 0.03
        return TrainingArguments(**values)


def _normalise_text(value: Any) -> str:
    return _WHITESPACE.sub(" ", str(value).strip().lower())


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {_normalise_text(value)} if value.strip() else set()
    if isinstance(value, dict):
        return {f"{_normalise_text(k)}={_normalise_text(v)}" for k, v in value.items()}
    if isinstance(value, Iterable):
        return {_normalise_text(v) for v in value if str(v).strip()}
    return {_normalise_text(value)}


def f1(predicted: set[str], gold: set[str]) -> float:
    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0
    matched = len(predicted & gold)
    if matched == 0:
        return 0.0
    precision = matched / len(predicted)
    recall = matched / len(gold)
    return 2 * precision * recall / (precision + recall)


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        return _normalise_text(value)
    if isinstance(value, dict):
        return {_normalise_text(key): _canonical(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(val) for val in value]
    return _normalise_text(value)


def _call_keys(calls: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(calls, dict):
        calls = [calls]
    if not isinstance(calls, list | tuple):
        return keys
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if isinstance(function, dict):
            call = function
        name = call.get("name", "")
        arguments = call.get("arguments", call.get("parameters", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {"_raw": arguments}
        if not str(name).strip():
            continue
        packed = json.dumps(_canonical(arguments), sort_keys=True, separators=(",", ":"))
        keys.add(f"{_normalise_text(name)}({packed})")
    return keys


def _parse_calls(output: Any) -> Any:
    if isinstance(output, dict | list):
        return output
    text = _CALL_NOISE.sub(" ", str(output if output is not None else "")).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    starts = sorted(index for index in (text.find("{"), text.find("[")) if index >= 0)
    for start in starts:
        try:
            value, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        return value
    return None


def _gold_cases(gold: Any) -> list[dict[str, Any]]:
    if isinstance(gold, str):
        try:
            gold = json.loads(gold)
        except ValueError:
            return []
    if not isinstance(gold, dict):
        return []
    if "tool_calls" in gold or "rubric" in gold:
        return [gold]
    cases = gold.get("tests")
    if isinstance(cases, list | tuple):
        return [case for case in cases if isinstance(case, dict)]
    return []


def rubric_f1_tool_calls(output: Any, gold: Any) -> float:
    cases = _gold_cases(gold)
    gold_calls: set[str] = set()
    gold_points: set[str] = set()
    for case in cases:
        gold_calls |= _call_keys(case.get("tool_calls"))
        gold_points |= _as_set(case.get("rubric"))
    if not gold_calls and not gold_points:
        return 0.0
    parsed = _parse_calls(output)
    if isinstance(parsed, dict):
        pred_calls = _call_keys(parsed.get("tool_calls", parsed if "name" in parsed else []))
        pred_points = _as_set(parsed.get("covers"))
    else:
        pred_calls = _call_keys(parsed)
        pred_points = set()
    if not gold_calls:
        return f1(pred_points, gold_points)
    calls_score = f1(pred_calls, gold_calls)
    if not gold_points:
        return calls_score
    return 0.5 * f1(pred_points, gold_points) + 0.5 * calls_score


def evaluate_model(model: Any, tokenizer: Any, rows: Sequence[Row], limit: int = 0) -> tuple[float, float]:
<<<<<<< HEAD
    """Legacy support-track F1. Prefer `evaluate_guard` for Arena 6."""
=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
    device = next(model.parameters()).device
    model.eval()
    scores: list[float] = []
    counts: list[int] = []
    selected = list(rows[:limit] if limit else rows)
    with torch.inference_mode():
        for row in selected:
            prompt = tokenizer.apply_chat_template(
                row.messages[:-1], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            continuation = generated[0, inputs["input_ids"].shape[1] :]
            output = tokenizer.decode(continuation, skip_special_tokens=True)
            scores.append(rubric_f1_tool_calls(output, row.completion))
            counts.append(int(continuation.numel()))
    return sum(scores) / len(scores), sum(counts) / len(counts)
<<<<<<< HEAD


def evaluate_guard(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Row],
    *,
    limit: int = 0,
    max_new_tokens: int = 40,
) -> dict[str, float]:
    """Holdout metrics for hallucination detection.

    Returns exact-match (span_accuracy after JSON parse), decision accuracy,
    copy fidelity on correctly flagged positives, specificity, sensitivity,
    mean output tokens, and mean wall ms per example (device-dependent).
    """
    from microtensor.scoring.metrics import span_accuracy

    device = next(model.parameters()).device
    model.eval()
    selected = list(rows[:limit] if limit else rows)
    exact = 0
    decisions = 0
    copy_ok = 0
    flagged_correct = 0
    tp = fp = tn = fn = 0
    tokens = 0
    times_ms: list[float] = []

    with torch.inference_mode():
        for row in selected:
            prompt = tokenizer.apply_chat_template(
                row.messages[:-1], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - start) * 1000.0)
            continuation = generated[0, inputs["input_ids"].shape[1] :]
            output = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            tokens += int(continuation.numel())

            score = span_accuracy(output, row.completion)
            if score >= 1.0 - 1e-9:
                exact += 1

            try:
                pred = json.loads(output)
                pred_spans = pred.get("unsupported", []) if isinstance(pred, dict) else None
                if not isinstance(pred_spans, list):
                    pred_spans = None
            except (json.JSONDecodeError, TypeError):
                pred_spans = None
            try:
                gold = json.loads(row.completion)
                gold_spans = list(gold.get("unsupported", []))
            except (json.JSONDecodeError, TypeError, AttributeError):
                gold_spans = []

            gold_pos = bool(gold_spans)
            pred_pos = bool(pred_spans) if pred_spans is not None else False
            if pred_spans is not None and pred_pos == gold_pos:
                decisions += 1
            if gold_pos and pred_pos:
                tp += 1
                flagged_correct += 1
                if pred_spans is not None and score >= 1.0 - 1e-9:
                    copy_ok += 1
            elif gold_pos and not pred_pos:
                fn += 1
            elif (not gold_pos) and pred_pos:
                fp += 1
            else:
                tn += 1

    n = max(len(selected), 1)
    times_ms.sort()
    p95 = times_ms[min(len(times_ms) - 1, int(0.95 * (len(times_ms) - 1)))] if times_ms else 0.0
    return {
        "n": float(len(selected)),
        "exact_match": exact / n,
        "decision_acc": decisions / n,
        "copy_given_flag": (copy_ok / flagged_correct) if flagged_correct else 0.0,
        "specificity": (tn / (tn + fp)) if (tn + fp) else 0.0,
        "sensitivity": (tp / (tp + fn)) if (tp + fn) else 0.0,
        "mean_output_tokens": tokens / n,
        "mean_ms": (sum(times_ms) / n) if times_ms else 0.0,
        "p95_ms": p95,
    }
=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
