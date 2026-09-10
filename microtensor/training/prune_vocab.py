from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SYSTEM = (
    "You are a helpful assistant that can use tools. "
    "You are developed by Salesforce xLAM team."
)
TEMPLATE = (
    "<|im_start|>system\n"
    + SYSTEM
    + "\n<|im_end|>"
    + "<|im_start|>user\n{prompt}<|im_end|>"
    + "<|im_start|>assistant\n{answer}<|im_end|>"
)


class PruneError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TextCase:
    ref: str
    text: str


@dataclass(frozen=True, slots=True)
class Selection:
    regular_ids: tuple[int, ...]
    added_ids: tuple[int, ...]
    cutoff: int
    closure: frozenset[int]
    ascii_ids: frozenset[int]
    byte_ids: frozenset[int]

    @property
    def old_order(self) -> tuple[int, ...]:
        return self.regular_ids + self.added_ids


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte-to-unicode alphabet."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = list(byte_values)
    missing = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            chars.append(256 + missing)
            missing += 1
    return dict(zip(byte_values, (chr(value) for value in chars), strict=True))


def token_bytes(token: str, inverse: dict[str, int]) -> bytes | None:
    try:
        return bytes(inverse[char] for char in token)
    except KeyError:
        return None


def _answer(row: dict[str, Any]) -> str:
    for key in ("completion", "answer", "gold"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value.strip()
        return json.dumps(value, ensure_ascii=False)
    raise PruneError(f"task {row.get('ref')!r} has no answer/completion/gold")


def load_cases(corpus: Path) -> tuple[list[TextCase], list[dict[str, str]]]:
    if not corpus.is_file():
        raise PruneError(f"corpus {corpus} is missing")
    rows: list[dict[str, Any]] = []
    raw = corpus.read_text(encoding="utf-8")
    if raw.lstrip().startswith("{") and "\n" not in raw.strip():
        payload = json.loads(raw)
        rows = [row for row in payload.get("tasks", []) if isinstance(row, dict)]
    else:
        for number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PruneError(f"{corpus}:{number} is invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise PruneError(f"{corpus}:{number} is not a JSON object")
            rows.append(row)

    cases: list[TextCase] = []
    pairs: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        answer = _answer(row)
        ref = str(row.get("ref", f"row-{index}"))
        rendered = TEMPLATE.format(prompt=prompt, answer=answer)
        cases.append(TextCase(f"{ref}:rendered", rendered))
        cases.append(TextCase(f"{ref}:answer", answer))
        pairs.append({"ref": ref, "prompt": prompt, "answer": answer})
    if not pairs:
        raise PruneError(f"{corpus} contains no prompt/answer pairs")
    return cases, pairs


def _probe_source() -> str:
    requested = (
        Path("/home/ubuntu/belgian/application/helpers/microtensor-subnet")
        / "microtensor/envelope/probe.py"
    )
    if requested.is_file():
        return requested.read_text(encoding="utf-8")

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(  # noqa: S603
        ["git", "show", "origin/main:microtensor/envelope/probe.py"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout

    local = Path(__file__).resolve().parents[1] / "envelope" / "probe.py"
    if local.is_file():
        return local.read_text(encoding="utf-8")
    raise PruneError("could not read _WORDS locally or from origin/main")


def probe_words() -> list[str]:
    module = ast.parse(_probe_source())
    for node in module.body:
        target_name = ""
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                target_name, value = target.id, node.value
        if target_name == "_WORDS" and value is not None:
            words = ast.literal_eval(value)
            if isinstance(words, tuple) and all(isinstance(word, str) for word in words):
                return list(words)
    raise PruneError("_WORDS was not found in microtensor/envelope/probe.py")


def extra_cases(path: Path | None) -> list[TextCase]:
    lines = probe_words()
    if path is not None:
        if not path.is_file():
            raise PruneError(f"extra-text file {path} is missing")
        lines.extend(line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines())
    return [TextCase(f"extra:{index}", text) for index, text in enumerate(lines) if text]


def _source_file(model: str, revision: str | None, filename: str) -> Path | None:
    local = Path(model).expanduser()
    if local.is_dir():
        candidate = local / filename
        return candidate if candidate.is_file() else None
    try:
        from transformers.utils.hub import cached_file

        found = cached_file(model, filename, revision=revision, _raise_exceptions_for_missing_entries=False)
    except Exception:
        return None
    return Path(found) if found else None


def load_tokenizer_json(model: str, revision: str | None) -> tuple[dict[str, Any], Path]:
    path = _source_file(model, revision, "tokenizer.json")
    if path is None:
        raise PruneError(f"{model!r} has no tokenizer.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PruneError(f"{path} could not be read: {exc}") from exc
    if payload.get("model", {}).get("type") != "BPE":
        raise PruneError("only a tokenizer.json BPE model is supported")
    return payload, path


def merge_parents(tokenizer_json: dict[str, Any]) -> dict[int, tuple[int, int]]:
    vocab = tokenizer_json["model"]["vocab"]
    parents: dict[int, tuple[int, int]] = {}
    for merge in tokenizer_json["model"].get("merges", []):
        if isinstance(merge, str):
            parts = merge.split(" ", 1)
        elif isinstance(merge, list) and len(merge) == 2:
            parts = [str(merge[0]), str(merge[1])]
        else:
            raise PruneError(f"unsupported merge entry: {merge!r}")
        if len(parts) != 2:
            raise PruneError(f"malformed merge entry: {merge!r}")
        left, right = parts
        merged = left + right
        if left in vocab and right in vocab and merged in vocab:
            parents[int(vocab[merged])] = (int(vocab[left]), int(vocab[right]))
    return parents


def closure_of(used: set[int], parents: dict[int, tuple[int, int]]) -> set[int]:
    closure = set(used)
    stack = list(used)
    while stack:
        token_id = stack.pop()
        for parent in parents.get(token_id, ()):
            if parent not in closure:
                closure.add(parent)
                stack.append(parent)
    return closure


def select_tokens(
    tokenizer: Any,
    tokenizer_json: dict[str, Any],
    cases: list[TextCase],
    target_size: int,
) -> Selection:
    vocab = {str(token): int(token_id) for token, token_id in tokenizer_json["model"]["vocab"].items()}
    regular_ids = set(vocab.values())
    added = sorted(tokenizer_json.get("added_tokens", []), key=lambda item: int(item["id"]))
    added_ids = tuple(int(item["id"]) for item in added)

    used: set[int] = set()
    for case in cases:
        used.update(tokenizer(case.text, add_special_tokens=False)["input_ids"])
    parents = merge_parents(tokenizer_json)
    closure = closure_of(used, parents)

    inverse = {char: byte for byte, char in bytes_to_unicode().items()}
    decoded = {token_id: token_bytes(token, inverse) for token, token_id in vocab.items()}
    ascii_ids = {
        token_id
        for token_id, raw in decoded.items()
        if raw is not None and raw and all(byte < 128 for byte in raw)
    }
    byte_ids = {token_id for token_id, raw in decoded.items() if raw is not None and len(raw) == 1}
    if len(byte_ids) != 256:
        raise PruneError(f"expected 256 single-byte tokens, found {len(byte_ids)}")

    mandatory_regular = (closure & regular_ids) | byte_ids

    def keep_at(cutoff: int) -> set[int]:
        return mandatory_regular | {token_id for token_id in ascii_ids if token_id < cutoff}

    lo, hi = 0, max(regular_ids) + 1
    while lo < hi:
        mid = (lo + hi) // 2
        total = len(keep_at(mid)) + len(added_ids)
        if total < target_size:
            lo = mid + 1
        else:
            hi = mid
    candidates = {max(0, lo - 1), lo, min(max(regular_ids) + 1, lo + 1)}
    cutoff = min(
        candidates,
        key=lambda value: (abs(len(keep_at(value)) + len(added_ids) - target_size), value),
    )
    kept_regular = keep_at(cutoff)

    missing: list[tuple[int, int]] = []
    for token_id in kept_regular:
        for parent in parents.get(token_id, ()):
            if parent not in kept_regular:
                missing.append((token_id, parent))
    if missing:
        child, parent = missing[0]
        raise PruneError(
            f"KEEP is not merge-closed: token {child} requires parent {parent}; "
            "the cutoff rule is invalid for this tokenizer"
        )

    return Selection(
        regular_ids=tuple(sorted(kept_regular)),
        added_ids=added_ids,
        cutoff=cutoff,
        closure=frozenset(closure),
        ascii_ids=frozenset(ascii_ids),
        byte_ids=frozenset(byte_ids),
    )


def rebuild_tokenizer(
    payload: dict[str, Any],
    selection: Selection,
) -> tuple[dict[str, Any], dict[int, int]]:
    rebuilt = json.loads(json.dumps(payload))
    old_vocab = {str(token): int(token_id) for token, token_id in payload["model"]["vocab"].items()}
    id_to_token = {token_id: token for token, token_id in old_vocab.items()}
    mapping = {old_id: new_id for new_id, old_id in enumerate(selection.old_order)}
    regular_set = set(selection.regular_ids)

    rebuilt["model"]["vocab"] = {
        id_to_token[old_id]: mapping[old_id] for old_id in selection.regular_ids
    }
    kept_merges: list[Any] = []
    for merge in payload["model"].get("merges", []):
        if isinstance(merge, str):
            left, right = merge.split(" ", 1)
        else:
            left, right = str(merge[0]), str(merge[1])
        merged = left + right
        if (
            old_vocab.get(left) in regular_set
            and old_vocab.get(right) in regular_set
            and old_vocab.get(merged) in regular_set
        ):
            kept_merges.append(merge)
    rebuilt["model"]["merges"] = kept_merges

    new_added: list[dict[str, Any]] = []
    for item in sorted(payload.get("added_tokens", []), key=lambda row: int(row["id"])):
        rewritten = dict(item)
        rewritten["id"] = mapping[int(item["id"])]
        new_added.append(rewritten)
    rebuilt["added_tokens"] = new_added
    return rebuilt, mapping


def _remap_id(value: Any, mapping: dict[int, int], label: str) -> Any:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        if value not in mapping:
            raise PruneError(f"{label} references removed token id {value}")
        return mapping[value]
    if isinstance(value, list):
        return [_remap_id(item, mapping, label) for item in value]
    return value


def remap_config(value: Any, mapping: dict[int, int], path: str = "") -> Any:
    if isinstance(value, list):
        return [remap_config(item, mapping, f"{path}[]") for item in value]
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for key, item in value.items():
        location = f"{path}.{key}" if path else key
        if key.endswith(("_token_id", "_token_ids")):
            out[key] = _remap_id(item, mapping, location)
        elif key == "added_tokens_decoder" and isinstance(item, dict):
            out[key] = {
                str(_remap_id(int(old_id), mapping, location)): remap_config(entry, mapping, location)
                for old_id, entry in item.items()
            }
        else:
            out[key] = remap_config(item, mapping, location)
    return out


def write_tokenizer_files(
    model: str,
    revision: str | None,
    out: Path,
    tokenizer_json: dict[str, Any],
    mapping: dict[int, int],
    chat_template: str,
) -> None:
    (out / "tokenizer.json").write_text(
        json.dumps(tokenizer_json, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    for filename in ("tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        source = _source_file(model, revision, filename)
        if source is None:
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload = remap_config(payload, mapping)
        if filename == "tokenizer_config.json":
            payload["chat_template"] = chat_template
        (out / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (out / "chat_template.jinja").write_text(chat_template, encoding="utf-8")


def verify_tokenization(
    original: Any,
    pruned: Any,
    cases: list[TextCase],
    mapping: dict[int, int],
) -> None:
    reverse = {new: old for old, new in mapping.items()}
    for case in cases:
        old_ids = original(case.text, add_special_tokens=False)["input_ids"]
        new_ids = pruned(case.text, add_special_tokens=False)["input_ids"]
        old_tokens = original.convert_ids_to_tokens(old_ids)
        new_tokens = pruned.convert_ids_to_tokens(new_ids)
        if old_tokens != new_tokens:
            raise PruneError(f"token strings changed for {case.ref}")
        mapped_back = [reverse.get(token_id, -1) for token_id in new_ids]
        if mapped_back != old_ids:
            raise PruneError(f"token ids do not map back for {case.ref}")


def _generate(model: Any, tokenizer: Any, prompts: list[dict[str, str]], device: str) -> list[str]:
    import torch

    model.eval()
    model.to(device)
    outputs: list[str] = []
    with torch.inference_mode():
        for row in prompts[:20]:
            text = (
                "<|im_start|>system\n"
                + SYSTEM
                + "\n<|im_end|><|im_start|>user\n"
                + row["prompt"]
                + "<|im_end|><|im_start|>assistant\n"
            )
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            continuation = generated[0, encoded["input_ids"].shape[1] :]
            outputs.append(tokenizer.decode(continuation, skip_special_tokens=False))
    model.to("cpu")
    return outputs


def verify_generation(
    model_name: str,
    revision: str | None,
    out: Path,
    prompts: list[dict[str, str]],
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    auth = {"token": token} if token else {}
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    original_tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=True, **auth
    )
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=dtype, trust_remote_code=True, **auth
    )
    original_outputs = _generate(original_model, original_tokenizer, prompts, device)
    del original_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pruned_tokenizer = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
    pruned_model = AutoModelForCausalLM.from_pretrained(out, dtype=dtype, trust_remote_code=True)
    pruned_outputs = _generate(pruned_model, pruned_tokenizer, prompts, device)
    if original_outputs != pruned_outputs:
        for index, (before, after) in enumerate(zip(original_outputs, pruned_outputs, strict=True)):
            if before != after:
                raise PruneError(
                    f"greedy generation changed for {prompts[index]['ref']}: "
                    f"original={before!r}, pruned={after!r}"
                )


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def prune(
    model_name: str,
    corpus: Path,
    out: Path,
    *,
    revision: str | None = None,
    extra_text: Path | None = None,
    target_size: int = 76_000,
    verify_generations: int = 20,
) -> None:
    try:
        import torch
        from torch import nn
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise PruneError(f"missing pruning dependency: {exc}") from exc

    if out.exists() and any(out.iterdir()):
        raise PruneError(f"output directory {out} is not empty")
    out.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    auth = {"token": token} if token else {}
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=True, **auth
    )
    payload, _ = load_tokenizer_json(model_name, revision)
    corpus_cases, prompts = load_cases(corpus)
    cases = corpus_cases + extra_cases(extra_text)
    selection = select_tokens(tokenizer, payload, cases, target_size)
    rebuilt, mapping = rebuild_tokenizer(payload, selection)

    chat_template = tokenizer.chat_template
    if not isinstance(chat_template, str) or not chat_template:
        raise PruneError("original tokenizer has no chat template")
    write_tokenizer_files(model_name, revision, out, rebuilt, mapping, chat_template)
    (out / "vocab_map.json").write_text(
        json.dumps({str(old): new for old, new in sorted(mapping.items())}, indent=2) + "\n",
        encoding="utf-8",
    )

    pruned_tokenizer = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
    verify_tokenization(tokenizer, pruned_tokenizer, cases, mapping)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **auth,
    )
    embedding = model.get_input_embeddings()
    before_shape = tuple(embedding.weight.shape)
    order = torch.tensor(selection.old_order, dtype=torch.long, device=embedding.weight.device)
    tied = model.get_output_embeddings().weight.data_ptr() == embedding.weight.data_ptr()
    new_embedding = embedding.weight.detach().index_select(0, order).clone()
    embedding.weight = nn.Parameter(new_embedding, requires_grad=embedding.weight.requires_grad)
    embedding.num_embeddings = len(selection.old_order)

    output_head = model.get_output_embeddings()
    if tied:
        output_head.weight = embedding.weight
    else:
        new_output = output_head.weight.detach().index_select(0, order).clone()
        output_head.weight = nn.Parameter(new_output, requires_grad=output_head.weight.requires_grad)
        if hasattr(output_head, "out_features"):
            output_head.out_features = len(selection.old_order)

    model.config.vocab_size = len(selection.old_order)
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(model.config, key, None)
        setattr(model.config, key, _remap_id(value, mapping, f"config.{key}"))
    if getattr(model, "generation_config", None) is not None:
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(model.generation_config, key, None)
            setattr(model.generation_config, key, _remap_id(value, mapping, f"generation.{key}"))
    model.save_pretrained(out, safe_serialization=True)

    # save_pretrained rewrites configs; restore all tokenizer/config metadata
    # from source with token IDs remapped and no padded vocabulary.
    write_tokenizer_files(model_name, revision, out, rebuilt, mapping, chat_template)
    config_path = out / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["vocab_size"] = len(selection.old_order)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    reloaded_tokenizer = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
    verify_tokenization(tokenizer, reloaded_tokenizer, cases, mapping)
    after_shape = tuple(model.get_input_embeddings().weight.shape)
    del model
    gc.collect()

    if verify_generations:
        verify_generation(model_name, revision, out, prompts[:verify_generations])

    inverse = {char: byte for byte, char in bytes_to_unicode().items()}
    old_vocab = payload["model"]["vocab"]
    non_ascii = sum(
        1
        for token, token_id in old_vocab.items()
        if int(token_id) in selection.regular_ids
        and (raw := token_bytes(token, inverse)) is not None
        and any(byte >= 128 for byte in raw)
    )
    print(f"original size       {len(old_vocab) + len(payload.get('added_tokens', []))}")
    print(f"kept size           {len(selection.old_order)}")
    print(f"CUTOFF              {selection.cutoff}")
    print(f"|CLOSURE|           {len(selection.closure)}")
    print(f"non-ASCII kept      {non_ascii}")
    print(f"embedding before    {before_shape}")
    print(f"embedding after     {after_shape}")
    print(f"output bytes        {directory_size(out)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prune a Qwen2 BPE vocabulary without retokenizing task text")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--extra-text", type=Path)
    parser.add_argument("--target-size", type=int, default=76_000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--verify-generations",
        type=int,
        default=20,
        help="number of prompts for greedy equivalence; 0 skips only this expensive check",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prune(
            args.model,
            args.corpus,
            args.out,
            revision=args.revision,
            extra_text=args.extra_text,
            target_size=args.target_size,
            verify_generations=args.verify_generations,
        )
    except (PruneError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
