from __future__ import annotations

import json
import os
<<<<<<< HEAD
import random
=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microtensor.core.constants import PUBLIC_SERVER_URL
<<<<<<< HEAD
from microtensor.training.arena import (
    CORPUS_VERSION,
    DEFAULT_POSITIVE_RATE,
    GUARD_REPLY_INSTRUCTIONS,
    PUBLIC_CORPUS_PATH,
    TRACK,
)

SEED = 1240
=======
from microtensor.training.arena import CORPUS_VERSION, PUBLIC_CORPUS_PATH, TRACK
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779


class TrainError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SftExample:
    ref: str
    prompt: str
    completion: str
<<<<<<< HEAD
    origin: str = ""
    spans: tuple[str, ...] = ()

    @property
    def positive(self) -> bool:
        return bool(self.spans)
=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779


@dataclass(frozen=True, slots=True)
class DatasetStats:
    n_train: int
<<<<<<< HEAD
    n_positive: int
    n_negative: int
    positive_rate: float
    n_halueval: int
    n_ragtruth: int
=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
    prompt_chars_min: int
    prompt_chars_p50: int
    prompt_chars_p95: int
    prompt_chars_max: int
    recommended_tokens: int
<<<<<<< HEAD
    legal_ok: int
    legal_failed: int
=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779


def corpus_url(version: str = CORPUS_VERSION, api: str = PUBLIC_SERVER_URL) -> str:
    return api.rstrip("/") + PUBLIC_CORPUS_PATH.format(version=version)


def _get(url: str, timeout: int = 120) -> dict[str, Any]:
    if not url.startswith(("http://", "https://")):
        raise TrainError(f"refusing a non-http corpus url: {url}")
    request = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": "microtensor-train", "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:  # noqa: S310
            return json.loads(answer.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise TrainError(f"{url} could not be read: {exc}") from exc


<<<<<<< HEAD
def canonical_completion(spans: Sequence[str]) -> str:
    """The only legal assistant strings on this track.

    `json.dumps` with default separators yields `": "` and `", "`, straight
    double quotes, and no trailing whitespace — the shape the scorer's
    `_output_spans` parses. Spans are kept as plain text (the span text), so
    whole-statement HaluEval gold and sub-sentence RAGTruth gold share one
    target form.
    """
    return json.dumps({"unsupported": list(spans)}, ensure_ascii=False)


def parse_spans(gold: Any) -> list[str]:
    """Pull unsupported spans from a gold blob without assuming one shape."""
    if gold is None:
        return []
    if isinstance(gold, str):
        text = gold.strip()
        if not text:
            return []
        try:
            gold = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TrainError(f"gold is not json: {exc}") from exc
    if isinstance(gold, dict):
        for key in ("unsupported", "unsupported_spans", "spans"):
            if key in gold:
                value = gold[key]
                if value is None:
                    return []
                if isinstance(value, str):
                    return [value] if value else []
                if isinstance(value, list | tuple):
                    return [str(item) for item in value]
                raise TrainError(f"gold.{key} is not a list of spans")
        return []
    if isinstance(gold, list | tuple):
        return [str(item) for item in gold]
    raise TrainError(f"unrecognised gold type {type(gold).__name__}")


def assert_legal_completion(completion: str) -> list[str]:
    """Fail closed unless the target is byte-identical to a canonical dump."""
    text = completion.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TrainError(f"completion is not json: {exc}") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"unsupported"}:
        raise TrainError("completion must be exactly {\"unsupported\": [...]}")
    spans = parsed["unsupported"]
    if not isinstance(spans, list) or any(not isinstance(s, str) for s in spans):
        raise TrainError("unsupported must be a list of strings")
    canonical = canonical_completion(spans)
    if canonical != text:
        raise TrainError(
            "completion is not canonical; "
            f"got {text!r}, want {canonical!r}"
        )
    return spans


def _is_public_train(ref: str, row: dict[str, Any]) -> bool:
    """Keep the published train split; drop fixed/rotating leaks in /public."""
    ref_l = ref.lower()
    # Ref wins: the public endpoint has occasionally labelled rotating rows
    # as partition=train, and those prompts are not the HaluEval train prior.
    if "rotat" in ref_l or "fixed" in ref_l:
        return False
    partition = str(row.get("partition", "")).strip().lower()
    if partition:
        return partition == "train"
    return "train" in ref_l or not ref_l.startswith("halu-")


def examples_from_public(tasks: Sequence[dict[str, Any]]) -> list[SftExample]:
    out: list[SftExample] = []
    for row in tasks:
        ref = str(row.get("ref", ""))
        if not _is_public_train(ref, row):
            continue
        prompt = str(row.get("prompt", ""))
        if not prompt.strip():
            raise TrainError(f"task {ref!r} has no prompt")
        # Prompt is reused verbatim — including leading/trailing layout the
        # corpus published. Only the completion is canonicalised.
        spans = parse_spans(row.get("gold"))
        completion = canonical_completion(spans)
        assert_legal_completion(completion)
        origin = str((row.get("inputs") or {}).get("origin") or "halueval")
        out.append(
            SftExample(
                ref=ref,
                prompt=prompt,
                completion=completion,
                origin=origin,
                spans=tuple(spans),
=======
def _completion(gold: Any) -> str:
    """The assistant turn, byte-stable with the published gold.

    Re-serialising a parsed object would change `100.0` into `100` and shuffle
    keys. The scorer matches name, arguments and JSON types, so the gold string
    is the target, not a pretty-printed cousin of it.
    """
    if isinstance(gold, str):
        text = gold.strip()
        if not text:
            raise TrainError("gold string is empty")
        json.loads(text)
        return text
    return json.dumps(gold, ensure_ascii=False)


def examples_from(tasks: Sequence[dict[str, Any]]) -> list[SftExample]:
    out: list[SftExample] = []
    for row in tasks:
        if str(row.get("partition", "train")) != "train":
            continue
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            raise TrainError(f"task {row.get('ref')!r} has no prompt")
        out.append(
            SftExample(
                ref=str(row["ref"]),
                prompt=prompt,
                completion=_completion(row.get("gold")),
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
            )
        )
    if not out:
        raise TrainError("the public split published no train tasks")
    return out


<<<<<<< HEAD
_PROMPT_PREFIX = (
    "Below is a source passage and a statement generated from it.\n"
    "Return every part of the statement that the source does not support.\n"
)


def guard_prompt(*, source: str, statement: str, question: str = "") -> str:
    """Build a prompt that matches the published guard boilerplate."""
    source = source.strip()
    statement = statement.strip()
    if not source or not statement:
        raise TrainError("RAGTruth row is missing source or statement")
    parts = [_PROMPT_PREFIX, f"Source:\n{source}\n"]
    if question.strip():
        parts.append(f"\nQuestion: {question.strip()}\n")
    parts.append(f"\nStatement:\n{statement}\n")
    parts.append(f"\n{GUARD_REPLY_INSTRUCTIONS}")
    return "".join(parts)


def _ragtruth_spans(labels: Any, output: str) -> list[str]:
    if isinstance(labels, str):
        try:
            labels = json.loads(labels)
        except json.JSONDecodeError:
            return []
    if not isinstance(labels, list):
        return []
    spans: list[str] = []
    for item in labels:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if text is None and "start" in item and "end" in item:
            try:
                text = output[int(item["start"]) : int(item["end"])]
            except (TypeError, ValueError):
                continue
        if isinstance(text, str) and text:
            spans.append(text)
    return spans


def examples_from_ragtruth(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int | None = None,
    seed: int = SEED,
    max_prompt_chars: int = 2_000,
) -> list[SftExample]:
    """Wrap RAGTruth rows in the guard prompt; targets are the span texts.

    `max_prompt_chars` keeps mixed rows near the published HaluEval length
    distribution. Unfiltered RAGTruth summaries run to 10k+ chars and would
    dominate prefill cost on the single-thread CPU validator.
    """
    rng = random.Random(seed)
    indexed = list(enumerate(rows))
    rng.shuffle(indexed)
    out: list[SftExample] = []
    for index, row in indexed:
        if limit is not None and len(out) >= limit:
            break
        output = str(row.get("output") or "")
        context = str(row.get("context") or row.get("input_str") or "")
        query = str(row.get("query") or "")
        spans = _ragtruth_spans(row.get("hallucination_labels"), output)
        # Skip empties here — public HaluEval already supplies the negative
        # mass. RAGTruth's value is sub-sentence positive localisation.
        if not spans:
            continue
        # Drop rows whose spans are not substrings of the statement; those
        # cannot be copied character-for-character at score time.
        statement_norm = output
        if any(span not in statement_norm for span in spans):
            continue
        try:
            prompt = guard_prompt(source=context, statement=output, question=query)
        except TrainError:
            continue
        if len(prompt) > max_prompt_chars:
            continue
        completion = canonical_completion(spans)
        assert_legal_completion(completion)
        task = str(row.get("task_type") or "ragtruth")
        out.append(
            SftExample(
                ref=f"ragtruth-{task}-{index}",
                prompt=prompt,
                completion=completion,
                origin="ragtruth",
                spans=tuple(spans),
            )
        )
    return out


def load_ragtruth_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise TrainError(
            "pyarrow is required to mix RAGTruth; pip install pyarrow"
        ) from exc
    if not path.is_file():
        raise TrainError(f"RAGTruth parquet missing: {path}")
    return pq.read_table(path).to_pylist()


def reweight_positives(
    examples: Sequence[SftExample],
    positive_rate: float = DEFAULT_POSITIVE_RATE,
    *,
    seed: int = SEED,
) -> list[SftExample]:
    """Subsample positives so the train prior matches the scored prior.

    Scored fixed/rotating partitions are ~32–36% positive; public train is
    ~50%. At a realistic copy fidelity, each point of specificity lost must
    be repaid by ~2 points of sensitivity — so over-representing positives
    teaches the model to flag too eagerly.
    """
    if not 0.0 < positive_rate < 1.0:
        raise TrainError(f"positive_rate must be in (0,1), got {positive_rate}")
    positives = [ex for ex in examples if ex.positive]
    negatives = [ex for ex in examples if not ex.positive]
    if not negatives:
        raise TrainError("reweight needs at least one negative example")
    if not positives:
        raise TrainError("reweight needs at least one positive example")

    # pos / (pos + neg) = r  =>  pos = r/(1-r) * neg
    target_pos = int(round((positive_rate / (1.0 - positive_rate)) * len(negatives)))
    target_pos = max(1, min(len(positives), target_pos))
    rng = random.Random(seed)
    kept_pos = rng.sample(positives, target_pos) if target_pos < len(positives) else list(positives)
    merged = kept_pos + list(negatives)
    rng.shuffle(merged)
    return merged


=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
def stats_of(examples: Sequence[SftExample]) -> DatasetStats:
    lengths = sorted(len(ex.prompt) for ex in examples)
    n = len(lengths)
    p50 = lengths[n // 2]
    p95 = lengths[min(n - 1, int(n * 0.95))]
<<<<<<< HEAD
    # 4 chars/token plus chat-template and ~64-token completion headroom.
    recommended = int(((p95 / 4) + 160 + 63) // 64 * 64)
    recommended = max(512, min(2048, recommended))
    n_pos = sum(1 for ex in examples if ex.positive)
    legal_ok = 0
    legal_failed = 0
    for ex in examples:
        try:
            assert_legal_completion(ex.completion)
            legal_ok += 1
        except TrainError:
            legal_failed += 1
    return DatasetStats(
        n_train=n,
        n_positive=n_pos,
        n_negative=n - n_pos,
        positive_rate=(n_pos / n) if n else 0.0,
        n_halueval=sum(1 for ex in examples if "halu" in ex.origin),
        n_ragtruth=sum(1 for ex in examples if ex.origin == "ragtruth"),
=======
    # 4 chars/token plus chat-template and 256-token completion headroom.
    recommended = int(((p95 / 4) + 320 + 255) // 256 * 256)
    recommended = max(512, min(2048, recommended))
    return DatasetStats(
        n_train=n,
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
        prompt_chars_min=lengths[0],
        prompt_chars_p50=p50,
        prompt_chars_p95=p95,
        prompt_chars_max=lengths[-1],
        recommended_tokens=recommended,
<<<<<<< HEAD
        legal_ok=legal_ok,
        legal_failed=legal_failed,
    )


def build_guard_sft(
    *,
    public_tasks: Sequence[dict[str, Any]],
    ragtruth_rows: Sequence[dict[str, Any]] | None = None,
    positive_rate: float = DEFAULT_POSITIVE_RATE,
    ragtruth_limit: int | None = 2_000,
    max_ragtruth_prompt_chars: int = 2_000,
    seed: int = SEED,
) -> tuple[list[SftExample], DatasetStats]:
    """Public HaluEval prompts (verbatim) + optional RAGTruth span mix, reweighted."""
    examples = examples_from_public(public_tasks)
    if ragtruth_rows:
        examples.extend(
            examples_from_ragtruth(
                ragtruth_rows,
                limit=ragtruth_limit,
                seed=seed,
                max_prompt_chars=max_ragtruth_prompt_chars,
            )
        )
    # Dedup by ref after the mix.
    seen: set[str] = set()
    unique: list[SftExample] = []
    for ex in examples:
        if ex.ref in seen:
            continue
        seen.add(ex.ref)
        unique.append(ex)
    reweighted = reweight_positives(unique, positive_rate, seed=seed)
    stats = stats_of(reweighted)
    if stats.legal_failed:
        raise TrainError(f"{stats.legal_failed} completions failed the legal-string gate")
    return reweighted, stats


=======
    )


>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
def download_public(
    out_dir: Path,
    *,
    version: str = CORPUS_VERSION,
    api: str = PUBLIC_SERVER_URL,
    timeout: int = 120,
<<<<<<< HEAD
    positive_rate: float = DEFAULT_POSITIVE_RATE,
    ragtruth_parquet: Path | None = None,
    ragtruth_limit: int | None = 2_000,
    max_ragtruth_prompt_chars: int = 2_000,
    seed: int = SEED,
) -> tuple[Path, DatasetStats]:
    """Fetch the public train split and write a guard SFT jsonl."""
=======
) -> tuple[Path, DatasetStats]:
    """Fetch the public train split and write an SFT jsonl next to it."""
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
    payload = _get(corpus_url(version, api), timeout=timeout)
    track = str(payload.get("track") or TRACK)
    if track != TRACK:
        raise TrainError(f"corpus {version} is track {track!r}, not {TRACK!r}")

    tasks = [t for t in payload.get("tasks", []) if isinstance(t, dict)]
<<<<<<< HEAD
    ragtruth_rows: list[dict[str, Any]] | None = None
    if ragtruth_parquet is not None:
        ragtruth_rows = load_ragtruth_parquet(ragtruth_parquet)

    examples, stats = build_guard_sft(
        public_tasks=tasks,
        ragtruth_rows=ragtruth_rows,
        positive_rate=positive_rate,
        ragtruth_limit=ragtruth_limit,
        max_ragtruth_prompt_chars=max_ragtruth_prompt_chars,
        seed=seed,
    )
=======
    examples = examples_from(tasks)
    stats = stats_of(examples)
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "public.json"
    raw_path.write_text(json.dumps(payload), encoding="utf-8")

    sft_path = write_sft(out_dir / "sft.jsonl", examples)
    meta = {
        "track": track,
        "corpus_version": str(payload.get("version") or version),
        "reference_model": payload.get("reference_model") or "",
        "counts": payload.get("counts") or {},
        "n_sft": stats.n_train,
<<<<<<< HEAD
        "n_positive": stats.n_positive,
        "n_negative": stats.n_negative,
        "positive_rate": stats.positive_rate,
        "positive_rate_target": positive_rate,
        "n_halueval": stats.n_halueval,
        "n_ragtruth": stats.n_ragtruth,
        "legal_ok": stats.legal_ok,
=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
        "prompt_chars": {
            "min": stats.prompt_chars_min,
            "p50": stats.prompt_chars_p50,
            "p95": stats.prompt_chars_p95,
            "max": stats.prompt_chars_max,
        },
        "recommended_max_input_tokens": stats.recommended_tokens,
        "sft": str(sft_path),
<<<<<<< HEAD
        "ragtruth_parquet": str(ragtruth_parquet) if ragtruth_parquet else "",
=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sft_path, stats


def write_sft(path: Path, examples: Sequence[SftExample]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
<<<<<<< HEAD
            assert_legal_completion(ex.completion)
=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
            row = {
                "ref": ex.ref,
                "prompt": ex.prompt,
                "completion": ex.completion,
<<<<<<< HEAD
                "origin": ex.origin,
                "n_spans": len(ex.spans),
=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
                "messages": [
                    {"role": "user", "content": ex.prompt},
                    {"role": "assistant", "content": ex.completion},
                ],
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_sft(path: Path) -> list[SftExample]:
    if not path.is_file():
        raise TrainError(f"sft file {path} is missing; run `mt train download` first")
    examples: list[SftExample] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainError(f"{path}:{number} is not json: {exc}") from exc
<<<<<<< HEAD
            completion = str(row["completion"])
            spans = tuple(assert_legal_completion(completion))
=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
            examples.append(
                SftExample(
                    ref=str(row.get("ref", f"row-{number}")),
                    prompt=str(row["prompt"]),
<<<<<<< HEAD
                    completion=completion,
                    origin=str(row.get("origin", "")),
                    spans=spans,
=======
                    completion=str(row["completion"]),
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
                )
            )
    if not examples:
        raise TrainError(f"{path} holds no examples")
    return examples


def resolve_sft(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "sft.jsonl"
    if candidate.is_file():
        return candidate
    raise TrainError(f"no sft.jsonl at {path}")


def hf_token() -> str:
    return (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    )
<<<<<<< HEAD


# Legacy alias kept so older call sites that expected examples_from still work.
def examples_from(tasks: Sequence[dict[str, Any]]) -> list[SftExample]:
    return examples_from_public(tasks)
=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
