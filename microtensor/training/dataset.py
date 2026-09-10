from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microtensor.core.constants import PUBLIC_SERVER_URL
from microtensor.training.arena import CORPUS_VERSION, PUBLIC_CORPUS_PATH, TRACK


class TrainError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SftExample:
    ref: str
    prompt: str
    completion: str


@dataclass(frozen=True, slots=True)
class DatasetStats:
    n_train: int
    prompt_chars_min: int
    prompt_chars_p50: int
    prompt_chars_p95: int
    prompt_chars_max: int
    recommended_tokens: int


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
            )
        )
    if not out:
        raise TrainError("the public split published no train tasks")
    return out


def stats_of(examples: Sequence[SftExample]) -> DatasetStats:
    lengths = sorted(len(ex.prompt) for ex in examples)
    n = len(lengths)
    p50 = lengths[n // 2]
    p95 = lengths[min(n - 1, int(n * 0.95))]
    # 4 chars/token plus chat-template and 256-token completion headroom.
    recommended = int(((p95 / 4) + 320 + 255) // 256 * 256)
    recommended = max(512, min(2048, recommended))
    return DatasetStats(
        n_train=n,
        prompt_chars_min=lengths[0],
        prompt_chars_p50=p50,
        prompt_chars_p95=p95,
        prompt_chars_max=lengths[-1],
        recommended_tokens=recommended,
    )


def download_public(
    out_dir: Path,
    *,
    version: str = CORPUS_VERSION,
    api: str = PUBLIC_SERVER_URL,
    timeout: int = 120,
) -> tuple[Path, DatasetStats]:
    """Fetch the public train split and write an SFT jsonl next to it."""
    payload = _get(corpus_url(version, api), timeout=timeout)
    track = str(payload.get("track") or TRACK)
    if track != TRACK:
        raise TrainError(f"corpus {version} is track {track!r}, not {TRACK!r}")

    tasks = [t for t in payload.get("tasks", []) if isinstance(t, dict)]
    examples = examples_from(tasks)
    stats = stats_of(examples)

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
        "prompt_chars": {
            "min": stats.prompt_chars_min,
            "p50": stats.prompt_chars_p50,
            "p95": stats.prompt_chars_p95,
            "max": stats.prompt_chars_max,
        },
        "recommended_max_input_tokens": stats.recommended_tokens,
        "sft": str(sft_path),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sft_path, stats


def write_sft(path: Path, examples: Sequence[SftExample]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            row = {
                "ref": ex.ref,
                "prompt": ex.prompt,
                "completion": ex.completion,
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
            examples.append(
                SftExample(
                    ref=str(row.get("ref", f"row-{number}")),
                    prompt=str(row["prompt"]),
                    completion=str(row["completion"]),
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
