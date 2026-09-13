from __future__ import annotations

import argparse
import json
from pathlib import Path

from microtensor.cli.common import fail
from microtensor.core.constants import PUBLIC_SERVER_URL
from microtensor.training.arena import (
    BASE_MODEL,
    CORPUS_VERSION,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_POSITIVE_RATE,
    DEFAULT_QUANT,
    HARDWARE_CLASS,
    LORA_ALPHA,
    LORA_R,
    MAX_P95_MS,
    MAX_RSS_BYTES,
    MAX_SIZE_BYTES,
    TRACK,
)
from microtensor.training.dataset import TrainError, download_public
from microtensor.training.evaluate import evaluate_gguf
from microtensor.training.export import export_gguf
from microtensor.training.lora import train_lora
from microtensor.training.pipeline import run as run_pipeline
from microtensor.training.pipeline import work_dir
from microtensor.training.prune_vocab import PruneError, prune


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "train",
        help="download the guard corpus, prune/SFT Llama-3.2-3B, export a GGUF",
    )
    inner = parser.add_subparsers(dest="action", required=True)

    dl = inner.add_parser(
        "download",
        help="fetch the public guard split, mix RAGTruth spans, write sft.jsonl",
    )
    dl.add_argument("--out", type=Path, default=None, help="data directory (default work/guard/data)")
    dl.add_argument("--corpus-version", default=CORPUS_VERSION)
    dl.add_argument("--api", default=PUBLIC_SERVER_URL)
    dl.add_argument(
        "--positive-rate",
        type=float,
        default=DEFAULT_POSITIVE_RATE,
        help="subsample positives to this prior (scored partitions are ~0.35)",
    )
    dl.add_argument(
        "--ragtruth-parquet",
        type=Path,
        default=None,
        help="optional wandb/RAGTruth-processed parquet for sub-sentence positives",
    )
    dl.add_argument("--ragtruth-limit", type=int, default=2000)
    dl.add_argument(
        "--max-ragtruth-prompt-chars",
        type=int,
        default=2000,
        help="drop RAGTruth rows longer than this (HaluEval max is ~1600)",
    )
    dl.add_argument("--seed", type=int, default=1240)
    dl.set_defaults(handler=_download)

    prune_cmd = inner.add_parser(
        "prune",
        help="shrink the Llama vocabulary while preserving statement round-trips",
    )
    prune_cmd.add_argument("--model", default=BASE_MODEL.partition("@")[0])
    prune_cmd.add_argument("--revision", default=BASE_MODEL.partition("@")[2])
    prune_cmd.add_argument("--corpus", type=Path, default=None)
    prune_cmd.add_argument("--extra-text", type=Path)
    prune_cmd.add_argument("--target-size", type=int, default=48_000)
    prune_cmd.add_argument("--out", type=Path, default=None)
    prune_cmd.add_argument("--verify-generations", type=int, default=20)
    prune_cmd.set_defaults(handler=_prune)

    lora = inner.add_parser("lora", help="QLoRA SFT on the pinned Llama-3.2-3B-Instruct")
    lora.add_argument("--data", type=Path, default=None, help="sft.jsonl or the data directory")
    lora.add_argument("--output", type=Path, default=None, help="adapter directory")
    lora.add_argument("--model", default=None, help="pruned local base or the pinned HF model")
    lora.add_argument("--revision", default=None, help="HF revision; ignored for a local model")
    lora.add_argument("--epochs", type=float, default=3.0)
    lora.add_argument("--batch-size", type=int, default=2)
    lora.add_argument("--grad-accum", type=int, default=8)
    lora.add_argument("--lr", type=float, default=2e-4)
    lora.add_argument("--lora-r", type=int, default=LORA_R)
    lora.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    lora.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    lora.add_argument("--no-4bit", action="store_true", help="train bf16 instead of QLoRA")
    lora.set_defaults(handler=_lora)

    exp = inner.add_parser("export", help="merge LoRA and quantise to GGUF")
    exp.add_argument("--adapter", type=Path, default=None)
    exp.add_argument("--out", type=Path, default=None, help="artifact directory (model.gguf)")
    exp.add_argument("--merged", type=Path, default=None)
    exp.add_argument("--quant", default=DEFAULT_QUANT, help="default Q8_0 for guard; avoid Q4")
    exp.add_argument("--embedding-quant", default="Q8_0")
    exp.add_argument("--base-model", help="override the LoRA adapter's recorded base checkpoint")
    exp.add_argument(
        "--skip-merge",
        action="store_true",
        help="convert --merged HF checkpoint as-is (no LoRA merge)",
    )
    exp.set_defaults(handler=_export)

    evaluate = inner.add_parser(
        "eval",
        help="evaluate an exported GGUF like a validator on local held-out data",
    )
    evaluate.add_argument("--model", type=Path, default=None, help="model.gguf path")
    evaluate.add_argument("--data", type=Path, default=None, help="sft.jsonl or data directory")
    evaluate.add_argument("--limit", type=int, default=0, help="evaluate at most this many tasks")
    evaluate.add_argument("--heldout-fraction", type=float, default=0.08)
    evaluate.add_argument("--context-tokens", type=int, default=DEFAULT_MAX_SEQ_LEN)
    evaluate.add_argument("--show-failures", type=int, default=5)
    evaluate.set_defaults(handler=_eval)

    all_ = inner.add_parser("run", help="download + lora + export")
    all_.add_argument("--root", type=Path, default=None)
    all_.add_argument("--epochs", type=float, default=3.0)
    all_.add_argument("--quant", default=DEFAULT_QUANT)
    all_.set_defaults(handler=_run)


def _root(args: argparse.Namespace) -> Path:
    return work_dir(getattr(args, "root", None))


def _download(args: argparse.Namespace) -> int:
    out = args.out or (_root(args) / "data")
    try:
        sft, stats = download_public(
            out,
            version=args.corpus_version,
            api=args.api,
            positive_rate=args.positive_rate,
            ragtruth_parquet=args.ragtruth_parquet,
            ragtruth_limit=args.ragtruth_limit,
            max_ragtruth_prompt_chars=args.max_ragtruth_prompt_chars,
            seed=args.seed,
        )
    except TrainError as exc:
        return fail(str(exc))
    print(f"track          {TRACK}/{HARDWARE_CLASS}")
    print(f"base           {BASE_MODEL}")
    print(f"examples       {stats.n_train}")
    print(
        "class balance  "
        f"pos {stats.n_positive}  neg {stats.n_negative}  "
        f"rate {stats.positive_rate:.3f}  (target {args.positive_rate})"
    )
    print(f"origins        halueval {stats.n_halueval}  ragtruth {stats.n_ragtruth}")
    print(f"legal gate     {stats.legal_ok}/{stats.n_train} canonical completions")
    print(
        "prompt chars   "
        f"min {stats.prompt_chars_min}  p50 {stats.prompt_chars_p50}  "
        f"p95 {stats.prompt_chars_p95}  max {stats.prompt_chars_max}"
    )
    print(f"declare tokens {stats.recommended_tokens}  (p95 prompt + template + 64 out)")
    print(
        "envelope       "
        f"{MAX_SIZE_BYTES / 1024**3:.1f} GiB  "
        f"{MAX_RSS_BYTES / 1024**3:.0f} GiB rss  "
        f"{MAX_P95_MS} ms p95"
    )
    print(f"sft            {sft}")
    return 0


def _prune(args: argparse.Namespace) -> int:
    root = _root(args)
    corpus = args.corpus or (root / "data" / "sft.jsonl")
    output = args.out or (root / "pruned-base")
    try:
        prune(
            args.model,
            corpus,
            output,
            revision=args.revision,
            extra_text=args.extra_text,
            target_size=args.target_size,
            verify_generations=args.verify_generations,
        )
    except (PruneError, OSError, ValueError) as exc:
        return fail(str(exc))
    print(f"pruned base    {output}")
    return 0


def _lora(args: argparse.Namespace) -> int:
    root = _root(args)
    data = args.data or (root / "data")
    output = args.output or (root / "lora")
    try:
        path = train_lora(
            data,
            output,
            model_name=args.model or BASE_MODEL.partition("@")[0],
            revision=args.revision or BASE_MODEL.partition("@")[2],
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            max_seq_len=args.max_seq_len,
            load_4bit=not args.no_4bit,
        )
    except TrainError as exc:
        return fail(str(exc))
    print(f"adapter {path}")
    card = path / "train_card.json"
    if card.is_file():
        print(card.read_text(encoding="utf-8").rstrip())
    return 0


def _export(args: argparse.Namespace) -> int:
    root = _root(args)
    adapter = args.adapter or (root / "lora")
    artifact = args.out or (root / "artifact")
    merged = args.merged or (root / "merged")
    try:
        gguf = export_gguf(
            adapter,
            artifact,
            merged=merged,
            quant=args.quant,
            embedding_quant=args.embedding_quant,
            base_model=args.base_model,
            skip_merge=args.skip_merge,
        )
    except TrainError as exc:
        return fail(str(exc))
    size = gguf.stat().st_size
    print(f"gguf     {gguf}")
    print(f"size     {size / 1024**3:.2f} GiB  (ceiling {MAX_SIZE_BYTES / 1024**3:.1f} GiB)")
    print(f"quant    {args.quant}")
    print(
        "next     mt miner init --artifact",
        artifact,
        f"--track {TRACK} --hardware-class {HARDWARE_CLASS}",
    )
    envelope = artifact / "envelope.json"
    if envelope.is_file():
        print(json.dumps(json.loads(envelope.read_text()), indent=2))
    return 0


def _eval(args: argparse.Namespace) -> int:
    root = _root(args)
    model = args.model or (root / "artifact" / "model.gguf")
    data = args.data or (root / "data")
    try:
        result = evaluate_gguf(
            model,
            data,
            limit=args.limit,
            heldout_fraction=args.heldout_fraction,
            context_tokens=args.context_tokens,
            show_failures=args.show_failures,
        )
    except TrainError as exc:
        return fail(str(exc))
    print(f"tasks          {result.tasks}")
    print(f"completed      {result.completed}/{result.tasks}")
    print(f"valid JSON     {result.json_valid / result.tasks:.2%}")
    print(f"exact tasks    {result.exact_tasks / result.tasks:.2%}")
    print(f"tool-call F1   {result.tool_call_f1:.4f}")
    print(
        f"calls          TP {result.true_positive}  "
        f"FP {result.false_positive}  FN {result.false_negative}"
    )
    print(f"p95 TTFT       {result.ttft_p95_ms} ms  (arena ceiling {MAX_P95_MS} ms)")
    print(f"p95 total      {result.total_p95_ms} ms")
    return 0


def _run(args: argparse.Namespace) -> int:
    try:
        paths = run_pipeline(root=args.root, epochs=args.epochs, quant=args.quant)
    except TrainError as exc:
        return fail(str(exc))
    for role, path in paths.items():
        print(f"{role.value:<12} {path}")
    return 0
