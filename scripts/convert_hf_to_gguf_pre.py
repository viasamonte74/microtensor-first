#!/usr/bin/env python3
"""Run llama.cpp's converter with tokenizer.ggml.pre pinned.

Vocabulary pruning changes llama.cpp's tokenizer fingerprint even though the
pre-tokenizer algorithm did not change. The stock converter consequently
refuses the checkpoint. This wrapper changes only pre-tokenizer identification;
all conversion remains in the selected llama.cpp checkout.

Usage:
  convert_hf_to_gguf_pre.py --pre llama-bpe \\
    <llama.cpp/convert_hf_to_gguf.py> <model> [converter arguments...]
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pin tokenizer.ggml.pre then run llama.cpp HF→GGUF conversion"
    )
    parser.add_argument(
        "--pre",
        required=True,
        help="tokenizer.ggml.pre value (e.g. llama-bpe, qwen2)",
    )
    parser.add_argument("converter", type=Path, help="path to convert_hf_to_gguf.py")
    parser.add_argument(
        "converter_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to the converter (model path, --outfile, ...)",
    )
    args = parser.parse_args()

    converter = args.converter.resolve()
    if not converter.is_file():
        raise SystemExit(f"converter is missing: {converter}")

    forwarded = list(args.converter_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if not forwarded:
        raise SystemExit("missing converter arguments (model path required)")

    pre = args.pre
    sys.path.insert(0, str(converter.parent))
    from conversion import ModelBase, load_all_models

    def pinned_pre(self, tokenizer):  # type: ignore[no-untyped-def]
        return pre

    # The converter lazily imports architecture classes. Load them first, then
    # pin the method on every class so a model module that cached/inherited the
    # stock implementation cannot restore fingerprint-based detection.
    load_all_models()
    pending = [ModelBase]
    seen: set[type] = set()
    while pending:
        cls = pending.pop()
        if cls in seen:
            continue
        seen.add(cls)
        cls.get_vocab_base_pre = pinned_pre  # type: ignore[attr-defined, method-assign]
        pending.extend(cls.__subclasses__())

    sys.argv = [str(converter), *forwarded]
    runpy.run_path(str(converter), run_name="__main__")


if __name__ == "__main__":
    main()
