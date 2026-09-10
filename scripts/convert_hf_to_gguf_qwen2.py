#!/usr/bin/env python3
"""Run llama.cpp's converter with tokenizer.ggml.pre pinned to qwen2.

Vocabulary pruning changes llama.cpp's tokenizer fingerprint even though the
pre-tokenizer algorithm did not change. The stock converter consequently
refuses the checkpoint. This wrapper changes only pre-tokenizer identification;
all conversion remains in the selected llama.cpp checkout.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: convert_hf_to_gguf_qwen2.py "
            "<llama.cpp/convert_hf_to_gguf.py> <model> [converter arguments...]"
        )
    converter = Path(sys.argv[1]).resolve()
    if not converter.is_file():
        raise SystemExit(f"converter is missing: {converter}")

    sys.path.insert(0, str(converter.parent))
    from conversion import ModelBase, load_all_models

    def qwen2_pre(self, tokenizer):  # type: ignore[no-untyped-def]
        return "qwen2"

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
        cls.get_vocab_base_pre = qwen2_pre  # type: ignore[attr-defined, method-assign]
        pending.extend(cls.__subclasses__())
    sys.argv = [str(converter), *sys.argv[2:]]
    runpy.run_path(str(converter), run_name="__main__")


if __name__ == "__main__":
    main()
