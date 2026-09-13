#!/usr/bin/env python3
"""Backward-compatible wrapper: pin tokenizer.ggml.pre to qwen2.

Prefer scripts/convert_hf_to_gguf_pre.py --pre <name> for new conversions.
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
    pre = Path(__file__).with_name("convert_hf_to_gguf_pre.py")
    sys.argv = [str(pre), "--pre", "qwen2", *sys.argv[1:]]
    runpy.run_path(str(pre), run_name="__main__")


if __name__ == "__main__":
    main()
