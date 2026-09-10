from __future__ import annotations

from typing import Final

# Live Arena 5 (support / mt-3g). These are the numbers the coordinator
# serves, not the registered defaults in tracks.py. Training that ignores
# them produces an artifact that fails the gate.

TRACK: Final[str] = "support"
HARDWARE_CLASS: Final[str] = "mt-3g"

BASE_REPO: Final[str] = "Salesforce/xLAM-2-1b-fc-r"
BASE_REVISION: Final[str] = "6870fcf102f434bd01819302d85780b8e767925d"
BASE_MODEL: Final[str] = f"{BASE_REPO}@{BASE_REVISION}"

CORPUS_VERSION: Final[str] = (
    "sha256:caef393af24094d0d4f2ade03395ce85fd58def22d38e4085e9d9a9b188909f6"
)

MAX_SIZE_BYTES: Final[int] = 1_610_612_736  # 1.5 GiB
MAX_RSS_BYTES: Final[int] = 3_221_225_472  # 3 GiB
MAX_P95_MS: Final[int] = 8_000

# Scoring budget on this track. Completions longer than this are truncated
# before the JSON closes and score zero.
MAX_OUTPUT_TOKENS: Final[int] = 256

# Prefill dominates p95 on the CPU validator. 1536 covers the public train
# tail (max ~1.5k tokens with the chat wrapper) without declaring xLAM's 32k
# context, which would miss the 8s ceiling. Override if a selfcheck fails.
DEFAULT_MAX_INPUT_TOKENS: Final[int] = 1536
DEFAULT_MAX_SEQ_LEN: Final[int] = 1536

# Cost-first K-quant. Q8_0 for this 1.5B class sits on the 1.5 GiB size
# line; Q4_K_M is the point that still formats JSON and stays cheap.
DEFAULT_QUANT: Final[str] = "Q4_K_M"

LORA_R: Final[int] = 16
LORA_ALPHA: Final[int] = 32
LORA_DROPOUT: Final[float] = 0.05
LORA_TARGETS: Final[tuple[str, ...]] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

PUBLIC_CORPUS_PATH: Final[str] = "/v1/corpora/{version}/public"
