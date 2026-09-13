from __future__ import annotations

from typing import Final

<<<<<<< HEAD
# Live Arena 6 (guard / mt-4g). Numbers below are what the coordinator
# serves for the open round, not the registered defaults in tracks.py.
# Training that ignores them produces an artifact that fails the gate.

TRACK: Final[str] = "guard"
HARDWARE_CLASS: Final[str] = "mt-4g"

# Front model. Gemma-3-4b-it is multimodal (wrong layer path + SWA/RoPE);
# Phi-4-mini is reserved as the escalation specialist.
BASE_REPO: Final[str] = "meta-llama/Llama-3.2-3B-Instruct"
BASE_REVISION: Final[str] = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
BASE_MODEL: Final[str] = f"{BASE_REPO}@{BASE_REVISION}"

SPECIALIST_REPO: Final[str] = "microsoft/Phi-4-mini-instruct"
SPECIALIST_REVISION: Final[str] = "cfbefacb99257ffa30c83adab238a50856ac3083"
SPECIALIST_MODEL: Final[str] = f"{SPECIALIST_REPO}@{SPECIALIST_REVISION}"

CORPUS_VERSION: Final[str] = (
    "sha256:a47b022cbe8b7a39beb259a26d7ed4ac01e33c0a229d80d14d7cacd44ebf8aa6"
)

# Anchored arena ceilings (live API), not tracks.py registered defaults.
MAX_SIZE_BYTES: Final[int] = 3_221_225_728  # 3.0 GiB
MAX_RSS_BYTES: Final[int] = 4_294_967_296  # 4.0 GiB
MAX_P95_MS: Final[int] = 45_000

# Live arena publishes reference_cost_ms=10000; repo default is 2000.
# Cost = total task latency; at-or-above the ceiling clamps ci and earns
# zero exclusive hypervolume. Prefer the live figure when sizing.
REFERENCE_COST_MS: Final[float] = 10_000.0

# Scoring budget on this track. Legal answers are ~10–40 tokens; 512 is the
# harness default and every extra token is billed as latency. Cap near 40.
MAX_OUTPUT_TOKENS: Final[int] = 40

# Prefill dominates p95. Public train prompts peak ~534 tokens (4 chars/tok);
# 1024 covers the chat wrapper with headroom without declaring a 128k context.
DEFAULT_MAX_INPUT_TOKENS: Final[int] = 1024
DEFAULT_MAX_SEQ_LEN: Final[int] = 1024

# Q8_0: ~455 M params fit comfortably under the 3 GiB ceiling and use int8
# SIMD on the single-thread CPU validator.
DEFAULT_QUANT: Final[str] = "Q8_0"

# llama.cpp tokenizer.ggml.pre after vocab prune (fingerprint no longer matches).
# Llama-3.2 Instruct is still llama-bpe; do not leave this as qwen2.
GGUF_VOCAB_PRE: Final[str] = "llama-bpe"

# Scored partitions are ~32–36% positive; public train is ~50%. Subsample
# positives to this prior unless the operator sweeps it.
DEFAULT_POSITIVE_RATE: Final[float] = 0.35
=======
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
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779

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
<<<<<<< HEAD

# Verbatim reply boilerplate from the published guard corpus. RAGTruth rows
# are wrapped with this same instruction so the model never sees a second
# output schema at train time.
GUARD_REPLY_INSTRUCTIONS: Final[str] = (
    'Reply with only JSON in this form and nothing else:\n'
    '{"unsupported": ["exact text of each unsupported span"]}\n'
    'If every part of the statement is supported, reply {"unsupported": []}.'
)
=======
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779
