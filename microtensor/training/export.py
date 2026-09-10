from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from microtensor.training.arena import (
    BASE_REPO,
    BASE_REVISION,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_QUANT,
    MAX_SIZE_BYTES,
)
from microtensor.training.dataset import TrainError, hf_token
from microtensor.training.lora import TRAIN_HINT

log = logging.getLogger("microtensor.training.export")

LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"


def _tools_root() -> Path:
    home = Path(os.environ.get("MT_HOME", Path.home() / ".microtensor"))
    return home / "tools" / "llama.cpp"


def _auth_env() -> dict[str, str]:
    env = dict(os.environ)
    token = hf_token()
    if token:
        env.setdefault("HF_TOKEN", token)
        env.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return env


def merge_adapter(adapter: Path, merged: Path, base_model: str | None = None) -> Path:
    """Merge LoRA into the checkpoint it was trained from."""
    try:
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise TrainError(f"merge needs the train stack ({exc}). {TRAIN_HINT}") from exc

    if not adapter.is_dir():
        raise TrainError(f"adapter directory {adapter} is missing; run `mt train lora` first")

    token = hf_token()
    token_kwargs = {"token": token} if token else {}
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    adapter_config = PeftConfig.from_pretrained(str(adapter))
    source = base_model or str(adapter_config.base_model_name_or_path) or BASE_REPO
    source_path = Path(source).expanduser()
    revision = None if source_path.is_dir() else BASE_REVISION
    log.info("loading LoRA base %s%s", source, f"@{revision}" if revision else "")
    base = AutoModelForCausalLM.from_pretrained(
        source,
        revision=revision,
        dtype=dtype,
        trust_remote_code=True,
        device_map="cpu",
        **token_kwargs,
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    model = model.merge_and_unload()
    merged.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, trust_remote_code=True, **token_kwargs)
    tokenizer.save_pretrained(str(merged))
    log.info("merged weights at %s", merged)
    return merged


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    log.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, env=_auth_env(), check=False)  # noqa: S603
    if result.returncode != 0:
        raise TrainError(f"command failed ({result.returncode}): {' '.join(cmd)}")


def ensure_llama_cpp(root: Path | None = None) -> tuple[Path, Path]:
    """Clone llama.cpp if needed and build llama-quantize.

    Conversion is Python (`convert_hf_to_gguf.py`); quantisation is the
    compiled binary. CUDA is not required for either.
    """
    root = root or _tools_root()
    convert = root / "convert_hf_to_gguf.py"
    if not convert.is_file():
        root.parent.mkdir(parents=True, exist_ok=True)
        if root.exists():
            shutil.rmtree(root)
        _run(["git", "clone", "--depth", "1", LLAMA_REPO, str(root)])

    quantize = _quantize_bin(root)
    if not quantize.is_file():
        _run(
            [
                "cmake",
                "-S",
                str(root),
                "-B",
                str(root / "build"),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DGGML_NATIVE=OFF",
                "-DGGML_CUDA=OFF",
            ]
        )
        _run(
            [
                "cmake",
                "--build",
                str(root / "build"),
                "--target",
                "llama-quantize",
                "-j",
                str(os.cpu_count() or 4),
            ]
        )
        quantize = _quantize_bin(root)
    if not quantize.is_file():
        raise TrainError(f"llama-quantize was not built under {root}")
    return convert, quantize


def _quantize_bin(root: Path) -> Path:
    for candidate in (
        root / "build" / "bin" / "llama-quantize",
        root / "build" / "llama-quantize",
        root / "llama-quantize",
    ):
        if candidate.is_file():
            return candidate
    return root / "build" / "bin" / "llama-quantize"


def convert_and_quantize(
    merged: Path,
    artifact: Path,
    *,
    quant: str = DEFAULT_QUANT,
    embedding_quant: str = "Q8_0",
    keep_fp16: bool = False,
) -> Path:
    convert, quantize = ensure_llama_cpp()
    try:
        import gguf  # noqa: F401
    except ImportError as exc:
        raise TrainError(
            f"gguf is required to convert HuggingFace weights ({exc}). {TRAIN_HINT}"
        ) from exc

    staging = artifact.parent / ".gguf-staging"
    staging.mkdir(parents=True, exist_ok=True)
    fp16 = staging / "model-f16.gguf"
    wrapper = Path(__file__).resolve().parents[2] / "scripts" / "convert_hf_to_gguf_qwen2.py"
    _run(
        [
            sys.executable,
            str(wrapper),
            str(convert),
            str(merged),
            "--outfile",
            str(fp16),
            "--outtype",
            "f16",
        ]
    )
    if not fp16.is_file():
        raise TrainError(f"convert_hf_to_gguf wrote nothing at {fp16}")

    artifact.mkdir(parents=True, exist_ok=True)
    out = artifact / "model.gguf"
    quantize_cmd = [str(quantize)]
    if embedding_quant:
        quantize_cmd.extend(["--token-embedding-type", embedding_quant])
    quantize_cmd.extend([str(fp16), str(out), quant])
    _run(quantize_cmd)
    if not keep_fp16:
        fp16.unlink(missing_ok=True)
    size = out.stat().st_size
    log.info("quantised %s  %.2f GiB  (%s)", out, size / 1024**3, quant)
    if size > MAX_SIZE_BYTES:
        raise TrainError(
            f"{out} is {size} bytes, over the {MAX_SIZE_BYTES} byte class ceiling; "
            "try Q4_K_S or Q3_K_M"
        )
    (artifact / "envelope.json").write_text(
        (
            "{\n"
            f'  "quant": "{quant}",\n'
            f'  "embedding_quant": "{embedding_quant}",\n'
            f'  "entrypoint": "model.gguf",\n'
            f'  "max_input_tokens": {DEFAULT_MAX_INPUT_TOKENS},\n'
            f'  "size_bytes": {size}\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    return out


def export_gguf(
    adapter: Path,
    artifact: Path,
    *,
    merged: Path | None = None,
    quant: str = DEFAULT_QUANT,
    embedding_quant: str = "Q8_0",
    base_model: str | None = None,
    skip_merge: bool = False,
) -> Path:
    """Merge LoRA, convert to GGUF, quantise into the artifact directory."""
    merged = merged or artifact.parent / "merged"
    if not skip_merge:
        merge_adapter(adapter, merged, base_model=base_model)
    elif not merged.is_dir():
        raise TrainError(f"merged directory {merged} is missing")
    return convert_and_quantize(
        merged,
        artifact,
        quant=quant,
        embedding_quant=embedding_quant,
    )
