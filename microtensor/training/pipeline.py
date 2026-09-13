from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from microtensor.core.protocol import Role
from microtensor.training.arena import (
    BASE_REPO,
    BASE_REVISION,
    CORPUS_VERSION,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_QUANT,
    LORA_R,
)
from microtensor.training.dataset import download_public
from microtensor.training.export import export_gguf
from microtensor.training.lora import train_lora
from microtensor.training.prune_vocab import prune

log = logging.getLogger("microtensor.training.pipeline")


def work_dir(root: Path | None = None) -> Path:
<<<<<<< HEAD
    return (root or Path("work") / "guard").resolve()
=======
    return (root or Path("work") / "support").resolve()
>>>>>>> 83dd90a202f33179871ef4cbfa00b3f66a936779


def run(
    hook: Any = None,
    root: Path | None = None,
    *,
    epochs: float = 3.0,
    quant: str = DEFAULT_QUANT,
    lora_r: int = LORA_R,
) -> dict[Role, Path]:
    """Download → QLoRA → GGUF quantise. The `mt miner serve` entrypoint.

        mt miner serve --train microtensor.training.pipeline:run --epochs 3
    """
    root = work_dir(root)
    data = root / "data"
    pruned_base = root / "pruned-base"
    adapter = root / "lora"
    artifact = root / "artifact"
    try:
        sft, stats = download_public(data, version=CORPUS_VERSION)
        log.info(
            "train split %d examples; declare max_input around %d tokens",
            stats.n_train,
            stats.recommended_tokens,
        )
        if not (pruned_base / "config.json").is_file():
            prune(
                BASE_REPO,
                sft,
                pruned_base,
                revision=BASE_REVISION,
                target_size=76_000,
            )
        train_lora(
            sft,
            adapter,
            model_name=str(pruned_base),
            revision=None,
            epochs=epochs,
            max_seq_len=min(DEFAULT_MAX_SEQ_LEN, stats.recommended_tokens),
            lora_r=lora_r,
            hook=hook,
        )
        export_gguf(adapter, artifact, quant=quant)
        paths = {Role.FRONT: artifact}
        if hook is not None:
            hook.on_complete(paths)
        return paths
    except Exception as exc:
        if hook is not None:
            hook.on_failure(str(exc))
        raise
