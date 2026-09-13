# Converting a vocabulary-pruned checkpoint to GGUF

Vocabulary pruning changes token IDs and therefore changes the hash that
llama.cpp's `convert_hf_to_gguf.py` uses to identify a BPE pre-tokenizer. The
algorithm did not change — only the fingerprint. Pin `tokenizer.ggml.pre` to
the family the base model actually used:

| Base family | `--pre` |
|---|---|
| Llama 3 / 3.2 Instruct | `llama-bpe` |
| Qwen2 / xLAM | `qwen2` |

## Conversion

```bash
python scripts/convert_hf_to_gguf_pre.py --pre llama-bpe \
  /path/to/llama.cpp/convert_hf_to_gguf.py \
  work/guard/healed_a_8l_e6 \
  --outfile work/guard/.gguf-staging/model-f16.gguf \
  --outtype f16
```

`mt train export` uses this wrapper automatically (`GGUF_VOCAB_PRE` in
`microtensor/training/arena.py`). Legacy Qwen2 callers can still use
`scripts/convert_hf_to_gguf_qwen2.py`.

## Quantization (Arena 6 / guard)

Ship **Q8_0** body + Q8_0 embeddings (not Q4/Q6). ~455 M params stay under
the 3 GiB class ceiling:

```bash
/path/to/llama.cpp/build/bin/llama-quantize \
  --token-embedding-type Q8_0 \
  work/guard/.gguf-staging/model-f16.gguf \
  work/guard/artifact/model.gguf \
  Q8_0
```

Integrated (full HF checkpoint, no LoRA merge):

```bash
PYTHONPATH=. python -c "
from pathlib import Path
from microtensor.training.export import convert_and_quantize
convert_and_quantize(
    Path('work/guard/healed_a_8l_e6'),
    Path('work/guard/artifact'),
    quant='Q8_0',
    embedding_quant='Q8_0',
)
"
```

Or via CLI:

```bash
mt train export \
  --skip-merge \
  --merged work/guard/healed_a_8l_e6 \
  --out work/guard/artifact \
  --quant Q8_0 \
  --embedding-quant Q8_0
```
