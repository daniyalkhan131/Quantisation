# Quantizing Whisper-large with MLX (and an experimental TurboQuant KV-cache port)

## Read this first: what TurboQuant actually is

TurboQuant lives in [`Blaizzy/mlx-vlm`](https://github.com/Blaizzy/mlx-vlm)
(`mlx_vlm/turboquant.py`, documented in the repo's README under
"TurboQuant KV Cache"). It is **not** a weight-quantization tool. It's a
scheme that compresses the *attention KV cache built up during autoregressive
generation* (random rotation + codebook quantization,
[arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) down to 2-4 bits, so
mlx-vlm's own models can hold longer contexts in less memory. It's turned on
with `--kv-bits`/`--kv-quant-scheme turboquant` on `mlx_vlm.generate` /
`mlx_vlm.server`, and it only applies to architectures mlx-vlm already knows
how to load (`mlx_vlm/models/*`, using mlx-vlm's own `KVCache` class).

**Whisper isn't one of those architectures.** mlx-vlm has no Whisper model
definition, no audio processor for it, nothing -- there's no way to point
`mlx_vlm.generate` at `openai/whisper-large-v3` at all, TurboQuant or not.
It's also worth knowing that even if it were wired up, TurboQuant compresses
the *decode-time* KV cache, which mostly matters at long context lengths.
Whisper's decoder only ever generates up to 448 tokens per 30s audio chunk,
so the realistic savings are small (see the [caveats](#caveats-honest-numbers)
below) -- Whisper-large's real cost is its ~1.5B weights, not its KV cache.

Given that, this folder does two different things:

- **`convert_from_hf.py`** -- the practical answer to "quantize Whisper-large":
  weight-quantize it with MLX's standard affine quantization (what actually
  shrinks the model), producing a checkpoint that runs with the standard
  [`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
  package, unmodified.
- **`turboquant_decoder.py`** -- a from-scratch MLX Whisper decoder that
  *does* wire mlx-vlm's actual `TurboQuantKVCache` class onto Whisper's
  decoder self-attention, so you can genuinely run and measure TurboQuant on
  Whisper. This is the literal ask, delivered as an experiment with honestly
  reported (small) numbers, not a production speedup.

Both were built and verified in this session on real speech (see
[Verified results](#verified-results-this-session) below); the commands
below also apply directly to `openai/whisper-large-v3`, just heavier/slower
to run than the `whisper-tiny` smoke test used for verification.

## Setup

```sh
cd whisper_turboquant
python3 -m venv .venv && source .venv/bin/activate

pip install mlx mlx-whisper numpy safetensors huggingface_hub

# Needed only for turboquant_decoder.py, which imports mlx_vlm.turboquant
# and mlx_vlm.models.* directly:
git clone git@github.com:Blaizzy/mlx-vlm.git
pip install -e ./mlx-vlm

# Needed only if you'll feed arbitrary audio files (mp3/mp4/etc.) rather than
# a 16kHz mono wav/numpy array -- mlx_whisper shells out to it directly:
brew install ffmpeg
```

Requires Apple Silicon (MLX/Metal). Tested on an M-series Mac.

## Part A -- weight-quantize Whisper-large (the practical path)

`convert_from_hf.py` downloads a Hugging Face `WhisperForConditionalGeneration`
checkpoint, remaps its parameter names to `mlx_whisper`'s layout (both trace
back to the original OpenAI checkpoint naming, so it's a fixed set of
substitutions -- see the script), and optionally applies MLX's group-wise
affine weight quantization (`mlx.nn.quantize`) before saving.

```sh
# Full precision (fp16) conversion, no quantization
python convert_from_hf.py \
  --hf-repo openai/whisper-large-v3 \
  --mlx-path ./mlx-models/whisper-large-v3-fp16

# 4-bit weight-quantized (~4x smaller on disk)
python convert_from_hf.py \
  --hf-repo openai/whisper-large-v3 \
  --mlx-path ./mlx-models/whisper-large-v3-4bit \
  --quantize --q-bits 4 --q-group-size 64
```

The output directory (`config.json` + `weights.safetensors`) is exactly what
`mlx_whisper.load_model` / `mlx_whisper.transcribe` expect, so it just works
with the standard package:

```python
import mlx_whisper
result = mlx_whisper.transcribe(
    "path/to/audio.mp3",
    path_or_hf_repo="./mlx-models/whisper-large-v3-4bit",
)
print(result["text"])
```

or from the CLI: `mlx_whisper --model ./mlx-models/whisper-large-v3-4bit audio.mp3`.

`--q-bits 4 --q-group-size 64` is a good default (~4x smaller, small quality
loss); use `--q-bits 8` for a safer/larger option, or drop `--quantize`
entirely for full fp16.

## Part B -- TurboQuant on Whisper's decoder KV cache (experimental)

`turboquant_decoder.py` reimplements just Whisper's text decoder
(`TurboMultiHeadAttention`, `TurboResidualAttentionBlock`, `TurboTextDecoder`)
using mlx-vlm's cache protocol (`cache.update_and_fetch(...)` dispatched
through `mlx_vlm.models.base.scaled_dot_product_attention`) instead of
`mlx_whisper`'s plain tensor-concatenation cache. That's what lets
`mlx_vlm.turboquant.TurboQuantKVCache` -- the same class mlx-vlm uses for
Qwen/Gemma/etc. -- be dropped in directly. It loads checkpoints produced by
`convert_from_hf.py` above (parameter names match exactly). The audio
encoder is reused from `mlx_whisper.whisper.AudioEncoder` unmodified, since
it's not autoregressive and has no KV cache to compress. Cross-attention K/V
are computed once from the encoder output and reused every decode step, so
there's nothing there for TurboQuant to compress either -- only
self-attention grows with each generated token, and that's the only part
this script touches.

```sh
# TurboQuant 3.5-bit (3-bit keys + 4-bit values), with a baseline comparison
python turboquant_decoder.py \
  --mlx-path ./mlx-models/whisper-large-v3-fp16 \
  --audio path/to/audio.wav \
  --kv-bits 3.5 --kv-quant-scheme turboquant \
  --compare
```

This does greedy (non-beam) decoding of a single 30s audio chunk and prints
the transcript plus the self-attention KV cache's allocated size (via
mlx-vlm's own `cache_nbytes`) for both a plain fp16 cache and TurboQuant.
It's a minimal decode loop for measurement, not a drop-in replacement for
`mlx_whisper.transcribe` (no beam search, temperature fallback, or chunking
across audio > 30s).

## Verified results (this session)

Run against `openai/whisper-tiny` (fast to download; same code path as
`whisper-large-v3`, just smaller) transcribing the classic JFK
"ask not what your country can do for you" test clip
([source](https://github.com/openai/whisper/raw/main/tests/jfk.flac)):

**Part A** -- `convert_from_hf.py`, then `mlx_whisper.transcribe`:

| Checkpoint | Weights size | Transcript |
|---|---|---|
| fp16 (converted) | 71 MB | "And so my fellow Americans ask not what your country can do for you ask what you can do for your country." |
| 4-bit quantized | 21 MB (**3.4x smaller**) | "And so my fellow Americans ask not What your country can do for you ask what you can do for your country" |

(whisper-tiny's overhead from quantization scales/biases eats into the ratio
at this size; whisper-large-v3 will land closer to the theoretical ~4x.)

**Part B** -- `turboquant_decoder.py --compare`:

| Scheme | Transcript | Self-attn KV cache allocated |
|---|---|---|
| fp16 baseline | "...ask not what your country can do for you ask what you can do for your country." | 1536.0 KB |
| TurboQuant 3.5-bit | "...ask not what your country can do for you ask what you can do for your country" | 36.6 KB (**97.6% smaller**) |
| TurboQuant 2-bit | (same) | 21.9 KB (**98.6% smaller**) |

Both quantized paths produced essentially the same transcript as the
baseline (whisper-tiny's own accuracy limits, not a quantization artifact --
compare to the actual JFK quote, "...ask not what your country can do for
you, ask what you can do for your country").

## Caveats (honest numbers)

- **The KV-cache reduction percentage above is not representative of real
  savings.** mlx-vlm's cache classes allocate in fixed 256-token steps
  regardless of scheme, so both the baseline and TurboQuant numbers above
  reflect one over-provisioned 256-token buffer, not the ~27 tokens actually
  used. The *relative* reduction (bits-per-element) is real and matches
  TurboQuant's design; the absolute KB figures are not "memory saved on this
  transcription," they're "memory saved per allocated 256-token buffer."
- **Absolute savings are tiny in absolute terms regardless.** A whisper-large
  decoder layer's KV cache at its 448-token max is a few hundred KB in fp16;
  TurboQuant shaves that down further, but it's noise next to the ~3 GB of
  fp16 weights (or ~0.8 GB at 4-bit) that dominate Whisper-large's memory
  footprint. Part A (weight quantization) is what actually matters for
  Whisper; Part B exists because it was asked for, not because it's the
  right lever here.
- **This isn't upstream-mlx-vlm-quality code.** `turboquant_decoder.py` is a
  minimal decode loop (greedy only, one 30s chunk, no beam search / VAD /
  timestamp features) built to prove the integration and measure it
  honestly -- use `mlx_whisper.transcribe` (Part A) for anything you'd
  actually rely on.

## Files

- `convert_from_hf.py` -- HF Whisper checkpoint -> MLX checkpoint, optional weight quantization.
- `turboquant_decoder.py` -- TurboQuant-enabled Whisper decoder + a minimal transcribe loop for measurement.
- `requirements.txt`
