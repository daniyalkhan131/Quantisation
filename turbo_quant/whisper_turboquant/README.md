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

## Part C -- evaluating all four configs on whisper-large-v3

`evaluate.py` runs the full 2x2 matrix -- weights (fp16 vs 4-bit) x KV cache
(none vs TurboQuant) -- against every `.wav` in a data directory, and writes
one JSON report with timing, memory, and quality numbers. Each config runs
in its own subprocess (`run_one_config.py`) so peak-memory numbers reflect
one model in isolation, not a running max across four models loaded
back-to-back. All four configs go through the *same* minimal decode loop
from `turboquant_decoder.py` (not `mlx_whisper.transcribe`'s production
pipeline), so weight precision and KV scheme are the only two things
changing between runs -- see [Part B](#part-b----turboquant-on-whispers-decoder-kv-cache-experimental)
for why `mlx_whisper.transcribe` isn't used here.

```sh
python convert_from_hf.py --hf-repo openai/whisper-large-v3 \
  --mlx-path ./mlx-models/whisper-large-v3-fp16
python convert_from_hf.py --hf-repo openai/whisper-large-v3 \
  --mlx-path ./mlx-models/whisper-large-v3-4bit --quantize --q-bits 4 --q-group-size 64

python evaluate.py \
  --fp16-path ./mlx-models/whisper-large-v3-fp16 \
  --int4-path ./mlx-models/whisper-large-v3-4bit \
  --data-dir /Users/daniyal/Documents/projects/Quantisation/turbo_quant/data \
  --kv-bits 3.5 \
  --output report.json
```

Each per-file entry in `report.json` includes: inference time, real-time
factor, tokens/sec, peak MLX device memory during inference, self-attention
KV cache bytes allocated, the transcript, and (for every config except the
baseline) word error rate against the fp16-baseline transcript for that same
file -- there's no ground-truth transcript for this audio, so the baseline
stands in as the reference for how much quantization moves the output, not
absolute correctness. A `summary` block aggregates all of that per config,
plus size/memory/KV-cache reduction percentages against the baseline.

### Results (this session, real `openai/whisper-large-v3`, M-series Mac)

2 files from `data/` (25.6s and 17.3s of real speech), full detail in
[`report.json`](report.json):

| Config | Backend | Checkpoint | Avg RTF | Avg tok/s | Peak RSS | Avg KV cache allocated | Avg WER vs MLX fp16 |
|---|---|---|---|---|---|---|---|
| fp16 baseline | MLX (GPU) | 2940 MB | 8.28x | 34.2 | 3424 MB | 40.0 MB | 0 (reference) |
| 4-bit weights only | MLX (GPU) | 837 MB (**-71.5%**) | 14.87x (**+79.5%**) | 61.0 | 1344 MB (**-60.7%**) | 40.0 MB (unchanged) | 0.112 |
| TurboQuant 3.5-bit KV only | MLX (GPU) | 2940 MB | 8.78x (+6.0%) | 36.3 | 3430 MB (~unchanged) | 3.3 MB (**-91.8%**) | 0.006 |
| 4-bit weights + TurboQuant KV | MLX (GPU) | 837 MB (**-71.5%**) | 13.12x (**+58.4%**) | 54.6 | 1346 MB (**-60.7%**) | 3.3 MB (**-91.7%**) | 0.112 |
| faster-whisper, float32 | CTranslate2 (**CPU**) | 2948 MB | 2.48x (**-70.0%**) | 10.0 | 7175 MB (**+109.6%**) | n/a | 0.0 |
| faster-whisper, int8 | CTranslate2 (**CPU**) | 2948 MB | 2.63x (**-68.3%**) | 10.7 | 3461 MB (+1.1%) | n/a | 0.093 |

(Run-to-run note: the MLX numbers above are from a slightly later run than
the first table this README shipped with -- single-run timings vary a few
percent between invocations, e.g. this run's fp16 baseline came in at 8.28x
RTF vs an earlier run's 7.67x. Treat single-digit-percent differences as
noise; the qualitative story below is stable across runs.)

Takeaways:

- **Weight quantization is the lever that matters on MLX/GPU.** 4-bit
  weights alone cut checkpoint size and process memory by ~60-70%, and
  *increased* throughput by ~80% (34.2 -> 61.0 tok/s) -- 4-bit matmuls on
  Apple Silicon are memory-bandwidth bound, so a smaller checkpoint decodes
  faster too, not just smaller. Quality cost: ~11% WER against the fp16
  transcript, concentrated entirely in the harder of the two clips (the
  easier clip was byte-identical to baseline).
- **TurboQuant KV cache barely moves speed or memory for Whisper on MLX**,
  exactly as flagged in Part B: cache allocation is ~40MB either way at
  these sequence lengths (well under whisper-large's 448-token cap), noise
  against a multi-GB footprint dominated by weights and encoder activations.
  Its ~92% *relative* KV reduction is real and reproducible, it's just
  compressing a small number to begin with. Quality cost was lower than
  weight quantization (~0.6% WER) and cost no throughput (if anything this
  run shows a slight increase, likely noise -- see Part B's caveat that
  TurboQuant's kernels are tuned for far longer contexts than this).
- **Combining both** gets you weight quantization's size/speed win with
  TurboQuant's KV reduction stacked on top, at no additional quality cost
  beyond what 4-bit weights alone already cost (WER identical to
  weight-4bit-only on both files) -- the two are compressing independent
  things (weights vs. activations), so their effects don't compound
  negatively here.
- **faster-whisper (CTranslate2) is a CPU-only run on this machine** --
  there's no Metal/GPU backend for CTranslate2 on Apple Silicon, so both its
  configs are ~3.5-4x slower in real-time factor than *every* MLX config,
  including MLX's own unquantized fp16. This is not "MLX beats
  faster-whisper" -- it's "GPU beats CPU," a hardware-backend difference,
  not a quantization-quality one. See [Part D](#part-d----faster-whisper-ctranslate2-as-an-external-reference-point)
  for the full picture, including a genuinely useful cross-check this
  comparison turned up: faster-whisper's float32 output was **byte-identical**
  to MLX's fp16 output on both files (WER 0.0), independently corroborating
  that neither conversion pipeline introduced a correctness bug.

## Part D -- faster-whisper (CTranslate2) as an external reference point

`run_faster_whisper_config.py` runs the same audio through
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (the most widely
used optimized Whisper runtime, built on CTranslate2) for comparison, driven
by `evaluate.py --include-faster-whisper`. It downloads Systran's official
CTranslate2 conversion of large-v3 (`Systran/faster-whisper-large-v3`) rather
than converting anything itself -- there's no MLX/mlx-vlm code involved here
at all, it's purely an external reference point.

**Hardware caveat, stated plainly:** CTranslate2 has no Metal backend, so on
Apple Silicon faster-whisper always runs on CPU. Every MLX config in this
report runs on the GPU via Metal. The RTF/tok-s gap between faster-whisper
and any MLX config here is mostly a CPU-vs-GPU story on this machine, not a
statement about which quantization approach is better. If you run this on
an x86 box with a CUDA GPU, faster-whisper would use that GPU and the
comparison would look completely different.

**What CTranslate2's `compute_type` actually changes:** unlike
`convert_from_hf.py --quantize` (which writes an actually smaller
`weights.safetensors` to disk), CTranslate2 quantizes *at load time* from
the same downloaded checkpoint -- `float32` and `int8` here read identical
files (2948 MB either way; see the table above) and only differ in their
in-memory representation and compute path. So "checkpoint size" isn't a
fair axis to compare CTranslate2's `int8` against MLX's `--quantize`; peak
RSS is the meaningful number instead, and there int8 does cut memory
substantially (7175 MB -> 3461 MB, about half) without touching disk size.

```sh
pip install faster-whisper

# One config directly:
python run_faster_whisper_config.py --config-name faster_whisper_int8 \
  --compute-type int8 --data-dir ../data --output fw_int8.json

# Or as part of the full report:
python evaluate.py \
  --fp16-path ./mlx-models/whisper-large-v3-fp16 \
  --int4-path ./mlx-models/whisper-large-v3-4bit \
  --data-dir ../data \
  --include-faster-whisper --faster-whisper-compute-types float32,int8 \
  --output report.json
```

Per-file WER against the MLX fp16 baseline (from `report.json`) shows the
same content-dependent pattern MLX's own 4-bit weights showed in Part C,
now cross-validated on a completely independent implementation:

| File | faster-whisper float32 | faster-whisper int8 |
|---|---|---|
| en_0000_0078.wav (easier clip) | WER 0.0 | WER 0.024 |
| en_0001_0055.wav (harder clip) | WER 0.0 | WER 0.163 |

Two things worth taking away from this table specifically: (1)
faster-whisper's unquantized float32 run reproduces MLX's fp16 transcript
exactly on both files, which is a real cross-implementation correctness
check, not something either script asserts about itself; (2) int8
quantization degrades the harder clip roughly 7x more than the easier one,
same lopsided pattern as MLX's 4-bit weights (Part C) -- quantization
sensitivity tracking audio difficulty looks like a property of Whisper
quantization generally here, not an artifact of one specific
implementation.

## Verified results, smaller model (`whisper-tiny`, earlier smoke test)

Before running the full evaluation above on whisper-large-v3, the same code
paths were validated on `openai/whisper-tiny` (fast to download) transcribing
the classic JFK "ask not what your country can do for you" test clip
([source](https://github.com/openai/whisper/raw/main/tests/jfk.flac)), to
catch integration bugs cheaply before spending a 3GB download and a long-form
model's compute on them:

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

- **The KV-cache byte counts are allocated capacity, not tokens actually
  used.** mlx-vlm's cache classes allocate in fixed 256-token steps
  regardless of scheme; the large-v3 baseline's 40MB is exactly
  `32 layers x 2 (k+v) x 20 heads x 256-token step x 64 head_dim x 2 bytes`,
  not the ~25-40 tokens each clip actually decoded to. The *relative*
  reduction (~92%, matching TurboQuant's bits-per-dimension design) is real
  and reproducible; the absolute MB figures are "memory per allocated
  256-token buffer," not "memory saved on this specific transcription."
  Real long-form use (dictation, meeting transcripts spanning many 30s
  chunks with a carried-over cache) would show TurboQuant mattering more
  than it does here.
- **`evaluate.py`/`turboquant_decoder.py` are not `mlx_whisper.transcribe`.**
  They share one minimal greedy decode loop (built for a controlled A/B/C/D
  comparison) with no beam search, temperature fallback, VAD, or chunking
  past 30s -- use `mlx_whisper.transcribe` (Part A) for anything you'd
  actually rely on for transcription quality. The reported tok/s and RTF
  numbers are specific to this minimal loop and will differ from
  `mlx_whisper.transcribe`'s (the Part A sanity check above ran noticeably
  faster wall-clock for the same files, since it has less per-step Python
  overhead and no memory reset between calls).
- **WER here is "distance from the fp16 transcript," not accuracy.** There's
  no ground-truth transcript for `data/`'s audio, so it measures how much a
  quantized config's output diverges from the unquantized run on the same
  audio -- a proxy for quality drift, not a claim about correctness.

## Files

- `convert_from_hf.py` -- HF Whisper checkpoint -> MLX checkpoint, optional weight quantization.
- `turboquant_decoder.py` -- TurboQuant-enabled Whisper decoder + a minimal transcribe loop for measurement.
- `run_one_config.py` -- runs one MLX (weights x KV-scheme) config over a data dir; invoked as a subprocess by `evaluate.py`.
- `run_faster_whisper_config.py` -- runs one faster-whisper (CTranslate2) config over a data dir; invoked as a subprocess by `evaluate.py --include-faster-whisper`.
- `evaluate.py` -- orchestrates the MLX 2x2 matrix (and optionally faster-whisper) and writes `report.json`.
- `report.json` -- full per-file results from the run described above, across all 6 configs.
- `requirements.txt`
