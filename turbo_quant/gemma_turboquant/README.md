# TurboQuant KV cache on Gemma-4-E4B: with vs. without

This compares [`vivekvar/turboquant`](https://huggingface.co/vivekvar/turboquant)'s
`TurboQuantCache` (a from-scratch, community implementation of the
[TurboQuant](https://arxiv.org/abs/2504.19874) KV-cache compression paper for
plain HuggingFace Transformers) against a normal `DynamicCache`, on
`google/gemma-4-E4B-it` -- a genuine LLM comparison, unlike the earlier
Whisper work in `../whisper_turboquant/` which used mlx-vlm's own (different,
unrelated) TurboQuant implementation.

## Read this first: what actually had to be fixed to get this running

The exact snippet in the request looked like it should just work:
```python
from turboquant import TurboQuantCache
cache = TurboQuantCache(model.config, nbits=4, skip_layers=skip)
```
It doesn't, out of the box, for two independent reasons, plus a third bug
hit once actually running it against Gemma-4. All three are documented here
because they're the actual content of "analyse and tell":

1. **`pip install turboquant` is a different, incompatible package.** PyPI's
   `turboquant` 0.2.0 has a completely different API
   (`TurboQuantCache(bits=3)`, no `config`/`nbits`/`skip_layers`/
   `calibrate_skip_layers` at all). The snippet only matches the code hosted
   at `huggingface.co/vivekvar/turboquant` (cloned here as `turboquant_src/`
   and `pip install -e`d from there instead). Same project name, two
   unrelated implementations -- check which one you actually installed
   before assuming a README example applies.

2. **Gemma-4 checkpoints are multimodal, and the "obvious" fix is wrong.**
   `google/gemma-4-E4B-it` loads as `Gemma4ForConditionalGeneration`
   (vision + audio + text towers in one 16GB `model.safetensors`), not a
   plain causal LM. transformers *does* expose a separate
   `Gemma4ForCausalLM` class that looks like the right text-only fix -- it
   isn't. `AutoModelForCausalLM.from_pretrained(...)` actually resolves
   `model_type="gemma4"` back to `Gemma4ForConditionalGeneration`
   (`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["gemma4"]`), because the checkpoint
   stores decoder weights nested as `model.language_model.layers.*`, while
   `Gemma4ForCausalLM` expects flat `model.layers.*`. Loading
   `Gemma4ForCausalLM` directly against this checkpoint doesn't error --
   it silently reports every decoder weight as "MISSING" and
   random-initializes the entire language model, which then generates pure
   token-salad (`" Spe Spe Spe সমূহ mücade ..."`) with an exit code of 0.
   The fix: just use `AutoModelForCausalLM` (closer to the original
   snippet than my first "obvious" fix was) and call `.generate()` on the
   full omni model with text-only input -- no `pixel_values`/audio, so it
   behaves as a plain causal LM, and every weight matches.

3. **`TurboQuantCache.__init__` assumes one global `head_dim`; Gemma-4
   doesn't have one.** Gemma-4 uses transformers' new
   heterogeneous-per-layer-config system, and its `head_dim` genuinely
   varies (confirmed empirically: `{256, 512}` across the 42 layers of
   `google/gemma-4-E4B-it`). Reading `config.head_dim` now raises
   `AmbiguousGlobalPerLayerAttributeError` by design, specifically to catch
   code like this that assumes a uniform value. `run_one_llm_config.py`
   works around it with `build_gemma4_turboquant_cache()`, which
   reimplements just `TurboQuantCache.__init__`'s per-layer construction
   loop reading `text_config.per_layer_config[i].head_dim` -- the actual
   quantization code (`TurboQuantLayer` / `TurboQuantizer` / codebook /
   packing) from the library is untouched.

4. **`generate()` crashes after early EOS-stopping.** transformers' newer
   generation loop calls `cache.crop()` to undo any tokens generated past a
   stopping point once the model naturally hits EOS. `QuantizedLayer`
   (`TurboQuantLayer`'s base class) doesn't override `crop()`, and inherits
   `DynamicLayer`'s, which assumes `self.keys` is a populated tensor --
   crashes with `TypeError: 'NoneType' object is not subscriptable` for a
   quantized layer whose residual window is in a state `crop()` doesn't
   expect. Worked around by forcing `min_new_tokens == max_new_tokens`, so
   generation never stops early and that cleanup path never runs. This is
   likely a general limitation of `QuantizedLayer`-based caches with modern
   `generate()`, not specific to Gemma-4.

None of these are "the library is broken" -- (1) is a naming collision to
be aware of, (2) is a Gemma-4-specific loading quirk unrelated to
TurboQuant, and (3)/(4) are real gaps in an early-stage ("Alpha") library
being pointed at a brand-new architecture it was never tested against
(the library's own README only claims testing on Qwen2.5, Llama-3.1/3.3,
Gemma-2-9B, and Phi-4 -- all pre-Gemma-4, all on CUDA).

## Setup

```sh
cd gemma_turboquant
python3 -m venv .venv && source .venv/bin/activate  # or reuse ../.venv

pip install torch transformers scipy huggingface_hub

git clone https://huggingface.co/vivekvar/turboquant turboquant_src
pip install -e ./turboquant_src
```

Requires ~16GB of free disk for the checkpoint download and enough unified
memory to hold it (tested on a 24GB Apple Silicon Mac via PyTorch's MPS
backend; no CUDA on this machine, so this is not the library's tested/tuned
target, see caveats).

## Running it

```sh
python evaluate_gemma4.py --nbits 4 --max-new-tokens 300 --output report.json
```

This runs `run_one_llm_config.py` twice (baseline, then turboquant), each in
its own subprocess for isolated peak-memory accounting, on two prompts: a
short factual question, and a long (~4300-token) synthetic context followed
by an analysis request. The long prompt exists specifically because
`TurboQuantLayer` keeps the most recent `residual_length` tokens (default
128) in full precision and only quantizes older ones -- a short prompt with
a short generation never leaves that window, making "with vs. without"
trivially identical.

## Results (this session, real `google/gemma-4-E4B-it`, M-series Mac, MPS)

| | Baseline (DynamicCache) | TurboQuant (4-bit) |
|---|---|---|
| Model load time | 22.1s | 20.4s |
| Model load peak device memory | 15154 MB | 15154 MB (identical -- same weights) |
| `calibrate_skip_layers()` time | -- | 21.9s (one extra forward pass) |
| Layers skipped by calibration | -- | **none** (empty set -- see below) |

Per-prompt (`report.json` has full detail):

| Prompt | Baseline tok/s | TurboQuant tok/s | Speed change | Baseline KV bytes | TurboQuant KV bytes | KV reduction | First divergence | WER vs baseline |
|---|---|---|---|---|---|---|---|---|
| short (28 prompt tokens) | 5.78 | 10.55 | +82.5%* | 18.8 MB | 6.6 MB | 64.9% | token 9 | 0.807* |
| long_context (4327 prompt tokens) | 6.38 | 3.47 | **-45.6%** | 265.3 MB | 69.0 MB | **74.0%** | token 30 | 0.737 |

*See caveats -- the "short" prompt's numbers are a test-harness artifact,
not a real quantization signal.

### What actually happened, prompt by prompt

**short prompt:** both configs answer "What is the capital of Paris..."
correctly and *identically* for the first ~9 tokens, then, because this
script forces `min_new_tokens=300` (to dodge the crop() bug above) on a
question that naturally wants a two-sentence answer, **both configs
degenerate** once they've actually answered: baseline loops on
`"If you'd like to try a different question, feel free to ask!"` about 15
times, TurboQuant collapses into ~270 tokens of blank lines. Different
degenerate failure modes, but neither reflects TurboQuant quality -- this is
an artifact of forcing generation length past the natural stopping point,
and the resulting 82.5%-faster / 80.7%-WER numbers for this prompt shouldn't
be read as "TurboQuant is faster with worse quality." (Generating blank
lines is cheaper than generating a repeated phrase, which likely explains
the speed delta here -- not the cache scheme.)

**long_context prompt (the fair test):** both configs produce a full,
coherent, on-topic 300-token analysis with no degeneration in either.
They diverge starting at token 30 into two different but equally
reasonable framings of the same pattern -- baseline groups by
"Observation Cycle" first, TurboQuant by "System Type" first -- and stay
fluent and on-topic to the end. This is the real finding: a small
quantization-induced numeric perturbation flipped one greedy-decoding
argmax decision early on, and because greedy decoding is a hard decision
boundary (not a smooth degradation), everything downstream follows a
different but equally valid path. **This is the same phenomenon
`../whisper_turboquant/README.md` found on Whisper** -- quantization noise
occasionally flips an argmax choice; the two resulting continuations are
different but not obviously worse.

On this same fair prompt, TurboQuant is **46% slower**, despite compressing
the KV cache by 74%. This is very likely because `vivekvar/turboquant` is a
pure-PyTorch reference implementation with no fused kernels (unlike
mlx-vlm's TurboQuant, which ships custom Metal kernels specifically to avoid
full dequantization at attention time -- see `../whisper_turboquant/`'s
`turboquant.py`). Every decode step here pays real Python/PyTorch overhead
to quantize new tokens and dequantize the whole compressed history, on a
backend (MPS) the library was never tuned for (its own benchmarks are
CUDA-only). At the ~4600-token sequence length this prompt reaches, that
per-step overhead outweighs any memory-bandwidth savings a fused
implementation might realize.

### The calibration result is itself informative

`TurboQuantCache.calibrate_skip_layers()` found **zero** outlier layers for
Gemma-4-E4B-it (empty skip set), overriding the library's own hardcoded
default of skipping layer 0 (`skip_layers=None` -> `{0}`) when calibration
isn't used. Layer 0's "outlier key norm" problem, which the library was
designed around based on the models it was tested on, doesn't appear to
apply to this specific model -- worth knowing if you reuse this library on
other new architectures: don't assume the hardcoded default is either
necessary or sufficient without running calibration.

## Caveats (honest numbers)

- **Peak "device memory" (~15.2 GB) is dominated by the unused vision and
  audio towers**, not the ~4B-parameter text decoder actually being
  exercised. The whole 16GB omni checkpoint loads into memory because
  `AutoModelForCausalLM` resolves to the full `Gemma4ForConditionalGeneration`
  class (see point 2 above) -- there's no supported way to load only the
  text submodule's weights from this checkpoint layout. This means the KV
  cache's 74% reduction (265 MB -> 69 MB) is invisible against a 15+ GB
  total footprint; the story here is almost identical to Whisper's ("KV
  compression is real but small next to what's actually resident in
  memory"), just more extreme because most of that memory isn't even the
  relevant model.
- **This is a CPU-adjacent backend for this library.** MPS is not CUDA;
  `vivekvar/turboquant`'s own benchmarks, version requirements, and testing
  are all CUDA-specific. The -46% speed result on the long prompt should be
  read as "this reference implementation, unoptimized for MPS, has real
  per-step overhead" rather than "TurboQuant is inherently slower" --
  mlx-vlm's fused-Metal-kernel TurboQuant (`../whisper_turboquant/`) did NOT
  show this regression on Apple Silicon.
- **N=1 runs, N=2 prompts.** No repeated trials to average out timing noise,
  and the "short" prompt's numbers are explicitly flagged above as not
  meaningful due to the `min_new_tokens` forcing artifact. Treat this as a
  qualitative demonstration with two genuinely measured data points, not a
  statistically tight benchmark.
- **WER-vs-baseline here measures divergence, not error.** As with Whisper,
  there's no ground truth; a high WER after an early greedy-decoding fork
  reflects "generated different words," which is expected once one token
  differs -- not "generated wrong words." Read the actual text (in
  `report.json` or the tables above) rather than the WER percentage alone.

## Part 2 -- the same model on mlx-vlm: how much faster, how much less memory

Everything above uses PyTorch + transformers on MPS. This section runs the
exact same model (`google/gemma-4-E4B-it`) and the exact same two prompts
through **mlx-vlm** instead -- both its plain weights (fp16, 4-bit) and its
own *real*, Metal-kernel-backed TurboQuant (`mlx_vlm.turboquant`, a
completely different, production implementation from `vivekvar/turboquant`
above) -- to answer "how much does the MLX backend actually buy you."

```sh
python -m mlx_vlm convert --hf-path google/gemma-4-E4B-it \
  --mlx-path ./mlx-models/gemma-4-E4B-it-fp16 --dtype float16
python -m mlx_vlm convert --hf-path google/gemma-4-E4B-it \
  --mlx-path ./mlx-models/gemma-4-E4B-it-4bit -q --q-bits 4 --q-group-size 64

python evaluate_mlx_backend.py --kv-bits 3.5 --max-tokens 300 \
  --pytorch-report report.json --output mlx_report.json
```

One gotcha that mattered: mlx-vlm defaults `quantized_kv_start=5000` --
TurboQuant only compresses cache entries *past* the 5000th cached token, so
with either of our prompts (70 and 4627 total tokens) it silently never
activates unless you explicitly override it to something like `0`. The
first run of this comparison hit exactly that: TurboQuant's output came
back byte-identical to baseline, which looked like a great result but
actually meant "we tested baseline against baseline." `run_one_mlx_config.py`
passes `--quantized-kv-start 0` for the TurboQuant configs to actually
exercise it.

### Results (this session, same hardware, same two prompts)

**Model load:**

| | PyTorch/MPS | mlx-vlm | mlx-vlm speedup |
|---|---|---|---|
| Load time | 22.1s | 6.4s | **3.4x** |
| Peak device memory | 15154 MB | 15146 MB | ~same (same fp16 weights) |

**Generation throughput, same fp16 weights, no KV quantization** -- this
isolates the backend/kernel difference alone, nothing to do with
quantization:

| Prompt | PyTorch/MPS tok/s | mlx-vlm tok/s | Speedup |
|---|---|---|---|
| short | 5.78 | 12.99 | **2.25x** |
| long_context | 6.38 | 12.06 | **1.89x** |

**Adding mlx-vlm's own 4-bit weight quantization on top** -- now both
backend and weight-quantization gains combine:

| Prompt | PyTorch/MPS fp16 tok/s | mlx-vlm 4-bit tok/s | Combined speedup |
|---|---|---|---|
| short | 5.78 | 42.85 | **7.4x** |
| long_context | 6.38 | 39.60 | **6.2x** |

**TurboQuant implementation, apples to apples** (both actually compressing
the KV cache, `nbits=4`/`kv-bits 3.5`): mlx-vlm's fused-kernel version vs
`vivekvar/turboquant`'s pure-PyTorch reference version from Part 1 --

| Prompt | vivekvar/turboquant tok/s (PyTorch/MPS) | mlx-vlm TurboQuant tok/s | Speedup |
|---|---|---|---|
| short (~370 total tokens) | 10.55 | 13.10 | 1.24x |
| long_context (~4630 total tokens) | 3.47 | 11.81 | **3.4x** |

**This speedup gap widening with context length is the single most telling
number in this whole comparison.** It's direct empirical confirmation of
`notes.txt`'s explanation: the reference implementation re-dequantizes its
*entire* compressed history on every decode step (no incremental caching),
so its cost grows with sequence length; mlx-vlm's fused Metal kernels
compute attention scores directly from the packed representation and don't
pay that cost at all. At a short context the two are close (1.24x); at
~4600 tokens the naive implementation has fallen to a third of mlx-vlm's
speed. Stack 4-bit weights on top of mlx-vlm's TurboQuant and the gap
against the naive PyTorch reference implementation reaches **11x** on the
long prompt.

**Memory:** peak device memory tracks weights almost entirely at this
model size (15.1 GB fp16 -> 4.9 GB at 4-bit, a **67.6% reduction** either
with or without TurboQuant) -- KV cache is such a small fraction of total
footprint here (as in Part 1 and in `../whisper_turboquant/`) that
TurboQuant's compression doesn't move the peak-memory needle either way.
Load-time peak memory is identical between TurboQuant and non-TurboQuant at
the same weight precision, for the same reason.

**Quality:** with TurboQuant genuinely active (post-fix), mlx-vlm's output
diverges from its own fp16 baseline about as much as `vivekvar/turboquant`
did from its baseline (both land in the 0.6-0.8 WER-vs-self-baseline range
once quantization is actually exercised) -- same "small perturbation flips
an early greedy-decoding choice, then the rest of the completion follows a
different but not obviously worse path" story as everywhere else in this
project. Full per-prompt text is in `mlx_report.json`.

### Bottom line

For this specific model on this specific Mac: mlx-vlm's backend alone is
worth ~2x over PyTorch/MPS at the same precision; mlx-vlm's 4-bit weight
quantization is worth another ~3x on top of that (~6-7x combined); and
mlx-vlm's *implementation quality* of TurboQuant (not the algorithm --
the same algorithm, just with fused kernels) is worth up to ~11x over a
naive-but-correct reference implementation at realistic context lengths.
The single biggest lesson from this whole `gemma_turboquant/` folder: for
KV-cache quantization schemes specifically, the reference implementation's
engineering quality (does it avoid full re-dequantization every step?)
matters as much as which paper it implements.

## Files

- `turboquant_src/` -- the actual library (cloned from
  `huggingface.co/vivekvar/turboquant`, `pip install -e`'d), untouched.
- `run_one_llm_config.py` -- loads `google/gemma-4-E4B-it`, builds either a
  `DynamicCache` or a (patched, see point 3 above) TurboQuant cache, runs
  both prompts, writes one config's metrics to JSON. Invoked as a
  subprocess by `evaluate_gemma4.py`.
- `evaluate_gemma4.py` -- orchestrates both configs and writes `report.json`
  with the full per-prompt comparison.
- `report.json` -- full results from the PyTorch/transformers run (Part 1).
- `run_one_mlx_config.py` -- loads `google/gemma-4-E4B-it` via mlx-vlm,
  optionally with `kv_bits`/`kv_quant_scheme`/`quantized_kv_start` set for
  mlx-vlm's real TurboQuant, runs both prompts, writes one config's metrics
  to JSON. Invoked as a subprocess by `evaluate_mlx_backend.py`.
- `evaluate_mlx_backend.py` -- orchestrates the mlx-vlm 2x2 matrix and
  cross-compares against `report.json`'s PyTorch numbers, writing
  `mlx_report.json`.
- `mlx_report.json` -- full results from the mlx-vlm run (Part 2), including
  the `cross_backend_vs_pytorch` section.
- `mlx-models/` -- converted MLX checkpoints (fp16, 4-bit); gitignored, not
  committed (regenerate with the `mlx_vlm convert` commands above).
- `notes.txt` -- plain-language notes on why the reference TurboQuant
  implementation slows down on long contexts.
- `requirements.txt`
