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

## Files

- `turboquant_src/` -- the actual library (cloned from
  `huggingface.co/vivekvar/turboquant`, `pip install -e`'d), untouched.
- `run_one_llm_config.py` -- loads `google/gemma-4-E4B-it`, builds either a
  `DynamicCache` or a (patched, see point 3 above) TurboQuant cache, runs
  both prompts, writes one config's metrics to JSON. Invoked as a
  subprocess by `evaluate_gemma4.py`.
- `evaluate_gemma4.py` -- orchestrates both configs and writes `report.json`
  with the full per-prompt comparison.
- `report.json` -- full results from the run described above.
- `requirements.txt`
