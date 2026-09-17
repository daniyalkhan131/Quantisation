"""Run one KV-cache config (plain DynamicCache vs vivekvar/turboquant's
TurboQuantCache) against google/gemma-4-E4B-it and dump timing/memory/output
metrics to a JSON file. Invoked as a subprocess by evaluate_gemma4.py so each
config gets a fresh model load and isolated peak-memory accounting.

Why google/gemma-4-E4B-it specifically: see ../README.md's "Read this first"
section -- short version: Gemma-4 checkpoints are multimodal
(Gemma4ForConditionalGeneration, with vision+audio towers). Loading with
AutoModelForCausalLM still resolves to Gemma4ForConditionalGeneration
(transformers' own MODEL_FOR_CAUSAL_LM_MAPPING_NAMES maps model_type "gemma4"
there, not to the separate Gemma4ForCausalLM class -- a first attempt using
Gemma4ForCausalLM directly loaded with EVERY decoder weight missing, since
that class expects flat `model.layers.*` keys while this checkpoint stores
them nested under `model.language_model.*`; silently random-initialized
the whole thing). We call .generate() on the full omni model with text-only
input (no pixel_values/audio) to get plain causal LM behavior, and pass
model.config through to TurboQuantCache, which resolves the nested
text_config itself via config.get_text_config().

Why the prompt is long: TurboQuantLayer (via transformers' QuantizedLayer)
keeps the most recent `residual_length` tokens (default 128) in full
precision and only quantizes older tokens once that window is exceeded. A
short prompt with a short generation never exceeds the window, so the
"with TurboQuant" and "without" runs would be trivially identical -- not a
real test. This script uses a long seed context specifically so quantization
actually gets exercised during generation.

Two more confirmed integration bugs beyond the two above (see README.md for
the full list): (1) TurboQuantCache.__init__ assumes one global head_dim for
the whole model, but Gemma-4-E4B-it genuinely varies head_dim per layer
({256, 512} across its 42 layers) -- worked around here by
build_gemma4_turboquant_cache(), which rebuilds the same per-layer
construction reading text_config.per_layer_config[i] instead. (2) generate()
crashes calling cache.crop() after early EOS-stopping, because
QuantizedLayer (TurboQuantLayer's base class) doesn't override crop() and
inherits DynamicLayer's, which assumes self.keys is a populated tensor --
worked around by forcing min_new_tokens == max_new_tokens so that cleanup
path is never entered.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from turboquant import TurboQuantCache

MODEL_ID = "google/gemma-4-E4B-it"

LONG_CONTEXT = (
    "You are a careful research assistant. Here is a long background passage. "
    + " ".join(
        f"Fact {i}: the {['moon', 'ocean', 'forest', 'desert', 'mountain', 'river'][i % 6]} "
        f"system labeled {i} has a measured baseline value of {(i * 37) % 100} units, "
        f"recorded during observation cycle {i % 12}."
        for i in range(140)
    )
    + " Based on all the facts above, summarize the pattern you observe across the observation cycles, "
    "then continue with a detailed step-by-step explanation of how you would verify it independently."
)

PROMPTS = {
    "short": "What is the capital of France, and what is it known for? Answer in two sentences.",
    "long_context": LONG_CONTEXT,
}


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024**2) if sys.platform == "darwin" else r / 1024


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def mem_snapshot_mb(device: str) -> float:
    if device == "mps":
        return torch.mps.current_allocated_memory() / (1024**2)
    if device == "cuda":
        return torch.cuda.memory_allocated() / (1024**2)
    return 0.0


class MemorySampler:
    """A StoppingCriteria that never stops generation, only samples device
    memory once per decode step so we get a peak across the whole run --
    MPS has no built-in reset_peak_memory_stats like CUDA does."""

    def __init__(self, device: str):
        self.device = device
        self.peak_mb = mem_snapshot_mb(device)

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        self.peak_mb = max(self.peak_mb, mem_snapshot_mb(self.device))
        return False


def cache_bytes(cache) -> int:
    """Sum actual tensor storage across all layers of an HF Cache object,
    covering both DynamicLayer (keys/values) and TurboQuantLayer
    (keys/values residual + _quantized_keys/_quantized_values tuples)."""
    total = 0
    seen = set()

    def add(t):
        nonlocal total
        if isinstance(t, torch.Tensor) and id(t) not in seen:
            seen.add(id(t))
            total += t.element_size() * t.nelement()

    for layer in cache.layers:
        for attr in ("keys", "values", "_quantized_keys", "_quantized_values"):
            val = getattr(layer, attr, None)
            if val is None:
                continue
            if isinstance(val, tuple):
                for t in val:
                    add(t)
            else:
                add(val)
    return total


def build_gemma4_turboquant_cache(model_config, nbits: int, device: str, skip_layers, base_seed: int = 42):
    """vivekvar/turboquant's TurboQuantCache.__init__ assumes one global
    head_dim for the whole model (`text_config.hidden_size // num_attention_heads`,
    or a single `text_config.head_dim`). Gemma-4-E4B-it genuinely varies
    head_dim per layer (confirmed empirically: {256, 512} across its 42
    layers), which trips transformers' AmbiguousGlobalPerLayerAttributeError
    on `config.head_dim`. This rebuilds the same per-layer construction
    TurboQuantCache.__init__ does, but reading head_dim from
    text_config.per_layer_config[i] instead of a single global value -- the
    actual quantization code (TurboQuantLayer/TurboQuantizer/codebook/packing)
    is untouched, only this cache-assembly glue is patched.
    """
    from transformers.cache_utils import Cache, DynamicLayer
    from turboquant.cache import TurboQuantLayer

    text_config = model_config.get_text_config(decoder=True)
    num_layers = text_config.num_hidden_layers
    if skip_layers is None:
        skip_layers = {0}

    layers = []
    for i in range(num_layers):
        if i in skip_layers:
            layers.append(DynamicLayer())
        else:
            head_dim = text_config.per_layer_config[i].head_dim
            layers.append(TurboQuantLayer(dim=head_dim, nbits=nbits, device=device, layer_seed=base_seed + i))
    return Cache(layers=layers)


def build_cache(config_name: str, model_config, device: str, nbits: float, calibrated_skip):
    if config_name == "baseline":
        return DynamicCache()
    return build_gemma4_turboquant_cache(model_config, int(nbits), device, calibrated_skip)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, choices=["baseline", "turboquant"])
    parser.add_argument("--nbits", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = pick_device()

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()
    load_time_s = time.perf_counter() - t0
    load_peak_mb = mem_snapshot_mb(device)

    calibrated_skip = None
    calibration_time_s = 0.0
    if args.config_name == "turboquant":
        t0 = time.perf_counter()
        calibrated_skip = TurboQuantCache.calibrate_skip_layers(model, tokenizer)
        calibration_time_s = time.perf_counter() - t0

    per_prompt = []
    for prompt_name, prompt_text in PROMPTS.items():
        # google/gemma-4-E4B-it is instruction-tuned; a bare prompt (no chat
        # template) makes it degenerate into echoing the instruction instead
        # of answering it.
        chat_out = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        chat_input_ids = (chat_out.input_ids if hasattr(chat_out, "input_ids") else chat_out).to(device)
        inputs = {"input_ids": chat_input_ids}
        prompt_tokens = chat_input_ids.shape[-1]

        cache = build_cache(args.config_name, model.config, device, args.nbits, calibrated_skip)
        sampler = MemorySampler(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.max_new_tokens,  # see module docstring: avoids a
                # transformers bug where generate()'s post-EOS cache.crop() cleanup
                # crashes on QuantizedLayer-based caches (self.keys is None there)
                past_key_values=cache,
                do_sample=False,
                stopping_criteria=[sampler],
            )
        if device == "mps":
            torch.mps.synchronize()
        gen_time_s = time.perf_counter() - t0

        new_tokens = output.shape[-1] - prompt_tokens
        generated_text = tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True)

        per_prompt.append(
            {
                "prompt_name": prompt_name,
                "prompt_tokens": prompt_tokens,
                "new_tokens": new_tokens,
                "generation_time_s": round(gen_time_s, 4),
                "tokens_per_second": round(new_tokens / gen_time_s, 2) if gen_time_s > 0 else None,
                "generation_peak_device_memory_mb": round(sampler.peak_mb, 2),
                "final_cache_bytes": cache_bytes(cache),
                "generated_text": generated_text,
                "generated_token_ids": output[0, prompt_tokens:].tolist(),
            }
        )

    report = {
        "config_name": args.config_name,
        "model": MODEL_ID,
        "device": device,
        "nbits": args.nbits if args.config_name == "turboquant" else None,
        "skip_layers": sorted(calibrated_skip) if calibrated_skip is not None else None,
        "model_load_time_s": round(load_time_s, 4),
        "model_load_peak_device_memory_mb": round(load_peak_mb, 2),
        "calibration_time_s": round(calibration_time_s, 4),
        "process_peak_rss_mb": round(peak_rss_mb(), 2),
        "per_prompt": per_prompt,
    }

    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
