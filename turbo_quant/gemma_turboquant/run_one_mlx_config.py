"""Run one mlx-vlm config (weights: fp16 vs 4-bit) x (KV cache: none vs
mlx-vlm's own TurboQuant) against google/gemma-4-E4B-it and dump
timing/memory/output metrics to a JSON file. Invoked as a subprocess by
evaluate_mlx_backend.py so each config gets a fresh model load and isolated
peak-memory accounting -- same reasoning as run_one_llm_config.py.

This uses the *real*, Metal-kernel-backed mlx-vlm TurboQuant
(mlx_vlm.turboquant.TurboQuantKVCache), not vivekvar/turboquant's pure-Python
reference implementation from run_one_llm_config.py -- the two are unrelated
implementations of the same paper. mlx-vlm's version infers each layer's
head_dim from the actual K/V tensor shape at update time (not from a global
config field), and only wraps `KVCache` (global-attention) layers, leaving
`RotatingKVCache` (sliding-window) layers in their native format -- which is
exactly the pair of things that broke vivekvar/turboquant on this model's
per-layer head_dim and hybrid attention. See ../README.md and notes.txt for
the comparison against that reference implementation.

Prompts are identical to run_one_llm_config.py's for direct comparability.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

MLX_PATHS = {
    "fp16": "./mlx-models/gemma-4-E4B-it-fp16",
    "4bit": "./mlx-models/gemma-4-E4B-it-4bit",
}

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--weights", required=True, choices=["fp16", "4bit"])
    parser.add_argument("--kv-bits", type=float, default=None)
    parser.add_argument("--kv-quant-scheme", default="turboquant")
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=0,
        help="mlx-vlm defaults this to 5000 (only quantize past 5000 cached "
        "tokens); both our prompts are shorter than that, so it must be "
        "lowered or TurboQuant never actually activates.",
    )
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    mlx_path = MLX_PATHS[args.weights]
    weights_size_mb = sum(
        f.stat().st_size for f in Path(mlx_path).glob("*.safetensors")
    ) / (1024**2)

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    model, processor = load(mlx_path)
    mx.eval(model.parameters())
    load_time_s = time.perf_counter() - t0
    load_peak_mb = mx.get_peak_memory() / (1024**2)

    per_prompt = []
    for prompt_name, prompt_text in PROMPTS.items():
        # google/gemma-4-E4B-it is instruction-tuned; mlx_vlm.generate()'s
        # `prompt` arg is used verbatim, no chat template applied
        # automatically (unlike the CLI, which calls this itself) -- same
        # gotcha as run_one_llm_config.py hit on the PyTorch side.
        templated_prompt = apply_chat_template(
            processor,
            model.config,
            [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}],
        )
        mx.reset_peak_memory()
        result = generate(
            model,
            processor,
            templated_prompt,
            max_tokens=args.max_tokens,
            kv_bits=args.kv_bits,
            kv_quant_scheme=args.kv_quant_scheme if args.kv_bits is not None else "uniform",
            quantized_kv_start=args.quantized_kv_start,
            temperature=0.0,
            verbose=False,
        )
        gen_peak_mb = mx.get_peak_memory() / (1024**2)

        per_prompt.append(
            {
                "prompt_name": prompt_name,
                "prompt_tokens": result.prompt_tokens,
                "new_tokens": result.generation_tokens,
                "prompt_tps": result.prompt_tps,
                "generation_tps": result.generation_tps,
                "generation_peak_device_memory_mb": round(gen_peak_mb, 2),
                "generated_text": result.text,
                "generated_token_ids": result.token_ids if result.token_ids else None,
            }
        )

    report = {
        "config_name": args.config_name,
        "model": mlx_path,
        "weights": args.weights,
        "kv_bits": args.kv_bits,
        "kv_quant_scheme": args.kv_quant_scheme if args.kv_bits is not None else None,
        "weights_size_mb": round(weights_size_mb, 2),
        "model_load_time_s": round(load_time_s, 4),
        "model_load_peak_device_memory_mb": round(load_peak_mb, 2),
        "process_peak_rss_mb": round(peak_rss_mb(), 2),
        "per_prompt": per_prompt,
    }

    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
