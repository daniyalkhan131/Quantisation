"""
TurboQuant inference with Qwen models.

Demonstrates TurboQuant KV cache compression as a drop-in replacement
for the default DynamicCache during model.generate().
"""

import sys
sys.path.insert(0, "/home/azureuser/turboquant")

import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from turboquant.cache import TurboQuantCache


def load_model(model_name: str, load_in_4bit: bool = True):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model, tokenizer


def generate_with_cache(model, tokenizer, prompt: str, cache_type: str = "turboquant",
                        max_new_tokens: int = 100, nbits: int = 4,
                        skip_layers: set[int] | None = None):
    """Generate text using specified cache type."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    # Create cache
    if cache_type == "turboquant":
        cache = TurboQuantCache(
            model.config,
            nbits=nbits,
            residual_length=128,
            device=str(model.device),
            skip_layers=skip_layers,
        )
    else:
        cache = None  # Use default DynamicCache

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            past_key_values=cache,
            do_sample=False,
        )

    elapsed = time.time() - start
    mem_peak = torch.cuda.max_memory_allocated()
    mem_used = torch.cuda.memory_allocated() - mem_before

    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    n_tokens = outputs.shape[1] - input_len

    print(f"\n  Cache: {cache_type}")
    print(f"  Tokens: {n_tokens} in {elapsed:.2f}s ({n_tokens/elapsed:.1f} tok/s)")
    print(f"  Peak GPU memory: {mem_peak / 1024**3:.2f} GB")
    print(f"  Cache memory delta: {mem_used / 1024**2:.1f} MB")
    print(f"  Output: {generated[:200]}...")

    return generated, elapsed, mem_peak


def main():
    parser = argparse.ArgumentParser(description="TurboQuant inference")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model name (default: Qwen2.5-1.5B for testing)")
    parser.add_argument("--prompt", default="Explain quantum computing in simple terms.",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--nbits", type=int, default=4, choices=[2, 4])
    parser.add_argument("--no-4bit", action="store_true", help="Load in BF16 instead of 4-bit")
    parser.add_argument("--compare", action="store_true", help="Compare TurboQuant vs default cache")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, load_in_4bit=not args.no_4bit)

    # Auto-calibrate skip layers
    skip = TurboQuantCache.calibrate_skip_layers(model, tokenizer)
    print(f"Auto-detected skip layers: {skip} (kept in BF16 due to outlier KV norms)")

    if args.compare:
        print("\n" + "=" * 60)
        print("COMPARISON: Default DynamicCache vs TurboQuantCache")
        print("=" * 60)

        # Default cache
        gen_default, t_default, mem_default = generate_with_cache(
            model, tokenizer, args.prompt, "default", args.max_tokens
        )
        torch.cuda.empty_cache()

        # TurboQuant cache
        gen_tq, t_tq, mem_tq = generate_with_cache(
            model, tokenizer, args.prompt, "turboquant", args.max_tokens, args.nbits,
            skip_layers=skip,
        )

        print(f"\n  Memory savings: {(mem_default - mem_tq) / 1024**2:.1f} MB "
              f"({mem_default/max(mem_tq, 1):.2f}x)")
        print(f"  Outputs match: {gen_default == gen_tq}")

    else:
        generate_with_cache(
            model, tokenizer, args.prompt, "turboquant", args.max_tokens, args.nbits,
            skip_layers=skip,
        )


if __name__ == "__main__":
    main()
