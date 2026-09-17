"""
Comprehensive TurboQuant benchmark across model families and sizes.
Tests: Qwen, Llama, Gemma, Phi, Mistral — 7B to 72B.

For each model:
1. Architecture analysis (layers, heads, KV heads, head_dim)
2. Outlier layer detection (key norm distribution)
3. Output quality (greedy decode comparison)
4. Memory savings at multiple context lengths
5. Prefill logit fidelity
"""

import sys
sys.path.insert(0, "/home/azureuser/turboquant")

import torch
import time
import json
import gc
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from turboquant.cache import TurboQuantCache

RESULTS_FILE = "/home/azureuser/turboquant/benchmark_results.json"

MODELS = [
    # (name, hf_id, approx_4bit_size_gb)
    ("Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", 5),
    ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct", 5),
    ("Gemma-2-9B", "google/gemma-2-9b-it", 6),
    ("Phi-4-14B", "microsoft/phi-4", 9),
    ("Qwen2.5-32B", "Qwen/Qwen2.5-32B-Instruct", 19),
    ("Llama-3.3-70B", "meta-llama/Llama-3.3-70B-Instruct", 38),
    ("Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct", 40),
]

PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to check if a number is prime.",
    "What causes the northern lights?",
]

CONTEXT_LENGTHS = [1024, 4096, 8192]

PASSAGE = (
    "The history of artificial intelligence began in antiquity, with myths, stories "
    "and rumors of artificial beings endowed with intelligence or consciousness by "
    "master craftsmen. The seeds of modern AI were planted by philosophers who attempted "
    "to describe the process of human thinking as the mechanical manipulation of symbols. "
    "This work culminated in the invention of the programmable digital computer in the 1940s, "
    "a machine based on the abstract essence of mathematical reasoning. "
)


def cleanup_model():
    """Free GPU memory between model tests."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load_model(model_id):
    """Load model in 4-bit with bitsandbytes."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
    )
    return model, tokenizer


def get_architecture_info(model, config):
    """Extract architecture details."""
    tc = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
    info = {
        "num_layers": getattr(tc, "num_hidden_layers", None),
        "hidden_size": getattr(tc, "hidden_size", None),
        "num_attention_heads": getattr(tc, "num_attention_heads", None),
        "num_kv_heads": getattr(tc, "num_key_value_heads", getattr(tc, "num_attention_heads", None)),
        "head_dim": None,
        "model_type": getattr(tc, "model_type", "unknown"),
        "max_position_embeddings": getattr(tc, "max_position_embeddings", None),
        "rope_theta": getattr(tc, "rope_theta", None),
        "torch_dtype": str(getattr(tc, "torch_dtype", "unknown")),
    }
    # Some models (Gemma-2) have explicit head_dim different from hidden_size/num_heads
    info["head_dim"] = getattr(tc, "head_dim", None)
    if info["head_dim"] is None and info["hidden_size"] and info["num_attention_heads"]:
        info["head_dim"] = info["hidden_size"] // info["num_attention_heads"]
    info["model_memory_gb"] = torch.cuda.memory_allocated() / 1024**3
    return info


def analyze_layer_norms(model, tokenizer):
    """Run calibration to find outlier layer norms."""
    inputs = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(inputs.input_ids, use_cache=True)

    cache = out.past_key_values
    norms = []
    for i in range(len(cache.layers)):
        k = cache.layers[i].keys
        if k is not None and k.numel() > 0:
            norms.append(round(k.float().norm(dim=-1).mean().item(), 2))
        else:
            norms.append(0.0)

    median_norm = sorted(norms)[len(norms) // 2]
    outlier_layers = [i for i, n in enumerate(norms) if n > 5.0 * median_norm]
    max_norm = max(norms)
    max_layer = norms.index(max_norm)

    del out, cache
    cleanup_model()

    return {
        "median_norm": round(median_norm, 2),
        "max_norm": round(max_norm, 2),
        "max_norm_layer": max_layer,
        "max_to_median_ratio": round(max_norm / median_norm, 2) if median_norm > 0 else 0,
        "outlier_layers": outlier_layers,
        "all_norms_first5": norms[:5],
        "all_norms_last3": norms[-3:],
    }


def test_output_quality(model, tokenizer, skip_layers):
    """Compare outputs on test prompts."""
    results = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_input = inputs.input_ids.shape[1]

        with torch.no_grad():
            out_d = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        text_d = tokenizer.decode(out_d[0][n_input:], skip_special_tokens=True)
        cleanup_model()

        cache = TurboQuantCache(model.config, nbits=4, residual_length=128,
                                device="cuda", skip_layers=skip_layers)
        with torch.no_grad():
            out_t = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                                   past_key_values=cache)
        text_t = tokenizer.decode(out_t[0][n_input:], skip_special_tokens=True)
        cleanup_model()

        # Find divergence
        diverge = min(len(text_d), len(text_t))
        for i, (a, b) in enumerate(zip(text_d, text_t)):
            if a != b:
                diverge = i
                break

        # Token-level match
        toks_d = tokenizer.encode(text_d)
        toks_t = tokenizer.encode(text_t)
        matching = sum(a == b for a, b in zip(toks_d, toks_t))
        total = max(len(toks_d), len(toks_t))

        results.append({
            "prompt": prompt,
            "exact_match": text_d == text_t,
            "diverge_at_char": diverge,
            "total_chars": len(text_d),
            "token_match_pct": round(100 * matching / total, 1) if total > 0 else 100,
            "default_output": text_d[:200],
            "turboquant_output": text_t[:200],
            "both_coherent": True,  # Manual check flag
        })

    return results


def test_memory_savings(model, tokenizer, skip_layers, arch_info):
    """Measure memory at different context lengths."""
    results = []

    for target_ctx in CONTEXT_LENGTHS:
        n_repeats = target_ctx // len(tokenizer.encode(PASSAGE)) + 1
        long_prompt = PASSAGE * n_repeats + "\n\nSummarize the above in 2 sentences."
        inputs = tokenizer(long_prompt, return_tensors="pt", truncation=True,
                           max_length=target_ctx).to(model.device)
        actual_len = inputs.input_ids.shape[1]

        # Default
        cleanup_model()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            out_d = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        peak_d = torch.cuda.max_memory_allocated()
        text_d = tokenizer.decode(out_d[0][actual_len:], skip_special_tokens=True)
        cleanup_model()

        # TurboQuant
        cache = TurboQuantCache(model.config, nbits=4, residual_length=128,
                                device="cuda", skip_layers=skip_layers)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            out_t = model.generate(**inputs, max_new_tokens=30, do_sample=False,
                                   past_key_values=cache)
        peak_t = torch.cuda.max_memory_allocated()
        text_t = tokenizer.decode(out_t[0][actual_len:], skip_special_tokens=True)
        cleanup_model()

        saved_mb = (peak_d - peak_t) / 1024**2

        results.append({
            "context_length": actual_len,
            "peak_default_gb": round(peak_d / 1024**3, 2),
            "peak_turboquant_gb": round(peak_t / 1024**3, 2),
            "saved_mb": round(saved_mb, 0),
            "output_match": text_d[:100] == text_t[:100],
        })

    return results


def test_prefill_logits(model, tokenizer, skip_layers):
    """Compare prefill logits (should be near-identical since first call returns originals)."""
    prompt = "The meaning of life is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_d = model(inputs.input_ids, use_cache=True)
        logits_d = out_d.logits[0, -1].float()
        cleanup_model()

        cache = TurboQuantCache(model.config, nbits=4, residual_length=128,
                                device="cuda", skip_layers=skip_layers)
        out_t = model(inputs.input_ids, use_cache=True, past_key_values=cache)
        logits_t = out_t.logits[0, -1].float()
        cleanup_model()

    diff = (logits_d - logits_t).abs()
    top1_d = logits_d.argmax().item()
    top1_t = logits_t.argmax().item()

    return {
        "max_logit_diff": round(diff.max().item(), 6),
        "mean_logit_diff": round(diff.mean().item(), 6),
        "same_top1": top1_d == top1_t,
        "top1_token": tokenizer.decode([top1_d]),
    }


def benchmark_model(model_name, model_id, approx_size):
    """Run full benchmark for one model."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARKING: {model_name} ({model_id})")
    print(f"{'='*70}")

    # Check disk space
    import shutil
    free_gb = shutil.disk_usage("/").free / 1024**3
    if free_gb < approx_size + 10:
        print(f"  SKIP: Only {free_gb:.0f}GB free, need ~{approx_size+10}GB")
        return None

    result = {"model_name": model_name, "model_id": model_id}

    try:
        # Load
        print(f"  Loading model...")
        model, tokenizer = load_model(model_id)
        print(f"  Loaded: {torch.cuda.memory_allocated()/1024**3:.1f} GB on GPU")

        # Architecture
        print(f"  Analyzing architecture...")
        result["architecture"] = get_architecture_info(model, model.config)
        print(f"    Layers={result['architecture']['num_layers']}, "
              f"KV heads={result['architecture']['num_kv_heads']}, "
              f"head_dim={result['architecture']['head_dim']}")

        # Check head_dim compatibility
        head_dim = result["architecture"]["head_dim"]
        if head_dim is None or head_dim % 2 != 0:
            print(f"  SKIP: Unsupported head_dim={head_dim}")
            del model, tokenizer
            cleanup_model()
            return result

        # Layer norms
        print(f"  Analyzing layer norms...")
        result["layer_norms"] = analyze_layer_norms(model, tokenizer)
        skip = set(result["layer_norms"]["outlier_layers"])
        print(f"    Median={result['layer_norms']['median_norm']}, "
              f"Max={result['layer_norms']['max_norm']} (layer {result['layer_norms']['max_norm_layer']}), "
              f"Ratio={result['layer_norms']['max_to_median_ratio']}x, "
              f"Skip layers={skip}")

        # Prefill logits
        print(f"  Testing prefill logit fidelity...")
        result["prefill_logits"] = test_prefill_logits(model, tokenizer, skip)
        print(f"    Max diff={result['prefill_logits']['max_logit_diff']}, "
              f"Same top-1={result['prefill_logits']['same_top1']}")

        # Output quality
        print(f"  Testing output quality ({len(PROMPTS)} prompts)...")
        result["quality"] = test_output_quality(model, tokenizer, skip)
        for q in result["quality"]:
            print(f"    '{q['prompt'][:40]}...' → diverge@{q['diverge_at_char']}, "
                  f"tokens={q['token_match_pct']}%")

        # Memory
        print(f"  Testing memory savings...")
        result["memory"] = test_memory_savings(model, tokenizer, skip, result["architecture"])
        for m in result["memory"]:
            print(f"    {m['context_length']}tok: "
                  f"{m['peak_default_gb']}GB → {m['peak_turboquant_gb']}GB "
                  f"(saved {m['saved_mb']}MB)")

        result["status"] = "success"

    except Exception as e:
        print(f"  ERROR: {e}")
        result["status"] = "error"
        result["error"] = str(e)

    finally:
        # Cleanup
        try:
            del model, tokenizer
        except:
            pass
        cleanup_model()
        # Clear HF cache for this model to save disk
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        print(f"  Cleaned up GPU memory")

    return result


def main():
    all_results = []

    # Load existing results if any
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
        tested = {r["model_id"] for r in all_results if r.get("status") == "success"}
    else:
        tested = set()

    for model_name, model_id, approx_size in MODELS:
        if model_id in tested:
            print(f"\n  SKIP {model_name}: already tested")
            continue

        result = benchmark_model(model_name, model_id, approx_size)
        if result:
            # Remove any previous failed result for this model
            all_results = [r for r in all_results if r.get("model_id") != model_id]
            all_results.append(result)

            # Save after each model
            with open(RESULTS_FILE, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  Results saved to {RESULTS_FILE}")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"  SUMMARY: TurboQuant Benchmark Results")
    print(f"{'='*90}")
    print(f"{'Model':<20} {'Layers':>6} {'KV/Hd':>6} {'HeadDim':>7} "
          f"{'Outliers':>8} {'Prefill':>8} {'Quality':>8} {'Saved@8K':>10}")
    print("-" * 90)

    for r in all_results:
        if r.get("status") != "success":
            print(f"{r['model_name']:<20} {'ERROR':>6}")
            continue

        arch = r["architecture"]
        norms = r["layer_norms"]
        prefill = r["prefill_logits"]
        quality = r["quality"]
        mem = r.get("memory", [])

        avg_diverge = sum(q["diverge_at_char"] for q in quality) / len(quality) if quality else 0
        saved_8k = next((m["saved_mb"] for m in mem if m["context_length"] >= 8000), "N/A")

        prefill_str = "exact" if prefill["max_logit_diff"] == 0 else f"{prefill['max_logit_diff']:.4f}"
        saved_str = "N/A" if saved_8k == "N/A" else f"{saved_8k}MB"
        print(f"{r['model_name']:<20} {arch['num_layers']:>6} {arch['num_kv_heads']:>6} "
              f"{arch['head_dim']:>7} {len(norms['outlier_layers']):>8} "
              f"{prefill_str:>8} "
              f"{avg_diverge:>7.0f}ch {saved_str:>10}")


if __name__ == "__main__":
    main()
