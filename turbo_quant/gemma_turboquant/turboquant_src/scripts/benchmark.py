"""Benchmark TurboQuant memory savings and throughput."""

import sys
sys.path.insert(0, "/home/azureuser/turboquant")

import torch
import time
from types import SimpleNamespace
from transformers.cache_utils import DynamicCache, Cache, DynamicLayer
from turboquant.cache import TurboQuantCache, TurboQuantLayer


def benchmark_memory(num_layers: int = 64, num_kv_heads: int = 8, head_dim: int = 128,
                     context_lengths: list[int] = None, skip_layers: set[int] = None):
    """Compare memory usage between DynamicCache and TurboQuantCache."""
    if context_lengths is None:
        context_lengths = [1024, 4096, 8192, 16384, 32768]
    if skip_layers is None:
        skip_layers = {0, 1}

    device = "cuda"
    batch = 1

    print(f"{'Context':>8} | {'DynamicCache':>14} | {'TurboQuant':>14} | {'Compression':>12} | {'Savings':>10}")
    print("-" * 72)

    for seq_len in context_lengths:
        # --- DynamicCache ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

        dyn_cache = DynamicCache()
        for layer_idx in range(num_layers):
            k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
            v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
            dyn_cache.update(k, v, layer_idx)
        mem_dynamic = torch.cuda.memory_allocated() - mem_before
        del dyn_cache
        torch.cuda.empty_cache()

        # --- TurboQuantCache ---
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

        # Create cache with skip_layers
        layers = []
        for i in range(num_layers):
            if i in skip_layers:
                layers.append(DynamicLayer())
            else:
                layers.append(TurboQuantLayer(
                    dim=head_dim, nbits=4, residual_length=1, device=device, layer_seed=42 + i
                ))
        tq_cache = Cache(layers=layers)

        for layer_idx in range(num_layers):
            k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
            v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
            tq_cache.update(k, v, layer_idx)
        mem_tq = torch.cuda.memory_allocated() - mem_before
        del tq_cache
        torch.cuda.empty_cache()

        ratio = mem_dynamic / max(mem_tq, 1)
        savings = (mem_dynamic - mem_tq) / 1024**2

        print(f"{seq_len:>8} | {mem_dynamic/1024**2:>11.1f} MB | {mem_tq/1024**2:>11.1f} MB | "
              f"{ratio:>10.2f}x | {savings:>7.1f} MB")


def benchmark_throughput(num_layers: int = 64, num_kv_heads: int = 8, head_dim: int = 128):
    """Benchmark quantization and dequantization throughput."""
    device = "cuda"
    batch = 1

    print(f"\n{'Operation':>20} | {'Seq Len':>8} | {'Time (ms)':>10} | {'Throughput':>15}")
    print("-" * 65)

    quantizer_layer = TurboQuantLayer(dim=head_dim, nbits=4, residual_length=1, device=device, layer_seed=42)

    for seq_len in [1024, 4096, 16384, 32768]:
        k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(3):
            packed, norms = quantizer_layer.quantizer.quantize(k)
            _ = quantizer_layer.quantizer.dequantize(packed, norms)
        torch.cuda.synchronize()

        # Quantize timing
        start = time.perf_counter()
        for _ in range(10):
            packed, norms = quantizer_layer.quantizer.quantize(k)
            torch.cuda.synchronize()
        quant_time = (time.perf_counter() - start) / 10 * 1000

        # Dequantize timing
        start = time.perf_counter()
        for _ in range(10):
            _ = quantizer_layer.quantizer.dequantize(packed, norms)
            torch.cuda.synchronize()
        dequant_time = (time.perf_counter() - start) / 10 * 1000

        n_vectors = batch * num_kv_heads * seq_len
        print(f"{'Quantize':>20} | {seq_len:>8} | {quant_time:>8.2f} ms | {n_vectors/quant_time*1000:>12.0f} vec/s")
        print(f"{'Dequantize':>20} | {seq_len:>8} | {dequant_time:>8.2f} ms | {n_vectors/dequant_time*1000:>12.0f} vec/s")


if __name__ == "__main__":
    print("=" * 72)
    print("TurboQuant Memory Benchmark — Qwen2.5-32B Configuration")
    print("  64 layers, 8 KV heads, head_dim=128, 4-bit, skip layers {0,1}")
    print("=" * 72)

    benchmark_memory()

    print("\n" + "=" * 72)
    print("TurboQuant Throughput Benchmark (single layer)")
    print("=" * 72)

    benchmark_throughput()
