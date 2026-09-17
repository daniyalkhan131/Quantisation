"""Test TurboQuantCache integration with the HF Transformers cache API."""

import sys
sys.path.insert(0, "/home/azureuser/turboquant")

import torch
from types import SimpleNamespace
from turboquant.cache import TurboQuantCache, TurboQuantLayer


def test_cache_basic():
    """Test TurboQuantCache with mock model config, simulating Qwen2.5-32B."""
    print("=" * 60)
    print("TEST: TurboQuantCache basic operations")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mock Qwen2.5-32B config (just the fields we need)
    config = SimpleNamespace(
        num_hidden_layers=4,  # Use 4 layers for testing (not 64)
        hidden_size=5120,
        num_attention_heads=40,
    )
    # Mock get_text_config for compatibility
    config.get_text_config = lambda decoder=True: config

    cache = TurboQuantCache(config, nbits=4, residual_length=4, device=device)
    print(f"  Created cache with {len(cache.layers)} layers")

    batch, heads, head_dim = 1, 8, 128

    # Simulate prefill: 16 tokens at once
    for layer_idx in range(4):
        k = torch.randn(batch, heads, 16, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch, heads, 16, head_dim, device=device, dtype=torch.bfloat16)

        k_out, v_out = cache.update(k, v, layer_idx)
        print(f"  Layer {layer_idx} prefill: input ({k.shape}) → output ({k_out.shape})")
        assert k_out.shape == (batch, heads, 16, head_dim)
        assert k_out.dtype == torch.bfloat16

    # Simulate decode: 1 token at a time, 8 steps
    for step in range(8):
        for layer_idx in range(4):
            k = torch.randn(batch, heads, 1, head_dim, device=device, dtype=torch.bfloat16)
            v = torch.randn(batch, heads, 1, head_dim, device=device, dtype=torch.bfloat16)

            k_out, v_out = cache.update(k, v, layer_idx)

            expected_len = 16 + step + 1
            assert k_out.shape == (batch, heads, expected_len, head_dim), \
                f"Expected seq_len={expected_len}, got {k_out.shape[-2]}"
            assert k_out.dtype == torch.bfloat16

        if step == 0 or step == 7:
            print(f"  Decode step {step}: seq_len={k_out.shape[-2]}")

    # Check sequence length
    seq_len = cache.get_seq_length(0)
    print(f"  Final seq_length: {seq_len}")

    print("\n  PASS: Cache operations correct\n")


def test_cache_memory():
    """Compare memory usage: DynamicCache vs TurboQuantCache."""
    from transformers.cache_utils import DynamicCache

    print("=" * 60)
    print("TEST: Memory comparison vs DynamicCache")
    print("=" * 60)

    device = "cuda"
    if not torch.cuda.is_available():
        print("  SKIP: No CUDA available")
        return

    config = SimpleNamespace(
        num_hidden_layers=64,
        hidden_size=5120,
        num_attention_heads=40,
    )
    config.get_text_config = lambda decoder=True: config

    batch, heads, head_dim = 1, 8, 128
    seq_len = 4096

    # --- DynamicCache (BF16 baseline) ---
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated()

    dyn_cache = DynamicCache()
    for layer_idx in range(64):
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        dyn_cache.update(k, v, layer_idx)

    mem_dynamic = torch.cuda.memory_allocated() - mem_before
    del dyn_cache
    torch.cuda.empty_cache()

    # --- TurboQuantCache (4-bit) ---
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    tq_cache = TurboQuantCache(config, nbits=4, residual_length=1, device=device)
    for layer_idx in range(64):
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        tq_cache.update(k, v, layer_idx)

    mem_turboquant = torch.cuda.memory_allocated() - mem_before
    del tq_cache
    torch.cuda.empty_cache()

    ratio = mem_dynamic / max(mem_turboquant, 1)
    print(f"  Seq length:     {seq_len}")
    print(f"  Layers:         64")
    print(f"  DynamicCache:   {mem_dynamic / 1024**2:.1f} MB")
    print(f"  TurboQuantCache: {mem_turboquant / 1024**2:.1f} MB")
    print(f"  Compression:    {ratio:.2f}x")
    print(f"\n  PASS: Memory comparison done\n")


if __name__ == "__main__":
    test_cache_basic()
    test_cache_memory()
    print("=" * 60)
    print("ALL CACHE TESTS PASSED")
    print("=" * 60)
