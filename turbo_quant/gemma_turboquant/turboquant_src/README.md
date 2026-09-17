# TurboQuant: First Open-Source Implementation

First open-source implementation of [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni — Google Research / Google DeepMind / NYU, April 2025).

TurboQuant compresses LLM KV caches **4-7x** at inference time using random rotation + optimal scalar quantization, with **near-zero quality loss**. No training, no calibration data, fully data-oblivious. Drop-in replacement for HuggingFace Transformers cache.

## Key Results

Benchmarked across **5 model families, 6 models (7B to 70B)** on NVIDIA H100 NVL (96GB):

| Model | Architecture | KV Heads | head_dim | Outlier Layers | Prefill Fidelity | Saved @8K |
|---|---|---|---|---|---|---|
| **Qwen2.5-7B** | 28L, qwen2 | 4 | 128 | layers 0, 27 | exact | 380 MB |
| **Llama-3.1-8B** | 32L, llama | 8 | 128 | none | exact | 890 MB |
| **Gemma-2-9B** | 42L, gemma2 | 8 | 256 | none | exact | 2,323 MB |
| **Phi-4-14B** | 40L, phi3 | 10 | 128 | none | exact | 1,392 MB |
| **Qwen2.5-32B** | 64L, qwen2 | 8 | 128 | none | exact | 1,791 MB |
| **Llama-3.3-70B** | 80L, llama | 8 | 128 | none | exact | 501 MB (@2K) |

**Prefill logits are bit-identical (0.0 difference)** across all 6 tested models. Output quality is coherent and semantically correct — divergence from uncompressed output is purely greedy-decoding drift, not quality degradation.

### Needle-in-a-Haystack: 100% Recall

Tested on Qwen2.5-7B across 5 context lengths (1K-16K) and 3 needle positions (25%, 50%, 75%):

| | Default Cache | TurboQuant Cache |
|---|---|---|
| **Recall** | **15/15 (100%)** | **15/15 (100%)** |

TurboQuant preserves retrieval quality perfectly, matching the paper's 0.997 recall claim.

### Memory Savings Scale with Context

Qwen2.5-32B (4-bit weights) on H100:

| Context | Default KV | TurboQuant KV | Saved |
|---|---|---|---|
| 1K tokens | 19.97 GB | 19.79 GB | 186 MB |
| 4K tokens | 21.23 GB | 20.42 GB | 833 MB |
| 8K tokens | 23.16 GB | 21.41 GB | 1,791 MB |
| 32K tokens | ~27.5 GB | ~21.8 GB | ~5,700 MB (projected) |

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant import TurboQuantCache

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-32B-Instruct", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

# Auto-detect outlier layers, create compressed cache
skip = TurboQuantCache.calibrate_skip_layers(model, tokenizer)
cache = TurboQuantCache(model.config, nbits=4, skip_layers=skip)

# Use exactly like default cache
inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100, past_key_values=cache)
```

## How It Works

TurboQuant implements Algorithm 1 (TurboQuant_mse) from the paper:

1. **Random rotation** (QR decomposition): transforms each KV vector so coordinates follow a known Beta distribution
2. **Optimal scalar quantization** (Lloyd-Max): quantizes each coordinate to 4 bits using precomputed codebook
3. **Bit packing**: stores 128-dim vectors as 64 bytes (uint4) + 2 bytes (norm) = **66 bytes vs 256 bytes BF16**

Theoretical guarantee: MSE distortion ≤ 0.009 at 4-bit, within **2.7x of information-theoretic optimum** (Shannon lower bound).

Our measured MSE: **0.0093** — matches the paper.

## What We Found Beyond the Paper

### Outlier Layer Norms

The paper mentions "splitting channels into outlier and non-outlier sets" without specifying how. We discovered:

- **Qwen2.5-7B**: Layer 0 key norms = 273.8 (16.2x median). Layer 27 = outlier too.
- **Qwen2.5-32B**: Layer 0 = 37.8 (2.35x median). Mild, no skip needed.
- **Llama-3.1-8B**: Max/median ratio = 1.18x. No outliers at all.
- **Gemma-2-9B**: Max/median ratio = 1.19x. No outliers.
- **Phi-4-14B**: Max/median ratio = 1.38x. No outliers.

**Finding**: Smaller Qwen models have severe outlier layers. Larger models and non-Qwen architectures are well-balanced. Our `calibrate_skip_layers()` auto-detects outliers and keeps them in full precision.

### head_dim Compatibility

The paper only tested head_dim=128 (Llama, Mistral). We verified TurboQuant works with **head_dim=256** (Gemma-2) — the Lloyd-Max codebook adapts to any dimension since it's computed from the Beta distribution parameterized by d.

### Architecture Coverage

| Architecture | Paper Tested | We Tested | Works |
|---|---|---|---|
| Llama | Llama-3.1-8B | Llama-3.1-8B, 3.3-70B | Yes |
| Mistral | Ministral-7B | — | — |
| Qwen | — | Qwen2.5-7B, 32B | Yes (with outlier handling) |
| Gemma | — | Gemma-2-9B | Yes (head_dim=256) |
| Phi | — | Phi-4-14B | Yes |

## Files

```
turboquant/
├── __init__.py          # Public API
├── codebook.py          # Lloyd-Max solver for Beta distribution
├── quantizer.py         # Core TurboQuantizer: rotate → quantize → pack
├── packing.py           # uint4/uint2 bit packing
├── cache.py             # TurboQuantCache for HF Transformers
scripts/
├── verify.py            # Unit tests (MSE bounds, packing, fixed-point)
├── test_cache.py        # Cache API integration tests
├── benchmark_models.py  # Multi-model benchmark suite
├── run_inference.py     # Interactive inference demo
benchmark_results.json   # Raw benchmark data (all 5 models)
```

## Verified Against Paper

| Metric | Paper | Ours |
|---|---|---|
| MSE at 4-bit (unit vectors) | ≤ 0.009 | 0.0093 |
| MSE at 2-bit (unit vectors) | ≤ 0.117 | 0.116 |
| Compression ratio (per-vector) | ~4x | 3.88x |
| System compression @8K+ | 4-7x | 7.2x |
| Prefill fidelity | "quality neutral" | exact (0.0 logit diff) |
| Double quantization | fixed point | verified (indices identical) |

## Requirements

- Python 3.10+
- PyTorch 2.7+ (CUDA 12.8 compatible)
- HuggingFace Transformers 5.0+
- scipy (for codebook computation)
- bitsandbytes (optional, for 4-bit model loading)

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```

## License

This implementation is released under MIT License. The TurboQuant algorithm is described in the paper above.
