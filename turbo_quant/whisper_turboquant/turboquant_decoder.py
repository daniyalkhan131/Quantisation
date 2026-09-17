"""Experimental: run Whisper's decoder with its self-attention KV cache
quantized by mlx-vlm's TurboQuant scheme.

mlx-vlm's TurboQuant only ships wired into mlx-vlm's own model zoo (VLMs/LLMs
loaded through `mlx_vlm.load`); Whisper isn't one of those architectures. To
actually use `mlx_vlm.turboquant.TurboQuantKVCache` on Whisper, this script
reimplements Whisper's text decoder with mlx-vlm's cache protocol
(`cache.update_and_fetch(...)` + `mlx_vlm.models.base.scaled_dot_product_attention`)
in place of mlx_whisper's plain concatenated-tensor cache, so the same cache
classes mlx-vlm uses for e.g. Qwen/Gemma decoding can be dropped in here.

The audio encoder is untouched (reused from `mlx_whisper.whisper`) since it
has no autoregressive KV cache to compress; only the decoder's self-attention
is affected. Cross-attention K/V are computed once from the encoder output
and reused for every decoded token, so there's nothing growing there for
TurboQuant to compress either -- only self-attention grows with each token.

Read ../README.md first: for Whisper's short (<=448 token) decode sequences,
the realistic memory savings from this are small. This script exists to
measure that honestly, not to claim a speedup.

Usage:
    python turboquant_decoder.py --mlx-path ./mlx-models/whisper-large-v3-fp16 \
        --audio path/to/audio.wav --kv-bits 3.5 --kv-quant-scheme turboquant

    # Baseline (no KV quantization) for comparison:
    python turboquant_decoder.py --mlx-path ./mlx-models/whisper-large-v3-fp16 \
        --audio path/to/audio.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, load_audio, pad_or_trim
from mlx_whisper.tokenizer import get_tokenizer
from mlx_whisper.whisper import AudioEncoder, ModelDimensions, sinusoids

from mlx_vlm.kv_quant import from_legacy as kv_quant_from_legacy
from mlx_vlm.models.base import scaled_dot_product_attention
from mlx_vlm.models.cache import KVCache, cache_nbytes, create_causal_mask
from mlx_vlm.turboquant import TurboQuantKVCache, turboquant_enabled


class TurboMultiHeadAttention(nn.Module):
    """Self-attention that reads/writes through an mlx-vlm cache object
    instead of mlx_whisper's plain (k, v) tensor-concatenation cache."""

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def __call__(self, x, mask, cache):
        B, L, D = x.shape
        H = self.n_head
        q = self.query(x).reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(B, L, H, -1).transpose(0, 2, 1, 3)

        k, v = cache.update_and_fetch(k, v)
        scale = (D // H) ** -0.5
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=scale, mask=mask)
        # A boolean causal mask can make SDPA promote its output to float32;
        # cast back so every block's cache stays in the checkpoint's dtype.
        out = out.astype(x.dtype).transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out(out)


class CrossAttention(nn.Module):
    """Cross-attention over the (fixed) encoder output. Computed once per
    utterance and reused every decode step -- nothing here grows with the
    number of generated tokens, so there is no cache for TurboQuant to
    compress."""

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def project_kv(self, xa):
        B, S, D = xa.shape
        H = self.n_head
        k = self.key(xa).reshape(B, S, H, -1).transpose(0, 2, 1, 3)
        v = self.value(xa).reshape(B, S, H, -1).transpose(0, 2, 1, 3)
        return k, v

    def __call__(self, x, kv):
        B, L, D = x.shape
        H = self.n_head
        q = self.query(x).reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        k, v = kv
        scale = (D // H) ** -0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None)
        out = out.astype(x.dtype).transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out(out)


class TurboResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.attn = TurboMultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)
        self.cross_attn = CrossAttention(n_state, n_head)
        self.cross_attn_ln = nn.LayerNorm(n_state)
        n_mlp = n_state * 4
        self.mlp1 = nn.Linear(n_state, n_mlp)
        self.mlp2 = nn.Linear(n_mlp, n_state)
        self.mlp_ln = nn.LayerNorm(n_state)

    def __call__(self, x, cross_kv, mask, cache):
        x = x + self.attn(self.attn_ln(x), mask, cache)
        x = x + self.cross_attn(self.cross_attn_ln(x), cross_kv)
        x = x + self.mlp2(nn.gelu(self.mlp1(self.mlp_ln(x))))
        return x


class TurboTextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = mx.zeros((n_ctx, n_state))
        self.blocks = [TurboResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        self.ln = nn.LayerNorm(n_state)

    def __call__(self, x, cross_kvs, caches, offset):
        L = x.shape[-1]
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + L]
        mask = create_causal_mask(L, offset) if L > 1 else None
        for block, cross_kv, cache in zip(self.blocks, cross_kvs, caches):
            x = block(x, cross_kv, mask, cache)
        x = self.ln(x)
        return self.token_embedding.as_linear(x)


class TurboWhisper(nn.Module):
    """Whisper with a TurboQuant-capable decoder. Parameter names mirror
    mlx_whisper.whisper.Whisper exactly, so checkpoints produced by
    convert_from_hf.py (or mlx_whisper's own converter) load unmodified."""

    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            dims.n_mels, dims.n_audio_ctx, dims.n_audio_state, dims.n_audio_head, dims.n_audio_layer
        )
        self.decoder = TurboTextDecoder(
            dims.n_vocab, dims.n_text_ctx, dims.n_text_state, dims.n_text_head, dims.n_text_layer
        )

    def make_self_attn_caches(self, kv_bits, kv_quant_scheme, kv_group_size=64):
        policy = kv_quant_from_legacy(kv_bits, kv_quant_scheme, kv_group_size)
        if policy is not None and not policy.is_turboquant:
            raise NotImplementedError(
                "This script only wires up the TurboQuant KV cache; pass "
                "--kv-quant-scheme turboquant (mlx-vlm's uniform int KV "
                "quantization is not implemented here)."
            )
        caches = []
        for _ in self.decoder.blocks:
            if policy is None:
                caches.append(KVCache())
            else:
                caches.append(TurboQuantKVCache(bits=policy.bits))
        return caches


def load_turbo_whisper(mlx_path: str) -> TurboWhisper:
    model_path = Path(mlx_path)
    config = json.loads((model_path / "config.json").read_text())
    quantization = config.pop("quantization", None)
    config.pop("model_type", None)
    dims = ModelDimensions(**config)

    weights_file = model_path / "weights.safetensors"
    weights = mx.load(str(weights_file))
    weights.pop("alignment_heads", None)  # mlx_whisper.whisper.Whisper-only bookkeeping

    model = TurboWhisper(dims)
    if quantization is not None:
        class_predicate = (
            lambda p, m: isinstance(m, (nn.Linear, nn.Embedding)) and f"{p}.scales" in weights
        )
        nn.quantize(model, **quantization, class_predicate=class_predicate)

    from mlx.utils import tree_unflatten

    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model


def transcribe_with_turboquant(
    model: TurboWhisper,
    audio,
    kv_bits,
    kv_quant_scheme,
    max_new_tokens: int = 224,
):
    dims = model.dims
    tokenizer = get_tokenizer(
        multilingual=dims.n_vocab >= 51865, num_languages=dims.n_vocab - 51765 - int(dims.n_vocab >= 51865)
    )

    if isinstance(audio, (str, Path)):
        audio = load_audio(str(audio))
    mel = log_mel_spectrogram(audio, n_mels=dims.n_mels)
    mel = pad_or_trim(mel, N_FRAMES, axis=-2)  # (n_frames, n_mels) -> pad the frames axis
    mel = mel[None]

    audio_features = model.encoder(mel)
    cross_kvs = [block.cross_attn.project_kv(audio_features) for block in model.decoder.blocks]
    caches = model.make_self_attn_caches(kv_bits, kv_quant_scheme)

    initial_tokens = list(tokenizer.sot_sequence_including_notimestamps)
    tokens = mx.array([initial_tokens])
    logits = model.decoder(tokens, cross_kvs, caches, offset=0)
    next_token = int(mx.argmax(logits[:, -1], axis=-1).item())

    generated = [next_token]
    offset = len(initial_tokens)
    while next_token != tokenizer.eot and len(generated) < max_new_tokens:
        tokens = mx.array([[next_token]])
        logits = model.decoder(tokens, cross_kvs, caches, offset=offset)
        next_token = int(mx.argmax(logits[:, -1], axis=-1).item())
        generated.append(next_token)
        offset += 1

    text = tokenizer.decode([t for t in generated if t < tokenizer.eot])
    self_attn_cache_bytes = cache_nbytes(caches)
    return text, self_attn_cache_bytes, offset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mlx-path", required=True, help="Converted MLX Whisper checkpoint (see convert_from_hf.py).")
    parser.add_argument("--audio", required=True, help="Path to an audio file.")
    parser.add_argument("--kv-bits", type=float, default=None, help="e.g. 3.5 for TurboQuant, 8 for uniform.")
    parser.add_argument("--kv-quant-scheme", default="turboquant", choices=["turboquant"])
    parser.add_argument("--max-new-tokens", type=int, default=224)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run with an unquantized (fp16) KV cache and print both side by side.",
    )
    args = parser.parse_args()

    model = load_turbo_whisper(args.mlx_path)

    def run(kv_bits, label):
        text, cache_bytes, n_tokens = transcribe_with_turboquant(
            model, args.audio, kv_bits, args.kv_quant_scheme, args.max_new_tokens
        )
        print(f"[{label}] transcript: {text!r}")
        print(f"[{label}] decoded tokens: {n_tokens}")
        print(f"[{label}] self-attention KV cache allocated: {cache_bytes / 1024:.2f} KB")
        return cache_bytes

    if args.compare:
        baseline_bytes = run(None, "fp16 baseline")
        turbo_bytes = run(args.kv_bits or 3.5, f"{args.kv_quant_scheme}")
        if baseline_bytes:
            print(f"Reduction vs fp16 baseline: {100 * (1 - turbo_bytes / baseline_bytes):.1f}%")
    else:
        label = "fp16 baseline" if args.kv_bits is None else f"{args.kv_quant_scheme} @ {args.kv_bits} bits"
        run(args.kv_bits, label)


if __name__ == "__main__":
    main()
