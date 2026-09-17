"""Convert a Hugging Face Whisper checkpoint (WhisperForConditionalGeneration
format, e.g. ``openai/whisper-large-v3``) into the MLX checkpoint layout that
``mlx_whisper`` and ``turboquant_decode.py`` in this folder both load
(``config.json`` + ``weights.safetensors``), optionally weight-quantizing it
with MLX's standard affine quantization on the way out.

This is the "practical" quantization path: it shrinks the model's weights
(what actually dominates Whisper-large's footprint), unlike TurboQuant, which
only compresses the KV cache built up during decoding. See ../README.md.

Usage:
    python convert_from_hf.py --hf-repo openai/whisper-large-v3 --mlx-path ./mlx-models/whisper-large-v3
    python convert_from_hf.py --hf-repo openai/whisper-large-v3 --mlx-path ./mlx-models/whisper-large-v3-4bit \
        --quantize --q-bits 4 --q-group-size 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_unflatten
from safetensors import safe_open

from mlx_whisper.whisper import ModelDimensions, Whisper

# HF `WhisperForConditionalGeneration` parameter names -> mlx_whisper.whisper
# parameter names. Both trace back to the same original OpenAI checkpoint
# naming, so the mapping is a small, fixed set of substring substitutions
# applied in order.
_SUBSTITUTIONS = [
    ("model.encoder.", "encoder."),
    ("model.decoder.", "decoder."),
    ("encoder.layers.", "encoder.blocks."),
    ("decoder.layers.", "decoder.blocks."),
    ("self_attn.q_proj", "attn.query"),
    ("self_attn.k_proj", "attn.key"),
    ("self_attn.v_proj", "attn.value"),
    ("self_attn.out_proj", "attn.out"),
    ("self_attn_layer_norm", "attn_ln"),
    ("encoder_attn.q_proj", "cross_attn.query"),
    ("encoder_attn.k_proj", "cross_attn.key"),
    ("encoder_attn.v_proj", "cross_attn.value"),
    ("encoder_attn.out_proj", "cross_attn.out"),
    ("encoder_attn_layer_norm", "cross_attn_ln"),
    ("fc1", "mlp1"),
    ("fc2", "mlp2"),
    ("final_layer_norm", "mlp_ln"),
    ("encoder.layer_norm", "encoder.ln_post"),
    ("decoder.layer_norm", "decoder.ln"),
    ("decoder.embed_tokens", "decoder.token_embedding"),
]

# Positional embeddings are plain arrays in mlx_whisper, not nn.Embedding
# modules, so they need a distinct target name (no ".weight" suffix) and are
# handled explicitly rather than through nn.Module.update().
_POSITIONAL = {
    "model.encoder.embed_positions.weight": "encoder._positional_embedding",
    "model.decoder.embed_positions.weight": "decoder.positional_embedding",
}


def _remap_key(key: str) -> str | None:
    if key == "proj_out.weight":
        return None  # tied to decoder.token_embedding.weight, already covered
    for old, new in _SUBSTITUTIONS:
        key = key.replace(old, new)
    return key


def load_hf_state_dict(model_path: Path) -> dict[str, np.ndarray]:
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        shards = sorted(set(json.loads(index_file.read_text())["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    state_dict: dict[str, np.ndarray] = {}
    for shard in shards:
        with safe_open(str(model_path / shard), framework="numpy") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def convert_weights(hf_state_dict: dict[str, np.ndarray]) -> dict[str, mx.array]:
    mlx_weights: dict[str, mx.array] = {}
    for key, value in hf_state_dict.items():
        if key in _POSITIONAL:
            mlx_weights[_POSITIONAL[key]] = mx.array(value)
            continue
        new_key = _remap_key(key)
        if new_key is None:
            continue
        if key.endswith("conv1.weight") or key.endswith("conv2.weight"):
            # PyTorch Conv1d weight: (out_channels, in_channels, kernel).
            # MLX Conv1d weight:     (out_channels, kernel, in_channels).
            value = np.transpose(value, (0, 2, 1))
        mlx_weights[new_key] = mx.array(value)
    return mlx_weights


def dims_from_hf_config(config: dict) -> ModelDimensions:
    return ModelDimensions(
        n_mels=config["num_mel_bins"],
        n_audio_ctx=config["max_source_positions"],
        n_audio_state=config["d_model"],
        n_audio_head=config["encoder_attention_heads"],
        n_audio_layer=config["encoder_layers"],
        n_vocab=config["vocab_size"],
        n_text_ctx=config["max_target_positions"],
        n_text_state=config["d_model"],
        n_text_head=config["decoder_attention_heads"],
        n_text_layer=config["decoder_layers"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-repo",
        default="openai/whisper-large-v3",
        help="Hugging Face repo id of a `WhisperForConditionalGeneration` checkpoint.",
    )
    parser.add_argument(
        "--mlx-path", required=True, help="Output directory for the MLX checkpoint."
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Storage dtype for unquantized weights.",
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Weight-quantize with MLX affine quantization."
    )
    parser.add_argument("--q-bits", type=int, default=4, help="Bits per weight when --quantize.")
    parser.add_argument(
        "--q-group-size", type=int, default=64, help="Quantization group size when --quantize."
    )
    args = parser.parse_args()

    print(f"Downloading {args.hf_repo} from Hugging Face...")
    model_path = Path(snapshot_download(args.hf_repo, allow_patterns=["*.json", "*.safetensors"]))

    config = json.loads((model_path / "config.json").read_text())
    dims = dims_from_hf_config(config)
    print(f"Model dimensions: {dims}")

    hf_state_dict = load_hf_state_dict(model_path)
    dtype = getattr(mx, args.dtype)
    weights = {k: v.astype(dtype) for k, v in convert_weights(hf_state_dict).items()}

    model = Whisper(dims, dtype)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())

    quantization = None
    if args.quantize:
        print(f"Quantizing to {args.q_bits} bits (group size {args.q_group_size})...")
        nn.quantize(model, group_size=args.q_group_size, bits=args.q_bits)
        quantization = {"group_size": args.q_group_size, "bits": args.q_bits}

    out_dir = Path(args.mlx_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    flat_weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out_dir / "weights.safetensors"), flat_weights)

    out_config = {
        "n_mels": dims.n_mels,
        "n_audio_ctx": dims.n_audio_ctx,
        "n_audio_state": dims.n_audio_state,
        "n_audio_head": dims.n_audio_head,
        "n_audio_layer": dims.n_audio_layer,
        "n_vocab": dims.n_vocab,
        "n_text_ctx": dims.n_text_ctx,
        "n_text_state": dims.n_text_state,
        "n_text_head": dims.n_text_head,
        "n_text_layer": dims.n_text_layer,
        "model_type": "whisper",
    }
    if quantization is not None:
        out_config["quantization"] = quantization
    (out_dir / "config.json").write_text(json.dumps(out_config, indent=2))

    # mlx_whisper.transcribe() looks up the tokenizer purely from `is_multilingual`
    # / `num_languages`, both derived from n_vocab, so no extra tokenizer files
    # are needed here.

    print(f"Saved MLX Whisper checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
