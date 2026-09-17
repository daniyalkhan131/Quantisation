"""Run a single (weights precision x KV-cache scheme) configuration over a
directory of audio files and dump timing/memory/quality metrics to a JSON
file. Meant to be invoked as a fresh subprocess per config by evaluate.py, so
that peak-memory accounting (both MLX's device allocator and the process's
RSS) reflects one config in isolation rather than a running max across
several models loaded back-to-back in one long-lived process.

Every config goes through the same minimal decode loop
(turboquant_decoder.transcribe_with_turboquant), so weight precision and KV
scheme are the only two variables changing between runs -- see ../README.md
for why that matters for a fair comparison.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from turboquant_decoder import load_turbo_whisper, transcribe_with_turboquant


def load_audio_16k(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        data = resample_poly(data, 16000 // g, sr // g).astype(np.float32)
    return data


def peak_rss_mb() -> float:
    """Peak resident set size of this process since it started, in MB."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024**2) if sys.platform == "darwin" else r / 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--kv-bits", type=float, default=None)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=224)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    audio_files = sorted(Path(args.data_dir).glob("*.wav"))
    if not audio_files:
        raise SystemExit(f"No .wav files found in {args.data_dir}")

    weights_file = Path(args.mlx_path) / "weights.safetensors"
    weights_size_mb = weights_file.stat().st_size / (1024**2)

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    model = load_turbo_whisper(args.mlx_path)
    mx.eval(model.parameters())
    load_time_s = time.perf_counter() - t0
    load_peak_device_mb = mx.get_peak_memory() / (1024**2)

    per_file = []
    for audio_path in audio_files:
        audio = load_audio_16k(str(audio_path))
        duration_s = len(audio) / 16000

        mx.reset_peak_memory()
        t0 = time.perf_counter()
        text, cache_bytes, n_decoded = transcribe_with_turboquant(
            model, audio, args.kv_bits, "turboquant", args.max_new_tokens
        )
        infer_time_s = time.perf_counter() - t0
        infer_peak_device_mb = mx.get_peak_memory() / (1024**2)

        per_file.append(
            {
                "audio_file": audio_path.name,
                "audio_duration_s": round(duration_s, 3),
                "transcript": text,
                "decoded_tokens": n_decoded,
                "inference_time_s": round(infer_time_s, 4),
                "real_time_factor": round(duration_s / infer_time_s, 3) if infer_time_s > 0 else None,
                "tokens_per_second": round(n_decoded / infer_time_s, 2) if infer_time_s > 0 else None,
                "inference_peak_device_memory_mb": round(infer_peak_device_mb, 2),
                "self_attn_kv_cache_allocated_bytes": cache_bytes,
            }
        )

    report = {
        "config_name": args.config_name,
        "mlx_path": args.mlx_path,
        "kv_bits": args.kv_bits,
        "kv_quant_scheme": "turboquant" if args.kv_bits is not None else None,
        "weights_size_mb": round(weights_size_mb, 2),
        "model_load_time_s": round(load_time_s, 4),
        "model_load_peak_device_memory_mb": round(load_peak_device_mb, 2),
        "process_peak_rss_mb": round(peak_rss_mb(), 2),
        "per_file": per_file,
    }

    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
