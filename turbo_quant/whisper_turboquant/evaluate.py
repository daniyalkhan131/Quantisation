"""Evaluate weight quantization and TurboQuant KV-cache quantization on
Whisper-large-v3, individually and combined, across every .wav in --data-dir.

Runs a 2x2 matrix (weights: fp16 vs 4-bit) x (KV cache: none vs TurboQuant
3.5-bit), each in its own subprocess (via run_one_config.py) so timing and
peak-memory numbers reflect one config in isolation rather than a running
max across models loaded back-to-back in one process. Writes a single JSON
report with per-config, per-file metrics plus summary deltas against the
fp16/no-KV-quant baseline (speed, memory, disk size, and word error rate of
the quantized transcript against the baseline transcript -- there's no
ground-truth transcript for this audio, so the baseline stands in as the
reference for measuring how much quantization moves the output, not
absolute accuracy).

Usage:
    python evaluate.py \
        --fp16-path ./mlx-models/whisper-large-v3-fp16 \
        --int4-path ./mlx-models/whisper-large-v3-4bit \
        --data-dir ../data \
        --output report.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard word-level Levenshtein WER: edit_distance / len(reference)."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0 if not hyp else 1.0

    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        curr = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1] / len(ref)


def run_config(script_dir: Path, config_name: str, mlx_path: str, kv_bits, data_dir: str, max_new_tokens: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / "result.json"
        cmd = [
            sys.executable,
            str(script_dir / "run_one_config.py"),
            "--config-name", config_name,
            "--mlx-path", mlx_path,
            "--data-dir", data_dir,
            "--max-new-tokens", str(max_new_tokens),
            "--output", str(out_file),
        ]
        if kv_bits is not None:
            cmd += ["--kv-bits", str(kv_bits)]
        print(f"=== running {config_name} ===", flush=True)
        subprocess.run(cmd, check=True)
        return json.loads(out_file.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp16-path", required=True, help="Unquantized (fp16 weights) MLX checkpoint dir.")
    parser.add_argument("--int4-path", required=True, help="4-bit weight-quantized MLX checkpoint dir.")
    parser.add_argument("--data-dir", required=True, help="Directory of .wav files to transcribe.")
    parser.add_argument("--kv-bits", type=float, default=3.5, help="TurboQuant bit width to test.")
    parser.add_argument("--max-new-tokens", type=int, default=224)
    parser.add_argument("--output", default="report.json")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    configs = [
        ("fp16_baseline", args.fp16_path, None),
        ("weight_4bit_only", args.int4_path, None),
        ("turboquant_kv_only", args.fp16_path, args.kv_bits),
        ("weight_4bit_plus_turboquant_kv", args.int4_path, args.kv_bits),
    ]

    results = {}
    for name, mlx_path, kv_bits in configs:
        results[name] = run_config(script_dir, name, mlx_path, kv_bits, args.data_dir, args.max_new_tokens)

    baseline = results["fp16_baseline"]
    baseline_transcripts = {f["audio_file"]: f["transcript"] for f in baseline["per_file"]}

    for name, report in results.items():
        for f in report["per_file"]:
            ref = baseline_transcripts[f["audio_file"]]
            f["word_error_rate_vs_fp16_baseline"] = round(word_error_rate(ref, f["transcript"]), 4)

    def avg(report, key):
        vals = [f[key] for f in report["per_file"] if f.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {}
    for name, report in results.items():
        summary[name] = {
            "weights_size_mb": report["weights_size_mb"],
            "model_load_time_s": report["model_load_time_s"],
            "model_load_peak_device_memory_mb": report["model_load_peak_device_memory_mb"],
            "process_peak_rss_mb": report["process_peak_rss_mb"],
            "avg_real_time_factor": avg(report, "real_time_factor"),
            "avg_tokens_per_second": avg(report, "tokens_per_second"),
            "avg_inference_peak_device_memory_mb": avg(report, "inference_peak_device_memory_mb"),
            "avg_self_attn_kv_cache_allocated_bytes": avg(report, "self_attn_kv_cache_allocated_bytes"),
            "avg_word_error_rate_vs_fp16_baseline": avg(report, "word_error_rate_vs_fp16_baseline"),
        }

    baseline_summary = summary["fp16_baseline"]
    for name, s in summary.items():
        if name == "fp16_baseline":
            continue
        s["weights_size_reduction_pct"] = round(
            100 * (1 - s["weights_size_mb"] / baseline_summary["weights_size_mb"]), 2
        )
        if s["avg_self_attn_kv_cache_allocated_bytes"] and baseline_summary["avg_self_attn_kv_cache_allocated_bytes"]:
            s["kv_cache_reduction_pct"] = round(
                100
                * (
                    1
                    - s["avg_self_attn_kv_cache_allocated_bytes"]
                    / baseline_summary["avg_self_attn_kv_cache_allocated_bytes"]
                ),
                2,
            )
        if s["model_load_peak_device_memory_mb"]:
            s["load_peak_memory_change_pct"] = round(
                100
                * (
                    s["model_load_peak_device_memory_mb"] / baseline_summary["model_load_peak_device_memory_mb"]
                    - 1
                ),
                2,
            )

    report = {
        "device": platform.platform(),
        "mlx_version": mx.__version__ if hasattr(mx, "__version__") else None,
        "model": "openai/whisper-large-v3",
        "data_dir": str(Path(args.data_dir).resolve()),
        "audio_files": sorted(f.name for f in Path(args.data_dir).glob("*.wav")),
        "kv_bits_tested": args.kv_bits,
        "decode_settings": {"max_new_tokens": args.max_new_tokens, "decoding": "greedy, single 30s chunk"},
        "summary": summary,
        "configs": results,
    }

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"\nWrote report to {args.output}")

    print("\n=== Summary ===")
    for name, s in summary.items():
        print(f"\n{name}:")
        for k, v in s.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
