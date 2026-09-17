"""Evaluate weight quantization and TurboQuant KV-cache quantization on
Whisper-large-v3, individually and combined, across every .wav in --data-dir
-- and optionally faster-whisper (CTranslate2) as an external reference point.

Runs the MLX 2x2 matrix (weights: fp16 vs 4-bit) x (KV cache: none vs
TurboQuant 3.5-bit), each in its own subprocess (via run_one_config.py) so
timing and peak-memory numbers reflect one config in isolation rather than a
running max across models loaded back-to-back in one process. With
--include-faster-whisper, also runs faster-whisper (via
run_faster_whisper_config.py) at the given --faster-whisper-compute-types.

IMPORTANT: faster-whisper/CTranslate2 has no Metal backend, so on Apple
Silicon it always runs on CPU -- every MLX config here runs on GPU. This is
"the practical alternative someone would actually compare against," not an
apples-to-apples backend comparison. See ../README.md.

Writes a single JSON report with per-config, per-file metrics plus summary
deltas against the fp16/no-KV-quant MLX baseline (speed, memory, disk size,
and word error rate of each config's transcript against the baseline
transcript -- there's no ground-truth transcript for this audio, so the
baseline stands in as the reference for measuring how much a config moves
the output, not absolute accuracy).

Usage:
    python evaluate.py \
        --fp16-path ./mlx-models/whisper-large-v3-fp16 \
        --int4-path ./mlx-models/whisper-large-v3-4bit \
        --data-dir ../data \
        --include-faster-whisper \
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


def run_subprocess(cmd: list[str], config_name: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / "result.json"
        cmd = cmd + ["--output", str(out_file)]
        print(f"=== running {config_name} ===", flush=True)
        subprocess.run(cmd, check=True)
        return json.loads(out_file.read_text())


def run_mlx_config(script_dir: Path, config_name: str, mlx_path: str, kv_bits, data_dir: str, max_new_tokens: int) -> dict:
    cmd = [
        sys.executable,
        str(script_dir / "run_one_config.py"),
        "--config-name", config_name,
        "--mlx-path", mlx_path,
        "--data-dir", data_dir,
        "--max-new-tokens", str(max_new_tokens),
    ]
    if kv_bits is not None:
        cmd += ["--kv-bits", str(kv_bits)]
    report = run_subprocess(cmd, config_name)
    report["backend"] = "mlx (GPU/Metal)"
    return report


def run_faster_whisper_config(
    script_dir: Path, config_name: str, model_size: str, compute_type: str, beam_size: int, data_dir: str
) -> dict:
    cmd = [
        sys.executable,
        str(script_dir / "run_faster_whisper_config.py"),
        "--config-name", config_name,
        "--model-size", model_size,
        "--compute-type", compute_type,
        "--beam-size", str(beam_size),
        "--data-dir", data_dir,
    ]
    return run_subprocess(cmd, config_name)


def normalized(report: dict) -> dict:
    """A common view over MLX-shaped and faster-whisper-shaped reports, so the
    summary table below doesn't need to special-case every field."""
    return {
        "backend": report.get("backend", "?"),
        "checkpoint_size_mb": report.get("weights_size_mb", report.get("checkpoint_size_mb")),
        "model_load_time_s": report.get("model_load_time_s"),
        "model_load_peak_device_memory_mb": report.get("model_load_peak_device_memory_mb"),  # MLX only
        "process_peak_rss_mb": report.get("process_peak_rss_mb"),
    }


def avg(report: dict, key: str):
    vals = [f[key] for f in report["per_file"] if f.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp16-path", required=True, help="Unquantized (fp16 weights) MLX checkpoint dir.")
    parser.add_argument("--int4-path", required=True, help="4-bit weight-quantized MLX checkpoint dir.")
    parser.add_argument("--data-dir", required=True, help="Directory of .wav files to transcribe.")
    parser.add_argument("--kv-bits", type=float, default=3.5, help="TurboQuant bit width to test.")
    parser.add_argument("--max-new-tokens", type=int, default=224)
    parser.add_argument("--include-faster-whisper", action="store_true")
    parser.add_argument("--faster-whisper-model-size", default="large-v3")
    parser.add_argument("--faster-whisper-compute-types", default="float32,int8")
    parser.add_argument("--faster-whisper-beam-size", type=int, default=1)
    parser.add_argument("--output", default="report.json")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    mlx_configs = [
        ("fp16_baseline", args.fp16_path, None),
        ("weight_4bit_only", args.int4_path, None),
        ("turboquant_kv_only", args.fp16_path, args.kv_bits),
        ("weight_4bit_plus_turboquant_kv", args.int4_path, args.kv_bits),
    ]

    results = {}
    for name, mlx_path, kv_bits in mlx_configs:
        results[name] = run_mlx_config(script_dir, name, mlx_path, kv_bits, args.data_dir, args.max_new_tokens)

    if args.include_faster_whisper:
        for compute_type in args.faster_whisper_compute_types.split(","):
            name = f"faster_whisper_{compute_type}"
            results[name] = run_faster_whisper_config(
                script_dir, name, args.faster_whisper_model_size, compute_type, args.faster_whisper_beam_size, args.data_dir
            )

    baseline = results["fp16_baseline"]
    baseline_transcripts = {f["audio_file"]: f["transcript"] for f in baseline["per_file"]}

    for name, report in results.items():
        for f in report["per_file"]:
            ref = baseline_transcripts[f["audio_file"]]
            f["word_error_rate_vs_fp16_baseline"] = round(word_error_rate(ref, f["transcript"]), 4)

    summary = {}
    for name, report in results.items():
        n = normalized(report)
        summary[name] = {
            "backend": n["backend"],
            "checkpoint_size_mb": n["checkpoint_size_mb"],
            "model_load_time_s": n["model_load_time_s"],
            "model_load_peak_device_memory_mb": n["model_load_peak_device_memory_mb"],
            "process_peak_rss_mb": n["process_peak_rss_mb"],
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
        if s["checkpoint_size_mb"] and baseline_summary["checkpoint_size_mb"]:
            s["checkpoint_size_reduction_pct_vs_mlx_fp16"] = round(
                100 * (1 - s["checkpoint_size_mb"] / baseline_summary["checkpoint_size_mb"]), 2
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
        if s["model_load_peak_device_memory_mb"] and baseline_summary["model_load_peak_device_memory_mb"]:
            s["load_peak_memory_change_pct"] = round(
                100
                * (
                    s["model_load_peak_device_memory_mb"] / baseline_summary["model_load_peak_device_memory_mb"]
                    - 1
                ),
                2,
            )
        if s["process_peak_rss_mb"] and baseline_summary["process_peak_rss_mb"]:
            s["process_rss_change_pct_vs_mlx_fp16"] = round(
                100 * (s["process_peak_rss_mb"] / baseline_summary["process_peak_rss_mb"] - 1), 2
            )
        if s["avg_real_time_factor"] and baseline_summary["avg_real_time_factor"]:
            s["speed_change_pct_vs_mlx_fp16"] = round(
                100 * (s["avg_real_time_factor"] / baseline_summary["avg_real_time_factor"] - 1), 2
            )

    report = {
        "device": platform.platform(),
        "mlx_version": mx.__version__ if hasattr(mx, "__version__") else None,
        "model": "openai/whisper-large-v3",
        "data_dir": str(Path(args.data_dir).resolve()),
        "audio_files": sorted(f.name for f in Path(args.data_dir).glob("*.wav")),
        "kv_bits_tested": args.kv_bits,
        "decode_settings": {
            "mlx_configs": {"max_new_tokens": args.max_new_tokens, "decoding": "greedy, single 30s chunk"},
            "faster_whisper_configs": {"beam_size": args.faster_whisper_beam_size, "temperature": 0.0}
            if args.include_faster_whisper
            else None,
        },
        "summary": summary,
        "configs": results,
    }

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"\nWrote report to {args.output}")

    print("\n=== Summary ===")
    for name, s in summary.items():
        print(f"\n{name} ({s['backend']}):")
        for k, v in s.items():
            if k == "backend":
                continue
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
