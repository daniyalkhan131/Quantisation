"""Run mlx-vlm's 2x2 matrix (weights: fp16 vs 4-bit) x (KV cache: none vs
mlx-vlm's real TurboQuant) on google/gemma-4-E4B-it, and compare against the
PyTorch/transformers + vivekvar/turboquant numbers already in report.json
(from evaluate_gemma4.py) -- to answer "how much speed gain and memory
reduction does mlx-vlm give."

Each config runs in its own subprocess (run_one_mlx_config.py) for isolated
peak-memory accounting, same reasoning as evaluate_gemma4.py.

Usage:
    python evaluate_mlx_backend.py --kv-bits 3.5 --max-tokens 300 \
        --pytorch-report report.json --output mlx_report.json
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


def run_config(script_dir: Path, config_name: str, weights: str, kv_bits, max_tokens: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / "result.json"
        cmd = [
            sys.executable,
            str(script_dir / "run_one_mlx_config.py"),
            "--config-name", config_name,
            "--weights", weights,
            "--max-tokens", str(max_tokens),
            "--output", str(out_file),
        ]
        if kv_bits is not None:
            cmd += ["--kv-bits", str(kv_bits), "--quantized-kv-start", "0"]
        print(f"=== running {config_name} ===", flush=True)
        subprocess.run(cmd, check=True)
        return json.loads(out_file.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kv-bits", type=float, default=3.5)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--pytorch-report", default="report.json", help="Existing PyTorch-backend report.json to compare against.")
    parser.add_argument("--output", default="mlx_report.json")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    configs = [
        ("mlx_fp16_baseline", "fp16", None),
        ("mlx_4bit_only", "4bit", None),
        ("mlx_fp16_turboquant", "fp16", args.kv_bits),
        ("mlx_4bit_turboquant", "4bit", args.kv_bits),
    ]

    results = {}
    for name, weights, kv_bits in configs:
        results[name] = run_config(script_dir, name, weights, kv_bits, args.max_tokens)

    baseline = results["mlx_fp16_baseline"]
    baseline_by_prompt = {p["prompt_name"]: p for p in baseline["per_prompt"]}

    for name, report in results.items():
        for p in report["per_prompt"]:
            ref = baseline_by_prompt[p["prompt_name"]]["generated_text"]
            p["word_error_rate_vs_mlx_fp16_baseline"] = round(word_error_rate(ref, p["generated_text"]), 4)

    def avg(report, key):
        vals = [p[key] for p in report["per_prompt"] if p.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {}
    for name, report in results.items():
        summary[name] = {
            "weights_size_mb": report["weights_size_mb"],
            "model_load_time_s": report["model_load_time_s"],
            "model_load_peak_device_memory_mb": report["model_load_peak_device_memory_mb"],
            "process_peak_rss_mb": report["process_peak_rss_mb"],
            "avg_generation_tps": avg(report, "generation_tps"),
            "avg_generation_peak_device_memory_mb": avg(report, "generation_peak_device_memory_mb"),
            "avg_word_error_rate_vs_mlx_fp16_baseline": avg(report, "word_error_rate_vs_mlx_fp16_baseline"),
        }

    base_s = summary["mlx_fp16_baseline"]
    for name, s in summary.items():
        if name == "mlx_fp16_baseline":
            continue
        s["weights_size_reduction_pct"] = round(100 * (1 - s["weights_size_mb"] / base_s["weights_size_mb"]), 2)
        s["load_peak_memory_change_pct"] = round(
            100 * (s["model_load_peak_device_memory_mb"] / base_s["model_load_peak_device_memory_mb"] - 1), 2
        )
        s["speed_change_pct_vs_mlx_fp16"] = round(
            100 * (s["avg_generation_tps"] / base_s["avg_generation_tps"] - 1), 2
        )

    # Cross-backend comparison against the earlier PyTorch/transformers run,
    # if available.
    cross_backend = None
    pytorch_path = Path(args.pytorch_report)
    if pytorch_path.exists():
        pt = json.loads(pytorch_path.read_text())
        pt_baseline = pt["summary"]["baseline"]
        pt_by_prompt = {c["prompt_name"]: c for c in pt["per_prompt_comparison"]}
        pt_configs = pt["configs"]
        pt_baseline_prompts = {p["prompt_name"]: p for p in pt_configs["baseline"]["per_prompt"]}
        pt_turbo_prompts = {p["prompt_name"]: p for p in pt_configs["turboquant"]["per_prompt"]}

        cross_backend = {
            "model_load": {
                "pytorch_mps_load_time_s": pt_baseline["model_load_time_s"],
                "mlx_load_time_s": base_s["model_load_time_s"],
                "load_time_speedup_x": round(pt_baseline["model_load_time_s"] / base_s["model_load_time_s"], 2),
                "pytorch_mps_peak_device_memory_mb": pt_baseline["model_load_peak_device_memory_mb"],
                "mlx_peak_device_memory_mb": base_s["model_load_peak_device_memory_mb"],
            },
            "per_prompt": [],
        }
        for prompt_name in baseline_by_prompt:
            pt_base_p = pt_baseline_prompts[prompt_name]
            pt_turbo_p = pt_turbo_prompts[prompt_name]
            mlx_base_p = baseline_by_prompt[prompt_name]
            mlx_turbo_p = {p["prompt_name"]: p for p in results["mlx_fp16_turboquant"]["per_prompt"]}[prompt_name]
            mlx_4bit_p = {p["prompt_name"]: p for p in results["mlx_4bit_only"]["per_prompt"]}[prompt_name]
            cross_backend["per_prompt"].append(
                {
                    "prompt_name": prompt_name,
                    "pytorch_mps_fp16_tok_s": pt_base_p["tokens_per_second"],
                    "mlx_fp16_tok_s": mlx_base_p["generation_tps"],
                    "mlx_fp16_speedup_x_vs_pytorch": round(mlx_base_p["generation_tps"] / pt_base_p["tokens_per_second"], 2),
                    "mlx_4bit_tok_s": mlx_4bit_p["generation_tps"],
                    "mlx_4bit_speedup_x_vs_pytorch": round(mlx_4bit_p["generation_tps"] / pt_base_p["tokens_per_second"], 2),
                    "pytorch_reference_turboquant_tok_s": pt_turbo_p["tokens_per_second"],
                    "mlx_real_turboquant_tok_s": mlx_turbo_p["generation_tps"],
                    "mlx_turboquant_speedup_x_vs_pytorch_reference_turboquant": round(
                        mlx_turbo_p["generation_tps"] / pt_turbo_p["tokens_per_second"], 2
                    ),
                }
            )

    report = {
        "device": platform.platform(),
        "mlx_version": mx.__version__ if hasattr(mx, "__version__") else None,
        "model": "google/gemma-4-E4B-it",
        "kv_bits_tested": args.kv_bits,
        "summary": summary,
        "cross_backend_vs_pytorch": cross_backend,
        "configs": results,
    }

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"\nWrote report to {args.output}")

    print("\n=== Summary ===")
    for name, s in summary.items():
        print(f"\n{name}:")
        for k, v in s.items():
            print(f"  {k}: {v}")

    if cross_backend:
        print("\n=== Cross-backend (mlx-vlm vs PyTorch/MPS) ===")
        print(json.dumps(cross_backend, indent=2))


if __name__ == "__main__":
    main()
