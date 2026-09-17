"""Compare vivekvar/turboquant's TurboQuantCache against a plain HF
DynamicCache on google/gemma-4-E4B-it: speed, memory, and how much the
generated text diverges once quantization actually kicks in.

Runs run_one_llm_config.py twice, once per config, each in its own
subprocess so model load and peak-memory accounting start fresh (rather than
a running max across two models loaded back-to-back in one process -- same
reasoning as whisper_turboquant/evaluate.py).

See README.md for why the prompts are structured the way they are (one short
prompt that never leaves TurboQuant's full-precision residual window, one
long one that does), and for the full set of caveats about this comparison.

Usage:
    python evaluate_gemma4.py --nbits 4 --max-new-tokens 300 --output report.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


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


def first_divergence(ids_a: list[int], ids_b: list[int]) -> int | None:
    for i, (a, b) in enumerate(zip(ids_a, ids_b)):
        if a != b:
            return i
    return None if len(ids_a) == len(ids_b) else min(len(ids_a), len(ids_b))


def run_config(script_dir: Path, config_name: str, nbits: int, max_new_tokens: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / "result.json"
        cmd = [
            sys.executable,
            str(script_dir / "run_one_llm_config.py"),
            "--config-name", config_name,
            "--nbits", str(nbits),
            "--max-new-tokens", str(max_new_tokens),
            "--output", str(out_file),
        ]
        print(f"=== running {config_name} ===", flush=True)
        subprocess.run(cmd, check=True)
        return json.loads(out_file.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nbits", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--output", default="report.json")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    baseline = run_config(script_dir, "baseline", args.nbits, args.max_new_tokens)
    turboquant = run_config(script_dir, "turboquant", args.nbits, args.max_new_tokens)

    baseline_by_prompt = {p["prompt_name"]: p for p in baseline["per_prompt"]}
    comparisons = []
    for p in turboquant["per_prompt"]:
        base_p = baseline_by_prompt[p["prompt_name"]]
        divergence_idx = first_divergence(base_p["generated_token_ids"], p["generated_token_ids"])
        comparisons.append(
            {
                "prompt_name": p["prompt_name"],
                "prompt_tokens": p["prompt_tokens"],
                "new_tokens": p["new_tokens"],
                "first_divergence_token_index": divergence_idx,
                "word_error_rate_vs_baseline": round(
                    word_error_rate(base_p["generated_text"], p["generated_text"]), 4
                ),
                "baseline_tokens_per_second": base_p["tokens_per_second"],
                "turboquant_tokens_per_second": p["tokens_per_second"],
                "speed_change_pct": round(
                    100 * (p["tokens_per_second"] / base_p["tokens_per_second"] - 1), 2
                )
                if base_p["tokens_per_second"]
                else None,
                "baseline_cache_bytes": base_p["final_cache_bytes"],
                "turboquant_cache_bytes": p["final_cache_bytes"],
                "cache_reduction_pct": round(
                    100 * (1 - p["final_cache_bytes"] / base_p["final_cache_bytes"]), 2
                )
                if base_p["final_cache_bytes"]
                else None,
            }
        )

    report = {
        "device": platform.platform(),
        "torch_version": torch.__version__,
        "model": baseline["model"],
        "nbits_tested": args.nbits,
        "skip_layers": turboquant["skip_layers"],
        "calibration_time_s": turboquant["calibration_time_s"],
        "summary": {
            "baseline": {
                "model_load_time_s": baseline["model_load_time_s"],
                "model_load_peak_device_memory_mb": baseline["model_load_peak_device_memory_mb"],
                "process_peak_rss_mb": baseline["process_peak_rss_mb"],
            },
            "turboquant": {
                "model_load_time_s": turboquant["model_load_time_s"],
                "model_load_peak_device_memory_mb": turboquant["model_load_peak_device_memory_mb"],
                "process_peak_rss_mb": turboquant["process_peak_rss_mb"],
            },
        },
        "per_prompt_comparison": comparisons,
        "configs": {"baseline": baseline, "turboquant": turboquant},
    }

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"\nWrote report to {args.output}")

    print("\n=== Per-prompt comparison ===")
    for c in comparisons:
        print(f"\n{c['prompt_name']} ({c['prompt_tokens']} prompt tokens, {c['new_tokens']} generated):")
        for k, v in c.items():
            if k == "prompt_name":
                continue
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
