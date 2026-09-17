"""Run one faster-whisper (CTranslate2) config over a directory of audio
files and dump timing/memory/quality metrics to a JSON file, in the same
shape run_one_config.py produces for the MLX configs, so evaluate.py can
merge both into one report.

IMPORTANT hardware caveat: CTranslate2 has no Metal backend, so on Apple
Silicon faster-whisper always runs on CPU, never the GPU that MLX uses for
every other config in this report. This script is not measuring "another
way to quantize the same GPU inference path" -- it's measuring the most
popular optimized Whisper *runtime*, on the hardware it actually runs on
here (CPU). Treat this as "the practical alternative someone would compare
against," not an apples-to-apples backend comparison. See ../README.md.

Also note: unlike convert_from_hf.py's --quantize (which writes an actually
smaller weights.safetensors to disk), CTranslate2's `compute_type` quantizes
*at load time* from the same on-disk checkpoint -- "int8" and "float32" here
read identical files and produce identical download/disk sizes; only the
in-memory representation and compute path differ.

Usage:
    python run_faster_whisper_config.py --config-name faster_whisper_int8 \
        --compute-type int8 --data-dir ../data --output result.json
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024**2) if sys.platform == "darwin" else r / 1024


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--model-size", default="large-v3")
    parser.add_argument("--compute-type", required=True, choices=["float32", "int8", "int8_float32"])
    parser.add_argument("--cpu-threads", type=int, default=0, help="0 = ctranslate2 default")
    parser.add_argument("--beam-size", type=int, default=1, help="1 = greedy, to match the MLX configs' decode loop")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    audio_files = sorted(Path(args.data_dir).glob("*.wav"))
    if not audio_files:
        raise SystemExit(f"No .wav files found in {args.data_dir}")

    t0 = time.perf_counter()
    model = WhisperModel(
        args.model_size, device="cpu", compute_type=args.compute_type, cpu_threads=args.cpu_threads
    )
    load_time_s = time.perf_counter() - t0

    checkpoint_size_mb = None
    try:
        from huggingface_hub import snapshot_download

        model_dir = Path(snapshot_download(f"Systran/faster-whisper-{args.model_size}"))
        checkpoint_size_mb = dir_size_mb(model_dir)
    except Exception:
        pass  # not fatal -- disk size is a secondary metric here

    per_file = []
    for audio_path in audio_files:
        import soundfile as sf

        duration_s = sf.info(str(audio_path)).duration

        t0 = time.perf_counter()
        segments, info = model.transcribe(
            str(audio_path),
            language="en",
            task="transcribe",
            beam_size=args.beam_size,
            temperature=0.0,
            without_timestamps=True,
            vad_filter=False,
        )
        segments = list(segments)  # the generator is lazy; consume it to actually run inference
        infer_time_s = time.perf_counter() - t0

        text = "".join(s.text for s in segments)
        n_tokens = sum(len(s.tokens) for s in segments)

        per_file.append(
            {
                "audio_file": audio_path.name,
                "audio_duration_s": round(duration_s, 3),
                "transcript": text,
                "decoded_tokens": n_tokens,
                "inference_time_s": round(infer_time_s, 4),
                "real_time_factor": round(duration_s / infer_time_s, 3) if infer_time_s > 0 else None,
                "tokens_per_second": round(n_tokens / infer_time_s, 2) if infer_time_s > 0 and n_tokens else None,
            }
        )

    report = {
        "config_name": args.config_name,
        "backend": "faster-whisper (CTranslate2, CPU)",
        "model_size": args.model_size,
        "compute_type": args.compute_type,
        "beam_size": args.beam_size,
        "checkpoint_size_mb": round(checkpoint_size_mb, 2) if checkpoint_size_mb else None,
        "model_load_time_s": round(load_time_s, 4),
        "process_peak_rss_mb": round(peak_rss_mb(), 2),
        "per_file": per_file,
    }

    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
