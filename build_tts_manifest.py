#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable


def infer_lang(task: str) -> str:
    task = task.lower()
    if task.endswith("-en") or "-en" in task:
        return "en"
    if task.endswith("-zh") or task.endswith("-ch") or "-zh" in task or "-ch" in task:
        return "zh"
    return ""


def is_eval_task(task: str) -> bool:
    low = task.lower()
    if "hard" in low:
        return False
    return infer_lang(low) in {"zh", "en"}


def resolve_pred_wav(tts_root: Path, task: str, utt_id: str) -> Path:
    pred_wav = tts_root / task / utt_id / "pred.wav"
    if pred_wav.exists():
        return pred_wav

    match = re.fullmatch(r"zero_shot_uttid_(\d+)", utt_id)
    if match:
        pred_wav = tts_root / task / match.group(1) / "pred.wav"
        if pred_wav.exists():
            return pred_wav

    return tts_root / task / utt_id / "pred.wav"


def iter_rows(input_jsonl: Path, tts_root: Path, benchmark: str, tts_system: str) -> Iterable[Dict[str, Any]]:
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task = str(obj.get("task", ""))
            if not is_eval_task(task):
                continue

            utt_id = str(obj["id"])
            pred_wav = resolve_pred_wav(tts_root, task, utt_id)
            if not pred_wav.exists():
                raise FileNotFoundError(f"Missing pred.wav for line {line_no}: {pred_wav}")

            prompt_wav = str(obj["audio_path"])
            yield {
                "id": utt_id,
                "audio_path": str(pred_wav),
                "reference": str(obj["text"]),
                "lang": infer_lang(task),
                "benchmark": benchmark,
                "tts_system": tts_system,
                "meta": {
                    "task": task,
                    "prompt_audio_path": prompt_wav,
                    "prompt_text": obj.get("ref_text", ""),
                    "source_manifest": str(input_jsonl),
                },
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--tts-root", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--tts-system", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    tts_root = Path(args.tts_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out.open("w", encoding="utf-8") as f:
        tts_system = args.tts_system or args.benchmark
        for row in iter_rows(input_jsonl, tts_root, args.benchmark, tts_system):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} rows to {out}")


if __name__ == "__main__":
    main()
