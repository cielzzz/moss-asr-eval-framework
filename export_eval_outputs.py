#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


SUMMARY_FIELDS = [
    "asr_backend",
    "benchmark",
    "tts_system",
    "lang",
    "n",
    "avg_cer",
    "avg_wer",
    "avg_pinyin_cer",
    "avg_pinyin_wer",
    "avg_pinyin_notone_cer",
    "avg_pinyin_notone_wer",
    "avg_sim",
]

PREDICTION_FIELDS = [
    "utt_id",
    "benchmark",
    "tts_system",
    "ref_text",
    "hyp_text",
    "audio_path",
    "lang",
    "cer",
    "wer",
    "pinyin_cer",
    "pinyin_wer",
    "pinyin_notone_cer",
    "pinyin_notone_wer",
]

ERROR_FIELDS = [
    "utt_id",
    "benchmark",
    "tts_system",
    "lang",
    "audio_path",
    "ref_text",
    "hyp_text",
    "error",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def average(rows: List[Dict[str, Any]], key: str) -> float | None:
    vals = [row.get(key) for row in rows if row.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def grouped_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("asr_backend", "")),
            str(row.get("benchmark", row.get("tts_model", ""))),
            str(row.get("tts_system", "")),
            str(row.get("lang", "")),
        )
        groups.setdefault(key, []).append(row)

    summary: List[Dict[str, Any]] = []
    for (asr_backend, benchmark, tts_system, lang), group_rows in sorted(groups.items()):
        summary.append({
            "asr_backend": asr_backend,
            "benchmark": benchmark,
            "tts_system": tts_system,
            "lang": lang,
            "n": len(group_rows),
            "avg_cer": average(group_rows, "cer"),
            "avg_wer": average(group_rows, "wer"),
            "avg_pinyin_cer": average(group_rows, "pinyin_cer"),
            "avg_pinyin_wer": average(group_rows, "pinyin_wer"),
            "avg_pinyin_notone_cer": average(group_rows, "pinyin_notone_cer"),
            "avg_pinyin_notone_wer": average(group_rows, "pinyin_notone_wer"),
            "avg_sim": average(group_rows, "sim"),
        })
    return summary


def write_summary_csv(path: Path, summary_rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({
                key: ("nan" if row.get(key) is None else row.get(key))
                for key in SUMMARY_FIELDS
            })


def write_metrics_json(path: Path, summary_rows: List[Dict[str, Any]]) -> None:
    metrics: Dict[str, Dict[str, Any]] = {}
    for row in summary_rows:
        key = str(row.get("lang", ""))
        metrics[key] = row
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def write_errors_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_FIELDS)
        writer.writeheader()
        for row in rows:
            if row.get("error"):
                writer.writerow({key: row.get(key, "") for key in ERROR_FIELDS})


def write_predictions(path: Path, rows: List[Dict[str, Any]]) -> None:
    write_jsonl(
        path,
        ({key: row.get(key) for key in PREDICTION_FIELDS if key in row} for row in rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export compact ASR eval artifacts from results.jsonl.")
    parser.add_argument("--results", required=True, help="Path to full results.jsonl produced by asr_eval.py")
    parser.add_argument("--out-dir", required=True, help="Directory to write predictions/summary/metrics/errors")
    args = parser.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(results_path)
    summary_rows = grouped_summary(rows)

    write_predictions(out_dir / "predictions.jsonl", rows)
    write_summary_csv(out_dir / "summary.csv", summary_rows)
    write_metrics_json(out_dir / "metrics.json", summary_rows)
    write_errors_csv(out_dir / "errors.csv", rows)

    print(f"Read {len(rows)} rows from {results_path}")
    print(f"Wrote predictions.jsonl, summary.csv, metrics.json, errors.csv to {out_dir}")


if __name__ == "__main__":
    main()
