#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASR evaluation script for TTS benchmark.

Core flow:
  manifest.jsonl -> ASR transcription -> normalization -> CER/WER + pinyin CER/WER -> jsonl/csv report

Manifest line format:
  {
    "id": "case_001",
    "audio_path": "/path/to/audio.wav",
    "reference": "银行行长来到重庆。",
    "lang": "zh",
    "benchmark": "seed-tts-eval",
    "tts_system": "your_tts",
    "meta": {}
  }
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
import yaml
from jiwer import wer
from pypinyin import lazy_pinyin, Style
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class EvalItem:
    id: str
    audio_path: str
    reference: str
    lang: str = "zh"
    benchmark: str = ""
    tts_system: str = ""
    meta: Dict[str, Any] | None = None


@dataclass
class ASRBackend:
    name: str
    provider: str
    params: Dict[str, Any]


# ---------------------------
# Text normalization
# ---------------------------

ZH_PUNCT = "，。！？；：“”‘’、（）《》【】—…·"
EN_PUNCT = string.punctuation
# 一次性编译正则，避免在热路径中重复构建
PUNCT_RE = re.compile(f"[{re.escape(ZH_PUNCT + EN_PUNCT)}]")

def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_spaces_for_cer: bool = False,
) -> str:
    """Normalize transcript/reference before metric calculation."""
    if text is None:
        return ""
    text = str(text).strip()

    # 全角空格 \u3000 / 连续空白 → 单个半角空格，消除 ASR 输出格式差异
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = PUNCT_RE.sub("", text)

    text = text.strip()

    # CER 按字符逐一比较，空格对中文无语义，去掉避免中英混排时错误切分
    if remove_spaces_for_cer:
        text = text.replace(" ", "")

    return text


def char_error_rate(ref: str, hyp: str) -> float:
    """CER using jiwer.wer over character tokens."""
    # jiwer.wer 以空格分词，把字符串拆成单字列表再用空格拼接，即可复用 WER 逻辑算 CER
    ref_chars = " ".join(list(ref))
    hyp_chars = " ".join(list(hyp))
    if not ref_chars.strip():
        return 0.0 if not hyp_chars.strip() else 1.0
    return wer(ref_chars, hyp_chars)


def word_error_rate(ref: str, hyp: str) -> float:
    """WER for whitespace-tokenized text. For Chinese, prefer CER or pinyin CER/WER."""
    if not ref.strip():
        return 0.0 if not hyp.strip() else 1.0
    return wer(ref, hyp)


def to_pinyin(
    text: str,
    *,
    style_name: str = "TONE3",
    neutral_tone_with_five: bool = True,
) -> List[str]:
    """Convert Chinese text to pinyin tokens.

    TONE3: 中国 -> ["zhong1", "guo2"]
    NORMAL: 中国 -> ["zhong", "guo"]
    """
    style = getattr(Style, style_name.upper(), Style.TONE3)
    # pypinyin 多音字依赖词组上下文，精度有限；真实评测可替换为 g2pW 或自研 G2P
    return lazy_pinyin(
        text,
        style=style,
        neutral_tone_with_five=neutral_tone_with_five,
        errors="ignore",  # 非汉字（数字、英文）直接跳过，不产生拼音 token
    )


def pinyin_error_rates(
    ref: str,
    hyp: str,
    *,
    style_name: str = "TONE3",
    prefix: str = "pinyin",
) -> Dict[str, Any]:
    ref_py = to_pinyin(ref, style_name=style_name)
    hyp_py = to_pinyin(hyp, style_name=style_name)

    ref_py_str = " ".join(ref_py)
    hyp_py_str = " ".join(hyp_py)

    py_wer = word_error_rate(ref_py_str, hyp_py_str)
    py_cer = char_error_rate("".join(ref_py), "".join(hyp_py))

    return {
        f"{prefix}_wer": py_wer,
        f"{prefix}_cer": py_cer,
        f"ref_{prefix}": ref_py_str,
        f"hyp_{prefix}": hyp_py_str,
    }


def chinese_pinyin_error_rates(ref: str, hyp: str, *, tone_style_name: str = "TONE3") -> Dict[str, Any]:
    """Return pinyin error rates with and without tones.

    `pinyin_*` keeps backward-compatible tone-bearing TONE3 fields.
    `pinyin_notone_*` is useful when tone mistakes should not count as phone errors.
    """
    result = pinyin_error_rates(ref, hyp, style_name=tone_style_name, prefix="pinyin")
    result.update(pinyin_error_rates(ref, hyp, style_name="NORMAL", prefix="pinyin_notone"))
    return result


# ---------------------------
# ASR adapters
# ---------------------------

def nested_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def transcribe_http(audio_path: str, backend: ASRBackend) -> str:
    """Generic HTTP ASR API adapter.

    Expected:
      backend.params.endpoint
      backend.params.api_key_env optional
      backend.params.file_field default "file"
      backend.params.response_text_path default "text"
    """
    endpoint = backend.params["endpoint"]
    api_key_env = backend.params.get("api_key_env")
    file_field = backend.params.get("file_field", "file")
    response_text_path = backend.params.get("response_text_path", "text")

    headers = {}
    if api_key_env:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env: {api_key_env}")
        headers["Authorization"] = f"Bearer {api_key}"

    with open(audio_path, "rb") as f:
        resp = requests.post(endpoint, headers=headers, files={file_field: f}, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    text = nested_get(data, response_text_path)
    if text is None:
        raise RuntimeError(f"Cannot find response text at path={response_text_path}; response={data}")
    return str(text)


def transcribe_funasr_local(audio_path: str, backend: ASRBackend) -> str:
    """FunASR local adapter.

    Install example:
      pip install funasr modelscope

    Then replace this function with your preferred loading pattern.
    Keeping imports inside function avoids requiring FunASR for non-FunASR runs.
    """
    try:
        from funasr import AutoModel
    except Exception as e:
        raise RuntimeError("Please install FunASR first: pip install funasr modelscope") from e

    model_name = backend.params.get("model", "paraformer-zh")
    vad_model = backend.params.get("vad_model", "fsmn-vad")
    punc_model = backend.params.get("punc_model", "ct-punc")

    # 每次调用都重新加载模型，仅适合小批量调试；批量评测应在 backend 级做模型缓存
    model = AutoModel(model=model_name, vad_model=vad_model, punc_model=punc_model)
    res = model.generate(input=audio_path)
    if isinstance(res, list) and res:
        return str(res[0].get("text", ""))
    return str(res)


_QWEN_BACKEND_CACHE: Dict[str, Any] = {}


def _load_qwen_backend_class() -> Any:
    """Load package backend by file path to avoid the legacy asr_eval.py name clash."""
    module_path = PROJECT_ROOT / "asr_eval" / "backends" / "qwen_asr.py"
    spec = importlib.util.spec_from_file_location("asr_eval_qwen_backend", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Qwen ASR backend from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.QwenASRBackend


def transcribe_qwen_local(audio_path: str, backend: ASRBackend) -> Dict[str, Any]:
    """Qwen3-ASR local adapter.

    Uses the official qwen-asr package and caches the loaded model per configured
    backend name, so a batch run does not reload weights for every utterance.
    """
    cache_key = backend.name
    if cache_key not in _QWEN_BACKEND_CACHE:
        cls = _load_qwen_backend_class()
        _QWEN_BACKEND_CACHE[cache_key] = cls.from_params(
            backend.params,
            backend_name=backend.params.get("backend_name", backend.name),
        )
    return _QWEN_BACKEND_CACHE[cache_key].transcribe(audio_path)


def transcribe_dummy(audio_path: str, backend: ASRBackend) -> str:
    """Debug adapter: returns text from sidecar .txt, or empty string.

    把 audio.wav 替换为 audio.txt 读取预置文本，无需真实音频即可跑通整条 pipeline 逻辑。
    """
    txt = Path(audio_path).with_suffix(".txt")
    return txt.read_text(encoding="utf-8").strip() if txt.exists() else ""


def transcribe(audio_path: str, backend: ASRBackend) -> Any:
    provider = backend.provider.lower()
    if provider == "http":
        return transcribe_http(audio_path, backend)
    if provider == "funasr-local":
        return transcribe_funasr_local(audio_path, backend)
    if provider in {"qwen-local", "qwen_asr", "qwen-asr"}:
        return transcribe_qwen_local(audio_path, backend)
    if provider == "dummy":
        return transcribe_dummy(audio_path, backend)
    raise ValueError(f"Unsupported ASR provider: {backend.provider}")


# ---------------------------
# IO
# ---------------------------

def load_manifest(path: str) -> List[EvalItem]:
    items: List[EvalItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for k in ["id", "audio_path", "reference"]:
                if k not in obj:
                    raise ValueError(f"Manifest line {ln} missing required field: {k}")
            items.append(
                EvalItem(
                    id=str(obj["id"]),
                    audio_path=str(obj["audio_path"]),
                    reference=str(obj["reference"]),
                    lang=str(obj.get("lang", "zh")),
                    benchmark=str(obj.get("benchmark", obj.get("tts_model", obj.get("tts_system", "")))),
                    tts_system=str(obj.get("tts_system", obj.get("tts_model", obj.get("benchmark", "")))),
                    meta=obj.get("meta", {}),
                )
            )
    return items


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_backends(config: Dict[str, Any]) -> List[ASRBackend]:
    backends = []
    for b in config.get("asr_backends", []):
        name = b["name"]
        provider = b["provider"]
        params = {k: v for k, v in b.items() if k not in {"name", "provider"}}
        backends.append(ASRBackend(name=name, provider=provider, params=params))
    if not backends:
        raise ValueError("No asr_backends found in config.")
    return backends


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    # 按 (asr_backend, benchmark, tts_system, lang) 分组，支持多 ASR × 多 benchmark × 多 TTS 对比
    group: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        key = (
            r["asr_backend"],
            r.get("benchmark", r.get("tts_model", "")),
            r.get("tts_system", ""),
            r.get("lang", ""),
        )
        group.setdefault(key, []).append(r)

    fields = [
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

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (asr_backend, benchmark, tts_system, lang), rs in sorted(group.items()):
            def avg(key: str) -> float:
                vals = [x.get(key) for x in rs if x.get(key) is not None]
                return sum(vals) / len(vals) if vals else float("nan")

            w.writerow({
                "asr_backend": asr_backend,
                "benchmark": benchmark,
                "tts_system": tts_system,
                "lang": lang,
                "n": len(rs),
                "avg_cer": avg("cer"),
                "avg_wer": avg("wer"),
                "avg_pinyin_cer": avg("pinyin_cer"),
                "avg_pinyin_wer": avg("pinyin_wer"),
                "avg_pinyin_notone_cer": avg("pinyin_notone_cer"),
                "avg_pinyin_notone_wer": avg("pinyin_notone_wer"),
                "avg_sim": avg("sim"),
            })


# ---------------------------
# Main eval
# ---------------------------

_SIMILARITY_SCORER_CACHE: Dict[str, Any] = {}


def _load_similarity_scorer_class() -> Any:
    module_path = PROJECT_ROOT / "asr_eval" / "similarity.py"
    spec = importlib.util.spec_from_file_location("asr_eval_similarity", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load similarity scorer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SpeakerSimilarityScorer


def get_similarity_scorer(sim_cfg: Dict[str, Any]) -> Any:
    cache_key = json.dumps(sim_cfg, sort_keys=True)
    if cache_key not in _SIMILARITY_SCORER_CACHE:
        cls = _load_similarity_scorer_class()
        _SIMILARITY_SCORER_CACHE[cache_key] = cls.from_config(sim_cfg)
    return _SIMILARITY_SCORER_CACHE[cache_key]


def evaluate_one(
    item: EvalItem,
    backend: ASRBackend,
    norm_cfg: Dict[str, Any],
    sim_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    asr_output = transcribe(item.audio_path, backend)
    asr_extra: Dict[str, Any] = {}
    if isinstance(asr_output, dict):
        hyp_raw = str(asr_output.get("text", ""))
        asr_extra = {k: v for k, v in asr_output.items() if k != "text"}
    else:
        hyp_raw = str(asr_output)

    # CER 和 WER 对空格的处理不同，需要分别 normalize：
    # - CER: remove_spaces_for_cer=True，字符串变成无空格的字序列
    # - WER: 保留空格，用于英文/中英混合的 word-level 对比
    ref_norm_cer = normalize_text(
        item.reference,
        lowercase=norm_cfg.get("lowercase", True),
        remove_punctuation=norm_cfg.get("remove_punctuation", True),
        remove_spaces_for_cer=norm_cfg.get("remove_spaces_for_cer", True),
    )
    hyp_norm_cer = normalize_text(
        hyp_raw,
        lowercase=norm_cfg.get("lowercase", True),
        remove_punctuation=norm_cfg.get("remove_punctuation", True),
        remove_spaces_for_cer=norm_cfg.get("remove_spaces_for_cer", True),
    )

    ref_norm_wer = normalize_text(
        item.reference,
        lowercase=norm_cfg.get("lowercase", True),
        remove_punctuation=norm_cfg.get("remove_punctuation", True),
        remove_spaces_for_cer=False,
    )
    hyp_norm_wer = normalize_text(
        hyp_raw,
        lowercase=norm_cfg.get("lowercase", True),
        remove_punctuation=norm_cfg.get("remove_punctuation", True),
        remove_spaces_for_cer=False,
    )

    result = {
        "id": item.id,
        "utt_id": item.id,
        "audio_path": item.audio_path,
        "lang": item.lang,
        "benchmark": item.benchmark,
        "tts_system": item.tts_system,
        "asr_backend": backend.name,
        "reference_raw": item.reference,
        "ref_text": item.reference,
        "hypothesis_raw": hyp_raw,
        "hyp_text": hyp_raw,
        "reference_norm": ref_norm_cer,
        "hypothesis_norm": hyp_norm_cer,
        "cer": char_error_rate(ref_norm_cer, hyp_norm_cer),
        "wer": word_error_rate(ref_norm_wer, hyp_norm_wer),
        "meta": item.meta or {},
    }
    if asr_extra:
        result["asr_output"] = asr_extra

    if item.lang.lower().startswith("zh"):
        py = chinese_pinyin_error_rates(
            ref_norm_cer,
            hyp_norm_cer,
            tone_style_name=norm_cfg.get("chinese_to_pinyin_style", "TONE3"),
        )
        result.update(py)
    else:
        result.update({
            "pinyin_wer": None,
            "pinyin_cer": None,
            "ref_pinyin": None,
            "hyp_pinyin": None,
            "pinyin_notone_wer": None,
            "pinyin_notone_cer": None,
            "ref_pinyin_notone": None,
            "hyp_pinyin_notone": None,
        })

    if sim_cfg and sim_cfg.get("enabled", False):
        prompt_wav = (item.meta or {}).get("prompt_audio_path")
        if prompt_wav:
            result["prompt_audio_path"] = prompt_wav
            result["sim"] = get_similarity_scorer(sim_cfg).score(prompt_wav, item.audio_path)
        else:
            result["sim"] = None

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="results.jsonl")
    parser.add_argument("--summary", default="summary.csv")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    norm_cfg = config.get("normalization", {})
    sim_cfg = config.get("similarity", {})
    items = load_manifest(args.manifest)
    backends = load_backends(config)

    rows: List[Dict[str, Any]] = []

    for backend in backends:
        for item in tqdm(items, desc=f"ASR={backend.name}"):
            try:
                rows.append(evaluate_one(item, backend, norm_cfg, sim_cfg))
            except Exception as e:
                if not args.continue_on_error:
                    raise
                rows.append({
                    "id": item.id,
                    "audio_path": item.audio_path,
                    "lang": item.lang,
                    "benchmark": item.benchmark,
                    "tts_system": item.tts_system,
                    "asr_backend": backend.name,
                    "error": repr(e),
                })

    write_jsonl(args.out, rows)
    write_summary_csv(args.summary, rows)
    print(f"Wrote detail results to: {args.out}")
    print(f"Wrote summary to: {args.summary}")


if __name__ == "__main__":
    main()
