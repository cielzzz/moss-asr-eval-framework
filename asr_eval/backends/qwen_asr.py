#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, Optional


class QwenASRBackend:
    """Qwen3-ASR-1.7B backend using the official qwen-asr package."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-ASR-1.7B",
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        language: Optional[str] = None,
        max_inference_batch_size: int = 1,
        max_new_tokens: int = 256,
        attn_implementation: Optional[str] = None,
        backend_name: str = "qwen-asr-1.7b",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.language = language
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation
        self.backend_name = backend_name
        self._model = None

    @classmethod
    def from_params(cls, params: Dict[str, Any], *, backend_name: str = "qwen-asr-1.7b") -> "QwenASRBackend":
        return cls(
            model_path=str(params.get("model_path", "Qwen/Qwen3-ASR-1.7B")),
            device=str(params.get("device", "cuda:0")),
            dtype=str(params.get("dtype", "bfloat16")),
            language=params.get("language"),
            max_inference_batch_size=int(params.get("max_inference_batch_size", 1)),
            max_new_tokens=int(params.get("max_new_tokens", 256)),
            attn_implementation=params.get("attn_implementation"),
            backend_name=str(params.get("backend_name", backend_name)),
        )

    def _torch_dtype(self) -> Any:
        try:
            import torch
        except Exception as e:
            raise RuntimeError("Qwen ASR backend requires torch. Install qwen-asr first: pip install -U qwen-asr") from e

        dtype = self.dtype.lower()
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported Qwen ASR dtype: {self.dtype}")

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from qwen_asr import Qwen3ASRModel
        except Exception as e:
            raise RuntimeError(
                "Qwen ASR backend requires the official qwen-asr package. "
                "Install it in the runtime environment with: pip install -U qwen-asr"
            ) from e

        kwargs: Dict[str, Any] = {
            "dtype": self._torch_dtype(),
            "device_map": self.device,
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        self._model = Qwen3ASRModel.from_pretrained(self.model_path, **kwargs)
        return self._model

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        model = self._load_model()
        results = model.transcribe(audio=audio_path, language=self.language)
        result = results[0] if isinstance(results, list) else results

        text = getattr(result, "text", None)
        language = getattr(result, "language", None)
        if text is None and isinstance(result, dict):
            text = result.get("text", "")
            language = result.get("language", language)

        return {
            "text": "" if text is None else str(text),
            "backend": self.backend_name,
            "language": language,
            "model_path": self.model_path,
        }

