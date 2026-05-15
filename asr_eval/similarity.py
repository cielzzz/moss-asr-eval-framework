#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict


class SpeakerSimilarityScorer:
    """WavLM speaker similarity wrapper for seed-tts-eval verification code."""

    def __init__(
        self,
        *,
        verification_dir: str,
        checkpoint: str,
        model_name: str = "wavlm_large",
        device: str = "cuda:0",
        use_gpu: bool = True,
    ) -> None:
        self.verification_dir = Path(verification_dir).resolve()
        self.checkpoint = str(checkpoint)
        self.model_name = model_name
        self.device = device
        self.use_gpu = use_gpu
        self._verification = None
        self._model = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SpeakerSimilarityScorer":
        return cls(
            verification_dir=str(config["verification_dir"]),
            checkpoint=str(config["checkpoint"]),
            model_name=str(config.get("model_name", "wavlm_large")),
            device=str(config.get("device", "cuda:0")),
            use_gpu=bool(config.get("use_gpu", True)),
        )

    def _load_verification(self) -> Any:
        if self._verification is not None:
            return self._verification

        verification_py = self.verification_dir / "verification.py"
        if not verification_py.exists():
            raise FileNotFoundError(f"Cannot find verification.py: {verification_py}")
        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(f"Cannot find speaker similarity checkpoint: {self.checkpoint}")

        sys.path.insert(0, str(self.verification_dir))
        spec = importlib.util.spec_from_file_location("seed_tts_eval_verification", verification_py)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load verification module from {verification_py}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._verification = module.verification
        return self._verification

    def score(self, prompt_wav: str, generated_wav: str) -> float:
        verification = self._load_verification()
        sim, self._model = verification(
            self.model_name,
            prompt_wav,
            generated_wav,
            use_gpu=self.use_gpu,
            checkpoint=self.checkpoint,
            wav1_start_sr=0,
            wav2_start_sr=0,
            wav1_end_sr=-1,
            wav2_end_sr=-1,
            wav2_cut_wav1=False,
            model=self._model,
            device=self.device,
        )
        return float(sim.item())

