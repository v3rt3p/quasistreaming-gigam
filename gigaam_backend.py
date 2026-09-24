from __future__ import annotations

import asyncio
import logging
import math
import os
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any

import gigaam
import numpy as np
import torch

from cuda_probe import resolve_torch_device


class GigaAMBackend:
    def __init__(
        self,
        *,
        model_name: str | None = None,
        requested_device: str | None = None,
        download_root: str | None = None,
        max_window_seconds: float | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "GIGAAM_MODEL_NAME", "v3_e2e_rnnt"
        )
        self.requested_device = requested_device or os.getenv("GIGAAM_DEVICE", "auto")
        self.download_root = Path(
            download_root
            or os.getenv("GIGAAM_DOWNLOAD_ROOT", "/app/.cache/gigaam")
        )
        self.max_window_seconds = (
            max_window_seconds
            if max_window_seconds is not None
            else float(os.getenv("GIGAAM_MAX_WINDOW_SECONDS", "24"))
        )
        if not math.isfinite(self.max_window_seconds) or not (
            0 < self.max_window_seconds < 25
        ):
            raise ValueError(
                "GIGAAM_MAX_WINDOW_SECONDS must be finite and below 25 seconds"
            )

        self.device: str | None = None
        self.model: Any | None = None
        self._inference_lock = threading.Lock()

    def _load_on_device(self, device: str) -> Any:
        model = gigaam.load_model(
            self.model_name,
            device=device,
            download_root=str(self.download_root),
        )
        if not hasattr(model, "transcribe"):
            raise TypeError("loaded GigaAM model does not expose transcribe()")
        return model

    def load(self) -> None:
        if self.model is not None:
            return

        self.download_root.mkdir(parents=True, exist_ok=True)
        selected_device = resolve_torch_device(torch, self.requested_device)
        started_at = time.perf_counter()

        try:
            model = self._load_on_device(selected_device)
        except Exception:
            if selected_device != "cuda":
                raise
            logging.exception("failed to load GigaAM on CUDA; retrying on CPU")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            selected_device = "cpu"
            model = self._load_on_device(selected_device)

        self.model = model
        self.device = selected_device
        logging.info(
            "loaded GigaAM model=%s device=%s requested_device=%s duration_s=%.3f",
            self.model_name,
            self.device,
            self.requested_device,
            time.perf_counter() - started_at,
        )

    async def load_async(self) -> None:
        await asyncio.to_thread(self.load)

    def _decode_window(self, audio: np.ndarray, sample_rate: int) -> str:
        if self.model is None:
            raise RuntimeError("GigaAM model is not loaded")

        with self._inference_lock:
            audio_f32 = np.asarray(audio, dtype=np.float32).reshape(-1)
            pcm16 = (audio_f32 * 32767.0).clip(-32768, 32767).astype("<i2")
            file_descriptor, path = tempfile.mkstemp(suffix=".wav")
            try:
                os.close(file_descriptor)
                with wave.open(path, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(pcm16.tobytes())

                result = self.model.transcribe(path)
            finally:
                os.unlink(path)

        if isinstance(result, str):
            return result.strip()
        text = getattr(result, "text", None)
        return (text if isinstance(text, str) else str(result or "")).strip()

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        if self.model is None:
            raise RuntimeError("GigaAM model is not loaded")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return ""

        window_samples = max(1, int(self.max_window_seconds * sample_rate))
        texts: list[str] = []
        for offset in range(0, samples.size, window_samples):
            chunk = samples[offset : offset + window_samples]
            text = await asyncio.to_thread(self._decode_window, chunk, sample_rate)
            if text:
                texts.append(text)
        return " ".join(texts)
