from __future__ import annotations

import numpy as np

from asr_core.config import ASR_TARGET_SR


def pcm16_to_float32(x: np.ndarray) -> np.ndarray:
    assert x.dtype == np.int16
    return (x.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def resample_int16(
    sig: np.ndarray, sr_in: int, sr_out: int = ASR_TARGET_SR
) -> np.ndarray:
    """Lightweight linear resampler for int16 audio."""
    if sr_in == sr_out or sig.size == 0:
        return sig
    t_in = np.linspace(
        0.0, sig.size / sr_in, num=sig.size, endpoint=False, dtype=np.float64
    )
    n_out = int(round(sig.size * (sr_out / sr_in)))
    t_out = np.linspace(
        0.0, sig.size / sr_in, num=n_out, endpoint=False, dtype=np.float64
    )
    out = np.interp(t_out, t_in, sig.astype(np.float64)).astype(np.int16)
    return out
