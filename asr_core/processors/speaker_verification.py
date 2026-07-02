from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerVerificationResult:
    accepted: bool
    reason: str | None = None
    similarity: float | None = None
    threshold: float | None = None
    speaker_id: str | None = None
    label: str | None = None
    best_speaker_id: str | None = None
    best_label: str | None = None
