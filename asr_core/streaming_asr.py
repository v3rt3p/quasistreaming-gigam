from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import numpy as np

from asr_core.config import ASR_LANGUAGE, ASR_TARGET_SR
from asr_core.processors.speaker_verification import SpeakerVerificationResult
from asr_core.processors.vad_processor import StreamingVADBuffer
from asr_core.utils.log import fw_log

TranscribeFn = Callable[
    [np.ndarray, float, str], Awaitable[list[tuple[float, float, str]]]
]
VerifySpeakerFn = Callable[[np.ndarray], Awaitable[SpeakerVerificationResult]]
SeparateFn = Callable[
    [np.ndarray], Awaitable[tuple[np.ndarray, np.ndarray]]
]


@dataclass(frozen=True)
class StreamingAsrConfig:
    sample_rate: int = ASR_TARGET_SR
    language: str = ASR_LANGUAGE
    vad_threshold: float = 0.3
    eou_silence_ms: int = 800
    min_utterance_ms: int = 1500
    max_utterance_ms: int = 8_000
    speech_pad_ms: int = 200
    partial_results: bool = True
    partial_interval_ms: int = 1000
    min_partial_audio_ms: int = 1800
    verify_speaker: bool = False
    emit_speaker_rejections: bool = False
    emit_vad_events: bool = False
    unsafe_partial_before_speaker_verification: bool = False
    energy_fallback_enabled: bool = True
    energy_fallback_rms: float = 0.06
    energy_fallback_peak: float = 0.25
    streaming_mode: str = "streaming"
    quasistreaming_stable_repeats: int = 3

    @classmethod
    def from_env(
        cls,
        *,
        sample_rate: int = ASR_TARGET_SR,
        language: str = ASR_LANGUAGE,
        vad_threshold: float = 0.3,
        verify_speaker: bool = False,
        emit_speaker_rejections: bool = False,
    ) -> "StreamingAsrConfig":
        return cls(
            sample_rate=sample_rate,
            language=language,
            vad_threshold=vad_threshold,
            eou_silence_ms=_int_env("ASR_STREAM_EOU_SILENCE_MS", "800"),
            min_utterance_ms=_duration_ms_env(
                "ASR_STREAM_MIN_UTTERANCE_MS",
                "VAD_MIN_BATCH_DURATION_SECONDS",
                1500,
            ),
            max_utterance_ms=_duration_ms_env(
                "ASR_STREAM_MAX_UTTERANCE_MS",
                "VAD_BATCH_DURATION_SECONDS",
                8_000,
            ),
            speech_pad_ms=_int_env("ASR_STREAM_SPEECH_PAD_MS", "200"),
            partial_results=_bool_env("ASR_STREAM_PARTIAL_RESULTS", True),
            partial_interval_ms=_int_env("ASR_STREAM_PARTIAL_INTERVAL_MS", "1000"),
            min_partial_audio_ms=_int_env("ASR_STREAM_MIN_PARTIAL_AUDIO_MS", "1800"),
            verify_speaker=verify_speaker,
            emit_speaker_rejections=emit_speaker_rejections,
            emit_vad_events=_bool_env("ASR_STREAM_EMIT_VAD_EVENTS", False),
            unsafe_partial_before_speaker_verification=_bool_env(
                "ASR_STREAM_UNSAFE_PARTIAL_BEFORE_SPEAKER_VERIFICATION", False
            ),
            streaming_mode="streaming",
            quasistreaming_stable_repeats=_int_env(
                "ASR_QUASISTREAMING_STABLE_REPEATS", "3"
            ),
        )


class StreamingAsrSession:
    """Per-connection VAD endpointing plus GigaAM short-window decoding."""

    def __init__(
        self,
        *,
        config: StreamingAsrConfig,
        transcribe: TranscribeFn,
        verify_speaker: VerifySpeakerFn | None = None,
        session_id: str | None = None,
    ) -> None:
        self.config = config
        self.transcribe = transcribe
        self.verify_speaker = verify_speaker
        self.session_id = session_id or uuid.uuid4().hex
        self.vad = StreamingVADBuffer(
            sample_rate=config.sample_rate,
            threshold=config.vad_threshold,
            min_speech_duration_ms=200,
            min_silence_duration_ms=100,
            max_gap_ms=300,
            merge_segments=True,
            chunk_size_ms=500,
            silence_detection_window_ms=500,
            speech_pad_ms=config.speech_pad_ms,
        )
        self.phrase_id = uuid.uuid4().hex
        self.last_partial_samples = 0
        self.transcripts_sent = 0
        self.speech_started = False
        self.full_audio_chunks: list[np.ndarray] = []
        self.last_quasistreaming_text = ""
        self.quasistreaming_stable_count = 0

    async def add_audio(self, pcm16: np.ndarray) -> list[dict]:
        if pcm16.size == 0:
            return []
        chunk = pcm16.astype(np.int16, copy=False)
        self.full_audio_chunks.append(chunk.copy())
        self.vad.add_audio(chunk.tobytes())
        self.vad.process_available_audio(is_final=False)
        return await self._events_after_vad_update(is_final_flush=False)

    async def finish(self) -> list[dict]:
        full_audio = (
            np.concatenate(self.full_audio_chunks)
            if self.full_audio_chunks
            else np.empty(0, dtype=np.int16)
        )
        transcripts_before_flush = self.transcripts_sent
        events = await self.finalize_utterance(reason="final_flush")
        if (
            self.transcripts_sent == transcripts_before_flush
            and full_audio.size
            and not self.config.verify_speaker
        ):
            fw_log(
                "streaming_asr_vad_fallback",
                session_id=self.session_id,
                audio_samples=len(full_audio),
            )
            events.extend(
                await self._decode_audio(
                    full_audio,
                    is_final=True,
                    reason="vad_fallback",
                    offset_s=0.0,
                )
            )
        events.append(
            {"event": "done", "type": "done", "is_final": True, "final": True}
        )
        return events

    async def finalize_utterance(self, *, reason: str) -> list[dict]:
        """Flush the current VAD utterance without ending the connection."""
        self.vad.process_available_audio(is_final=True)
        events = await self._events_after_vad_update(
            is_final_flush=True,
            final_reason=reason,
        )
        return events

    async def _events_after_vad_update(
        self,
        *,
        is_final_flush: bool,
        final_reason: str = "final_flush",
    ) -> list[dict]:
        events: list[dict] = []
        audio = self.vad.get_accumulated_audio()
        if audio.size == 0:
            return events

        if self.config.emit_vad_events and not self.speech_started:
            self.speech_started = True
            events.append(
                {
                    "event": "speech_start",
                    "type": "speech_start",
                    "is_final": False,
                    "final": False,
                }
            )

        duration_ms = int(round(self.vad.get_duration_seconds() * 1000))
        silence_boundary = (
            self.vad.has_recent_silence_gap(self.config.eou_silence_ms)
            and duration_ms >= self.config.min_utterance_ms
        )
        max_reached = duration_ms >= self.config.max_utterance_ms
        if is_final_flush or silence_boundary or max_reached:
            reason = (
                final_reason
                if is_final_flush
                else "max_utterance"
                if max_reached
                else "eou_silence"
            )
            if self.config.emit_vad_events:
                events.append(
                    {
                        "event": "speech_end",
                        "type": "speech_end",
                        "is_final": True,
                        "final": True,
                        "eou": True,
                        "reason": reason,
                    }
                )
            events.extend(
                await self._decode_audio(
                    audio,
                    is_final=True,
                    reason=reason,
                    offset_s=self._utterance_offset_s(),
                )
            )
            self._reset_utterance()
            return events

        if not self._should_emit_partial(audio):
            return events
        partial_events = await self._decode_audio(
            audio,
            is_final=False,
            reason="partial",
            offset_s=self._utterance_offset_s(),
        )
        if self.config.streaming_mode == "quasistreaming":
            partial_events, stable = self._apply_quasistreaming_stability(
                partial_events
            )
            events.extend(partial_events)
            if stable:
                self._reset_utterance()
                return events
        else:
            events.extend(partial_events)
        self.last_partial_samples = len(audio)
        return events

    def _apply_quasistreaming_stability(
        self, events: list[dict]
    ) -> tuple[list[dict], bool]:
        text = " ".join(
            str(event.get("text") or "").strip()
            for event in events
            if event.get("type") == "transcript"
        ).strip()
        if not text:
            return events, False
        if text == self.last_quasistreaming_text:
            self.quasistreaming_stable_count += 1
        else:
            self.last_quasistreaming_text = text
            self.quasistreaming_stable_count = 1
        stable = (
            self.quasistreaming_stable_count
            >= self.config.quasistreaming_stable_repeats
        )
        if stable:
            for event in events:
                if event.get("type") != "transcript":
                    continue
                event["is_final"] = True
                event["final"] = True
                event["result_type"] = "final"
                event["eou"] = True
                event["reason"] = "quasistreaming_stable"
                phrase = event.get("phrase")
                if isinstance(phrase, dict):
                    phrase["is_final"] = True
        return events, stable

    def _should_emit_partial(self, audio: np.ndarray) -> bool:
        if not self.config.partial_results:
            return False
        if self.config.verify_speaker and (
            not self.config.unsafe_partial_before_speaker_verification
        ):
            return False
        min_samples = int(
            self.config.min_partial_audio_ms * self.config.sample_rate / 1000
        )
        interval_samples = int(
            self.config.partial_interval_ms * self.config.sample_rate / 1000
        )
        return len(audio) >= min_samples and (
            self.last_partial_samples == 0
            or len(audio) - self.last_partial_samples >= interval_samples
        )

    async def _decode_audio(
        self,
        audio: np.ndarray,
        *,
        is_final: bool,
        reason: str,
        offset_s: float,
    ) -> list[dict]:
        match: SpeakerVerificationResult | None = None
        if is_final and self.config.verify_speaker:
            if self.verify_speaker is None:
                raise RuntimeError("Speaker verification requested but unavailable")
            match = await self.verify_speaker(audio)
            if not match.accepted:
                fw_log(
                    "streaming_asr_speaker_rejected",
                    session_id=self.session_id,
                    reason=match.reason,
                    similarity=round(match.similarity or 0.0, 4)
                    if match.similarity is not None
                    else None,
                    threshold=match.threshold,
                    speaker_id=match.speaker_id,
                )
                if self.config.emit_speaker_rejections:
                    return [_speaker_rejection_event(match, is_final=True)]
                return []

        segments = await self.transcribe(
            audio.astype(np.int16, copy=False),
            offset_s,
            self.config.language,
        )
        events = []
        for start_s, end_s, text in segments:
            if not text:
                continue
            event = _transcript_event(
                text=text,
                start_s=start_s,
                end_s=end_s,
                is_final=is_final,
                phrase_id=self.phrase_id,
                match=match,
            )
            event["result_type"] = "final" if is_final else "partial"
            event["eou"] = is_final
            event["reason"] = reason
            events.append(event)
            self.transcripts_sent += 1
        fw_log(
            "streaming_asr_decode_done",
            session_id=self.session_id,
            reason=reason,
            is_final=is_final,
            events=len(events),
            audio_samples=len(audio),
        )
        return events

    def _utterance_offset_s(self) -> float:
        segments = getattr(self.vad, "voice_segments", [])
        if not segments:
            return 0.0
        start_sample = int(getattr(segments[0], "start_sample", 0) or 0)
        return start_sample / float(self.config.sample_rate)

    def _reset_utterance(self) -> None:
        self.vad.clear_accumulated_audio()
        self.last_partial_samples = 0
        self.speech_started = False
        self.phrase_id = uuid.uuid4().hex
        self.last_quasistreaming_text = ""
        self.quasistreaming_stable_count = 0
        self.full_audio_chunks.clear()


class SeparatedStreamingAsrSession:
    """Run SepFormer before two VAD streams and stop on the first VAD end."""

    def __init__(
        self,
        *,
        config: StreamingAsrConfig,
        transcribe: TranscribeFn,
        separate: SeparateFn,
        verify_speaker: VerifySpeakerFn | None = None,
        session_id: str | None = None,
        separation_window_ms: int = 1000,
    ) -> None:
        self.config = config
        self.separate = separate
        self.session_id = session_id or uuid.uuid4().hex
        self.window_samples = max(
            1,
            int(config.sample_rate * separation_window_ms / 1000),
        )
        self.pending_chunks: list[np.ndarray] = []
        self.pending_samples = 0
        self.channels = (
            StreamingAsrSession(
                config=config,
                transcribe=transcribe,
                verify_speaker=verify_speaker,
                session_id=f"{self.session_id}-speaker-1",
            ),
            StreamingAsrSession(
                config=config,
                transcribe=transcribe,
                verify_speaker=verify_speaker,
                session_id=f"{self.session_id}-speaker-2",
            ),
        )

    async def add_audio(self, pcm16: np.ndarray) -> list[dict]:
        chunk = np.asarray(pcm16, dtype=np.int16).reshape(-1)
        if chunk.size == 0:
            return []
        self.pending_chunks.append(chunk.copy())
        self.pending_samples += int(chunk.size)
        if self.pending_samples < self.window_samples:
            return []
        return await self._process_pending()

    async def finish(self) -> list[dict]:
        events: list[dict] = []
        if self.pending_samples:
            events.extend(await self._process_pending())
        for channel in self.channels:
            channel_events = await channel.finish()
            events.extend(
                event for event in channel_events if event.get("type") != "done"
            )
        events.append(
            {"event": "done", "type": "done", "is_final": True, "final": True}
        )
        return events

    async def _process_pending(self) -> list[dict]:
        mixture = np.concatenate(self.pending_chunks)
        self.pending_chunks.clear()
        self.pending_samples = 0
        first, second = await self.separate(mixture)
        separated = (first, second)
        channel_events = [
            await channel.add_audio(source)
            for channel, source in zip(self.channels, separated, strict=True)
        ]
        ended = [
            index
            for index, events in enumerate(channel_events)
            if _contains_utterance_end(events)
        ]
        if ended:
            first_ended = ended[0]
            peer = 1 - first_ended
            if not _contains_utterance_end(channel_events[peer]):
                channel_events[peer].extend(
                    await self.channels[peer].finalize_utterance(
                        reason="peer_vad_end",
                    )
                )
            fw_log(
                "sepformer_first_vad_end",
                session_id=self.session_id,
                first_ended_channel=first_ended + 1,
            )
        events: list[dict] = []
        for index, items in enumerate(channel_events, start=1):
            for event in items:
                event["separated_speaker"] = index
                events.append(event)
        return events


def _contains_utterance_end(events: list[dict]) -> bool:
    return any(
        event.get("type") == "speech_end"
        or (
            event.get("type") == "transcript"
            and (event.get("is_final") is True or event.get("final") is True)
        )
        for event in events
    )


def _transcript_event(
    *,
    text: str,
    start_s: float,
    end_s: float,
    is_final: bool,
    phrase_id: str,
    match: SpeakerVerificationResult | None = None,
) -> dict:
    event: dict[str, Any] = {
        "event": "transcript",
        "type": "transcript",
        "text": text,
        "is_final": is_final,
        "final": is_final,
        "start_ms": int(round(start_s * 1000)),
        "end_ms": int(round(end_s * 1000)),
        "phrase": {
            "id": phrase_id,
            "text": text,
            "start_time": float(start_s),
            "end_time": float(end_s),
            "quality": "Quality",
            "is_final": is_final,
        },
    }
    if match is not None and match.accepted:
        metadata = {
            "speaker_id": match.speaker_id,
            "speaker_label": match.label,
            "speaker_verified": True,
            "speaker_similarity": round(match.similarity or 0.0, 4),
            "speaker_threshold": match.threshold,
        }
        event.update(metadata)
        event["phrase"].update(metadata)
    return event


def _speaker_rejection_event(
    match: SpeakerVerificationResult,
    *,
    is_final: bool,
) -> dict:
    return {
        "event": "speaker_rejected",
        "type": "speaker_rejected",
        "reason": match.reason or "unknown_speaker",
        "speaker_id": None,
        "speaker_label": None,
        "best_speaker_id": match.best_speaker_id,
        "best_speaker_label": match.best_label,
        "similarity": round(match.similarity or 0.0, 4)
        if match.similarity is not None
        else None,
        "threshold": match.threshold,
        "is_final": is_final,
        "final": is_final,
    }


def _int_env(name: str, default: str, *, multiplier: int = 1) -> int:
    try:
        return int(float(os.getenv(name, default)) * multiplier)
    except (TypeError, ValueError):
        return int(float(default) * multiplier)


def _duration_ms_env(ms_name: str, seconds_name: str, default_ms: int) -> int:
    raw_ms = os.getenv(ms_name)
    if raw_ms is not None:
        try:
            return int(float(raw_ms))
        except ValueError:
            return default_ms
    raw_seconds = os.getenv(seconds_name)
    if raw_seconds is not None:
        try:
            return int(float(raw_seconds) * 1000)
        except ValueError:
            return default_ms
    return default_ms


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}
