"""Voice Activity Detection (VAD) processor using TEN VAD."""

from __future__ import annotations

import os
import platform
from ctypes import CDLL
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple, overload

import numpy as np

from asr_core.utils.dsp import resample_int16
from asr_core.utils.log import fw_log

TEN_VAD_SAMPLE_RATE = 16000
TEN_VAD_HOP_SIZE = 256


@dataclass
class VoiceSegment:
    """Represents a detected voice segment"""

    start_sample: int
    end_sample: int
    audio_data: np.ndarray  # PCM16 audio data for this segment


class TenVAD:
    """Wrapper for TEN VAD frame inference with Silero-compatible helpers."""

    detector_class: Optional[type] = None
    sample_rate: int = 48000

    @classmethod
    def initialize(cls, sample_rate: int = 48000) -> None:
        """Load TEN VAD's native binding and validate the configured sample rate."""
        try:
            fw_log("vad_init_start", sample_rate=sample_rate)
            if sample_rate <= 0:
                raise ValueError("VAD sample rate must be positive")
            if cls.detector_class is None:
                from ten_vad import TenVad  # type: ignore[import-untyped]

                if platform.system() == "Linux":
                    try:
                        CDLL("libc++.so.1")
                    except OSError as exc:
                        raise RuntimeError(
                            "TEN VAD requires the Linux libc++1 package"
                        ) from exc
                probe = TenVad(hop_size=TEN_VAD_HOP_SIZE, threshold=0.5)
                probe.process(np.zeros(TEN_VAD_HOP_SIZE, dtype=np.int16))
                del probe
                cls.detector_class = TenVad
            cls.sample_rate = sample_rate
            fw_log(
                "vad_init_success",
                engine="ten_vad",
                sample_rate=sample_rate,
                inference_sample_rate=TEN_VAD_SAMPLE_RATE,
                hop_size=TEN_VAD_HOP_SIZE,
            )
        except Exception as e:
            fw_log("vad_init_error", error=str(e))
            raise

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if VAD is initialized"""
        return cls.detector_class is not None

    @classmethod
    def _new_detector(cls, threshold: float) -> Any:
        if cls.detector_class is None:
            cls.initialize()
        if cls.detector_class is None:  # pragma: no cover - guarded by initialize
            raise RuntimeError("TEN VAD is unavailable")
        return cls.detector_class(hop_size=TEN_VAD_HOP_SIZE, threshold=threshold)

    @classmethod
    @overload
    def detect_voice_segments(
        cls,
        audio_pcm16: np.ndarray,
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        window_size_samples: int = TEN_VAD_HOP_SIZE,
        return_seconds: Literal[False] = False,
        _detector: Any | None = None,
    ) -> List[Tuple[int, int]]: ...

    @classmethod
    @overload
    def detect_voice_segments(
        cls,
        audio_pcm16: np.ndarray,
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        window_size_samples: int = TEN_VAD_HOP_SIZE,
        return_seconds: Literal[True],
        _detector: Any | None = None,
    ) -> List[Tuple[float, float]]: ...

    @classmethod
    def detect_voice_segments(
        cls,
        audio_pcm16: np.ndarray,
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        window_size_samples: int = TEN_VAD_HOP_SIZE,
        return_seconds: bool = False,
        _detector: Any | None = None,
    ) -> List[Tuple[int, int]] | List[Tuple[float, float]]:
        """
        Detect voice segments in audio using TEN VAD.

        Args:
            audio_pcm16: PCM16 audio data (int16 numpy array)
            threshold: Speech probability threshold (0.0-1.0)
            min_speech_duration_ms: Minimum duration of speech segment
            min_silence_duration_ms: Minimum duration of silence between segments
            window_size_samples: Deprecated compatibility argument. TEN VAD uses
                its optimized 256-sample hop at 16 kHz.
            return_seconds: If True, return times in seconds; otherwise samples

        Returns:
            List of (start, end) tuples representing voice segments
        """
        if cls.detector_class is None:
            cls.initialize()

        try:
            del window_size_samples
            original_audio = np.asarray(audio_pcm16, dtype=np.int16)
            vad_audio = resample_int16(
                original_audio,
                sr_in=cls.sample_rate,
                sr_out=TEN_VAD_SAMPLE_RATE,
            )
            detector = _detector or cls._new_detector(threshold)
            segments_16k = _detect_ten_vad_segments(
                detector,
                vad_audio,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
            )
            segments = _map_segments_to_source_rate(
                segments_16k,
                source_samples=len(original_audio),
                source_rate=cls.sample_rate,
            )

            fw_log(
                "vad_detection",
                engine="ten_vad",
                audio_samples=len(audio_pcm16),
                segments_detected=len(segments),
                threshold=threshold,
            )

            if return_seconds:
                return [
                    (start / cls.sample_rate, end / cls.sample_rate)
                    for start, end in segments
                ]
            return segments

        except Exception as e:
            fw_log("vad_detection_error", error=str(e))
            # If VAD fails, return the entire audio as one segment
            if return_seconds:
                return [(0.0, len(audio_pcm16) / cls.sample_rate)]
            return [(0, len(audio_pcm16))]

    @classmethod
    def extract_voice_segments(
        cls,
        audio_pcm16: np.ndarray,
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        merge_segments: bool = True,
        max_gap_ms: int = 300,
        speech_pad_ms: int = 0,
        _detector: Any | None = None,
    ) -> List[VoiceSegment]:
        """
        Extract voice segments from audio, removing silence.

        Args:
            audio_pcm16: PCM16 audio data (int16 numpy array)
            threshold: Speech probability threshold
            min_speech_duration_ms: Minimum speech duration
            min_silence_duration_ms: Minimum silence duration
            merge_segments: If True, merge segments with small gaps
            max_gap_ms: Maximum gap between segments to merge (in milliseconds)
            speech_pad_ms: Context kept before and after detected speech

        Returns:
            List of VoiceSegment objects containing detected voice audio
        """
        # Detect segments in samples
        segments = cls.detect_voice_segments(
            audio_pcm16,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=False,
            _detector=_detector,
        )

        if not segments:
            energy_segment = _energy_fallback_segment(audio_pcm16)
            if energy_segment is not None:
                segments = [energy_segment]
                fw_log(
                    "vad_energy_fallback_detected",
                    audio_samples=len(audio_pcm16),
                    rms=round(_pcm16_rms(audio_pcm16), 5),
                    peak=round(_pcm16_peak(audio_pcm16), 5),
                )

        if not segments:
            fw_log("vad_no_segments_detected", audio_samples=len(audio_pcm16))
            return []

        # Merge segments if requested
        if merge_segments and len(segments) > 1:
            max_gap_samples = int(max_gap_ms * cls.sample_rate / 1000)
            original_count = len(segments)
            merged = []
            current_start, current_end = segments[0]

            for start, end in segments[1:]:
                gap = start - current_end
                if gap <= max_gap_samples:
                    # Merge with current segment
                    current_end = end
                else:
                    # Save current segment and start new one
                    merged.append((current_start, current_end))
                    current_start, current_end = start, end

            # Add final segment
            merged.append((current_start, current_end))
            segments = merged

            fw_log(
                "vad_segments_merged",
                original_count=original_count,
                merged_count=len(merged),
                max_gap_ms=max_gap_ms,
            )

        pad_samples = max(0, int(speech_pad_ms * cls.sample_rate / 1000))

        # Extract audio for each segment
        voice_segments = []
        for start_sample, end_sample in segments:
            # Ensure indices are within bounds
            start_sample = max(0, int(start_sample) - pad_samples)
            end_sample = min(len(audio_pcm16), int(end_sample) + pad_samples)

            if end_sample > start_sample:
                segment_audio = audio_pcm16[start_sample:end_sample]
                voice_segments.append(
                    VoiceSegment(
                        start_sample=start_sample,
                        end_sample=end_sample,
                        audio_data=segment_audio,
                    )
                )

        fw_log(
            "vad_segments_extracted",
            segments_count=len(voice_segments),
            total_audio_samples=len(audio_pcm16),
            voiced_samples=sum(
                seg.end_sample - seg.start_sample for seg in voice_segments
            ),
        )

        return voice_segments


def _detect_ten_vad_segments(
    detector: Any,
    audio_pcm16: np.ndarray,
    *,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
) -> list[tuple[int, int]]:
    """Run TEN VAD frame inference and return segments on its 16 kHz timeline."""
    if audio_pcm16.size == 0:
        return []

    min_speech_samples = int(min_speech_duration_ms * TEN_VAD_SAMPLE_RATE / 1000)
    min_silence_samples = int(min_silence_duration_ms * TEN_VAD_SAMPLE_RATE / 1000)
    speech_start: int | None = None
    silence_start: int | None = None
    segments: list[tuple[int, int]] = []

    for frame_start in range(0, len(audio_pcm16), TEN_VAD_HOP_SIZE):
        frame = audio_pcm16[frame_start : frame_start + TEN_VAD_HOP_SIZE]
        if len(frame) < TEN_VAD_HOP_SIZE:
            frame = np.pad(frame, (0, TEN_VAD_HOP_SIZE - len(frame)))
        _probability, speech_flag = detector.process(frame.astype(np.int16, copy=False))
        frame_end = min(frame_start + TEN_VAD_HOP_SIZE, len(audio_pcm16))

        if speech_flag:
            if speech_start is None:
                speech_start = frame_start
            silence_start = None
            continue

        if speech_start is None:
            continue
        if silence_start is None:
            silence_start = frame_start
        if frame_end - silence_start < min_silence_samples:
            continue
        if silence_start - speech_start >= min_speech_samples:
            segments.append((speech_start, silence_start))
        speech_start = None
        silence_start = None

    if speech_start is not None:
        speech_end = silence_start if silence_start is not None else len(audio_pcm16)
        if speech_end - speech_start >= min_speech_samples:
            segments.append((speech_start, speech_end))
    return segments


def _map_segments_to_source_rate(
    segments: list[tuple[int, int]],
    *,
    source_samples: int,
    source_rate: int,
) -> list[tuple[int, int]]:
    scale = source_rate / float(TEN_VAD_SAMPLE_RATE)
    return [
        (
            max(0, min(source_samples, int(np.floor(start * scale)))),
            max(0, min(source_samples, int(np.ceil(end * scale)))),
        )
        for start, end in segments
    ]


class StreamingVADBuffer:
    """
    Streaming VAD buffer that accumulates audio and extracts voice segments.
    Maintains continuity of voiced audio across streaming chunks.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        max_gap_ms: int = 300,
        merge_segments: bool = True,
        chunk_size_ms: int = 1000,
        silence_detection_window_ms: int = 500,
        speech_pad_ms: int = 0,
    ):
        """
        Initialize streaming VAD buffer.

        Args:
            sample_rate: Audio sample rate
            threshold: VAD threshold
            min_speech_duration_ms: Minimum speech duration
            min_silence_duration_ms: Minimum silence duration
            max_gap_ms: Maximum gap to merge segments
            chunk_size_ms: Process audio in chunks of this size
            silence_detection_window_ms: Window to detect recent silence for clean batch boundaries
            speech_pad_ms: Context kept before and after detected speech
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.max_gap_ms = max_gap_ms
        self.merge_segments = merge_segments
        self.chunk_size_samples = int(chunk_size_ms * sample_rate / 1000)
        self.silence_detection_window_samples = int(
            silence_detection_window_ms * sample_rate / 1000
        )
        self.speech_pad_ms = speech_pad_ms

        # Audio buffer
        self.audio_buffer = bytearray()
        self.total_samples_received = 0
        self.total_samples_processed = 0

        # Voice segments (accumulated clean audio)
        self.voice_segments: List[VoiceSegment] = []
        self.accumulated_voice_audio = bytearray()

        # Track last processed segment to detect silence gaps
        self.last_voice_segment_end = 0  # Sample position of last voice segment
        self.recent_silence_gap_ms = 0.0  # Duration of recent silence gap

        # Initialize VAD
        TenVAD.initialize(sample_rate)
        self.detector = TenVAD._new_detector(threshold)

        fw_log(
            "streaming_vad_buffer_init",
            sample_rate=sample_rate,
            threshold=threshold,
            chunk_size_ms=chunk_size_ms,
            silence_detection_window_ms=silence_detection_window_ms,
            speech_pad_ms=speech_pad_ms,
        )

    def add_audio(self, audio_bytes: bytes) -> None:
        """Add audio chunk to buffer"""
        self.audio_buffer.extend(audio_bytes)
        self.total_samples_received += (
            len(audio_bytes) // 2
        )  # PCM16 = 2 bytes per sample

    def process_available_audio(self, is_final: bool = False) -> Optional[np.ndarray]:
        """
        Process available audio in buffer and extract voice segments.

        Args:
            is_final: If True, process all remaining audio in buffer

        Returns:
            PCM16 numpy array of accumulated voice audio, or None if no new audio
        """
        buffer_samples = len(self.audio_buffer) // 2

        # Determine how much to process
        if is_final:
            samples_to_process = buffer_samples
        else:
            # Only process complete chunks
            samples_to_process = (
                buffer_samples // self.chunk_size_samples
            ) * self.chunk_size_samples

        if samples_to_process == 0:
            return None

        # Extract audio to process
        bytes_to_process = samples_to_process * 2
        audio_to_process = bytes(self.audio_buffer[:bytes_to_process])
        del self.audio_buffer[:bytes_to_process]

        # Convert to numpy array
        audio_pcm16 = np.frombuffer(audio_to_process, dtype=np.int16)

        # Extract voice segments
        voice_segments = TenVAD.extract_voice_segments(
            audio_pcm16,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            merge_segments=self.merge_segments,
            max_gap_ms=self.max_gap_ms,
            speech_pad_ms=self.speech_pad_ms,
            _detector=self.detector,
        )

        # Calculate silence gap before first voice segment
        if voice_segments:
            first_voice_start = (
                self.total_samples_processed + voice_segments[0].start_sample
            )
            if self.last_voice_segment_end > 0:
                silence_gap_samples = first_voice_start - self.last_voice_segment_end
                self.recent_silence_gap_ms = (
                    silence_gap_samples / self.sample_rate
                ) * 1000

            # Update last voice segment end
            last_voice_end = (
                self.total_samples_processed + voice_segments[-1].end_sample
            )
            self.last_voice_segment_end = last_voice_end
        else:
            # No voice segments in this chunk - accumulate silence
            if self.last_voice_segment_end > 0:
                silence_gap_samples = (
                    self.total_samples_processed
                    + samples_to_process
                    - self.last_voice_segment_end
                )
                self.recent_silence_gap_ms = (
                    silence_gap_samples / self.sample_rate
                ) * 1000

        # Accumulate voice audio
        new_voice_audio = bytearray()
        for segment in voice_segments:
            segment_bytes = segment.audio_data.tobytes()
            new_voice_audio.extend(segment_bytes)
            self.accumulated_voice_audio.extend(segment_bytes)

            # Store segment info (adjust sample positions to global timeline)
            global_start = self.total_samples_processed + segment.start_sample
            global_end = self.total_samples_processed + segment.end_sample
            self.voice_segments.append(
                VoiceSegment(
                    start_sample=global_start,
                    end_sample=global_end,
                    audio_data=segment.audio_data,
                )
            )

        self.total_samples_processed += samples_to_process

        if len(new_voice_audio) > 0:
            fw_log(
                "streaming_vad_processed",
                processed_samples=samples_to_process,
                voice_segments=len(voice_segments),
                new_voice_samples=len(new_voice_audio) // 2,
                total_voice_samples=len(self.accumulated_voice_audio) // 2,
                recent_silence_gap_ms=round(self.recent_silence_gap_ms, 2),
            )

            return np.frombuffer(bytes(new_voice_audio), dtype=np.int16)

        return None

    def has_recent_silence_gap(self, min_gap_ms: float = 300.0) -> bool:
        """
        Check if there's a recent silence gap suitable for batch boundary.

        Args:
            min_gap_ms: Minimum silence gap duration in milliseconds

        Returns:
            True if there's a sufficient silence gap for clean batch boundary
        """
        return self.recent_silence_gap_ms >= min_gap_ms

    def get_accumulated_audio(self) -> np.ndarray:
        """Get all accumulated voice audio as PCM16 numpy array"""
        if len(self.accumulated_voice_audio) == 0:
            return np.array([], dtype=np.int16)
        return np.frombuffer(bytes(self.accumulated_voice_audio), dtype=np.int16)

    def get_duration_seconds(self) -> float:
        """Get duration of accumulated voice audio in seconds"""
        return len(self.accumulated_voice_audio) // 2 / self.sample_rate

    def clear_accumulated_audio(self) -> None:
        """Clear accumulated voice audio after processing a batch"""
        self.accumulated_voice_audio.clear()
        self.voice_segments.clear()
        self.recent_silence_gap_ms = 0.0

        fw_log(
            "vad_accumulated_audio_cleared",
            message="Cleared accumulated audio after batch processing",
        )

    def clear(self) -> None:
        """Clear all buffers"""
        self.audio_buffer.clear()
        self.voice_segments.clear()
        self.accumulated_voice_audio.clear()
        self.total_samples_received = 0
        self.total_samples_processed = 0
        self.last_voice_segment_end = 0
        self.recent_silence_gap_ms = 0.0


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _pcm16_rms(audio_pcm16: np.ndarray) -> float:
    if len(audio_pcm16) == 0:
        return 0.0
    audio_f32 = audio_pcm16.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(np.square(audio_f32))))


def _pcm16_peak(audio_pcm16: np.ndarray) -> float:
    if len(audio_pcm16) == 0:
        return 0.0
    audio_f32 = audio_pcm16.astype(np.float32) / 32768.0
    return float(np.max(np.abs(audio_f32)))


def _energy_fallback_segment(audio_pcm16: np.ndarray) -> Optional[Tuple[int, int]]:
    if os.getenv("VAD_ENERGY_FALLBACK_ENABLED", "true").lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return None
    rms = _pcm16_rms(audio_pcm16)
    peak = _pcm16_peak(audio_pcm16)
    min_rms = _float_env("VAD_ENERGY_FALLBACK_RMS", 0.06)
    min_peak = _float_env("VAD_ENERGY_FALLBACK_PEAK", 0.25)
    if rms < min_rms or peak < min_peak:
        return None
    return 0, len(audio_pcm16)
