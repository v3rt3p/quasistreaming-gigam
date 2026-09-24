from __future__ import annotations

import asyncio
import array
import json
import logging
import os
import signal
import sys
import uuid

import numpy as np
import sentry_sdk
import soundfile as sf
import websockets
from websockets.exceptions import ConnectionClosed

from asr_core.streaming_asr import StreamingAsrConfig, StreamingAsrSession
from gigaam_backend import GigaAMBackend

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8080"))

INPUT_GAIN = float(os.environ.get("INPUT_GAIN", "1"))
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.3"))
VAD_MIN_SILENCE_DURATION = float(os.environ.get("VAD_MIN_SILENCE_DURATION", "0.8"))
VAD_MIN_SPEECH_DURATION = float(os.environ.get("VAD_MIN_SPEECH_DURATION", "1.5"))
VAD_MAX_SPEECH_DURATION = float(os.environ.get("VAD_MAX_SPEECH_DURATION", "8"))
ASR_STREAMING_MODE = os.environ.get("ASR_STREAMING_MODE", "streaming")

SENTRY_DSN = os.environ.get("SENTRY_DSN", "https://example@o0.ingest.sentry.io/0")

sentry_sdk.init(
    dsn=SENTRY_DSN,
    traces_sample_rate=1,
    default_integrations=False,
)

LOG_PATH = os.environ.get("LOG_PATH")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

base_sample_rate = 16000

recognizer: GigaAMBackend | None = None


def save_buffer(samples: np.ndarray) -> None:
    if LOG_PATH is None:
        return
    path = f"{uuid.uuid4()}.wav"
    sf.write(os.path.join(LOG_PATH, path), samples, base_sample_rate)
    logging.info("saved samples to %s", path)


def create_streaming_config(sample_rate: int) -> StreamingAsrConfig:
    return StreamingAsrConfig(
        sample_rate=sample_rate,
        language=os.environ.get("ASR_LANGUAGE", "ru"),
        vad_threshold=VAD_THRESHOLD,
        eou_silence_ms=int(VAD_MIN_SILENCE_DURATION * 1000),
        min_utterance_ms=int(VAD_MIN_SPEECH_DURATION * 1000),
        max_utterance_ms=int(VAD_MAX_SPEECH_DURATION * 1000),
        speech_pad_ms=int(os.environ.get("ASR_STREAM_SPEECH_PAD_MS", "200")),
        partial_results=os.environ.get("ASR_STREAM_PARTIAL_RESULTS", "true").lower()
        not in {"0", "false", "no", "off"},
        partial_interval_ms=int(os.environ.get("ASR_STREAM_PARTIAL_INTERVAL_MS", "1000")),
        min_partial_audio_ms=int(os.environ.get("ASR_STREAM_MIN_PARTIAL_AUDIO_MS", "1800")),
        emit_vad_events=os.environ.get("ASR_STREAM_EMIT_VAD_EVENTS", "false").lower()
        in {"1", "true", "yes", "on"},
        streaming_mode=ASR_STREAMING_MODE,
        quasistreaming_stable_repeats=int(
            os.environ.get("ASR_QUASISTREAMING_STABLE_REPEATS", "3")
        ),
    )


async def recognize_audio(
    audio_pcm16: np.ndarray,
    offset_s: float,
    _language: str,
) -> list[tuple[float, float, str]]:
    if recognizer is None:
        raise RuntimeError("GigaAM recognizer is not initialized")

    samples = (audio_pcm16.astype(np.float32) / 32768.0) * INPUT_GAIN
    if samples.size == 0:
        return []

    with sentry_sdk.start_span(op="recognize", name="GigaAM recognizing") as span:
        text = await recognizer.transcribe(samples, base_sample_rate)
        span.set_data("text", text)
        span.set_data("gigaam_model", recognizer.model_name)
        span.set_data("gigaam_device", recognizer.device)
        span.set_data("audio_duration_s", len(samples) / base_sample_rate)

    text = text.strip()
    if not text:
        return []
    duration_s = len(audio_pcm16) / float(base_sample_rate)
    return [(offset_s, offset_s + duration_s, text)]


async def transcribe(websocket: websockets.ServerConnection) -> None:
    request = getattr(websocket, "request", None)
    headers = getattr(request, "headers", None)
    if headers is None:
        headers = getattr(websocket, "request_headers", {})
    transaction = sentry_sdk.continue_trace(headers, op="stt", name="STT session")
    overall_buffer: list[np.ndarray] = []

    with sentry_sdk.start_transaction(transaction):
        try:
            config_message = json.loads(await websocket.recv())
            sample_rate = config_message["sample_rate"]
            if sample_rate != base_sample_rate:
                logging.warning(
                    "sample rate mismatch, expected %d, got %d",
                    base_sample_rate,
                    sample_rate,
                )
                await close_socket(websocket)
                return

            session = StreamingAsrSession(
                config=create_streaming_config(sample_rate),
                transcribe=recognize_audio,
            )

            async for message in websocket:
                if isinstance(message, str):
                    continue

                samples = np.array(array.array("h", message), dtype=np.int16)
                overall_buffer.append(samples.copy())
                events = await session.add_audio(samples)
                if not await send_transcript_events(websocket, events):
                    continue

                # An end-of-utterance is a boundary, not a connection boundary.
                # Streaming clients can send the next utterance on this socket.
                session = StreamingAsrSession(
                    config=create_streaming_config(sample_rate),
                    transcribe=recognize_audio,
                )

            # Async iteration ends when the peer closes. Never run a final
            # decode/send here: the peer cannot receive it anymore.
        except ConnectionClosed:
            logging.info("STT peer closed the WebSocket")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            logging.warning("invalid STT session: %s", error)
        finally:
            if overall_buffer:
                save_buffer(np.concatenate(overall_buffer))


def websocket_is_open(websocket: websockets.ServerConnection) -> bool:
    state = getattr(websocket, "state", None)
    if state is not None:
        return state.name == "OPEN" if hasattr(state, "name") else state == 1
    return not bool(getattr(websocket, "closed", False))


async def close_socket(websocket: websockets.ServerConnection) -> None:
    if not websocket_is_open(websocket):
        return
    try:
        await websocket.close()
    except ConnectionClosed:
        pass


async def send_transcript_events(
    websocket: websockets.ServerConnection,
    events: list[dict],
) -> bool:
    if not websocket_is_open(websocket):
        return False

    for event in events:
        if event.get("type") != "transcript":
            continue
        text = str(event.get("text") or "").strip()
        if not text:
            continue
        is_final = bool(event.get("is_final") or event.get("final") or event.get("eou"))
        logging.info(
            "%s recognized text: '%s'",
            "final" if is_final else "partial",
            text,
        )
        try:
            await websocket.send(
                json.dumps(
                    {
                        "end_of_utt": is_final,
                        "text": text,
                        "result_type": event.get("result_type"),
                        "reason": event.get("reason"),
                    }
                )
            )
        except ConnectionClosed:
            return False
        if is_final:
            return True
    return False


async def _windows_cancel(stop_event: asyncio.Event) -> None:
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        stop_event.set()


async def main() -> None:
    global recognizer
    recognizer = GigaAMBackend()
    await recognizer.load_async()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
    else:
        loop.create_task(_windows_cancel(stop))

    async with websockets.serve(transcribe, HOST, PORT):
        logging.info("started on :%d", PORT)
        await stop.wait()


if __name__ == "__main__":
    asyncio.run(main())
