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
import sherpa_onnx
import soundfile as sf
import websockets

from asr_core.streaming_asr import StreamingAsrConfig, StreamingAsrSession

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8080"))

INPUT_GAIN = float(os.environ.get("INPUT_GAIN", "1"))

RECOGNIZER_ONNX_PROVIDER = os.environ.get("RECOGNIZER_ONNX_PROVIDER", "cpu")
RECOGNIZER_MODEL_PATH = os.environ.get("RECOGNIZER_MODEL_PATH", "v2_ctc.onnx")
RECOGNIZER_TOKENS_PATH = os.environ.get("RECOGNIZER_TOKENS_PATH", "tokens.txt")

VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.3"))
VAD_MIN_SILENCE_DURATION = float(os.environ.get("VAD_MIN_SILENCE_DURATION", "0.8"))
VAD_MIN_SPEECH_DURATION = float(os.environ.get("VAD_MIN_SPEECH_DURATION", "1.5"))
VAD_MAX_SPEECH_DURATION = float(os.environ.get("VAD_MAX_SPEECH_DURATION", "8"))
ASR_STREAMING_MODE = os.environ.get("ASR_STREAMING_MODE", "quasistreaming")

SENTRY_DSN = os.environ.get("SENTRY_DSN", "https://example@o0.ingest.sentry.io/0")

sentry_sdk.init(
    dsn=SENTRY_DSN,
    traces_sample_rate=1,
    default_integrations=False
)

LOG_PATH = os.environ.get("LOG_PATH")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

base_sample_rate = 16000

recognizer = None


def create_recognizer() -> sherpa_onnx.OfflineRecognizer:
    logging.info(f"creating recognizer for provider '{RECOGNIZER_ONNX_PROVIDER}'")
    return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=RECOGNIZER_MODEL_PATH,
        tokens=RECOGNIZER_TOKENS_PATH,
        debug=False,
        provider=RECOGNIZER_ONNX_PROVIDER
    )


def save_buffer(samples) -> None:
    if LOG_PATH is None:
        return
    path = f"{uuid.uuid4()}.wav"
    sf.write(os.path.join(LOG_PATH, path), samples, base_sample_rate)
    logging.info(f"saved samples to {path}")


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
        raise RuntimeError("Recognizer is not initialized")

    samples = (audio_pcm16.astype(np.float32) / 32768.0) * INPUT_GAIN
    if samples.size == 0:
        return []

    with sentry_sdk.start_span(op="recognize", name="GigaAM recognizing") as span:
        stream = recognizer.create_stream()
        stream.accept_waveform(base_sample_rate, samples)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        span.set_data("text", text)

    if not text:
        return []
    duration_s = len(audio_pcm16) / float(base_sample_rate)
    return [(offset_s, offset_s + duration_s, text)]


async def transcribe(websocket: websockets.ServerConnection) -> None:
    with sentry_sdk.start_transaction(sentry_sdk.continue_trace(websocket.request.headers, op="stt", name="STT session")):
        config_message = json.loads(await websocket.recv())
        sample_rate = config_message['sample_rate']
        if sample_rate != base_sample_rate:
            logging.warning(f"sample rate mismatch, expected {base_sample_rate}, got {sample_rate}")
            await websocket.close()
            return

        session = StreamingAsrSession(
            config=create_streaming_config(sample_rate),
            transcribe=recognize_audio,
        )
        overall_buffer: list[np.ndarray] = []

        async for message in websocket:
            if type(message) is str:
                continue

            samples = np.array(array.array("h", message), dtype=np.int16)
            overall_buffer.append(samples.copy())
            events = await session.add_audio(samples)
            should_close = await send_transcript_events(websocket, events)
            if should_close:
                save_buffer(np.concatenate(overall_buffer))
                await websocket.close()
                return

        events = await session.finish()
        await send_transcript_events(websocket, events)
        if overall_buffer:
            save_buffer(np.concatenate(overall_buffer))


async def send_transcript_events(
    websocket: websockets.ServerConnection,
    events: list[dict],
) -> bool:
    for event in events:
        if event.get("type") != "transcript":
            continue
        text = str(event.get("text") or "").strip()
        if not text:
            continue
        is_final = bool(event.get("is_final") or event.get("final") or event.get("eou"))
        logging.info("%s recognized text: '%s'", "final" if is_final else "partial", text)
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
    recognizer = create_recognizer()

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
