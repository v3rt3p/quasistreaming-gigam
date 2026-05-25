import asyncio
import logging
import os
import signal
import sys
import websockets
import json
import sherpa_onnx
import numpy as np
import array
import soundfile as sf
import uuid
import sentry_sdk

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8080))

RECOGNIZER_ONNX_PROVIDER = os.environ.get("RECOGNIZER_ONNX_PROVIDER", "cpu")
VAD_ONNX_PROVIDER = os.environ.get("VAD_ONNX_PROVIDER", "cpu")

INPUT_GAIN = float(os.environ.get("INPUT_GAIN", "1"))

RECOGNIZER_MODEL_PATH = os.environ.get("RECOGNIZER_MODEL_PATH", "v2_ctc.onnx")
RECOGNIZER_TOKENS_PATH = os.environ.get("RECOGNIZER_TOKENS_PATH", "tokens.txt")

VAD_MODEL_PATH = os.environ.get("VAD_MODEL_PATH", "silero_vad.onnx")
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.3"))
VAD_MIN_SILENCE_DURATION = float(os.environ.get("VAD_MIN_SILENCE_DURATION", "1"))
VAD_MIN_SPEECH_DURATION = float(os.environ.get("VAD_MIN_SPEECH_DURATION", "0.25"))
VAD_MAX_SPEECH_DURATION = float(os.environ.get("VAD_MAX_SPEECH_DURATION", "8"))

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


def create_vad() -> sherpa_onnx.VoiceActivityDetector:
    logging.info(f"creating vad for provider '{VAD_ONNX_PROVIDER}'")
    config = sherpa_onnx.VadModelConfig(provider=VAD_ONNX_PROVIDER)
    config.silero_vad.model = VAD_MODEL_PATH
    config.silero_vad.threshold = VAD_THRESHOLD
    config.silero_vad.min_silence_duration = VAD_MIN_SILENCE_DURATION
    config.silero_vad.min_speech_duration = VAD_MIN_SPEECH_DURATION
    config.silero_vad.max_speech_duration = VAD_MAX_SPEECH_DURATION
    config.sample_rate = base_sample_rate

    window_size = config.silero_vad.window_size

    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)
    return (vad, window_size)


def save_buffer(samples) -> None:
    if LOG_PATH is None:
        return
    path = f"{uuid.uuid4()}.wav"
    sf.write(os.path.join(LOG_PATH, path), samples, base_sample_rate)
    logging.info(f"saved samples to {path}")


async def transcribe(websocket: websockets.ServerConnection) -> None:
    with sentry_sdk.start_transaction(sentry_sdk.continue_trace(websocket.request.headers, op="stt", name="STT session")):
        global recognizer

        config_message = json.loads(await websocket.recv())
        sample_rate = config_message['sample_rate']
        if sample_rate != base_sample_rate:
            logging.warning(f"sample rate mismatch, expected {base_sample_rate}, got {sample_rate}")
            await websocket.close()
            return

        with sentry_sdk.start_span(op="create-vad", name="VAD creation"):
            vad, window_size = create_vad()

        buffer = []
        started = False
        started_time = None
        offset = 0

        logging.info("vad created")

        current_time = 0

        overall_buffer = []

        texts = []

        async for message in websocket:
            if type(message) is str:
                continue

            samples = np.array(array.array('h', message)) / 32767.0 * INPUT_GAIN

            buffer = np.concatenate([buffer, samples])
            overall_buffer = np.concatenate([overall_buffer, samples])

            while offset + window_size < len(buffer):
                vad.accept_waveform(buffer[offset : offset + window_size])
                if not started and vad.is_speech_detected():
                    logging.info("speech started")
                    started = True
                    started_time = current_time
                offset += window_size

            if not started:
                if len(buffer) > 10 * window_size:
                    offset -= len(buffer) - 10 * window_size
                    buffer = buffer[-10 * window_size :]

            if started and current_time - started_time > 1:
                with sentry_sdk.start_span(op="recognize", name="GigaAM recognizing") as span:
                    stream = recognizer.create_stream()
                    stream.accept_waveform(sample_rate, buffer)
                    recognizer.decode_stream(stream)
                    text = stream.result.text.strip()
                    span.set_data("text", text)

                if text:
                    texts.append(text)
                    logging.info(f"recognized text: '{text}'")

                    end_of_utt = len(texts) > 3 and texts[-1] == texts[-2] and texts[-2] == texts[-3]

                    await websocket.send(json.dumps({
                        "end_of_utt": end_of_utt,
                        "text": text
                    }))
                    if end_of_utt:
                        break

                started_time = current_time

            while not vad.empty():
                with sentry_sdk.start_span(op="recognize", name="GigaAM recognizing") as span:
                    stream = recognizer.create_stream()
                    stream.accept_waveform(sample_rate, vad.front.samples)
                    recognizer.decode_stream(stream)
                    vad.pop()
                    text = stream.result.text.strip()
                    span.set_data("text", text)

                logging.info(f"final recognized text: '{text}'")
                save_buffer(overall_buffer)
                await websocket.send(json.dumps({
                    "end_of_utt": True,
                    "text": text
                }))

                buffer = []
                offset = 0
                started = False
                started_time = None

                await websocket.close()
                return

            current_time += len(samples) / base_sample_rate

        save_buffer(overall_buffer)

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
