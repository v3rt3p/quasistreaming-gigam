# quasistreaming-gigam

Streaming WebSocket ASR using GigaAM `multilingual_large_ctc` and TEN VAD.
The connection stays open across utterances; `end_of_utt` marks an utterance
boundary. The public protocol remains mono 16 kHz PCM16 input with JSON
responses containing `text` and `end_of_utt`.

The transport and endpointing are streaming, but this pinned GigaAM API only
provides file-based `transcribe()`. Each partial therefore re-decodes a bounded
short window in a worker thread rather than using a stateful decoder.

The streaming ASR endpointing code and TEN VAD are ported from Alina_Service.
On Linux, TEN VAD requires `libc++1`; the bundled Dockerfiles install it.

## Models

GigaAM downloads `multilingual_large_ctc` on first startup. Set
`GIGAAM_DOWNLOAD_ROOT` to a persistent directory so container restarts reuse
the checkpoint. The first load requires outbound network access and enough disk
space for the model.

TEN VAD is installed as a Python dependency and does not require a separate
ONNX model file.

## CPU

```shell
docker build -f Dockerfile.cpu -t quasistreaming-gigam:cpu .
docker run --rm \
  -p 8080:8080 \
  -v gigaam-cache:/app/.cache/gigaam \
  quasistreaming-gigam:cpu
```

## CUDA

The CUDA image uses Python 3.11, PyTorch 2.6, and the CUDA 11.8 wheel stack.
The host needs a compatible NVIDIA driver and the NVIDIA Container Toolkit.

```shell
docker compose build
docker compose up
```

Compose reserves GPU 0 and persists the GigaAM checkpoint in the
`gigaam-multilingual-large-ctc` volume. Startup logs must report
`model=multilingual_large_ctc device=cuda`. A CPU fallback means the host
driver, GPU passthrough, PyTorch wheel, or GPU architecture needs attention.

To prewarm the model cache before rollout, start a container from the same image
with the cache mounted and let model initialization finish once. Do not copy the
old GigaAM ONNX model or token file into this cache; the Python GigaAM package
downloads and verifies its own checkpoint.

## Configuration

- `GIGAAM_MODEL_NAME`: checkpoint name, default `multilingual_large_ctc`.
- `GIGAAM_DEVICE`: `auto`, `cpu`, or `cuda`; unavailable or incompatible CUDA
  falls back to CPU, including one CPU retry after CUDA model-load failure.
- `GIGAAM_DOWNLOAD_ROOT`: model cache, default `/app/.cache/gigaam`.
- `GIGAAM_MAX_WINDOW_SECONDS`: maximum decode window, default `24`; values must
  be finite, positive, and below GigaAM's 25-second short-form limit.
- `VAD_THRESHOLD`, `VAD_MIN_SILENCE_DURATION`,
  `VAD_MIN_SPEECH_DURATION`, `VAD_MAX_SPEECH_DURATION`: endpointing controls.
- `INPUT_GAIN`: input multiplier.
- `ASR_STREAMING_MODE=quasistreaming|streaming`: partial output mode; the
  Docker Compose deployment uses `streaming` so final results only follow VAD
  endpointing, while both modes use bounded re-decoding with this GigaAM API.
- `ASR_QUASISTREAMING_STABLE_REPEATS`: matching partial results required to end
  an utterance in quasi-streaming mode, default `3`.

## Client

The input WAV must be 16 kHz. Stereo audio is mixed to mono by the client.

```shell
python client.py input.wav ws://localhost:8080
```

## Tests

Tests mock the large model and CUDA hardware:

```shell
python -m pip install -r requirements.dev.txt
python -m pytest
```

For deployment verification, also run the client against both images with a
known speech fixture. On a GPU host, verify `torch.cuda.is_available()` inside
the running container and confirm the startup log reports `device=cuda`.
