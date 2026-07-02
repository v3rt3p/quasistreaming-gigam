# quasistreaming-gigam

Simple quasistreaming server with GigaAM and VAD.

The server uses the streaming ASR endpointing code and TEN VAD ported from
Alina_Service. On Linux, TEN VAD requires `libc++1`; the bundled Dockerfiles
install it.

## CPU/CUDA Inference

This repository includes two subsets of files:

- `Dockerfile.cpu`, `requirements.cpu.txt`: For CPU inference.
- `Dockerfile.cuda`, `requirements.cuda.txt`: For NVIDIA CUDA inference.

## Client

A simple client is provided in `client.py` to test the server:

``` shell
python client.py input.wav ws://localhost:8080
```

Useful streaming environment variables:

- `ASR_STREAMING_MODE=quasistreaming|streaming`
- `ASR_QUASISTREAMING_STABLE_REPEATS=3`
- `VAD_THRESHOLD=0.3`
- `VAD_MIN_SILENCE_DURATION=0.8`
- `VAD_MIN_SPEECH_DURATION=1.5`
- `VAD_MAX_SPEECH_DURATION=8`
