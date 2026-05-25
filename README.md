# quasistreaming-gigam

Simple quasistreaming server with GigaAM and VAD.

## CPU/CUDA Inference

This repository includes two subsets of files:

- `Dockerfile.cpu`, `requirements.cpu.txt`: For CPU inference.
- `Dockerfile.cuda`, `requirements.cuda.txt`: For NVIDIA CUDA inference.

## Client

A simple client is provided in `client.py` to test the server:

``` shell
python client.py input.wav ws://localhost:8080
```
