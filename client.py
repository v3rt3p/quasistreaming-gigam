#!/usr/bin/env python3
import asyncio
import array
import json
import sys
import numpy as np
import soundfile as sf
import websockets

SERVER_URL = "ws://localhost:8080"
CHUNK_SAMPLES = 1600  # 100 ms at 16 kHz


async def run(wav_path: str, url: str) -> None:
    data, sample_rate = sf.read(wav_path, dtype="int16", always_2d=False)

    # Mix down to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.int16)

    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"sample_rate": sample_rate}))

        # Stream audio in chunks
        for i in range(0, len(data), CHUNK_SAMPLES):
            chunk = data[i : i + CHUNK_SAMPLES]
            await ws.send(array.array("h", chunk).tobytes())

        # Keep the socket open across utterance boundaries. The server uses
        # end_of_utt to delimit results and closes only when the peer is done.
        try:
            async for message in ws:
                result = json.loads(message)
                print(result.get("text", ""), flush=True)
        except websockets.exceptions.ConnectionClosedOK:
            pass


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.wav> [server_url]", file=sys.stderr)
        sys.exit(1)

    wav_path = sys.argv[1]
    url = sys.argv[2] if len(sys.argv) > 2 else SERVER_URL

    asyncio.run(run(wav_path, url))


if __name__ == "__main__":
    main()
