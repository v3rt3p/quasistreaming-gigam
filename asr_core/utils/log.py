from __future__ import annotations
import time
import json
import logging
from asr_core.config import LOG_JSON, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format="%(message)s")


def fw_log(event: str, **fields) -> None:
    payload = {"ts": time.time(), "evt": event, **fields}
    try:
        if LOG_JSON:
            logging.info(json.dumps(payload, ensure_ascii=False))
        else:
            kv = ", ".join(f"{k}={v}" for k, v in fields.items())
            logging.info(f"[ASR] {event}: {kv}")
    except Exception:
        pass
