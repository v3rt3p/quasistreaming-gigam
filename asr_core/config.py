from __future__ import annotations

import logging
import os

ASR_TARGET_SR = int(os.environ.get("ASR_TARGET_SR", "16000"))
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "ru")
LOG_JSON = os.environ.get("LOG_JSON", "false").lower() in {"1", "true", "yes", "on"}
LOG_LEVEL = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
