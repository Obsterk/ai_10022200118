"""
logger_config.py  —  structured logging for every pipeline stage.
ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)

Satisfies Part-D requirement: "Implement logging at each stage" with a
uniform JSON-style line format so we can grep/aggregate later.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import settings


class StageFormatter(logging.Formatter):
    """Pretty JSON-line formatter — easy to parse and easy to read."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts"   : datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "stage": getattr(record, "stage", "-"),
            "msg"  : record.getMessage(),
        }
        # attach any structured extras
        for key in ("query", "top_k", "chunk_ids", "scores", "latency_ms",
                    "prompt_tokens", "feedback"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "rag") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(settings.log_level)

    # stdout handler (human readable)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(StageFormatter())
    logger.addHandler(sh)

    # file handler (append-only)
    fh = logging.FileHandler(settings.log_file, encoding="utf-8")
    fh.setFormatter(StageFormatter())
    logger.addHandler(fh)

    logger.propagate = False
    return logger


def log_stage(logger: logging.Logger, stage: str, msg: str, **extra) -> None:
    """Helper that always adds a ``stage`` attribute."""
    logger.info(msg, extra={"stage": stage, **extra})


class Timer:
    """Context manager that logs elapsed time of a stage."""

    def __init__(self, logger: logging.Logger, stage: str, msg: str, **extra):
        self.logger, self.stage, self.msg, self.extra = logger, stage, msg, extra

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = (time.perf_counter() - self.t0) * 1000
        log_stage(self.logger, self.stage, f"{self.msg} done",
                  latency_ms=round(dt, 2), **self.extra)
