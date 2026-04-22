"""Shared FastAPI dependencies: logger and rate limiter."""

from __future__ import annotations

import logging

import structlog
from slowapi import Limiter
from slowapi.util import get_remote_address

# JSON-ish structured logging. Easy to grep and friendly to log aggregators.
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("nexus.api")

# Per-IP rate limiter. Exempts local dev traffic if you ever want to by
# customizing the key function, but for now we rate-limit everything.
limiter = Limiter(key_func=get_remote_address)
