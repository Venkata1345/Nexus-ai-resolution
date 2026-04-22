"""Shared fixtures for the Nexus test suite.

Most tests avoid touching disk, MLflow, or sentence-transformers; where
those would be pulled in, they're replaced with stand-ins that behave
just enough like the real thing.
"""

from __future__ import annotations

import os

# Ensure required env vars exist before Settings loads. Tests never hit the
# real Gemini API, but the Settings class declares GEMINI_API_KEY as required.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
