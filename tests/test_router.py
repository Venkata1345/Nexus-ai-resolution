"""Tests for the intent router. We only exercise pure logic -- the real
sentence-transformer and XGBoost classifier are replaced with fakes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from langchain_core.messages import HumanMessage

import src.agents.router as router_mod
from src.agents.router import _predict, predict_intent_node


class _FakeEncoder:
    """Returns a fixed 3-d vector per call; shape matches real encoder output."""

    def __init__(self, vec: list[float]):
        self.vec = np.array([vec], dtype=np.float32)

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        # Real encoder returns (N, d); mirror that for the batch of len(texts).
        return np.vstack([self.vec] * len(texts))


class _FakeModel:
    """predict_proba returns a fixed probability row."""

    def __init__(self, probs: list[float]):
        self.probs = np.array([probs])

    def predict_proba(self, X):
        # Return one prob row per input row; X is (N, d).
        return np.repeat(self.probs, X.shape[0], axis=0)


class _FakeLabelEncoder:
    def __init__(self, classes: list[str]):
        self.classes = classes

    def inverse_transform(self, idxs):
        return np.array([self.classes[i] for i in idxs])


def test_predict_returns_argmax_class_and_its_probability():
    encoder = _FakeEncoder([0.1, 0.2, 0.3])
    model = _FakeModel([0.1, 0.7, 0.2])
    labels = _FakeLabelEncoder(["a", "b", "c"])

    intent, confidence = _predict("hi", encoder, model, labels)
    assert intent == "b"
    assert confidence == 0.7


def test_predict_tie_takes_first_argmax():
    encoder = _FakeEncoder([0.0])
    model = _FakeModel([0.5, 0.5])
    labels = _FakeLabelEncoder(["first", "second"])

    intent, confidence = _predict("x", encoder, model, labels)
    assert intent == "first"
    assert confidence == 0.5


def test_predict_returns_plain_python_types():
    encoder = _FakeEncoder([0.0])
    model = _FakeModel([0.2, 0.8])
    labels = _FakeLabelEncoder(["a", "b"])

    intent, conf = _predict("x", encoder, model, labels)
    # Downstream graph state expects plain str/float, not numpy scalars.
    assert type(intent) is str
    assert type(conf) is float


def test_predict_intent_node_writes_state_and_uses_latest_message(monkeypatch):
    encoder = _FakeEncoder([0.1])
    model = _FakeModel([0.2, 0.8])
    labels = _FakeLabelEncoder(["order", "refund"])
    monkeypatch.setattr(router_mod, "_get_deps", lambda: (encoder, model, labels))

    state = {
        "messages": [
            HumanMessage(content="older message that should be ignored"),
            HumanMessage(content="I want my money back"),
        ]
    }
    out = predict_intent_node(state)

    assert out == {"intent": "refund", "intent_confidence": 0.8}


def test_predict_intent_node_calls_encoder_with_latest_text_only(monkeypatch):
    encoder = MagicMock(wraps=_FakeEncoder([0.0]))
    model = _FakeModel([1.0])
    labels = _FakeLabelEncoder(["only"])
    monkeypatch.setattr(router_mod, "_get_deps", lambda: (encoder, model, labels))

    predict_intent_node({"messages": [HumanMessage(content="hello")]})

    # Encoder was called once, with a list containing exactly the latest text.
    encoder.encode.assert_called_once()
    args, kwargs = encoder.encode.call_args
    assert args[0] == ["hello"]
    assert kwargs.get("normalize_embeddings") is True
