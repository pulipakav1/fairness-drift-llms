import socket

import pandas as pd
import pytest

from src.config import CONTEXT_TRANSITIONS, DATASETS, FAMILIES, RELEASE_CHAINS, STRICT_TRANSITIONS, release_transitions
from src.data import filter_responses, response_issue
from src.metrics import add_scores, sentiment_score, stereotype_score, toxicity_score


def test_model_and_transition_counts():
    models = [m for family in FAMILIES.values() for m in family]
    chain_models = [m for chain in RELEASE_CHAINS.values() for m in chain]
    transitions = list(release_transitions())
    assert len(models) == len(set(models)) == 19
    assert len(chain_models) == 11 and len(set(models) - set(chain_models)) == 8
    assert len(transitions) == 8
    assert STRICT_TRANSITIONS <= set(transitions) and len(set(transitions) - STRICT_TRANSITIONS) == 5
    assert any("claude-opus-4-1-20250805" in t for t in CONTEXT_TRANSITIONS)
    assert DATASETS == sorted(DATASETS) and len(DATASETS) == 7


def test_toxicity_counts_each_term_once():
    assert toxicity_score("A calm reply.") == 0.0
    assert toxicity_score("Hate, hate and more HATE.") == 0.1
    assert toxicity_score("Racism and sexism are discrimination.") == pytest.approx(0.3)


def test_stereotype_counts_each_pattern_once():
    assert stereotype_score("A plain sentence.") == 0.0
    assert stereotype_score("They always, always say that.") == 0.2
    assert stereotype_score("All cats are typical and never every dog is.") == 0.8


def test_sentiment_range():
    assert sentiment_score("") == 0.5
    assert 0.5 < sentiment_score("This is wonderful.") <= 1.0
    assert 0.0 <= sentiment_score("This is terrible.") < 0.5


def test_scoring_ignores_api_keys_and_network(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("scoring opened a network connection")
    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setenv("PERSPECTIVE_API_KEY", "must-be-ignored")
    scores = add_scores(pd.DataFrame({"response": ["They always hate violence.", "Fine."]}))
    assert scores["toxicity_score"].tolist() == [0.2, 0.0]
    assert scores["stereotype_score"].tolist() == [0.2, 0.0]


def test_filter_drops_unscorable_and_keeps_refusals():
    raw = pd.DataFrame({"response": [
        "A normal answer.", "", None, "Error: insufficient_quota", "[BLOCKED: SAFETY]",
        "[No text returned]", "I cannot help with that.",
    ]})
    issues = raw["response"].map(response_issue).tolist()
    assert issues == [None, "empty", "empty", "api_error", "policy_blocked", "no_text", "refusal"]
    assert filter_responses(raw)["response"].tolist() == ["A normal answer.", "I cannot help with that."]
