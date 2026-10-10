"""Analyzer + Verdict: evidence, explanation, privacy and the team contract (proposal FR7-FR9)."""
import json
from pathlib import Path

import jsonschema
import pytest

from component2_hate_speech_detection.src.engine.analyzer import Analyzer, Observation, TextSpan
from component2_hate_speech_detection.src.engine.policy import default_policy
from component2_hate_speech_detection.src.engine.verdict import redact

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[2] / "docs/interface-contracts/comp2_to_comp3.schema.json").read_text()
)


@pytest.fixture
def analyzer(lexicon_scorer):
    return Analyzer(scorer=lexicon_scorer)


def test_detection_produces_schema_valid_trigger_without_raw_text(analyzer):
    text = "nobody likes you go back to your country"
    v = analyzer.analyze_text(text, age=9)
    assert v.rung == "L3" and v.top_category == "bullying"
    payload = v.to_trigger_payload(platform="group_chat")
    jsonschema.validate(instance=payload, schema=SCHEMA)
    assert payload["risk_category"] == "cyberbullying"
    assert text not in json.dumps(payload)


def test_clean_chat_does_not_wake_c3(analyzer):
    v = analyzer.analyze_text("want to play minecraft after school?", age=9)
    assert v.rung == "L0" and v.to_trigger_payload() is None


def test_self_harm_maps_to_team_schema_and_is_urgent(analyzer):
    payload = analyzer.analyze_text("i don't want to live anymore", age=14).to_trigger_payload()
    jsonschema.validate(instance=payload, schema=SCHEMA)
    assert payload["risk_category"] == "self_harm_language"
    assert payload["context_metadata"]["urgent"] is True


def test_verdict_carries_explanation_evidence_and_versions(analyzer):
    v = analyzer.analyze_text("you should kys", age=10)
    assert "Matched rule" in v.explanation
    assert any(e.rule_id == "kys" for e in v.evidence)
    assert v.policy_version == default_policy().version and v.engine_version


def test_oldest_persona_keeps_no_snippets(analyzer):
    v = analyzer.analyze_text("you should kys", age=15)
    assert all(e.redacted_snippet == "" for e in v.evidence)


def test_screen_observation_blurs_only_the_harmful_block(analyzer):
    obs = Observation(source="screen", spans=[
        TextSpan("Inbox  Settings  Help", "ocr", [0.0, 0.0, 0.3, 0.05]),
        TextSpan("you should kys", "ocr", [0.2, 0.5, 0.6, 0.55]),
        TextSpan("see you at practice", "ocr", [0.2, 0.6, 0.6, 0.65]),
    ])
    v = analyzer.analyze(obs, age=10)
    assert v.alerts and v.regions == [[0.2, 0.5, 0.6, 0.55]]


def test_repeated_screen_text_is_scored_once(analyzer):
    obs = Observation(source="screen", spans=[TextSpan("hello there", "ocr"), TextSpan("you should kys", "ocr")])
    analyzer.analyze(obs, age=10)
    analyzer.analyze(obs, age=10)
    assert analyzer.cache.misses == 2 and analyzer.cache.hits == 2


def test_redaction_masks_pii():
    out = redact("email me at kid@example.com or call +94 77 123 4567, password: hunter2")
    assert "kid@example.com" not in out and "hunter2" not in out and "4567" not in out
