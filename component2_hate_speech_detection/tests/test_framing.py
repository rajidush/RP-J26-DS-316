"""Framing guard (proposal SO2, FR4; Guardian invariant I-3). MVP cases plus the safety fix."""
import pytest

from component2_hate_speech_detection.src.engine.framing import FramingGuard


@pytest.mark.parametrize("text", [
    "someone in the group chat told me to kys and i'm scared",
    "miss he keeps calling me a retard what do i do",
    "you should never tell someone to kys thats awful",
    "we learned about hate speech in class today",
])
def test_reporting_language_is_detected(text):
    assert FramingGuard().detect(text).reporting


@pytest.mark.parametrize("text", ["you should kys", "shut up faggot", "i'm going to hurt you after school"])
def test_direct_abuse_is_not_reporting(text):
    assert not FramingGuard().detect(text).reporting


def test_invariant_I3_framing_caps_below_lowest_threshold_but_never_zeroes(decision_params):
    guard = FramingGuard()
    for raw in (0.6, 0.88, 0.98):
        capped, _ = guard.apply(raw, "he called me a retard what do i do")
        assert 0 < capped < min(decision_params["persona_theta"].values())


def test_already_low_score_is_not_raised():
    capped, _ = FramingGuard().apply(0.08, "we learned about hate speech in class today")
    assert capped == 0.08
