"""Decision by age (proposal SO4, FR6, FR10; Guardian §6.5, invariants I-4, I-5)."""
from component2_hate_speech_detection.src.engine.decide import PERSONAS, RUNGS, decide, persona_for_age


def test_same_words_alert_for_a_young_child_but_not_a_teen():
    assert decide("bullying", 0.60, "P1_PROTECT").rung == "L3"
    assert decide("bullying", 0.60, "P3_RESPECT").rung == "L0"


def test_teen_gets_a_softer_rung_for_the_same_bullying():
    assert decide("bullying", 0.90, "P1_PROTECT").rung == "L3"
    assert decide("bullying", 0.90, "P3_RESPECT").rung == "L2"


def test_critical_categories_use_one_threshold_for_every_age_and_are_urgent():
    for persona in PERSONAS:
        d = decide("self_harm", 0.58, persona)
        assert d.rung == "L4" and d.urgent and d.recommended_action == "block"


def test_threat_is_urgent_only_when_very_confident():
    assert not decide("threat", 0.80, "P1_PROTECT").urgent
    assert decide("threat", 0.95, "P1_PROTECT").urgent


def test_profanity_only_informs_the_parent():
    d = decide("profanity", 0.95, "P1_PROTECT")
    assert d.rung == "L1" and d.recommended_action == "notify"


def test_invariant_I4_abstained_items_never_exceed_L1():
    for persona in PERSONAS:
        assert RUNGS.index(decide("threat", 0.99, persona, abstained=True).rung) <= RUNGS.index("L1")


def test_invariant_I5_decision_is_a_pure_function():
    for args in [("bullying", 0.7, "P2_GUIDE"), ("self_harm", 0.6, "P3_RESPECT"), ("threat", 0.95, "P1_PROTECT")]:
        assert len({decide(*args) for _ in range(20)}) == 1


def test_parent_cap_limits_the_rung():
    assert decide("self_harm", 0.9, "P1_PROTECT", parent_caps={"P1_PROTECT": "L2"}).rung == "L2"


def test_persona_for_age_bands():
    assert [persona_for_age(a) for a in (8, 10, 11, 13, 14, 15)] == [
        "P1_PROTECT", "P1_PROTECT", "P2_GUIDE", "P2_GUIDE", "P3_RESPECT", "P3_RESPECT"]
