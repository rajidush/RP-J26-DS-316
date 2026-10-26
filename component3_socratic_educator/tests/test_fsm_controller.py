"""
Function 1 tests. Run with: pytest component3_socratic_educator/tests -v
"""
import json
import sys
from pathlib import Path
import pytest
from component3_socratic_educator.src.fsm_controller import FSMController, SessionTranscript
from component3_socratic_educator.src import config
from component3_socratic_educator.src.schemas import RiskCategory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component3_socratic_educator.src.fsm_controller import DialogueState, FSMController
from component3_socratic_educator.src.schemas import RiskCategory, TriggerPayload


def make_trigger(confidence=0.9, risk_category=RiskCategory.EXPLICIT_VISUAL):
    return TriggerPayload.new(
        source_component="component1_screen_monitoring",
        content_type="image",
        risk_category=risk_category,
        confidence_score=confidence,
        app_or_window="browser",
    )


def test_full_session_completes_and_validates():
    controller = FSMController()
    trigger = make_trigger()
    responses = iter(["I'm not sure", "I guess I was just curious", "I should have closed it instead"])
    result, transcript = controller.run(trigger, respond=lambda _: next(responses))

    assert result.session_id == trigger.session_id
    assert result.dialogue_turns >= 1
    # will raise if it doesn't match docs/interface-contracts/comp3_to_comp4.schema.json
    result.validate()


def test_inquire_stops_at_max_attempts_even_with_short_answers():
    controller = FSMController()
    trigger = make_trigger()
    # every child response is a single word -> must still terminate via
    # MAX_INQUIRE_MOVES, not loop forever
    result, transcript = controller.run(trigger, respond=lambda _: "no")
    inquire_turns = [t for t in transcript.turns if t.state == DialogueState.INQUIRE]
    assert len(inquire_turns) <= 3


def test_high_confidence_trigger_produces_escalation():
    controller = FSMController()
    trigger = make_trigger(confidence=0.95)
    result, _ = controller.run(trigger, respond=lambda _: "whatever")
    assert result.risk_level.value == "high"
    assert result.escalation_flag is True


def test_low_confidence_trigger_does_not_escalate():
    controller = FSMController()
    trigger = make_trigger(confidence=0.2)
    result, _ = controller.run(trigger, respond=lambda _: "I was just looking around, nothing serious")
    assert result.escalation_flag is False

@pytest.mark.parametrize("rc", list(RiskCategory))
def test_intercept_never_names_flagged_content(rc):
    controller = FSMController()
    trigger = make_trigger(risk_category=rc)
    text = controller.opening_question(trigger).lower()
    # neither the category label nor its words may appear
    for word in rc.value.split("_"):
        assert word not in text

def test_intercept_is_reproducible():
    controller = FSMController()
    trigger = make_trigger()
    assert controller.opening_question(trigger) == controller.opening_question(trigger)
    
def test_inquire_stops_at_max_moves_with_short_answers():
    controller = FSMController()
    trigger = make_trigger()
    transcript = SessionTranscript(trigger=trigger)
    responses = controller.inquire_loop(trigger, respond=lambda _: "no", transcript=transcript)
    assert len(responses) == config.MAX_INQUIRE_MOVES


def test_inquire_logs_turns_to_transcript():
    controller = FSMController()
    trigger = make_trigger()
    transcript = SessionTranscript(trigger=trigger)
    controller.inquire_loop(trigger, respond=lambda _: "no", transcript=transcript)
    assert len(transcript.turns) == config.MAX_INQUIRE_MOVES
    
def test_run_actually_uses_inquire_loop():
    controller = FSMController()
    trigger = make_trigger()
    _, transcript = controller.run(trigger, respond=lambda _: "no")
    inquire_turns = [t for t in transcript.turns if t.state.name == "INQUIRE"]
    assert len(inquire_turns) == config.MAX_INQUIRE_MOVES


# --- 14 Oct fixes -------------------------------------------------------------

def _inquire_replies(replies):
    """Run a session; return the replies the child was asked for during INQUIRE."""
    it = iter(replies)
    _, transcript = FSMController().run(make_trigger(confidence=0.6), respond=lambda _: next(it, "ok"))
    return [t.child_response for t in transcript.turns if t.state == DialogueState.INQUIRE]


def test_a_short_reply_with_nothing_in_it_is_still_followed_up():
    assert _inquire_replies(["hmm", "ok", "no", "fine"]) == ["ok", "no", "fine"]


REFUSAL_CLOSE = "That's okay, you don't have to talk right now. I'm here if you want to talk later."


@pytest.mark.parametrize("refusal", ["leave me alone", "I don't want to talk", "I do not want to talk about it",
                                     "stop asking", "go away"])
def test_a_refusal_ends_inquire_at_once(refusal):
    # 12 Oct run: the withdrawn child was asked again after "leave me alone".
    assert _inquire_replies(["hmm", refusal, "never asked"]) == [refusal]


def test_a_refusal_to_the_opening_line_skips_inquire():
    assert _inquire_replies(["leave me alone", "never asked"]) == []


def test_a_refusing_child_who_is_not_escalated_hears_that_it_is_okay_not_to_talk():
    it = iter(["hmm", "leave me alone"])
    result, transcript = FSMController().run(make_trigger(confidence=0.6), respond=lambda _: next(it))
    assert result.escalation_flag is False
    assert transcript.turns[-1].system_text == REFUSAL_CLOSE


def test_a_refusing_child_who_is_escalated_still_hears_the_parent_line():
    it = iter(["hmm", "leave me alone"])
    result, transcript = FSMController().run(make_trigger(confidence=0.9), respond=lambda _: next(it))
    assert result.escalation_flag is True
    assert "parent" in transcript.turns[-1].system_text


def test_a_distressed_child_is_escalated_even_at_moderate_risk():
    # 12 Oct run: "I can't stop crying about it" at confidence 0.71 was not escalated (ADR 0005).
    it = iter(["It was really scary", "I can't stop crying about it"])
    result, transcript = FSMController().run(make_trigger(confidence=0.6), respond=lambda _: next(it, "ok"))
    assert result.risk_level.value == "moderate" and result.emotional_state.value == "distressed"
    assert result.escalation_flag is True
    assert "parent" in transcript.turns[-1].system_text
    assert result.dialogue_summary.endswith("escalated: emotional_state=distressed")


def test_the_summary_says_why_a_high_risk_session_escalated():
    result, _ = FSMController().run(make_trigger(confidence=0.9), respond=lambda _: "I clicked a link by accident")
    assert result.dialogue_summary.endswith("escalated: risk_level=high")


def test_a_calm_moderate_risk_session_is_not_escalated_and_says_nothing_about_it():
    result, _ = FSMController().run(make_trigger(confidence=0.6), respond=lambda _: "I clicked a link by accident")
    assert result.escalation_flag is False and "escalated" not in result.dialogue_summary


def test_a_refusal_with_distress_ends_inquire_and_still_escalates():
    # Agreed 14 Oct: the refusal is respected, and the distress gets a parent involved.
    it = iter(["hmm", "leave me alone, I'm so scared", "never asked"])
    result, transcript = FSMController().run(make_trigger(confidence=0.6), respond=lambda _: next(it))
    assert [t.child_response for t in transcript.turns if t.state == DialogueState.INQUIRE] == [
        "leave me alone, I'm so scared"]
    assert result.emotional_state.value == "distressed" and result.escalation_flag is True
    assert "parent" in transcript.turns[-1].system_text


class FakeAssessor:
    """Returns a fixed session result, labels every reply with `stance` (or the
    stance `stances` maps it to), and remembers the turns it was last given."""

    def __init__(self, state, plan, stance="neutral", stances=None):
        self.state, self.plan, self.stance, self.stances = state, plan, stance, stances or {}
        self.seen = None

    def assess(self, turns):
        from component3_socratic_educator.src.assessor import STANCE, ReplyAssessment, SessionAssessment
        self.seen = [t.child_response for t in turns if t.child_response is not None]
        replies = [ReplyAssessment(r, {STANCE: self.stances.get(r, self.stance)}, {STANCE: []})
                   for r in self.seen]
        return SessionAssessment(replies=replies, emotional_state=self.state, constructed_plan=self.plan)


def test_evaluate_takes_the_emotional_state_and_plan_from_the_injected_assessor():
    from component3_socratic_educator.src.schemas import EmotionalState
    assessor = FakeAssessor(EmotionalState.DISTRESSED, plan=True)
    result, transcript = FSMController(assessor=assessor).run(
        make_trigger(confidence=0.6), respond=lambda _: "I clicked a link by accident")
    assert assessor.seen == [t.child_response for t in transcript.turns if t.child_response is not None]
    assert result.emotional_state.value == "distressed" and result.self_regulation_shown is True
    assert result.escalation_flag is True and "parent" in transcript.turns[-1].system_text


def test_the_assessor_not_the_phrase_rules_decides_escalation():
    from component3_socratic_educator.src.schemas import EmotionalState
    result, _ = FSMController(assessor=FakeAssessor(EmotionalState.CALM, plan=False)).run(
        make_trigger(confidence=0.6), respond=lambda _: "I'm so scared and alone and it hurts")
    assert result.emotional_state.value == "calm" and result.escalation_flag is False
    assert result.self_regulation_shown is False


# --- PP2 ticket 03: Socratic moves per age band --------------------------------

import jsonschema  # noqa: E402

from component3_socratic_educator.src.grammar_decoder import INQUIRE_TEMPLATES  # noqa: E402
from component3_socratic_educator.src.schemas import _CONTRACTS_DIR, EmotionalState  # noqa: E402

COMP3_TO_COMP4 = json.loads((_CONTRACTS_DIR / "comp3_to_comp4.schema.json").read_text(encoding="utf-8"))


def _session(replies, assessor=None, **run_kwargs):
    it = iter(replies)
    return FSMController(assessor=assessor).run(make_trigger(confidence=0.6), respond=lambda _: next(it, "ok"),
                                                **run_kwargs)


def _inquire(transcript):
    return [t for t in transcript.turns if t.state == DialogueState.INQUIRE]


def _moves(transcript):
    return [(t.move, t.easier) for t in _inquire(transcript)]


ENGAGED = ["hmm", "I was playing a game", "It made me feel bad inside", "ok"]   # no refusal, plan or "don't know"


def test_inquire_asks_clarify_then_consequences_then_alternatives():
    _, transcript = _session(ENGAGED)
    assert _moves(transcript) == [("clarify", False), ("consequences", False), ("alternatives", False)]


def test_each_move_is_worded_from_its_template_for_the_band():
    for band in config.AGE_BANDS:
        _, transcript = _session(ENGAGED, age_band=band)
        assert [t.system_text for t in _inquire(transcript)] == [
            INQUIRE_TEMPLATES[move][band][0] for move in config.SOCRATIC_MOVES]


@pytest.mark.parametrize("band", [None, "", "16-18", "8\u201310", "adult", 9])
def test_a_missing_or_invalid_band_gives_the_8_10_wording(band):
    _, transcript = _session(ENGAGED, age_band=band)
    assert [t.system_text for t in _inquire(transcript)] == [
        INQUIRE_TEMPLATES[move]["8-10"][0] for move in config.SOCRATIC_MOVES]
    assert {t.age_band for t in transcript.turns} == {"8-10"}


def test_dont_know_repeats_the_move_in_its_easier_form_and_counts_toward_the_cap():
    _, transcript = _session(["hmm", "idk", "I was playing a game", "It made me feel bad inside"], age_band="11-13")
    assert _moves(transcript) == [("clarify", False), ("clarify", True), ("consequences", False)]
    assert _inquire(transcript)[1].system_text == INQUIRE_TEMPLATES["clarify"]["11-13"][1]


def test_dont_know_to_the_easier_form_moves_on_rather_than_asking_it_again():
    _, transcript = _session(["hmm", "idk", "not sure", "I was playing a game"])
    assert _moves(transcript) == [("clarify", False), ("clarify", True), ("consequences", False)]


@pytest.mark.parametrize("reply", ["I don't know, it was so scary", "idk it wasn't my fault"])
def test_dont_know_with_a_feeling_or_denial_in_it_is_answered_by_the_next_move(reply):
    _, transcript = _session(["hmm", reply, "ok"])
    assert _moves(transcript)[:2] == [("clarify", False), ("consequences", False)]


def test_inquire_never_asks_more_than_three_moves():
    _, transcript = _session(["hmm"] + ["idk"] * 6)
    assert len(_inquire(transcript)) == config.MAX_INQUIRE_MOVES == 3


def test_a_refusal_mid_inquire_ends_it_at_once():
    _, transcript = _session(["hmm", "I was playing a game", "leave me alone", "never asked"])
    assert _moves(transcript) == [("clarify", False), ("consequences", False)]


@pytest.mark.parametrize("plan", ["Next time I'll close it and tell my mum", "I should close it instead"])
def test_a_safer_plan_in_the_childs_own_words_ends_inquire_after_the_current_move(plan):
    _, transcript = _session(["hmm", plan, "never asked"])
    assert _moves(transcript) == [("clarify", False)]


def test_the_plan_rule_is_the_assessors_change_talk_stance_not_a_keyword():
    stated = "I will be careful from now on"
    assessor = FakeAssessor(EmotionalState.CALM, plan=True, stances={stated: "change_talk"})
    _, transcript = _session(["hmm", "I was playing a game", stated, "never asked"], assessor=assessor)
    assert _moves(transcript) == [("clarify", False), ("consequences", False)]

    keyword = "next time I will do exactly the same thing instead"     # the rules would fire; this assessor says no
    _, transcript = _session(["hmm", keyword], assessor=FakeAssessor(EmotionalState.CALM, plan=False))
    assert len(_inquire(transcript)) == config.MAX_INQUIRE_MOVES


def test_every_turn_records_its_state_move_band_and_where_its_line_came_from():
    _, transcript = _session(["hmm", "idk", "I was playing a game", "ok"], age_band="14-15")
    rows = [(t.state.name, t.move, t.easier, t.age_band, t.source, t.fallback_reason) for t in transcript.turns]
    assert rows == [
        ("INTERCEPT", None, False, "14-15", "fallback", "model_unreachable"),   # tests block the live model
        ("INQUIRE", "clarify", False, "14-15", "template", None),
        ("INQUIRE", "clarify", True, "14-15", "template", None),
        ("INQUIRE", "consequences", False, "14-15", "template", None),
        ("EVALUATE", None, False, "14-15", None, None),                          # no child-facing line
        ("CONTRACT", None, False, "14-15", "template", None),
    ]


def test_the_band_can_be_set_once_on_the_controller():
    it = iter(ENGAGED)
    _, transcript = FSMController(age_band="14-15").run(make_trigger(confidence=0.6), lambda _: next(it, "ok"))
    assert _inquire(transcript)[0].system_text == INQUIRE_TEMPLATES["clarify"]["14-15"][0]
    it = iter(ENGAGED)
    _, transcript = FSMController(age_band="14-15").run(make_trigger(confidence=0.6), lambda _: next(it, "ok"),
                                                        age_band="11-13")
    assert _inquire(transcript)[0].system_text == INQUIRE_TEMPLATES["clarify"]["11-13"][0]   # per run wins


CHILDREN = {
    "engaged": ENGAGED,
    "dont_know": ["idk", "idk", "not sure", "no idea", "dunno"],
    "refuses_at_once": ["leave me alone"],
    "refuses_later": ["hmm", "whatever", "stop asking"],
    "plans": ["I was gaming", "I'll tell my dad next time"],
    "distressed": ["It was really scary", "I can't stop crying about it", "I feel sick"],
    "silent": [""],
}


@pytest.mark.parametrize("band", [*config.AGE_BANDS, None])
@pytest.mark.parametrize("child", CHILDREN)
@pytest.mark.parametrize("confidence", [0.3, 0.6, 0.9])
def test_every_session_record_validates_against_the_component_4_contract(band, child, confidence):
    it = iter(CHILDREN[child])
    record, _ = FSMController().run(make_trigger(confidence=confidence), lambda _: next(it, "ok"), age_band=band)
    jsonschema.validate(record.to_dict(), COMP3_TO_COMP4)
