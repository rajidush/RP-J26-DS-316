"""
Seam 2 tests: the session assessor EVALUATE calls (src/assessor.py).

The contract suite runs against every arm in ARMS; a new classifier arm joins
by being added there. The derivation tests cover the one rule every arm shares
(ADR 0006). The rules-arm tests below them pin the codebook rules
(src/emotion_rules.py, ticket 02), including the PP1 panel's examples.
"""
import itertools
from collections import Counter

import pytest

from component3_socratic_educator.src.assessor import (
    CURRENT_DISTRESS, DISTRESS_SEVERITY, DISTRESS_SEVERITIES, EXPRESSED_EMOTION, EXPRESSED_EMOTIONS, LABEL_DEFAULTS, STANCE, STANCES,
    ReplyAssessment, RulesAssessor, SessionAssessment, derive_emotional_state,
)
from component3_socratic_educator.src.fsm_controller import DialogueState, Turn
from component3_socratic_educator.src.schemas import EmotionalState as E

ARMS = [pytest.param(RulesAssessor(), id="rules")]
VOCABULARIES = {EXPRESSED_EMOTION: EXPRESSED_EMOTIONS, DISTRESS_SEVERITY: DISTRESS_SEVERITIES, STANCE: STANCES}


def session(*replies):
    """Turns as the controller records them: an INTERCEPT line, INQUIRE lines, then EVALUATE."""
    turns = [Turn(DialogueState.INTERCEPT if i == 0 else DialogueState.INQUIRE, f"question {i}?", r)
             for i, r in enumerate(replies)]
    return turns + [Turn(DialogueState.EVALUATE, "[internal evaluation, no child-facing text]")]


SESSIONS = [
    session("I clicked a link by accident", "I closed it after that"),
    session("It's not my fault.", "I feel so scared."),
    session("idk", "leave me alone"),
    session("It was really scary.", "I think I should talk to my mom about it."),
    session("lol I got lost in the game", "I'm really angry at them", "none of your business"),
    session("", "ok"),
    session(),
]


# -- contract: every arm --------------------------------------------------

@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("turns", SESSIONS)
def test_one_result_per_child_reply_in_order(arm, turns):
    result = arm.assess(turns)
    assert isinstance(result, SessionAssessment)
    assert [r.reply for r in result.replies] == [t.child_response for t in turns if t.child_response is not None]
    assert all(isinstance(r, ReplyAssessment) for r in result.replies)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("turns", SESSIONS)
def test_every_reply_has_the_three_parts_from_the_codebook(arm, turns):
    for r in arm.assess(turns).replies:
        assert set(r.labels) == set(VOCABULARIES)
        assert all(r.labels[part] in values for part, values in VOCABULARIES.items()), r.labels


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("turns", SESSIONS)
def test_every_label_but_the_default_comes_with_evidence(arm, turns):
    # The phrases that fired, or a model's top classes and scores (spec story 24).
    for r in arm.assess(turns).replies:
        assert set(r.evidence) == set(r.labels)
        assert all(isinstance(e, list) and all(isinstance(x, str) for x in e) for e in r.evidence.values())
        assert all(r.evidence[part] for part, value in r.labels.items() if value != LABEL_DEFAULTS[part]), r


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("turns", SESSIONS)
def test_the_session_state_is_the_shared_derivation_of_the_reply_labels(arm, turns):
    result = arm.assess(turns)
    assert result.emotional_state == derive_emotional_state(result.replies)
    assert isinstance(result.constructed_plan, bool)


# -- the derivation every arm shares (ADR 0006, dominant stance agreed 29 Oct) --

def labelled(*parts):
    """Reply results from (distress, stance) pairs; the expressed emotion is neutral."""
    return [ReplyAssessment("", {EXPRESSED_EMOTION: "neutral", DISTRESS_SEVERITY: d, STANCE: s}, {EXPRESSED_EMOTION: [], DISTRESS_SEVERITY: [], STANCE: []})
            for d, s in parts]


PAIRS = list(itertools.product(DISTRESS_SEVERITIES, STANCES))
SEQUENCES = [seq for n in (1, 2, 3) for seq in itertools.product(PAIRS, repeat=n)]


def expected_state(parts):
    """The rule written out once more, independently: any current distress; otherwise the
    stance group with most replies (neutral does not vote), ties in the ADR order."""
    if any(d in CURRENT_DISTRESS for d, _ in parts):
        return E.DISTRESSED
    groups = Counter({"sustain_talk": "defensive", "discord": "defensive", "disengaged": "unclear",
                      "change_talk": "calm"}.get(s) for _, s in parts)
    groups.pop(None, None)
    if not groups:
        return E.CALM
    top = max(groups.values())
    return next(E(g) for g in ("defensive", "unclear", "calm") if groups[g] == top)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_the_derivation_over_every_combination_of_up_to_three_replies(n):
    for parts in (seq for seq in SEQUENCES if len(seq) == n):
        assert derive_emotional_state(labelled(*parts)) == expected_state(parts), parts


def test_the_expressed_emotion_never_changes_the_session_state():
    for d, s in PAIRS:
        states = set()
        for emotion in EXPRESSED_EMOTIONS:
            reply = ReplyAssessment("", {EXPRESSED_EMOTION: emotion, DISTRESS_SEVERITY: d, STANCE: s}, {EXPRESSED_EMOTION: [], DISTRESS_SEVERITY: [],
                                                                                   STANCE: []})
            states.add(derive_emotional_state([reply]))
        assert len(states) == 1, (d, s)


@pytest.mark.parametrize("parts,state", [
    ([("current_mild", "change_talk")], E.DISTRESSED),                      # distress outranks a plan
    ([("past", "neutral")], E.CALM),                                        # past distress is not distressed
    ([("none", "disengaged"), ("none", "change_talk")], E.UNCLEAR),         # a tie goes in the ADR order
    ([("none", "neutral"), ("none", "neutral"), ("none", "change_talk")], E.CALM),
    ([("none", "sustain_talk"), ("none", "change_talk"), ("none", "change_talk")], E.CALM),   # dominant, not any
    ([("none", "disengaged"), ("none", "disengaged"), ("none", "discord")], E.UNCLEAR),
    ([], E.UNCLEAR),                                                        # nothing to judge
])
def test_worked_examples_of_the_derivation(parts, state):
    assert derive_emotional_state(labelled(*parts)) == state


# -- rules arm: the codebook rules --------------------------------------------

def labels_of(reply):
    (result,) = RulesAssessor().assess(session(reply)).replies
    return result.labels[EXPRESSED_EMOTION], result.labels[DISTRESS_SEVERITY], result.labels[STANCE]


# The PP1 panel's examples (23 Oct viva) and ticket 02's distress-with-a-plan case.
@pytest.mark.parametrize("reply,expected", [
    ("I'm not scared at all", ("neutral", "none", "neutral")),                    # negation
    ("lol I got lost in the game", ("joy", "none", "neutral")),                   # a word out of context
    ("I feel like nobody cares about me", ("sadness", "current_mild", "neutral")),  # implicit feeling
    ("I'm really angry at them", ("anger", "none", "neutral")),                   # anger is not distress
    ("I'm really scared but I'll tell my mum", ("fear", "current_mild", "change_talk")),
])
def test_the_panels_examples(reply, expected):
    assert labels_of(reply) == expected


# Known failures: kept, not removed, so a later arm can be measured against them.
@pytest.mark.xfail(strict=True, reason="rules arm: implicit fear with no feeling word")
def test_known_failure_implicit_fear():
    assert labels_of("everyone at school is going to see it") == ("fear", "current_mild", "neutral")


@pytest.mark.xfail(strict=True, reason="rules arm: rumination with no feeling word")
def test_known_failure_rumination():
    assert labels_of("I keep thinking about it and I can't sleep") == ("fear", "current_mild", "neutral")


@pytest.mark.parametrize("reply,expected", [
    # distress severity: current unless the child says it is over; self-harm language never downgrades
    ("It was kinda scary", ("fear", "current_mild", "neutral")),                 # softened still counts (ADR 0006)
    ("I was scared but I'm fine now", ("fear", "past", "neutral")),
    ("I can't stop crying", ("sadness", "current_acute", "neutral")),
    ("I wanted to disappear but I'm fine now", ("sadness", "current_acute", "neutral")),
    ("I don't know what to do anymore", ("sadness", "current_mild", "neutral")),
    ("ew that was gross", ("disgust", "none", "neutral")),
    ("wait what, that was so weird", ("surprise", "none", "neutral")),
    # stance
    ("Next time I'll close it and tell my mum", ("neutral", "none", "change_talk")),
    ("I should have closed it instead", ("neutral", "none", "change_talk")),
    ("next time I will do exactly the same thing", ("neutral", "none", "sustain_talk")),
    ("I think I saw something weird", ("surprise", "none", "neutral")),           # a hedge is not a plan
    ("It's not my fault", ("neutral", "none", "sustain_talk")),
    ("It’s not my fault", ("neutral", "none", "sustain_talk")),               # curly apostrophe
    ("none of your business", ("neutral", "none", "discord")),
    ("leave me alone", ("neutral", "none", "disengaged")),                        # opting out, not loneliness
    ("idk", ("neutral", "none", "disengaged")),
    ("nothing", ("neutral", "none", "disengaged")),
    ("", ("neutral", "none", "disengaged")),
    ("hmm", ("neutral", "none", "neutral")),                                      # MISC: a filler is follow/neutral
    ("I clicked a link by accident", ("neutral", "none", "neutral")),
])
def test_rules_reply_labels(reply, expected):
    assert labels_of(reply) == expected


# From /code-review's probes (29 Oct), written from the codebook, not from the
# evaluation data: missing any of these is the costliest error under ADR 0006.
@pytest.mark.parametrize("reply,expected", [
    # self-harm language is current acute distress, however it is worded, and a negation
    # in front of it does not cancel it
    ("I wanna die", ("sadness", "current_acute", "neutral")),
    ("I don't want to live anymore", ("sadness", "current_acute", "neutral")),
    ("I wish I was dead", ("sadness", "current_acute", "neutral")),
    ("I keep thinking about killing myself", ("sadness", "current_acute", "neutral")),
    ("I've been cutting again", ("sadness", "current_acute", "neutral")),
    ("kms", ("sadness", "current_acute", "neutral")),
    ("I didn't mean to hurt myself", ("sadness", "current_acute", "sustain_talk")),
    ("I panicked", ("fear", "current_acute", "neutral")),
    ("I couldn't breathe", ("fear", "current_acute", "neutral")),
    # a negation cancels a feeling only when it negates that feeling
    ("I've never been so scared", ("fear", "current_mild", "neutral")),
    ("no im scared", ("fear", "current_mild", "neutral")),
    ("i dont know im scared", ("fear", "current_mild", "disengaged")),
    ("I'm not even that scared", ("neutral", "none", "neutral")),
    ("I don't feel sad", ("neutral", "none", "neutral")),
    # the feeling is past only when the child says it is over after it, and not "still"
    ("I'm still scared of the people I used to talk to", ("fear", "current_mild", "neutral")),
    ("I used to be fine but now I'm scared all the time", ("fear", "current_mild", "neutral")),
    ("I'm scared, it's not better now", ("fear", "current_mild", "neutral")),
    # implicit distress with a fixed wording
    ("I'm not okay", ("sadness", "current_mild", "neutral")),
    ("I don't feel safe", ("fear", "current_mild", "neutral")),
    ("I hate myself", ("sadness", "current_mild", "neutral")),
    ("everyone hates me", ("sadness", "current_mild", "neutral")),
    ("I want it to stop, they keep messaging me", ("sadness", "current_mild", "neutral")),
    # not the child's distress
    ("my mum would panic", ("neutral", "none", "neutral")),
    ("I was home alone", ("neutral", "none", "neutral")),
    # a plan the child turns down or only imagines is not change talk
    ("I could tell my mum but I won't", ("neutral", "none", "sustain_talk")),
    ("If I tell my mum she'll take my phone", ("neutral", "none", "neutral")),
])
def test_rules_safety_regressions(reply, expected):
    assert labels_of(reply) == expected


def test_rules_evidence_is_the_phrases_that_fired():
    (r,) = RulesAssessor().assess(session("I'm really scared but I'll tell my mum")).replies
    assert r.evidence == {EXPRESSED_EMOTION: ["scared"], DISTRESS_SEVERITY: ["scared"], STANCE: ["i'll tell"]}


@pytest.mark.parametrize("replies,expected", [
    (["I clicked a link by accident", "I closed it after that"], E.CALM),
    (["It's not my fault.", "I didn't do anything."], E.DEFENSIVE),
    (["I feel so alone.", "My heart feels heavy."], E.DISTRESSED),
    (["I don't know.", "Just... quiet."], E.UNCLEAR),
    (["I'm not sure", "I'm not sure", "I'm not sure"], E.UNCLEAR),
    (["I guess I should tell my dad"], E.CALM),
    (["I do not want to talk"], E.UNCLEAR),
    (["can you stop asking me that"], E.UNCLEAR),
    ([], E.UNCLEAR),
    (["", "   "], E.UNCLEAR),
    (["no", "ok"], E.UNCLEAR),
    (["hmm", "I was gaming", "I'll tell my mum"], E.CALM),
    (["idk", "I'll tell my mum"], E.UNCLEAR),                                  # a tie goes in the ADR order
    (["It's not my fault.", "I don't know."], E.DEFENSIVE),
    # changed by ADR 0006: current distress now outranks a plan, and softened feelings count
    (["It was really scary.", "I think I should talk to my mom about it."], E.DISTRESSED),
    (["I'm so scared and alone.", "Maybe I should tell someone."], E.DISTRESSED),
    (["It was kinda scary.", "It made me a little upset."], E.DISTRESSED),
    (["I'm really angry at them", "I'm not scared at all"], E.CALM),
])
def test_rules_session_states(replies, expected):
    assert RulesAssessor().assess(session(*replies)).emotional_state == expected


@pytest.mark.parametrize("replies,expected", [
    (["hmm", "I should have closed it instead"], True),
    (["I couldn't stop watching"], False),   # whole words: "i could" not inside "i couldn't"
    (["I clicked a link by accident"], False),
])
def test_rules_plan_is_still_the_pp1_self_regulation_keywords(replies, expected):
    # Ticket 04 replaces this with constructed vs chosen vs echoed plans.
    assert RulesAssessor().assess(session(*replies)).constructed_plan is expected
