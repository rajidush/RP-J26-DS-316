"""
Unit tests for the EVALUATE-state emotional_state classifier (src/emotion_rules.py).
Agreement on the simulated dataset is measured separately by
evidence/evaluate_emotion_rules.py (dev C001-C020, held-out C021-C040).
"""
import pytest

from component3_socratic_educator.src.emotion_rules import classify
from component3_socratic_educator.src.schemas import EmotionalState as E


@pytest.mark.parametrize("replies,expected", [
    # one clear case per label
    (["I clicked a link by accident", "I closed it after that"], E.CALM),
    (["It's not my fault.", "I didn't do anything."], E.DEFENSIVE),
    (["I feel so alone.", "My heart feels heavy."], E.DISTRESSED),
    (["I don't know.", "Just... quiet."], E.UNCLEAR),
    # no answer / bare refusal -> unclear, never defensive
    ([], E.UNCLEAR),
    (["", "   "], E.UNCLEAR),
    (["no", "ok"], E.UNCLEAR),
    # naming a feeling AND planning a safer step is regulation -> calm
    (["It was really scary.", "I think I should talk to my mom about it."], E.CALM),
    # distress that outweighs the planning still wins (3 distress vs 2 reflective)
    (["I'm so scared and alone and it hurts.", "Maybe I should tell someone."], E.DISTRESSED),
    # ...but an equal amount of planning counts as regulating (2 vs 2)
    (["I'm so scared and alone.", "Maybe I should tell someone."], E.CALM),
    # softened feelings are not distress
    (["It was kinda scary.", "It made me a little upset."], E.CALM),
    # "leave me alone" is opting out, not loneliness
    (["Just... leave me alone."], E.UNCLEAR),
    # "I don't know what to do" is distress, not a non-answer
    (["I don't know what to do anymore."], E.DISTRESSED),
    # curly apostrophes from the LLM are handled
    (["It’s not my fault.", "I didn’t see it."], E.DEFENSIVE),
    # ties go distressed > defensive > unclear
    (["It's not my fault.", "It's scary."], E.DISTRESSED),
    (["It's not my fault.", "I don't know."], E.DEFENSIVE),
])
def test_classify(replies, expected):
    assert classify(replies).state == expected


def test_verdict_explains_itself():
    verdict = classify(["It's not my fault.", "I feel so scared."])
    assert verdict.hits == {"distress": ["scared"], "defensive": ["not my fault"]}
    assert verdict.counts == {"distress": 1, "defensive": 1}


def test_every_label_is_reachable():
    """The old word-count placeholder could never output `distressed`."""
    outputs = {classify(r).state for r in (
        ["I was just curious about it"], ["whatever, not my fault"],
        ["I feel so scared"], ["idk"],
    )}
    assert outputs == set(E)
