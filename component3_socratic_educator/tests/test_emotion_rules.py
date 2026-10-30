"""
The reply types INQUIRE reacts to (src/emotion_rules.py): a refusal and "I don't know".
The rules arm's labels are tested through the assessor, in tests/test_assessor.py;
agreement on the simulated dataset is measured by evidence/evaluate_emotion_rules.py.
"""
import pytest

from component3_socratic_educator.src.emotion_rules import is_dont_know, is_refusal


@pytest.mark.parametrize("reply,expected", [
    ("leave me alone", True), ("I don't want to talk", True), ("I do not want to talk about it", True),
    ("can you stop asking me that", True), ("please just go away now", True), ("I need some space", True),
    ("I feel so alone", False), ("I want to talk to my mum", False), ("", False),
])
def test_refusal(reply, expected):
    assert is_refusal(reply) is expected


@pytest.mark.parametrize("reply,expected", [
    ("idk", True), ("I don't know", True), ("I'm not sure", True), ("dunno", True), ("no idea", True),
    ("I dont really know", True), ("I don't get it", True),
    # it also says something, so the next move answers it, not the easier form
    ("I don't know, it was so scary", False), ("idk it wasn't my fault", False), ("idk, I'll tell my mum", False),
    # other non-answers and the lookahead's exceptions are not "I don't know"
    ("I don't know what to do", False), ("nothing", False), ("weird", False), ("ok", False), ("", False),
])
def test_dont_know_is_the_reply_type_inquire_repeats_a_move_for(reply, expected):
    assert is_dont_know(reply) is expected
