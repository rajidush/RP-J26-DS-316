"""
The measurement-only audit of delivered INTERCEPT lines (plan, Mon 12 Oct).
The post-check decides what the child sees; the audit only measures, with
broader rules written before the unseen run, so a violation rate is not 0 by
construction. Run with: pytest component3_socratic_educator/tests -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component3_socratic_educator.src import grammar_decoder as gd
from component3_socratic_educator.src.intercept_audit import AUDIT_RULES, audit

OPENER = gd.INTERCEPT_OPENERS[0]


@pytest.mark.parametrize("question, rule", [
    ("What's happening on your screen right now?", "invites_description"),    # passed live, 12 Oct
    ("Can you tell me what it looked like?", "invites_description"),
    ("Was the picture you found upsetting?", "points_at_content"),
    ("Did that seem scary or dangerous to you?", "hints_at_nature"),
    ("Why did you look at that?", "blames_child"),
    ("Do your parents know about this?", "reveals_intervention"),
    ("Can we keep this just between us?", "asks_secrecy"),
])
def test_the_audit_flags_lines_the_post_check_lets_through(question, rule):
    line = f"{OPENER} {question}"
    assert gd.delivery_violation(line) is None       # the gap: both layers pass it
    assert rule in audit(line)


@pytest.mark.parametrize("question", [
    "What were you doing just now?",
    "How are you feeling right now?",
    "What were you thinking about a moment ago?",
])
def test_the_audit_passes_questions_that_turn_to_the_child(question):
    assert audit(f"{OPENER} {question}") == []


def test_every_fallback_line_passes_the_audit():
    for line in gd.GrammarConstrainedGenerator._CANDIDATES["INTERCEPT"]:
        assert audit(line) == [], line


def test_the_audit_reads_the_question_only_not_the_fixed_opener():
    # Both openers mention the screen or what was seen; they are fixed by design.
    for opener in gd.INTERCEPT_OPENERS:
        assert audit(f"{opener} How are you feeling right now?") == []


def test_every_flag_is_a_named_rule():
    flags = audit(f"{OPENER} Why did you look at that scary picture on your screen?")
    assert flags and set(flags) <= set(AUDIT_RULES)
