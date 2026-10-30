"""
EVALUATE's rules arm (ticket 02): one child reply -> the codebook's three parts
(src/codebook.py), each with the phrases that fired as its evidence. The
session's emotional state is derived from these labels by the rule every arm
shares (assessor.derive_emotional_state), not here.

A transparent baseline, not a trained model. The rules handle what phrase rules
can: a negation just before a phrase ("not scared at all"), words out of context
("lost in the game" is not distress), implicit feelings that have a fixed
wording ("nobody cares about me"), and a resolved feeling ("I was scared but I'm
fine now" is past distress). Implicit feelings with no fixed wording are not
caught; tests/test_assessor.py keeps those as known failures.

Also here: the reply types INQUIRE reacts to (a refusal, "I don't know") and the
PP1 self-regulation keywords, which ticket 04 replaces.
"""
from __future__ import annotations

import re

from .codebook import CHANGE_TALK, DISTRESS_SEVERITY, EXPRESSED_EMOTION, STANCE

_APOS = "['’]?"  # "don't", "dont", "don’t"


def _rx(*phrases: str) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(phrases) + r")\b")


# --- reply types INQUIRE reacts to -------------------------------------------

# A refusal: the child opts out of talking. INQUIRE stops asking (14 Oct: the
# withdrawn child was asked again after "leave me alone"), and its stance is disengaged.
_REFUSAL_PHRASES = (
    r"leave me alone", rf"(don{_APOS}t|do not) (really )?(want|feel like) (to )?talk(ing)?", r"stop asking",
    r"go away", r"i (need|want) (some )?space",
)
REFUSAL = _rx(*_REFUSAL_PHRASES)
# "I don't know": INQUIRE asks the same Socratic move again in its easier form. The
# uncertainty phrases were added 14 Oct from the 12 Oct end-to-end bug: three "I'm not
# sure" replies had no signal and fell through to calm.
_DONT_KNOW_PHRASES = (
    rf"i don{_APOS}t (really )?know(?! (what to do|why|how))", r"idk", rf"i don{_APOS}t (get|understand)",
    r"not sure", r"dunno", r"no idea",
)
DONT_KNOW = _rx(*_DONT_KNOW_PHRASES)

# The PP1 self-regulation keywords behind `self_regulation_shown` (moved here
# from the controller on 26 Oct, unchanged). Whole words: "i could" must not
# fire inside "i couldn't".
SELF_REGULATION = re.compile(r"\b(instead|next time|should have|i could)\b")


# --- expressed emotion and distress severity ------------------------------------
# One set of fear and sadness phrases serves both parts, so they cannot disagree:
# fear or sadness now is current distress (the ReachOut triage branches), and
# softened wording ("kinda scary") counts (ADR 0006).
# Self-harm language and overwhelm: current acute, counted even after a negation
# ("I didn't mean to hurt myself") and never downgraded by "I'm fine now".
_ACUTE_SADNESS = (
    r"want(ed)? to die", r"wanna die", r"wish i (was|were) dead", r"better off dead",
    rf"(don{_APOS}t|do not) want to (live|be here)", r"kill(ing)? myself", r"kms", r"hurt(ing)? myself",
    r"cut(ting)? myself", r"(been|keep|started) cutting", r"cutting again", r"self harm(ing)?",
    r"(want(ed)? to |wanna )?disappear", r"end it all", r"want(ed)? (it|everything) all to stop",
    r"want(ed)? everything to stop", rf"can{_APOS}t (stop crying|cope|take it|do this)", r"falling apart",
    r"overwhelmed",
)
_ACUTE_FEAR = (r"i (panicked|was panicking|am panicking)", rf"i{_APOS}m panicking", r"panic attacks?",
               rf"(can|could)n{_APOS}t breathe")
_FEAR = (
    r"scared", r"scary", r"afraid", r"frighten(ed|ing)", r"terrif(ied|ying)", r"worr(ied|y|ying)", r"nervous",
    r"anxious", r"creepy", r"creeped out", r"freak(ed|ing) out", r"uncomfortable",
    rf"(don{_APOS}t|do not) feel safe", r"not safe",
)
_SADNESS = (
    r"sad(ness)?", r"upset(ting)?", r"lonely", r"(?<!leave me )(?<!leave it )(?<!home )alone", r"cry(ing)?",
    r"cried", r"hurts?", r"heavy", r"my heart", r"miss (my|you|them|him|her)", r"nobody cares",
    r"no one cares", r"nobody likes me", r"(everyone|everybody) hates me", r"hate myself", r"left out",
    r"disappointed", r"embarrassed", r"ashamed", r"awful", r"terrible", r"feel(s|ing)? (bad|wrong|sick)",
    r"so wrong", r"bad dreams?", r"nightmares?", r"judged", rf"don{_APOS}t know what to do", r"depressed",
    r"not (okay|ok|alright|fine)", r"want(ed)? it to stop",
)
# Most phrases wins; a tie goes in this order (the distress-related groups first).
EMOTIONS = {
    "fear": _rx(*_ACUTE_FEAR, *_FEAR),
    "sadness": _rx(*_ACUTE_SADNESS, *_SADNESS),
    "anger": _rx(r"angry", r"mad", r"furious", r"annoy(ed|ing)", r"hate", r"pissed", r"irritated", r"unfair",
                 r"so rude"),
    "disgust": _rx(r"gross", r"disgusting", r"ew+", r"yuck", r"nasty", r"(made|makes) me sick"),
    "surprise": _rx(r"surpris(ed|ing)", r"shock(ed)?", r"wow", r"omg", rf"didn{_APOS}t expect", r"out of nowhere",
                    r"weird", r"confus(ed|ing)", rf"don{_APOS}t (get|understand)", rf"what{_APOS}s that",
                    r"wait what"),
    "joy": _rx(r"lol", r"haha+", r"lmao", r"fun", r"funny", r"happy", r"glad", r"excited", r"love (it|that|this)",
               r"cool", r"awesome", r"yay"),
}
ACUTE_DISTRESS_PHRASES = _rx(*_ACUTE_SADNESS, *_ACUTE_FEAR)
# The child says the feeling is over, after naming it, and not "still": past distress.
RESOLVED = _rx(r"(fine|okay|ok|better|alright) now", r"not anymore", r"over it", r"back then")
_STILL = _rx(r"still")

# --- stance: MISC client language plus "disengaged" ------------------------------
_ACTIONS = (r"(tell|ask|talk|close|block|report|stop|leave|exit|mute|delete|ignore|show|turn it off|log off"
            r"|switch it off|get help|be careful|told|asked|talked|closed|blocked|reported|stopped|left|muted"
            r"|deleted|ignored|showed)")
# Toward a safer step: a commitment, a step taken, or a better alternative.
CHANGE = _rx(
    rf"i{_APOS}ll (just |probably |definitely )?{_ACTIONS}", rf"i will (just )?{_ACTIONS}",
    rf"i{_APOS}m (going to|gonna) {_ACTIONS}",
    rf"i (should|could|can|need to|would|might|will) (have )?{_ACTIONS}",
    r"i (told|asked|closed|blocked|reported|stopped|left|muted|deleted)",
    r"next time", r"instead", r"from now on", r"trusted adult", r"be (more |extra )?careful",
    r"(tell|talk to|ask) (someone|a grown-?up|an adult|a teacher|my \w+|mum|mom|dad)",
)
# Only imagined ("if I tell my mum..."): not change talk.
_CONDITIONAL = re.compile(r"\bif\s+(?:[\w']+\s+){0,2}$")
# Arguing for the old behaviour, or turning the step down: it overrides change-talk
# wording ("next time I will do the same thing", "I could tell my mum but I won't").
SAME_AGAIN = _rx(
    r"(do|doing|did) (it |that |this )?(exactly |just )?the same( thing)?", r"do it again",
    r"keep (doing|watching|going|looking)", rf"i (don{_APOS}t|do not) care",
    rf"but i (won{_APOS}t|will not|(don{_APOS}t|do not) want to)",
)
# Friction with the educator, without opting out.
DISCORD = _rx(
    r"none of your business", r"mind your own", r"why do you (care|want to know|keep asking|need to know)",
    rf"you{_APOS}re not my (mum|mom|dad|parent|teacher)", rf"you (don{_APOS}t|do not) (get|understand)",
    r"(stupid|dumb|pointless) question", r"this is (stupid|dumb|pointless|annoying)",
    r"stop (lecturing|judging)", r"who are you", r"shut up",
)
# Minimising, denying, deflecting.
SUSTAIN = _rx(
    r"not (really )?my (fault|choice|thing)", rf"wasn{_APOS}t (really )?(me|my)", r"not me",
    rf"i didn{_APOS}t (really )?(do|see|mean|look)", rf"didn{_APOS}t do anything", rf"don{_APOS}t worry about it",
    r"whatever", r"so what", r"it was nothing", r"(no|not a) big deal", r"just a (little|bit)", r"bit of fun",
    r"no way", rf"didn{_APOS}t (bother|happen to) me", rf"wasn{_APOS}t (even )?looking",
    r"not (that |very )?important", r"leave it alone", r"everyone does it",
)
# Non-answers. A filler ("hmm", "ok") is MISC's follow/neutral, not disengaged.
NON_ANSWER = _rx(*_DONT_KNOW_PHRASES, r"nothing", r"nope", r"quiet", r"boring", rf"what{_APOS}s that",
                 r"i guess(?! i (should|need))")
_BARE_NO = re.compile(r"^\W*(no|nah)\W*$")

# A negation cancels the phrase it negates: "not scared", "wasn't even that scared",
# "don't feel sad". Only softening words may stand between, so "never been so scared",
# "no, I'm scared" and "I don't know, I'm scared" are not negations.
_NEGATED = re.compile(r"\b(not|never|\w+n't|dont|didnt|doesnt|isnt|wasnt|cant|wont|aint)\s+"
                      r"(?:(really|even|that|very|too|feel|feeling|felt|be|been|being|get|getting|got|a|bit|at|all"
                      r"|as)\s+){0,2}$")


def _matches(pattern: re.Pattern, text: str) -> list[re.Match]:
    """The matches of `pattern` in `text` that are not negated."""
    return [m for m in pattern.finditer(text) if not _NEGATED.search(text[:m.start()])]


def _fired(pattern: re.Pattern, text: str) -> list[str]:
    return [m.group(0) for m in _matches(pattern, text)]


def _emotion(text: str) -> tuple[str, list[str]]:
    hits = {group: _fired(rx, text) for group, rx in EMOTIONS.items()}
    best = max(EMOTIONS, key=lambda g: (len(hits[g]), -list(EMOTIONS).index(g)))
    return (best, hits[best]) if hits[best] else ("neutral", [])


def _distress(text: str) -> tuple[str, list[str]]:
    if acute := [m.group(0) for m in ACUTE_DISTRESS_PHRASES.finditer(text)]:
        return "current_acute", acute
    hits = sorted(_matches(EMOTIONS["fear"], text) + _matches(EMOTIONS["sadness"], text), key=lambda m: m.start())
    if not hits:
        return "none", []
    over = any(m.start() >= hits[-1].end() for m in _matches(RESOLVED, text)) and not _STILL.search(text)
    return ("past" if over else "current_mild"), [m.group(0) for m in hits]


def _stance(text: str) -> tuple[str, list[str]]:
    if refusal := _fired(REFUSAL, text):
        return "disengaged", refusal
    if same := _fired(SAME_AGAIN, text):
        return "sustain_talk", same
    if change := [m.group(0) for m in _matches(CHANGE, text) if not _CONDITIONAL.search(text[:m.start()])]:
        return CHANGE_TALK, change
    for stance, pattern in (("discord", DISCORD), ("sustain_talk", SUSTAIN), ("disengaged", NON_ANSWER)):
        if hits := _fired(pattern, text):
            return stance, hits
    if not re.search(r"\w", text):
        return "disengaged", ["(no reply)"]
    if bare := _BARE_NO.match(text):
        return "disengaged", [bare.group(1)]
    return "neutral", []


def label_reply(reply: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    """One reply's three parts and, for each, the phrases that fired (empty for the
    default value). Stance precedence: a refusal, "the same again", change talk,
    discord, sustain talk, a non-answer."""
    text = (reply or "").lower().replace("’", "'")
    parts = {EXPRESSED_EMOTION: _emotion(text), DISTRESS_SEVERITY: _distress(text), STANCE: _stance(text)}
    return {k: v for k, (v, _) in parts.items()}, {k: e for k, (_, e) in parts.items()}


def is_refusal(reply: str) -> bool:
    return bool(REFUSAL.search((reply or "").lower()))


def is_dont_know(reply: str) -> bool:
    """An "I don't know" reply: INQUIRE asks the move again in its easier form. One
    that also says something ("idk, it was so scary", "idk, not my fault") is
    answered by the next move."""
    labels, _ = label_reply(reply)
    return (bool(DONT_KNOW.search((reply or "").lower())) and labels[STANCE] == "disengaged"
            and labels[DISTRESS_SEVERITY] == "none")
