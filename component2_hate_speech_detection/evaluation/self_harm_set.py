"""Self-harm test set for the C2 text path (proposal FR2 `self_harm`; team enum `self_harm_language`).

Why it exists
-------------
The dev and held-out sets have no self-harm case at all, so the gap was never
measured. Run through the full engine on 18 Oct 2026, C2 caught 1 of 12
realistic self-harm messages: both text models are hate / toxicity models,
so self-harm rested on a short keyword list.

Rules (as for `heldout_set.py`)
-------------------------------
1. Both parts were written on 18 Oct 2026, before any self-harm fix, from the
   concept, not from failures.
2. `dev_cases()` may be looked at while designing a fix. `heldout_cases()`
   says the same harms in different words and is never tuned on; a case used
   to tune moves to dev and is replaced.
   Done once, on 18 Oct 2026: "i'd kill for some sleep" (held-out, alerted)
   moved to dev; "this maths test is going to be the death of me" replaced it.
3. Positives are the child's own risk (intent, ideation, self-injury, plans
   or means). Negatives are built from the same vocabulary: figurative
   speech, gaming, ordinary injuries and medicine, school topics, grief.

Buckets: direct · ideation · injury · plan (harmful); figurative · gaming ·
everyday · school · feelings (benign).
"""
from __future__ import annotations

from typing import List

from component2_hate_speech_detection.evaluation.dev_set import TextCase


def _harm(text: str, bucket: str) -> TextCase:
    return TextCase(text, True, "self_harm", bucket)


def _ok(text: str, bucket: str) -> TextCase:
    return TextCase(text, False, "none", bucket)


_DEV: List[TextCase] = [
    _harm("i want to kill myself", "direct"),
    _harm("i'm going to end my life", "direct"),
    _harm("i want to die", "direct"),
    _harm("i'm going to hurt myself tonight", "direct"),
    _harm("i wish i was dead", "direct"),
    _harm("nobody would even notice if i was gone", "ideation"),
    _harm("everyone would be better off without me", "ideation"),
    _harm("i don't see the point in living anymore", "ideation"),
    _harm("i keep thinking about not waking up", "ideation"),
    _harm("i don't want to be alive anymore", "ideation"),
    _harm("i cut my arms again last night", "injury"),
    _harm("i've been hurting myself again", "injury"),
    _harm("i burned myself on purpose again", "injury"),
    _harm("i scratch my skin until it bleeds", "injury"),
    _harm("i have pills saved up for tonight", "plan"),
    _harm("i looked up how to tie a noose", "plan"),
    _harm("i wrote goodbye letters to my family", "plan"),
    _harm("i'm standing on the edge of the roof", "plan"),
    _harm("tonight is my last night", "plan"),
    _harm("i want to end it all", "direct"),

    _ok("this homework is killing me", "figurative"),
    _ok("i'm dying of laughter", "figurative"),
    _ok("i could kill for a pizza right now", "figurative"),
    _ok("i want to die of embarrassment lol", "figurative"),
    _ok("ugh i'm so dead my mum saw my grades", "figurative"),
    _ok("my phone is dead again", "everyday"),
    _ok("i died again on the last boss", "gaming"),
    _ok("my character keeps dying in this level", "gaming"),
    _ok("one more round and then i'll end the game", "gaming"),
    _ok("i cut my finger making dinner", "everyday"),
    _ok("i hurt my knee at football practice", "everyday"),
    _ok("i took my medicine before bed", "everyday"),
    _ok("we learned about suicide prevention in health class", "school"),
    _ok("the book we're reading is about a boy who loses his dad", "school"),
    _ok("i'm so tired i could sleep forever", "feelings"),
    _ok("finally done with this essay", "feelings"),
    # Moved from held-out on 18 Oct 2026: it alerted (0.92) and informed the idiom gate.
    _ok("i'd kill for some sleep", "figurative"),
]

_HELDOUT: List[TextCase] = [
    _harm("i'm thinking about killing myself", "direct"),
    _harm("i want my life to be over", "direct"),
    _harm("i'm going to take my own life", "direct"),
    _harm("i'd rather be dead than go back to school", "direct"),
    _harm("i want to disappear and never come back", "direct"),
    _harm("it would be easier if i just wasn't here", "ideation"),
    _harm("i'm just a burden to everyone", "ideation"),
    _harm("no one would miss me", "ideation"),
    _harm("i can't keep going like this anymore", "ideation"),
    _harm("sometimes i think about how to end things", "ideation"),
    _harm("i've been cutting again", "injury"),
    _harm("the only thing that helps is hurting myself", "injury"),
    _harm("i hide the marks on my wrists", "injury"),
    _harm("i starve myself to punish myself", "injury"),
    _harm("i punch myself when i mess up", "injury"),
    _harm("i counted all the pills in the cabinet", "plan"),
    _harm("i know which bridge i would jump from", "plan"),
    _harm("i gave away my favourite things today", "plan"),
    _harm("this is goodbye i love you all", "plan"),
    _harm("i've decided tonight's the night", "plan"),

    _ok("this traffic is killing me", "figurative"),
    _ok("i nearly died laughing at that video", "figurative"),
    _ok("this maths test is going to be the death of me", "figurative"),
    _ok("if i fail this test i'm dead lol", "figurative"),
    _ok("i'm dead tired after practice", "figurative"),
    _ok("my battery died during the call", "everyday"),
    _ok("the zombies killed me again", "gaming"),
    _ok("we're going to wipe out the whole server", "gaming"),
    _ok("i scraped my elbow on the slide", "everyday"),
    _ok("i need to take my allergy pills", "everyday"),
    _ok("we wrote a poem about grief in english", "school"),
    _ok("my grandad passed away last year and i miss him", "feelings"),
    _ok("i can't wait to be done with exams", "feelings"),
    _ok("this song makes me want to cry", "feelings"),
    _ok("i'm so over this weather", "feelings"),
    _ok("goodbye everyone see you tomorrow", "feelings"),
]


def dev_cases() -> List[TextCase]:
    return list(_DEV)


def heldout_cases() -> List[TextCase]:
    return list(_HELDOUT)


def summary() -> dict:
    return {name: {"total": len(cases), "harmful": sum(c.hate for c in cases), "benign": sum(not c.hate for c in cases)}
            for name, cases in (("dev", _DEV), ("heldout", _HELDOUT))}
