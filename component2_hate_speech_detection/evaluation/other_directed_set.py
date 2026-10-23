"""Other-directed violence test set: threats, film lines and pleas that must not read as self-harm.

Why it exists
-------------
On 19 Oct 2026 the real engine turned film dialogue into urgent self-harm
alerts: "I'm going to find you, and I'm going to kill you" (suicidality 0.96),
"Drop the gun or I'll shoot!" (0.94). The self-harm specialist
(sentinet/suicidality) learned from suicide posts against r/teenagers, so
violent first-person words appear only on its positive side. The gate in
`src/engine/self_harm.py` blocks "yourself" and "you should", not "kill you".

Rules (as for `self_harm_set.py`)
---------------------------------
1. Written on 23 Oct 2026, before the self-harm model was fine-tuned and
   before its training data was built, from the concept, not from failures.
2. `dev_cases()` are the five lines found on 19 Oct; they motivated the fix.
   `heldout_cases()` are never tuned on; a case used to tune moves to dev and
   is replaced. Before the real run they were scored only in CPU code checks
   of the pipeline on stand-in data (fake replay rows, 1 seed, 1 epoch;
   23 Oct); nothing in the data or the recipe was changed after them.
3. Negatives (`hate=False` for self-harm) are first person on purpose, so the
   gate lets the model's reading through and the test is about the model.
   Many are real threats: C2 should still alert on them, as `threat`, not as
   `self_harm`. Positives are the writer's own risk, mostly with the same
   violent verbs, so a model that just learned "violent verb -> 0" fails.

Buckets: film · threat_chat · plea · protective (not self-harm);
self_violent · self_indirect · self_injury (self-harm).
"""
from __future__ import annotations

from typing import List

from component2_hate_speech_detection.evaluation.dev_set import TextCase

SELF_HARM_BUCKETS = ("self_violent", "self_indirect", "self_injury")


def _other(text: str, bucket: str) -> TextCase:
    return TextCase(text, False, "none", bucket)


def _self(text: str, bucket: str) -> TextCase:
    return TextCase(text, True, "self_harm", bucket)


# Found 19 Oct 2026 on a scripted week of film audio (presentations/c4_handoff): all five were L4 self_harm.
_DEV: List[TextCase] = [
    _other("I'm going to find you, and I'm going to kill you", "film"),
    _other("Drop the gun or I'll shoot!", "film"),
    _other("I will kill every last one of you", "film"),
    _other("Somebody help me! Please don't hurt me!", "plea"),
    _other("Nobody threatens my family.", "protective"),
]

_HELDOUT: List[TextCase] = [
    _other("Put the knife down or I'll put you in the ground.", "film"),
    _other("If you come any closer, I'll blow your head off.", "film"),
    _other("I swear I will hunt down every one of you.", "film"),
    _other("Get on your knees or I'll shoot your partner.", "film"),
    _other("You took my daughter, and now I'm going to make you pay.", "film"),
    _other("I'm going to tear this whole city apart to find him.", "film"),
    _other("One more step and I'll pull the trigger.", "film"),
    _other("Tell your boss I'm coming for him tonight.", "film"),
    _other("I'll burn this place to the ground with you in it.", "film"),
    _other("I'm going to end you, just like I ended your brother.", "film"),

    _other("i'm going to beat you up tomorrow at lunch", "threat_chat"),
    _other("say that again and i'll smash your face in", "threat_chat"),
    _other("i know where you live and i'm coming for you", "threat_chat"),
    _other("i'll stab you if you tell anyone", "threat_chat"),
    _other("i'm gonna make you bleed after school", "threat_chat"),
    _other("i will hurt your little sister if you don't pay", "threat_chat"),
    _other("i'll break every bone in your body", "threat_chat"),
    _other("i'm going to shoot up the school", "threat_chat"),

    _other("please stop, you're hurting me", "plea"),
    _other("don't shoot, i have kids!", "plea"),
    _other("i'm scared he's going to hurt me again", "plea"),
    _other("help me, someone is following me home", "plea"),
    _other("leave me alone or i'll call the police", "plea"),
    _other("i thought he was going to kill me", "plea"),

    _other("nobody touches my little brother", "protective"),
    _other("i would never let anyone hurt my sister", "protective"),
    _other("if anyone bullies her they'll have to deal with me", "protective"),
    _other("i'll protect my family no matter what", "protective"),
    _other("my dad would kill anyone who hurt me", "protective"),
    _other("i'd fight anyone who tried to take my dog", "protective"),

    _self("i want to shoot myself", "self_violent"),
    _self("i keep thinking about stabbing myself", "self_violent"),
    _self("i'm going to jump off the bridge tonight", "self_violent"),
    _self("i'm going to hang myself", "self_violent"),
    _self("i want to put a bullet in my head", "self_violent"),
    _self("i just want to slit my wrists", "self_violent"),
    _self("i'm going to overdose on my mum's pills", "self_violent"),
    _self("i want to throw myself in front of a train", "self_violent"),

    _self("i don't deserve to live", "self_indirect"),
    _self("i'm done, i can't do this anymore, goodbye", "self_indirect"),
    _self("i'm scared of what i might do to myself tonight", "self_indirect"),
    _self("i wish i could fall asleep and never wake up", "self_indirect"),
    _self("i feel like everyone would be happier if i died", "self_indirect"),
    _self("there's no reason for me to stay alive", "self_indirect"),
    _self("i've been planning how to do it", "self_indirect"),
    _self("i won't be here next week", "self_indirect"),

    _self("i hit myself until i bruise", "self_injury"),
    _self("i bang my head against the wall when i'm upset", "self_injury"),
    _self("i used a razor on my legs again", "self_injury"),
    _self("i make myself bleed so i can feel something", "self_injury"),
    _self("i hurt myself again because of them", "self_injury"),
    _self("i burned my arm with a lighter on purpose", "self_injury"),
    _self("i want to hurt myself so bad right now", "self_injury"),
    _self("i deserve the pain so i cut", "self_injury"),
]


def dev_cases() -> List[TextCase]:
    return list(_DEV)


def heldout_cases() -> List[TextCase]:
    return list(_HELDOUT)


def summary() -> dict:
    return {name: {"total": len(cases), "self_harm": sum(c.hate for c in cases), "other": sum(not c.hate for c in cases)}
            for name, cases in (("dev", _DEV), ("heldout", _HELDOUT))}
