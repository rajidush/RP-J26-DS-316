"""Grooming test set for the C2 text path (proposal FR2 `grooming`; team enum `grooming_language`).

Why it exists
-------------
The dev and held-out sets have no grooming bucket: their two secrecy cases are
labelled `sexual_harassment`, so whether C2 ever says "grooming" was never
measured.

What counts as harmful here
---------------------------
Grooming is a conversation-level pattern (Guardian; O'Connell 2003), and many
single cues are ordinary chat from a friend ("add me on discord", "are your
parents home?"). So a harmful case is one message that carries a grooming
tactic on its own, by the online-grooming stages (O'Connell 2003; Winters &
Jeglic 2017):

    secrecy     hiding the contact from caregivers
    isolation   "only i understand you", setting the child against parents or friends
    risk        checking who can see the chat, then pushing on
    sexual      a sexual or image request wrapped in secrecy, flattery or a bribe
    meeting     arranging to meet alone or in secret

Negatives use the same vocabulary in ordinary family, friend, school and
gaming chat: surprises, crushes, lifts from parents, homework photos.

Rules (as for `heldout_set.py`)
-------------------------------
1. Both parts were written on 18 Oct 2026, before any grooming fix, from the
   concept, not from failures.
2. `dev_cases()` may be looked at while designing a fix. `heldout_cases()`
   says the same tactics in different words and is never tuned on; a case
   used to tune moves to dev and is replaced.

Buckets: secrecy · isolation · risk · sexual · meeting (harmful);
surprise · friends · family · school · gaming (benign).

Results (`evaluation/grooming.py`, 18 Oct 2026)
-----------------------------------------------
The grooming tactic families in `packs/policy/lexicon.json` (pack 1.2.0),
written on dev only, against pack 1.1.0 (commit 4dbdd14). The same at ages
9, 12 and 15; every grooming alert is urgent.

    dev        18 / 20 as grooming (was 0; 3 alerted as sexual harassment)   0 / 16 false alerts (was 1)
    held-out    4 / 20 as grooming (was 0)                                    1 / 16 (unchanged; see below)
    107 existing cases: 3 of 321 decisions changed, all "don't tell your
    parents this is our secret", now grooming (urgent) instead of sexual harassment.

Word rules written on 20 dev lines do not generalise: held-out says the same
tactics in other words and 16 of 20 slip past. The rules are kept because
what they catch is right and they add no false alert, but per-message
grooming needs a model that reads meaning, not more patterns. The held-out
false alert is not a grooming rule: the hate head reads "i'll trade you my
skin for your pickaxe" as hate (0.99).

Both parts and the rules were written by the same person, so even the
held-out number is optimistic; an independently written or real-data set is
needed before quoting it.
"""
from __future__ import annotations

from typing import List

from component2_hate_speech_detection.evaluation.dev_set import TextCase

HARM_BUCKETS = ("secrecy", "isolation", "risk", "sexual", "meeting")


def _harm(text: str, bucket: str) -> TextCase:
    return TextCase(text, True, "grooming", bucket)


def _ok(text: str, bucket: str) -> TextCase:
    return TextCase(text, False, "none", bucket)


_DEV: List[TextCase] = [
    _harm("don't tell your parents we talk", "secrecy"),
    _harm("delete our messages so your mum doesn't see them", "secrecy"),
    _harm("this is just between you and me, your parents wouldn't understand", "secrecy"),
    _harm("if anyone asks just say i'm a friend from school", "secrecy"),
    _harm("you can't tell anyone about us or we'll both get in trouble", "secrecy"),
    _harm("your parents don't understand you like i do", "isolation"),
    _harm("i'm the only one who really cares about you", "isolation"),
    _harm("your friends are just jealous of what we have", "isolation"),
    _harm("you're so mature for your age, not like the other kids", "isolation"),
    _harm("are your parents asleep yet? turn your camera on", "risk"),
    _harm("is anyone else in the room? good, don't let them see this", "risk"),
    _harm("does your mum ever check your phone?", "risk"),
    _harm("send me pics and don't tell your parents", "sexual"),
    _harm("if you send me a photo i'll give you robux", "sexual"),
    _harm("you'd look so pretty in a swimsuit, send one just for me", "sexual"),
    _harm("have you ever kissed anyone? you can tell me, i won't tell", "sexual"),
    _harm("i can pick you up from school, just don't tell your mum", "meeting"),
    _harm("let's meet at the park, come alone", "meeting"),
    _harm("i'll buy you the new phone if you come see me this weekend", "meeting"),
    _harm("tell your parents you're at a friend's house and come meet me", "meeting"),

    _ok("don't tell mum about her birthday present", "surprise"),
    _ok("keep the surprise party a secret from dad ok", "surprise"),
    _ok("omg don't tell anyone i have a crush on him", "friends"),
    _ok("pls delete that ugly photo of me lol", "friends"),
    _ok("you can tell me anything, that's what best friends are for", "friends"),
    _ok("tell your parents i said thanks for dinner", "family"),
    _ok("your mum said you can come over after school", "family"),
    _ok("my dad can pick you up from practice tomorrow", "family"),
    _ok("send me pics of the homework sheet please", "school"),
    _ok("we learned about online safety and strangers in class", "school"),
    _ok("the teacher said keep your answers secret until the quiz ends", "school"),
    _ok("add me on discord so we can play later", "gaming"),
    _ok("can you gift me robux for my birthday", "gaming"),
    _ok("let's meet at the park after school with everyone", "friends"),
    _ok("are your parents home? i want to drop off your jumper", "family"),
    _ok("i'm on my phone in my room, want to call?", "friends"),
]

_HELDOUT: List[TextCase] = [
    _harm("make sure your parents never find out about our chats", "secrecy"),
    _harm("clear your history after we talk ok", "secrecy"),
    _harm("our friendship is special, nobody else needs to know", "secrecy"),
    _harm("if your dad sees this he'll stop us talking, so hide it", "secrecy"),
    _harm("promise me you won't mention me to your family", "secrecy"),
    _harm("your mum is too strict, she doesn't get you like i do", "isolation"),
    _harm("nobody understands you the way i do", "isolation"),
    _harm("you don't need your friends, you've got me", "isolation"),
    _harm("you're way more grown up than other girls your age", "isolation"),
    _harm("is your door locked? i want to show you something", "risk"),
    _harm("can anyone else read your messages?", "risk"),
    _harm("when are your parents out of the house?", "risk"),
    _harm("send me a selfie in bed but keep it secret", "sexual"),
    _harm("i'll send you a gift card for a picture of you", "sexual"),
    _harm("you're so cute, show me what you look like without the hoodie", "sexual"),
    _harm("do you ever think about older guys? it's ok you can tell me", "sexual"),
    _harm("i'll wait for you outside school, don't tell anyone", "meeting"),
    _harm("let's hang out, just us, don't bring anyone", "meeting"),
    _harm("i'll pay for your train ticket if you visit me", "meeting"),
    _harm("say you're sleeping over at a friend's and come to mine", "meeting"),

    _ok("shh don't tell your sister what we got her", "surprise"),
    _ok("it's a secret until grandma's party on saturday", "surprise"),
    _ok("don't tell anyone but i failed the spelling test", "friends"),
    _ok("delete that video of me falling over pls", "friends"),
    _ok("i promise i won't tell anyone your secret", "friends"),
    _ok("say hi to your mum for me", "family"),
    _ok("your dad said he'll drive us to the cinema", "family"),
    _ok("my parents are out tonight so i'm at my nan's", "family"),
    _ok("can you send me a photo of page 12 for the homework", "school"),
    _ok("our teacher told us never to share passwords", "school"),
    _ok("we have to keep our science project a secret till friday", "school"),
    _ok("join my server so we can play minecraft", "gaming"),
    _ok("i'll trade you my skin for your pickaxe", "gaming"),
    _ok("want to meet at the library with the whole class", "friends"),
    _ok("is your mum home? she left her keys at ours", "family"),
    _ok("are you alone at the shop? i'm coming to join you", "friends"),
]


def dev_cases() -> List[TextCase]:
    return list(_DEV)


def heldout_cases() -> List[TextCase]:
    return list(_HELDOUT)


def summary() -> dict:
    return {name: {"total": len(cases), "harmful": sum(c.hate for c in cases), "benign": sum(not c.hate for c in cases)}
            for name, cases in (("dev", _DEV), ("heldout", _HELDOUT))}
