"""
The reply-label codebook (ADR 0006): the three parts every assessor arm labels
each child reply with, and their values. Research and sources:
evidence/research/emotion_taxonomy_research_2026-10-25.md.

    expressed emotion  GoEmotions' Ekman grouping, or neutral
    distress severity  after the ReachOut triage branches (CLPsych 2016)
    stance             Motivational Interviewing client language (MISC 2.5),
                       plus "disengaged", a project extension for non-answers
                       and opting out
"""
EXPRESSED_EMOTION = "expressed_emotion"
DISTRESS_SEVERITY = "distress_severity"
STANCE = "stance"

EXPRESSED_EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral")
DISTRESS_SEVERITIES = ("none", "past", "current_mild", "current_acute")
CHANGE_TALK = "change_talk"
STANCES = (CHANGE_TALK, "neutral", "sustain_talk", "discord", "disengaged")

# Any current distress, mild or acute, makes the session distressed.
CURRENT_DISTRESS = ("current_mild", "current_acute")
# The value a part takes when nothing supports another one; every other value needs evidence.
LABEL_DEFAULTS = {EXPRESSED_EMOTION: "neutral", DISTRESS_SEVERITY: "none", STANCE: "neutral"}
