"""
The stated-emotion check (safety gate S3): an educator line may name only a
feeling the child used verbatim in their latest reply. Shared by the format
check of the fine-tuning data (inquire_finetune.py) and the post-check of
model-written INQUIRE lines (src/grammar_decoder.py), so the two cannot drift.
"""
import re

# Fixed list for the stated-emotion check: words that attribute a feeling. Words
# that are mostly used another way ("hurt someone", "don't worry", "weird")
# are left out, so a paraphrase can slip through (see README limitations).
EMOTION_WORDS = (
    "afraid", "angry", "anger", "annoyed", "anxious", "anxiety", "ashamed", "bored",
    "confused", "creeped out", "depressed", "disappointed", "disgusted", "distressed",
    "embarrassed", "excited", "fear", "frightened", "frightening", "frustrated",
    "frustrating", "frustration", "furious", "grossed out", "guilty", "happy",
    "heartbroken", "helpless", "hopeless", "horrified", "irritated", "jealous",
    "lonely", "mad", "miserable", "nervous", "overwhelmed", "panicked", "sad",
    "sadness", "scared", "scary", "shaken", "shocked", "stressed", "terrified",
    "uncomfortable", "uneasy", "unhappy", "upset", "upsetting", "worried",
)
# The stricter list for model-written INQUIRE lines and the PP2 rows trained for them:
# words the 21 Nov live capture saw a model name as an unstated feeling
# (evidence/inquire/inquire_model_capture_2026-11-21). It blocks "don't worry" too, which
# EMOTION_WORDS leaves out on purpose; a wrong block costs only a template line. The PP1
# format check keeps EMOTION_WORDS, so the recorded PP1 data checks as it did.
POST_CHECK_EMOTION_WORDS = EMOTION_WORDS + (
    "curious", "curiosity", "uncertain", "uncertainty", "unsettled", "unsettling", "unsure",
    "worry", "worrying",
)


def _normalise(text):
    return text.replace("’", "'").replace("‘", "'").lower()


def _has_phrase(phrase, text):
    return re.search(rf"(?<![a-z']){re.escape(phrase)}(?![a-z'])", text) is not None


def unstated_emotions(text, child_reply, words=EMOTION_WORDS):
    """Emotion words (from `words`) in `text` that the child's reply does not contain verbatim."""
    said, child = _normalise(text), _normalise(child_reply)
    return [w for w in words if _has_phrase(w, said) and not _has_phrase(w, child)]
