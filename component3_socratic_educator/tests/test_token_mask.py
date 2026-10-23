"""
The finite-automaton token mask (src/token_mask.py): Component 3's own
implementation of token masking for the INTERCEPT controller grammar, used by
evidence/trace_token_masking.py to show the masking step by step.

These tests use a toy vocabulary, so no model is needed. The automaton must
accept exactly what the controller grammar (grammar_decoder._GRAMMAR) accepts.
"""
import random

import pytest

from component3_socratic_educator.src.grammar_decoder import (
    _GRAMMAR, INTERCEPT_OPENERS, SCREEN_INTERCEPT_OPENERS)
from component3_socratic_educator.src.token_mask import (
    InterceptAutomaton, MaskingLogitsProcessor, TokenMask, piece_text)

A = InterceptAutomaton()
# Token-mask evidence uses the original screen openers; text openers are also
# in INTERCEPT_OPENERS for the shared grammar enum.
OPENER_1, OPENER_2 = SCREEN_INTERCEPT_OPENERS
assert OPENER_1 in INTERCEPT_OPENERS and OPENER_2 in INTERCEPT_OPENERS


def accepts(text: str) -> bool:
    state = A.run(text)
    return state is not None and A.is_accepting(state)


# --- the automaton -----------------------------------------------------------

@pytest.mark.parametrize("line", [
    f"{OPENER_1} Can you tell me what you were thinking?",
    f"{OPENER_2} How are you feeling right now?",
    f"{OPENER_1} What's one word, maybe two, for it?",
    f"{OPENER_2} Really?",         # the shortest question: capital + 5 + "?"
])
def test_accepts_grammatical_lines(line):
    assert accepts(line)


@pytest.mark.parametrize("line, why", [
    ("Hello there. What were you doing?", "not a fixed opener"),
    (f"{OPENER_1}What were you doing?", "no space after the opener"),
    (f"{OPENER_1} what were you doing?", "question starts lower-case"),
    (f"{OPENER_1} What were you doing on day 2?", "digit"),
    (f"{OPENER_1} What’s happening?", "curly apostrophe"),
    (f"{OPENER_1} What were you doing!", "no question mark"),
    (f"{OPENER_1} Okay?", "question too short"),
    (f"{OPENER_1} W{'a' * 99}?", "question too long"),
    (f"{OPENER_1} What were you doing? Okay?", "text after the question mark"),
    (f"{OPENER_1} What were you doing", "unfinished"),
])
def test_rejects_ungrammatical_lines(line, why):
    assert not accepts(line), why


def test_dead_as_soon_as_a_prefix_cannot_be_completed():
    assert A.run("I noticed") is not None
    assert A.run("I noticed x") is None
    assert A.run("{") is None


def test_agrees_with_the_controller_grammar_on_random_lines():
    rng = random.Random(20261018)
    alphabet = "AWhatyou ,'?!.1’x"
    for _ in range(3000):
        opener = rng.choice([OPENER_1, OPENER_2, OPENER_1[:-1], "I noticed"])
        sep = rng.choice([" ", "", "  "])
        body = "".join(rng.choice(alphabet) for _ in range(rng.choice([3, 6, 7, 20, 99, 100, 101])))
        line = opener + sep + rng.choice(["W", "w", ""]) + body
        assert accepts(line) == bool(_GRAMMAR["INTERCEPT"].match(line)), line


# --- tokenizer pieces ------------------------------------------------------------

@pytest.mark.parametrize("piece, text", [
    ("▁What", " What"), ("noticed", "noticed"), ("<0x41>", "A"), ("<0x0A>", "\n"), ("▁▁", "  "),
])
def test_piece_text(piece, text):
    assert piece_text(piece) == text


def test_special_pieces_have_no_text():
    assert piece_text("<start_of_turn>", special=True) is None


# --- the token mask over a toy vocabulary ------------------------------------

VOCAB = ["I", " noticed", "I noticed", "Hello", "{", "S", "omething", " something on your screen just now.",
         " What", " were you doing", "?", "!", " what", "<eos>", "", None, " doing?"]
EOS = VOCAB.index("<eos>")
ids = {t: i for i, t in enumerate(VOCAB) if t is not None}


def make_mask():
    return TokenMask(VOCAB, end_ids={EOS})


def test_start_allows_only_tokens_that_can_begin_an_opener():
    allowed = make_mask().allowed(A.start)
    assert {ids["I"], ids["I noticed"], ids["S"]} <= allowed
    assert not allowed & {ids["Hello"], ids["{"], ids[" noticed"], EOS, ids[""], VOCAB.index(None)}


def test_a_token_may_span_the_opener_and_the_question():
    mask = make_mask()
    state = mask.state_after([ids["I noticed"], ids[" something on your screen just now."]])
    allowed = mask.allowed(state)
    assert ids[" What"] in allowed          # " What" is the space plus the question's first letter
    assert ids[" what"] not in allowed      # lower-case start
    assert EOS not in allowed


def test_end_of_generation_only_once_the_line_is_complete():
    mask = make_mask()
    prefix = [ids["I noticed"], ids[" something on your screen just now."], ids[" What"], ids[" were you doing"]]
    allowed = mask.allowed(mask.state_after(prefix))
    assert ids["?"] in allowed and ids["!"] not in allowed and EOS not in allowed
    done = mask.state_after(prefix + [ids["?"]])
    assert mask.allowed(done) == {EOS}


# --- the logits processor ------------------------------------------------------

def test_processor_blocks_disallowed_tokens_and_records_the_step():
    torch = pytest.importorskip("torch")
    mask = make_mask()
    proc = MaskingLogitsProcessor(mask, prompt_len=2, top_k=3)
    scores = torch.zeros(1, len(VOCAB))
    scores[0, ids["Hello"]] = 5.0     # the model's favourite is not allowed
    scores[0, ids["{"]] = 4.0
    scores[0, ids["I"]] = 1.0
    out = proc(torch.tensor([[7, 8]]), scores.clone())
    assert out[0, ids["Hello"]] == float("-inf") and out[0, ids["{"]] == float("-inf")
    assert out[0, ids["I"]] == 1.0
    step = proc.trace[0]
    assert [c["token_id"] for c in step["top_before_mask"]] == [ids["Hello"], ids["{"], ids["I"]]
    assert [c["allowed"] for c in step["top_before_mask"]] == [False, False, True]
    assert step["allowed_count"] == len(mask.allowed(A.start))


def test_greedy_decoding_under_the_mask_always_yields_a_grammatical_line():
    """A toy 'model' that always prefers forbidden tokens still ends up with a
    line the controller grammar accepts, because the mask leaves it no choice."""
    torch = pytest.importorskip("torch")
    mask = make_mask()
    preference = torch.tensor([[0.1 * i for i in range(len(VOCAB))]])
    for bad in ("Hello", "{", "!", " what"):
        preference[0, ids[bad]] = 100.0
    preference[0, ids["S"]] = -1.0    # the toy vocabulary cannot finish the second opener
    proc = MaskingLogitsProcessor(mask, prompt_len=0)
    generated = []
    for _ in range(60):   # "Hello" is a valid question word, so the length limit is what forces the "?"
        scores = proc(torch.tensor([generated], dtype=torch.long), preference.clone())
        nxt = int(scores[0].argmax())
        if nxt == EOS:
            break
        generated.append(nxt)
    else:
        pytest.fail("the mask never forced an end of generation")
    line = "".join(VOCAB[i] for i in generated)
    assert _GRAMMAR["INTERCEPT"].match(line), line
