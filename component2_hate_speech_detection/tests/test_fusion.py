"""Fusion (proposal design rule 3, FR5; Guardian invariants I-1, I-2)."""
from component2_hate_speech_detection.src.engine.fusion import FullTextReread, Signal, fuse


def test_invariant_I1_uncalibrated_signal_cannot_move_the_score():
    # The MVP measured zero-shot CLIP at 0.324-0.393 for every image; averaged in,
    # it cleared a confirmed "you should kys" for ages 14-15.
    with_noise = fuse([Signal("text", 0.88, True), Signal("vision", 0.39, False)])
    assert with_noise.fused == 0.88 and "vision" in with_noise.ignored


def test_invariant_I2_fusion_never_lowers_the_strongest_signal():
    for text, vision in [(0.88, 0.10), (0.40, 0.95), (0.50, 0.50), (0.99, 0.0)]:
        result = fuse([Signal("text", text, True), Signal("vision", vision, True)])
        assert result.fused >= max(text, vision)


def test_two_mid_band_signals_corroborate_upward():
    result = fuse([Signal("text", 0.50, True), Signal("vision", 0.50, True)])
    assert result.agreement and result.fused > 0.50


def test_reread_finds_harm_past_the_first_window():
    long_text = "lorem ipsum dolor " * 40 + "the bad part"
    scorer = lambda chunk: 0.9 if "bad part" in chunk else 0.1  # noqa: E731
    assert FullTextReread(scorer).score(long_text, 0.1) == 0.9


def test_reread_skips_short_text():
    assert FullTextReread(lambda c: 0.99).score("short", 0.2) == 0.2
