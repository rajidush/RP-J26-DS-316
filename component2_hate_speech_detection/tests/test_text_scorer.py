"""Text scorer: corroboration between heads (proposal design rule 1, FR3) and the gaming cap (I-6)."""


def test_two_agreeing_heads_are_trusted(make_scorer):
    assert make_scorer((0.80, "bullying"), (0.60, None)).score("plain words").model_score == 0.80


def test_one_very_confident_head_is_trusted_alone(make_scorer):
    assert make_scorer((0.95, "threat"), (0.01, None)).score("plain words").model_score == 0.95


def test_lone_moderately_confident_head_is_damped(make_scorer, decision_params):
    d = make_scorer((0.80, "bullying"), (0.10, None)).score("plain words")
    assert d.model_score == round(0.80 * decision_params["corroboration"]["solo_damp"], 4)
    assert d.corroboration == "damped"


def test_damping_can_drop_a_lone_head_below_the_youngest_threshold(make_scorer, decision_params):
    d = make_scorer((0.72, "bullying"), (0.10, None)).score("plain words")
    assert d.score < decision_params["persona_theta"]["P1_PROTECT"]


def test_corroboration_ablation_switch_restores_max(make_scorer):
    d = make_scorer((0.80, "bullying"), (0.10, None), use_corroboration=False).score("plain words")
    assert d.model_score == 0.80


def test_explicit_abuse_skips_the_heads(make_scorer):
    d = make_scorer((0.10, None), (0.10, None)).score("you should kys")
    assert d.model_score is None and d.score >= 0.88


def test_head_labels_and_scores_reach_the_evidence(make_scorer):
    d = make_scorer((0.95, "threat", {"threat": 0.95}), (0.10, None, {"toxic": 0.1})).score("no lexicon hit here")
    assert d.model_labels == {"threat": 0.95}
    assert d.head_scores == {"fake:0": 0.95, "fake:1": 0.10}


def test_invariant_I6_gaming_context_caps_heads_below_the_stage1_gate(make_scorer, decision_params):
    assert decision_params["gaming_model_cap"] < decision_params["stage1_theta"]
    d = make_scorer((0.93, "bullying"), (0.88, "threat")).score("you're trash at this game lol")
    assert d.gaming_capped and d.score < decision_params["stage1_theta"]


def test_corroboration_constants_stay_ordered(decision_params):
    c = decision_params["corroboration"]
    assert c["floor"] < c["solo_trust"] and c["solo_damp"] < 1.0


# -- framing applied inside the scorer (proposal SO2) ----------------------------


def test_reported_abuse_is_held_below_every_persona_threshold(lexicon_scorer, decision_params):
    d = lexicon_scorer.score("miss he keeps calling me a retard what do i do")
    assert d.discounted_from and d.discounted_from > 0.80
    assert d.score < min(decision_params["persona_theta"].values())
    assert d.framing_reason.startswith("quoted_or_reported:")


def test_framing_never_discounts_self_harm_intent(lexicon_scorer):
    """MVP gap: 'i'm scared' read as reporting and capped a suicide statement at 0.50."""
    d = lexicon_scorer.score("im scared and i want to kill myself")
    assert d.category == "self_harm" and d.score >= 0.88 and not d.discounted


def test_framing_never_discounts_grooming_coercion(lexicon_scorer):
    d = lexicon_scorer.score("don't tell your parents about this, i'm scared you'll get in trouble")
    assert d.category == "sexual_harassment" and not d.discounted
