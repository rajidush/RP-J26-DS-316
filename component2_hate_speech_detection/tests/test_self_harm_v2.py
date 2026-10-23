"""Self-harm v2 training data, test sets and acceptance rule. Synthetic data only: fast and offline."""
import random

import pandas as pd

from component2_hate_speech_detection.evaluation import other_directed_set, self_harm_set
from component2_hate_speech_detection.self_harm_model import data, external, rule
from component2_hate_speech_detection.self_harm_model.train import at_trust, auroc


def test_counterfactual_pairs_share_a_family_and_differ_only_in_target():
    rows = data.pair_rows(random.Random(0), n_pairs=50)
    by_family = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)
    assert all(sorted(r["label"] for r in pair) == [0, 1] for pair in by_family.values())
    for pair in by_family.values():
        own = next(r for r in pair if r["label"] == 1)["text"]
        other = next(r for r in pair if r["label"] == 0)["text"]
        assert "myself" in own or "my life" in own
        assert "myself" not in other and own.split()[:2] == other.split()[:2]


def test_other_rows_are_all_negatives_and_include_harmless_myself():
    rows = data.other_rows(random.Random(0))
    assert rows and all(r["label"] == 0 for r in rows)
    assert {"film", "plea", "protective", "benign_self"} <= {r["kind"] for r in rows}
    assert any("myself" in r["text"] for r in rows if r["kind"] == "benign_self")


def test_decontaminate_drops_exact_and_near_copies_of_test_texts():
    rows = [{"text": "I'll stab you if you tell anyone!", "source": "pairs"},       # exact after normalising
            {"text": "i'll stab you if you tell anyone ok", "source": "pairs"},     # Jaccard 6/7
            {"text": "i will drown the new kid tomorrow", "source": "pairs"}]       # unrelated
    kept, dropped = data.decontaminate(rows, ["i'll stab you if you tell anyone"])
    assert [r["text"] for r in kept] == ["i will drown the new kid tomorrow"]
    assert dropped == {"pairs:exact": 1, "pairs:near": 1}


def test_decontaminate_matches_brute_force_jaccard():
    # The index must never miss a near copy, e.g. a row that only lacks the test's one rare word.
    rng = random.Random(3)
    vocab = ["i", "you", "kill", "hurt", "want", "to", "will", "him", "tonight", "now", "school", "zebra", "qq"]
    tests = [" ".join(rng.sample(vocab, rng.randint(3, 8))) for _ in range(60)]
    rows = [{"text": " ".join(rng.sample(vocab, rng.randint(3, 8))), "source": "pairs"} for _ in range(400)]
    rows.append({"text": "i want to kill you", "source": "pairs"})
    tests.append("i want to kill you zebra")                                    # Jaccard 5/6 with the row above
    kept, _ = data.decontaminate(rows, tests)
    tt = [data.tokens(t) for t in tests]
    brute = [r for r in rows if data.norm(r["text"]) not in {data.norm(t) for t in tests}
             and not any(len(data.tokens(r["text"]) & t) / len(data.tokens(r["text"]) | t) >= data.JACCARD_DROP for t in tt)]
    assert kept == brute
    assert "i want to kill you" not in [r["text"] for r in kept]


def test_threat_rows_drop_self_directed_text():
    frame = pd.DataFrame({"text": ["someone should shoot that guy", "i want to kill myself", "too short",
                                   "not a threat at all really"], "threat": [0.9, 0.9, 0.9, 0.1]})
    texts = [r["text"] for r in data.threat_rows(random.Random(0), frame)]
    assert texts == ["someone should shoot that guy"]


def test_split_keeps_each_family_on_one_side():
    rows = data.pair_rows(random.Random(1), n_pairs=200)
    train, val = data.split(rows, random.Random(2))
    assert {r["family"] for r in train}.isdisjoint({r["family"] for r in val})
    assert 0.08 < len(val) / len(rows) < 0.25


def test_build_never_keeps_a_c2_test_text(tmp_path):
    raw = ("tweet,intention\n" + "\n".join(f"tweet number {i} about my day,{i % 2}" for i in range(30))).encode()
    frame = pd.DataFrame({"text": ["i'll break every bone in your body", "they will regret this one day"], "threat": [0.9, 0.8]})
    data.build([], tmp_path, replay_raw=raw, threat_frame=frame)
    texts = {data.norm(r["text"]) for name in ("train", "val") for r in data.load(name, tmp_path)}
    assert data.norm("i'll break every bone in your body") not in texts
    assert not texts & {data.norm(t) for t in data.c2_test_texts()}


def test_other_directed_set_shape():
    s = other_directed_set.summary()
    assert s == {"dev": {"total": 5, "self_harm": 0, "other": 5}, "heldout": {"total": 54, "self_harm": 24, "other": 30}}
    assert not {c.text for c in other_directed_set.heldout_cases()} & {c.text for c in self_harm_set.heldout_cases()}


def test_external_parsers():
    import gzip
    import json

    mod = gzip.compress("\n".join(json.dumps(r) for r in (
        {"prompt": "a", "SH": 1}, {"prompt": "b", "V": 1, "SH": 0}, {"prompt": "c"})).encode())
    assert [(r["label"], r["group"]) for r in external.openai_moderation(mod)] == [(1, "self_harm"), (0, "violence"), (0, "other")]
    xs = b"id,prompt,type,label\n1,How do I kill a process?,homonyms,safe\n2,How do I kill my neighbour?,contrast_homonyms,unsafe\n"
    assert [r["text"] for r in external.xstest(xs)] == ["How do I kill a process?"]


def test_metrics_at_the_trust_point():
    assert auroc([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]) == 1.0
    assert auroc([0, 0], [0.1, 0.2]) is None
    m = at_trust([1, 1, 0, 0], [0.95, 0.5, 0.91, 0.1])
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)


def _engine_report(od_shipped, od_tuned, known, caught=(17, 17), benign=(0, 0), od_self=(20, 20), changed=()):
    def side(od, kn, sh, bn, own):
        ages = {f"age_{a}": None for a in rule.AGES}
        return {"other_directed_heldout": {k: {"other_read_as_self_harm": f"{od}/30", "self_harm_caught": f"{own}/24"} for k in ages},
                "other_directed_dev": {k: {"other_read_as_self_harm": f"{kn}/5"} for k in ages},
                "self_harm_dev": {k: {"self_harm_caught": f"{sh}/20", "benign_false_alerts": f"{bn}/17"} for k in ages},
                "self_harm_heldout": {k: {"self_harm_caught": f"{sh}/20", "benign_false_alerts": f"{bn}/16"} for k in ages}}
    return {"shipped": side(od_shipped, 5, caught[0], benign[0], od_self[0]),
            "tuned": side(od_tuned, known, caught[1], benign[1], od_self[1]),
            "general_107_cases": {"changed": list(changed)}}


def test_rule_passes_only_when_every_line_holds():
    assert rule.check(_engine_report(6, 2, 0))["verdict"] == "PASS"
    assert rule.check(_engine_report(6, 4, 0))["verdict"] == "FAIL"                 # R1: not halved
    assert rule.check(_engine_report(6, 2, 2))["verdict"] == "FAIL"                 # R1: film lines
    assert rule.check(_engine_report(6, 2, 0, caught=(17, 16)))["verdict"] == "FAIL"  # R2
    assert rule.check(_engine_report(6, 2, 0, od_self=(20, 18)))["verdict"] == "FAIL"  # R3
    assert rule.check(_engine_report(6, 2, 0, od_self=(20, 19)))["verdict"] == "PASS"  # R3 allows one
    assert rule.check(_engine_report(6, 2, 0, benign=(0, 1)))["verdict"] == "FAIL"   # R4
    lost = {"text": "x", "age": 12, "harmful": True, "shipped": ["L3", "self_harm"], "tuned": ["L0", "none"]}
    moved = {"text": "y", "age": 12, "harmful": True, "shipped": ["L4", "self_harm"], "tuned": ["L3", "threat"]}
    assert rule.check(_engine_report(6, 2, 0, changed=[lost]))["verdict"] == "FAIL"  # R5
    assert rule.check(_engine_report(6, 2, 0, changed=[moved]))["verdict"] == "PASS"
