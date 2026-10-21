"""Tests for the decision rule and size-bucket recall in src/compare_models.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.compare_models import (decide, deployment_check,  # noqa: E402
                                                             previous_measurements, recall_by_size)


def _m(recall, fpr, f1=0.5):
    return {"recall": recall, "false_positive_rate": fpr, "f1": f1}


def _table(a, b50, b25):
    """a (V1) / b50, b25 (V2) = {'full': metrics, 'clean': metrics} for the tiles strategy."""
    t = {}
    for (m, thr), d in {("V1", 0.5): a, ("V2", 0.5): b50, ("V2", 0.25): b25}.items():
        for s in ("full", "clean"):
            t[(m, thr, s, "tiles")] = d[s]
    return t


GOOD_RLVS = {("V1", 0.5): {"accuracy": 0.99}, ("V2", 0.5): {"accuracy": 0.99}, ("V2", 0.25): {"accuracy": 0.99}}
A = {"full": _m(0.40, 0.05), "clean": _m(0.30, 0.04)}


def test_choose_b_when_all_checks_pass_and_prefer_better_f1():
    b50 = {"full": _m(0.60, 0.06, f1=0.70), "clean": _m(0.50, 0.05, f1=0.60)}
    b25 = {"full": _m(0.70, 0.09, f1=0.75), "clean": _m(0.60, 0.08, f1=0.65)}
    d = decide(_table(A, b50, b25), GOOD_RLVS)
    assert d["choose"] == "V2" and d["threshold"] == 0.25
    assert d["verdicts"][0.5]["passes"] and d["verdicts"][0.25]["passes"]


def test_recall_must_be_higher_on_both_sets():
    b = {"full": _m(0.60, 0.05), "clean": _m(0.30, 0.04)}        # equal on clean → not higher
    d = decide(_table(A, b, b), GOOD_RLVS)
    assert d["choose"] == "V1" and not d["verdicts"][0.5]["checks"]["recall_higher_clean"]


def test_false_alarm_rate_at_most_double():
    ok = {"full": _m(0.60, 0.10), "clean": _m(0.50, 0.08)}          # exactly 2x → allowed
    too_many = {"full": _m(0.90, 0.11), "clean": _m(0.80, 0.08)}    # > 2x on full → fails
    d = decide(_table(A, ok, too_many), GOOD_RLVS)
    assert d["verdicts"][0.5]["passes"] and not d["verdicts"][0.25]["checks"]["fpr_within_2x_full"]
    assert d["choose"] == "V2" and d["threshold"] == 0.5


def test_rlvs_accuracy_floor():
    b = {"full": _m(0.60, 0.05), "clean": _m(0.50, 0.04)}
    rlvs = {**GOOD_RLVS, ("V2", 0.5): {"accuracy": 0.979}, ("V2", 0.25): {"accuracy": 0.97}}
    assert decide(_table(A, b, b), rlvs)["choose"] == "V1"


def test_recall_by_size_counts_violent_composites_only():
    rows = [
        {"file": "composite/violence/a.png", "source": "composite", "label": "1", "pred": "1", "strategy": "tiles"},
        {"file": "composite/violence/b.png", "source": "composite", "label": "1", "pred": "0", "strategy": "tiles"},
        {"file": "composite/violence/c.png", "source": "composite", "label": "1", "pred": "1", "strategy": "tiles"},
        {"file": "composite/violence/c.png", "source": "composite", "label": "1", "pred": "0", "strategy": "full"},
        {"file": "composite/non_violence/d.png", "source": "composite", "label": "0", "pred": "1", "strategy": "tiles"},
    ]
    buckets = {"composite/violence/a.png": "under30", "composite/violence/b.png": "under30",
               "composite/violence/c.png": "45plus", "composite/non_violence/d.png": "under30"}
    r = recall_by_size(rows, buckets, "tiles")
    assert r["under30"] == {"n": 2, "hit": 1, "recall": 0.5}
    assert r["30-45"] == {"n": 0, "hit": 0, "recall": None}
    assert r["45plus"]["recall"] == 1.0


def _deploy_table(v1_tiles, v2_full):
    t = {}
    for s_ in ("full", "clean"):
        t[("V1", 0.5, s_, "tiles")] = v1_tiles[s_]
        t[("V2", 0.5, s_, "full")] = v2_full[s_]
    return t


LAT = {"V1": {"tiles": {"median": 243.4}}, "V2": {"full": {"median": 51.3}}}


def test_deployment_check_v2_full_vs_v1_tiles():
    v1 = {"full": _m(0.361, 0.048), "clean": _m(0.230, 0.050)}
    ok = {"full": _m(0.548, 0.093), "clean": _m(0.508, 0.072)}
    d = deployment_check(_deploy_table(v1, ok), GOOD_RLVS, LAT)
    assert d["passes"] and d["speedup"] == 4.7 and d["detail"]["full"]["fpr_ratio"] == 1.94
    too_noisy = {"full": _m(0.548, 0.10), "clean": _m(0.508, 0.072)}
    assert not deployment_check(_deploy_table(v1, too_noisy), GOOD_RLVS, LAT)["checks"]["fpr_within_2x_full"]


def test_previous_measurements_reads_old_ab_format_and_new_format():
    old = {"models": {"A": "models/violence-vit-screen-ft-v1", "B": "models/violence-vit-screen-ft-v2"},
           "rlvs_test": {"A|0.5": {"accuracy": 0.991}, "B|0.5": {"accuracy": 0.9928}, "B|0.25": {"accuracy": 0.9952}},
           "latency": {"A": {"full": {"median": 51.0}}, "B": {"full": {"median": 51.3}}}}
    prev = previous_measurements(old)
    assert prev["violence-vit-screen-ft-v1"]["rlvs"] == {0.5: {"accuracy": 0.991}}
    assert prev["violence-vit-screen-ft-v2"]["rlvs"][0.25]["accuracy"] == 0.9952
    assert prev["violence-vit-screen-ft-v2"]["latency"]["full"]["median"] == 51.3
    new = {"measurements": {"violence-vit-pretrained-hf": {"rlvs": {"0.5": {"accuracy": 0.9964}},
                                                          "latency": {"full": {"median": 50.0}}}}}
    assert previous_measurements(new)["violence-vit-pretrained-hf"]["rlvs"][0.5]["accuracy"] == 0.9964
