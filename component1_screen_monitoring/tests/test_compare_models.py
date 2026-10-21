"""Tests for the decision rule and size-bucket recall in src/compare_models.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.compare_models import decide, recall_by_size  # noqa: E402


def _m(recall, fpr, f1=0.5):
    return {"recall": recall, "false_positive_rate": fpr, "f1": f1}


def _table(a, b50, b25):
    """a / b50 / b25 = {'full': metrics, 'clean': metrics} for the tiles strategy."""
    t = {}
    for (m, thr), d in {("A", 0.5): a, ("B", 0.5): b50, ("B", 0.25): b25}.items():
        for s in ("full", "clean"):
            t[(m, thr, s, "tiles")] = d[s]
    return t


GOOD_RLVS = {("A", 0.5): {"accuracy": 0.99}, ("B", 0.5): {"accuracy": 0.99}, ("B", 0.25): {"accuracy": 0.99}}
A = {"full": _m(0.40, 0.05), "clean": _m(0.30, 0.04)}


def test_choose_b_when_all_checks_pass_and_prefer_better_f1():
    b50 = {"full": _m(0.60, 0.06, f1=0.70), "clean": _m(0.50, 0.05, f1=0.60)}
    b25 = {"full": _m(0.70, 0.09, f1=0.75), "clean": _m(0.60, 0.08, f1=0.65)}
    d = decide(_table(A, b50, b25), GOOD_RLVS)
    assert d["choose"] == "B" and d["threshold"] == 0.25
    assert d["verdicts"][0.5]["passes"] and d["verdicts"][0.25]["passes"]


def test_recall_must_be_higher_on_both_sets():
    b = {"full": _m(0.60, 0.05), "clean": _m(0.30, 0.04)}        # equal on clean → not higher
    d = decide(_table(A, b, b), GOOD_RLVS)
    assert d["choose"] == "A" and not d["verdicts"][0.5]["checks"]["recall_higher_clean"]


def test_false_alarm_rate_at_most_double():
    ok = {"full": _m(0.60, 0.10), "clean": _m(0.50, 0.08)}          # exactly 2x → allowed
    too_many = {"full": _m(0.90, 0.11), "clean": _m(0.80, 0.08)}    # > 2x on full → fails
    d = decide(_table(A, ok, too_many), GOOD_RLVS)
    assert d["verdicts"][0.5]["passes"] and not d["verdicts"][0.25]["checks"]["fpr_within_2x_full"]
    assert d["choose"] == "B" and d["threshold"] == 0.5


def test_rlvs_accuracy_floor():
    b = {"full": _m(0.60, 0.05), "clean": _m(0.50, 0.04)}
    rlvs = {**GOOD_RLVS, ("B", 0.5): {"accuracy": 0.979}, ("B", 0.25): {"accuracy": 0.97}}
    assert decide(_table(A, b, b), rlvs)["choose"] == "A"


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
