"""Step 07 paired renderer audit on a fresh synthetic cohort."""
import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import re
import subprocess
from time import perf_counter
import tracemalloc
from types import ModuleType
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from .evaluate_adaptive_profile import scenario_rows
from .evaluate_anomaly_optimization import digest
from .evaluate_temporal_explain import controlled
from .evaluate_xai_runtime import examples, setup
from .src.behavioral_adaptive_profile import adaptive_profiles
from .src.behavioral_explanation_reliability import assess_temporal
from .src.behavioral_profile import ANCHORS, BUNDLE
from .src.behavioral_synthetic import build_dataset
from .src.behavioral_xai_runtime import ExplanationRequest
from ui import services
from ui.pages import _behavioral_upgrade as current

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "parent_friendly_explanation_step07"
BASE = "d19f9edf13acea6685e6e5237cb633d493dc915e"
PAGE = "ui/pages/_behavioral_upgrade.py"
SEED = 2026112641
REPEATS = 5
JARGON = re.compile(r"\b(anomaly|model|eligible|profile|contributing factor|unusual-activity assessment)\b", re.I)
TREND = re.compile(r"change.{0,30}continued across (?:recent )?days", re.I)
SECTIONS = {"f2-overview", "f2-observed", "f2-recent", "f2-xai", "f2-guidance", "f2-safety", "f2-availability"}


def original():
    source = subprocess.check_output(["git", "show", f"{BASE}:{PAGE}"], cwd=ROOT.parent)
    previous = ModuleType("step07_before")
    exec(compile(source, f"{BASE}:{PAGE}", "exec"), previous.__dict__)
    return previous


class Capture:
    def __init__(self, day):
        self.day = day
        self.section = "alerts"
        self.text = defaultdict(list)
        self.sections = set()

    def selectbox(self, label, options, key=None):
        return self.day if label == "Activity date" else options[0]

    def columns(self, count):
        return [nullcontext() for _ in range(count)]

    def expander(self, title):
        return nullcontext()

    @contextmanager
    def card(self, key, title=None):
        previous = self.section
        self.section = key
        self.sections.add(key)
        try:
            yield
        finally:
            self.section = previous

    def write(self, text):
        self.text[self.section].append(str(text).replace("**", ""))

    markdown = caption = warning = write


def render(module, rows, bundle):
    child, day = rows.iloc[-1][["child_id", "date"]]
    capture = Capture(day)
    with patch.object(module, "st", capture), patch.object(module, "theme", capture):
        module.render({"daily": rows, "child_ids": [child]}, bundle)
    assert capture.sections - {"f2-selection"} == SECTIONS
    return dict(capture.text)


def evidence(rows, bundle):
    matrix, _, history = setup(rows, bundle)
    index = rows.index[-1]
    child, day = rows.loc[index, ["child_id", "date"]]
    score, flag = history.loc[index, ["score", "flag"]]
    assessed = bool(pd.notna(flag))
    flagged = bool(flag) if assessed else False
    request = ExplanationRequest(bundle)
    raw, probes = request.daily(matrix.loc[[index], bundle["features"]], score)
    review = request.review_daily(raw, score, flagged, probes)
    temporal, variants = request.temporal(history, matrix, child, day)
    temporal_review = assess_temporal(temporal, variants)
    guard = adaptive_profiles(rows, matrix, history["flag"]).loc[index]
    usable = bool(guard["available"] and current.adaptive_history_ready(guard))
    dominant = ANCHORS[int(np.argmax(np.abs(guard["robust_z"])))] if usable and guard["abrupt"] else None
    delta = int(np.sign(guard["robust_z"][ANCHORS.index(dominant)])) if dominant else 0
    source = "Language activity" if dominant and dominant.startswith("language_") else "Screen activity"
    opposite = bool(dominant and review.status == "supported" and review.group == source
                    and delta != (1 if flagged else -1))
    urgent = services.behavioral_urgent_report(rows.tail(1), child)
    return {
        "assessed": assessed, "flagged": flagged, "daily": review.status, "group": review.group,
        "corroborated": review.corroborated, "lime_r2": review.diagnostics.get("lime_r2"),
        "temporal": temporal.status, "temporal_review": temporal_review.status,
        "sustained": current.recent_behavior(history, day, bundle["profile"]["minimum"])[1],
        "baseline_usable": usable, "baseline_shift": bool(guard["shift"]),
        "baseline_feature": dominant, "baseline_direction": delta, "opposite": opposite,
        "urgent": not urgent["urgent"].empty, "safety_incomplete": urgent["urgent_incomplete"],
        "recorded_observation": bool(matrix.at[index, "language_bullying_flags"] > 0
                                     or matrix.at[index, "screen_escalated_incidents"] > 0),
    }


def cases(data, bundle):
    yield from examples(data, bundle)
    anchor = data.loc[data["split"].eq("train") & data["is_anomaly"].eq(0)].iloc[0]
    end = pd.Timestamp(data["date"].max())
    for pattern in ("sustained", "spike"):
        rows = controlled(anchor, end, pattern, "language")
        yield pattern, rows.assign(date=rows["date"].dt.date)
    for pattern in ("screen_concern", "abrupt", "urgent_gap"):
        rows, _ = scenario_rows(anchor, "stable" if pattern == "urgent_gap" else pattern, end, 1, 5)
        if pattern == "urgent_gap":
            rows.loc[rows.index[-1], "screen_monitor_observed"] = False
            rows.loc[rows.index[-1], "screen_self_harm_imagery_incidents"] = 1
        rows["date"] = rows["date"].dt.date
        if pattern == "abrupt":
            matrix, _, history = setup(rows, bundle)
            shifts = adaptive_profiles(rows, matrix, history["flag"])["shift"]
            if shifts.any():
                rows = rows.loc[:shifts.idxmax()]
        yield pattern, rows
    # Prespecified evidence-stratification: same source, opposite direction.
    # Search only by computed evidence, with a fixed cap on daily probes.
    checks = 0
    for _, rows in data.groupby("child_id", sort=True):
        rows = rows.assign(date=pd.to_datetime(rows["date"]).dt.date).reset_index(drop=True)
        matrix, _, history = setup(rows, bundle)
        guard = adaptive_profiles(rows, matrix, history["flag"])
        candidates = guard.loc[guard["abrupt"] & history["flag"].notna()]
        for index, profile in candidates.iterrows():
            deviations = profile["robust_z"]
            leading = int(np.argmax(np.abs(deviations)))
            flagged = bool(history.at[index, "flag"])
            if flagged == (deviations[leading] > 0):
                continue
            request = ExplanationRequest(bundle)
            score = history.at[index, "score"]
            raw, probes = request.daily(matrix.loc[[index], bundle["features"]], score)
            review = request.review_daily(raw, score, flagged, probes)
            expected = "Language activity" if ANCHORS[leading].startswith("language_") else "Screen activity"
            if review.status == "supported" and review.group == expected:
                yield "opposite_evidence", rows.loc[:index]
                return
            checks += 1
            if checks >= 64:
                return


def audit(messages, context):
    lookup = lambda section: " ".join(messages.get(section, ()))
    xai, advice = lookup("f2-xai"), lookup("f2-guidance")
    baseline, safety = lookup("f2-observed"), lookup("f2-safety")
    overview, recent = lookup("f2-overview"), lookup("f2-recent")
    issues = []
    daily_claim = ("unusual" if "an unusual" in overview.lower() else
                   "usual" if "no unusual" in overview.lower() else "unassessed")
    expected_daily = "unassessed" if not context["assessed"] else "unusual" if context["flagged"] else "usual"
    if daily_claim != expected_daily:
        issues.append("daily assessment")
    if ("look more unusual" in xai or "look less unusual" in xai) != (context["daily"] == "supported"):
        issues.append("daily factor outside reviewed evidence")
    if context["daily"] == "supported" and (
        ("look more unusual" in xai) != context["flagged"]
    ):
        issues.append("daily factor direction")
    if ("Both explanation checks" in xai or "second explanation" in xai) != context["corroborated"]:
        issues.append("corroboration")
    if "AI explanation highlighted" in advice and context["daily"] != "supported":
        issues.append("unsupported guidance topic")
    observation_advice = "records include language flagged" in advice or "marked for review" in advice
    if observation_advice and not context["recorded_observation"]:
        issues.append("unsupported recorded observation")
    if context["urgent"] and ("Urgent safety review" not in lookup("alerts") or "Try Asking" in advice):
        issues.append("urgent alert or priority guidance lost")
    if context["safety_incomplete"] and "Safety monitoring was incomplete for the selected day" not in safety:
        issues.append("coverage warning lost")
    if bool(TREND.search(recent)) != context["sustained"]:
        issues.append("continuing-change headline")
    if bool(re.search(r"(?:On one recent|Across several)", recent)) != (
        context["temporal_review"] == "supported" and context["temporal"] in {"isolated", "sustained"}
    ):
        issues.append("named temporal effect outside reviewed evidence")
    expected_direction = {1: "More ", -1: "Fewer "}.get(context["baseline_direction"])
    directional = baseline.startswith(("More ", "Fewer "))
    if directional and (not expected_direction or not baseline.startswith(expected_direction)):
        issues.append("unsupported personal direction")
    if not context["baseline_usable"] and (directional or "similar to" in baseline):
        issues.append("unsupported personal comparison")
    flattened = " ".join(line for lines in messages.values() for line in lines)
    words = re.findall(r"\b[\w’]+\b", flattened)
    sentences = max(1, len(re.findall(r"[.!?](?:[”\s]|$)", flattened)))
    shared_trend = sum(bool(TREND.search(lookup(section))) for section in ("f2-overview", "f2-recent"))
    return {"issues": issues, "words": len(words), "words_per_sentence": len(words) / sentences,
            "jargon_hits": len(JARGON.findall(flattened)), "duplicated_trend": int(shared_trend > 1),
            "guided_topic": bool("AI explanation highlighted" in advice or observation_advice),
            "directional_personal": directional,
            "opposite_evidence_explicit": bool(context["opposite"] and directional and "your child" in baseline
                                               and "AI’s assessment" in xai)}


def evaluate(data, phase):
    bundle = joblib.load(BUNDLE)
    old = original()
    records, timings = [], []
    for name, rows in cases(data, bundle):
        context = evidence(rows, bundle)
        for label, module in (("before", old), ("after", current)):
            messages = render(module, rows, bundle)
            records.append({"case": name, "arm": label, "evidence": context,
                            "messages": messages, "checks": audit(messages, context)})
        for index in range(REPEATS):
            order = (("before", old), ("after", current))
            for label, module in (order if index % 2 == 0 else order[::-1]):
                start = perf_counter()
                render(module, rows, bundle)
                timings.append({"case": name, "arm": label, "repetition": index,
                                "milliseconds": (perf_counter() - start) * 1000})
        for label, module in (("before", old), ("after", current)):
            with patch.object(bundle["pipeline"], "score_samples", wraps=bundle["pipeline"].score_samples) as spy:
                render(module, rows, bundle)
                calls = spy.call_count
            tracemalloc.start()
            render(module, rows, bundle)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            next(record for record in reversed(records) if record["arm"] == label).update(
                prediction_calls=calls, peak_bytes=peak,
            )
    (OUT / f"{phase}_messages.json").write_text(json.dumps(records, indent=2, ensure_ascii=False, default=str),
                                                encoding="utf-8")
    latency = pd.DataFrame(timings)
    latency.to_csv(OUT / f"{phase}_runtime.csv", index=False)
    summary = {}
    for arm in ("before", "after"):
        own = [row for row in records if row["arm"] == arm]
        checks = [row["checks"] for row in own]
        values = latency.loc[latency["arm"].eq(arm), "milliseconds"]
        summary[arm] = {"cases": len(own), "issues": sum(len(item["issues"]) for item in checks),
                        **{name: float(np.mean([item[name] for item in checks]))
                           for name in ("words", "words_per_sentence", "jargon_hits")},
                        **{name: sum(item[name] for item in checks)
                           for name in ("duplicated_trend", "guided_topic", "directional_personal",
                                        "opposite_evidence_explicit")},
                        "median_ms": float(values.median()), "p95_ms": float(values.quantile(.95)),
                        "median_peak_bytes": int(np.median([row["peak_bytes"] for row in own])),
                        "total_prediction_calls": sum(row["prediction_calls"] for row in own)}
    for left, right in zip(records[::2], records[1::2], strict=True):
        assert left["prediction_calls"] == right["prediction_calls"], left["case"]
        assert left["messages"].get("alerts", []) == right["messages"].get("alerts", []), left["case"]
    (OUT / f"{phase}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "lock", "holdout"))
    stage = parser.parse_args().stage
    OUT.mkdir(exist_ok=True)
    files = (Path(__file__), ROOT.parent / PAGE, BUNDLE)
    hashes = {path.name: digest(path) for path in files}
    lock_path = OUT / "selection_lock.json"
    if stage == "validate":
        evaluate(pd.read_csv(ROOT / "data" / "behavioral_daily.csv"), "development")
    elif stage == "lock":
        lock_path.write_text(json.dumps({
            "seed": SEED, "baseline_ref": BASE, "sha256": hashes,
            "criteria": "Trace daily directions, corroboration, guidance, baseline, urgent labels and coverage to evidence. "
                        "Count repeated trend claims, fixed jargon phrases, words and words per sentence. "
                        "Compare identical prediction counts and paired runtime and traced Python memory. "
                        "Report observed results without tuning on holdout.",
        }, indent=2), encoding="utf-8")
    else:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock["sha256"] != hashes or lock["baseline_ref"] != BASE:
            raise RuntimeError("Step 07 evaluation lock does not match")
        data = build_dataset(lock["seed"]).daily
        (OUT / "holdout_fingerprint.json").write_text(json.dumps({
            "rows": len(data), "sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
        }, indent=2), encoding="utf-8")
        evaluate(data, "holdout")


if __name__ == "__main__":
    main()
