"""Expanded synthetic validation of the frozen exact group-aware Shapley explainer."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .evaluate_behavioral_shap import DATA, benchmark, fidelity, scored_test_rows
from .src.behavioral_anomaly import ARTIFACT_DIR, METADATA_FILE, PIPELINE_FILE, anomaly_scores, feature_matrix
from .src.behavioral_shap import (
    BehavioralShap, FALLBACK, GROUPS, TOLERANCE, format_parent_explanation,
    mask_violations, normal_training_rows, ranked, select_background, top_overlap, verified_groups,
)

HOLDOUT = DATA.parent / "step4a_heldout_seed20261020" / DATA.name
PILOT = DATA.parent.parent / "outputs" / "behavioral_shap_pilot_v1.json"
BACKGROUNDS = [(32, 17), (32, 29), (64, 17), (64, 29)]
SAMPLE_SEED = 22026
SCREEN, LANGUAGE, CALENDAR = GROUPS
EVENT_COLUMNS = ["screen_incidents", "language_flags"]
CATEGORIES = ["bullying_run", "grooming_burst", "late_night_violence", "self_harm_day", "volume_spike"]


def keys(frame):
    return list(frame[["child_id", "date"]].itertuples(index=False, name=None))


def draw_panel(test, excluded):
    """Reserve boundary rows, then balance labels, then supplement rare categories."""
    ordered = test.sort_values(["absolute_margin", "child_id", "date"], kind="stable")
    pool = ordered.loc[[key not in excluded for key in keys(ordered)]]
    plan = {}
    for flagged in (True, False):
        rows = pool.loc[pool.margin.gt(0).eq(flagged)].head(6)
        plan.update(dict.fromkeys(rows.index, "near_flagged" if flagged else "near_unflagged"))
    rng = np.random.default_rng(SAMPLE_SEED)
    for label in ("TP", "FP", "FN", "TN"):
        rows = pool.loc[pool.outcome.eq(label)].drop(index=list(plan), errors="ignore")
        selected = rng.choice(rows.index.to_numpy(), size=min(12, len(rows)), replace=False)
        plan.update(dict.fromkeys(map(int, selected), label))
    for category in CATEGORIES:
        rows = pool.loc[pool.anomaly_type.eq(category)]
        available = rows.drop(index=list(plan), errors="ignore")
        needed = max(0, 4 - (len(rows) - len(available)))
        selected = rng.choice(available.index, size=min(needed, len(available)), replace=False)
        plan.update(dict.fromkeys(map(int, selected), "category_supplement"))
    return plan


def challenge_panel(test, excluded, explainer, background):
    """Select two per screen/calendar lead, after an input-defined candidate screen."""
    ordered = test.sort_values(["child_id", "date"])
    pool = ordered.loc[[key not in excluded for key in keys(ordered)]]
    rng = np.random.default_rng(SAMPLE_SEED + 1)
    patterns = {
        SCREEN: pool.screen_incidents.gt(0) & pool.language_flags.eq(0),
        CALENDAR: pool.total_risk_events.eq(0),
    }
    selected = {}
    screened = {}
    for group, pattern in patterns.items():
        candidates = rng.permutation(pool.index[pattern])
        attempted = 0
        found = 0
        for index in candidates[:32]:
            attempted += 1
            result = explainer.explain(test.loc[[index]], background, method="group_aware")
            if result is not None and result.reconstructs and ranked(result.group_values, 1) == [group]:
                selected[int(index)] = "lead_challenge"
                found += 1
                if found == 2:
                    break
        screened[group] = {"candidate_pool": len(candidates), "screened": attempted, "selected": found}
    return selected, screened


def activity_context(frame, *, exact):
    event_values = frame[EVENT_COLUMNS] if exact else frame[EVENT_COLUMNS].gt(0)
    return list(event_values.assign(is_weekend=frame["is_weekend"]).itertuples(index=False, name=None))


def masked_support(explainer, features, background, frequent, observed):
    """Count contextual combinations for the same eight masks used by the explainer."""
    counts = Counter()
    for size in (1, 2):  # exclude all-donor and all-target endpoints
        for subset in combinations(GROUPS, size):
            masked = explainer._replace_groups(features, background.values, subset)
            counts["mixed_rows"] += len(masked)
            counts["presence_context_unseen"] += sum(key not in observed for key in activity_context(masked, exact=False))
            counts["exact_context_frequency_below_5"] += sum(key not in frequent for key in activity_context(masked, exact=True))
    return dict(counts)


def validate_text(row_flag, runs, deletion, text):
    """Independently check the labels, sign, materiality, and fallback behavior."""
    if not row_flag:
        return text is None
    eligible = {}
    if len(runs) == len(BACKGROUNDS) and all(r.reconstructs and not r.invalid_rows for r in runs):
        for group in GROUPS:
            values = [r.group_values[group] for r in runs] + [deletion.get(group, 0)]
            if all(group in ranked(r.group_values, 2) for r in runs):
                if min(values) >= 0.01:
                    eligible[group] = "increased"
                if max(values) <= -0.01:
                    eligible[group] = "decreased"
    if not eligible:
        return text == FALLBACK
    if text is None or text == FALLBACK:
        return False
    seen = {}
    for sentence in text.split(". "):
        for direction in ("increased", "decreased"):
            if f" {direction} the model's score" in sentence:
                seen.update({group: direction for group in GROUPS if group in sentence})
    return seen == eligible


def evaluate_case(test, index, stratum, state):
    explainer, backgrounds, donor, frequent, observed = state
    row = test.loc[[index]]
    data = row.iloc[0]
    item = {name: data[name] for name in ("child_id", "date", "anomaly_type", "episode_id", "outcome", "score", "margin")}
    if pd.isna(item["episode_id"]):
        item["episode_id"] = None
    item.update(stratum=stratum, runs=[], failures=[])
    features = feature_matrix(row, explainer.metadata["features"])
    flagged = bool(data.margin > 0)
    try:
        deletion = explainer.deletion_effects(row, donor)
    except ValueError as error:
        item["failures"].append({"stage": "deletion", "error": str(error)})
        deletion = {}
    item["deletion_effects"] = deletion
    evidence = []
    for (size, seed), background in backgrounds.items():
        try:
            result, timing = benchmark(lambda: explainer.explain(row, background, method="group_aware"))
            if result is None:
                raise ValueError("explanation withheld for assessable row")
            direct_match = bool(abs(result.anomaly_score - data.score) <= TOLERANCE)
            flag_match = result.is_anomaly == flagged
            if not (direct_match and flag_match and result.reconstructs and not result.invalid_rows):
                item["failures"].append({"stage": "validation", "size": size, "seed": seed})
            run = {
                "size": size, "seed": seed, **timing, "groups": result.group_values,
                "leading": ranked(result.group_values, 1)[0], "expected_score": result.expected_score,
                "reconstruction_error": result.reconstruction_error, "score_match": direct_match,
                "threshold_match": flag_match and result.reconstructs,
                "audited_rows": result.audited_rows, "invalid_rows": result.invalid_rows,
                "model_calls": result.model_calls, "model_rows": result.model_rows,
                "mask_support": masked_support(explainer, features, background, frequent, observed),
            }
            if deletion:
                run["fidelity"] = fidelity(result.group_values, deletion)
            item["runs"].append(run)
            evidence.append(result)
        except (ArithmeticError, ValueError) as error:
            evidence.append(None)
            item["failures"].append({"stage": "explanation", "size": size, "seed": seed, "error": str(error)})

    ready = [result for result in evidence if result is not None]
    if len(ready) == len(backgrounds):
        pairs = list(combinations(ready, 2))
        item["stability"] = {
            "lead_agreement": float(np.mean([ranked(a.group_values, 1) == ranked(b.group_values, 1) for a, b in pairs])),
            "top2_overlap": float(np.mean([top_overlap(a.group_values, b.group_values, 2) for a, b in pairs])),
            "all_sign_agreement": float(np.mean([
                np.sign(a.group_values[g]) == np.sign(b.group_values[g]) for a, b in pairs for g in GROUPS
            ])),
            "max_group_span": {g: float(np.ptp([r.group_values[g] for r in ready])) for g in GROUPS},
            "expected_score_span": float(np.ptp([r.expected_score for r in ready])),
        }
    text = format_parent_explanation(evidence, deletion)
    item["parent_text"] = text
    item["verified_groups"] = verified_groups(evidence, deletion) if flagged else {}
    item["parent_grounded"] = validate_text(flagged, ready, deletion, text)
    if not item["parent_grounded"]:
        item["failures"].append({"stage": "parent_grounding"})
    if flagged and ready:
        item["missing_donor_fallback"] = format_parent_explanation(ready, {}) == FALLBACK
        item["single_background_fallback"] = format_parent_explanation(ready[:1], deletion) == FALLBACK
        item["coverage_suppressed"] = explainer.explain(
            row.assign(language_monitor_observed=False), next(iter(backgrounds.values())), method="group_aware"
        ) is None
    return item


def matched_one_event(test, train, explainer):
    """Compare observed screen-only and language-only one-event days at equal calendar context."""
    pairs = []
    rng = np.random.default_rng(SAMPLE_SEED + 2)
    for weekend in (False, True):
        quiet = train.loc[train.is_weekend.eq(weekend) & train.total_risk_events.eq(0)]
        zero_score = float(anomaly_scores(explainer.pipeline, feature_matrix(quiet, explainer.metadata["features"])).mean())
        single = test.loc[test.is_weekend.eq(weekend) & test.total_risk_events.eq(1)]
        screens = rng.permutation(single.index[single.screen_incidents.eq(1)])
        languages = rng.permutation(single.index[single.language_flags.eq(1)])
        for screen, language in zip(screens, languages):
            s, l = test.loc[screen], test.loc[language]
            pairs.append({
                "weekend": weekend, "quiet_reference_rows": len(quiet), "quiet_reference_score": zero_score,
                "screen_id": [s.child_id, s.date], "language_id": [l.child_id, l.date],
                "screen_score_shift": float(s.score - zero_score),
                "language_score_shift": float(l.score - zero_score),
            })
    return pairs


def distribution(test):
    return {
        str(category): {
            "rows": len(group), "screen_present": int(group.screen_incidents.gt(0).sum()),
            "language_present": int(group.language_flags.gt(0).sum()),
            "screen_mean_count": float(group.screen_incidents.mean()),
            "language_mean_count": float(group.language_flags.mean()),
            "flagged": int(group.margin.gt(0).sum()),
        } for category, group in test.groupby("anomaly_type", sort=True)
    }


def evaluate():
    paths = {"original_daily": DATA, "independent_daily": HOLDOUT, "pilot": PILOT,
             "pipeline": ARTIFACT_DIR / PIPELINE_FILE, "metadata": ARTIFACT_DIR / METADATA_FILE}
    before = {key: sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    explainer = BehavioralShap()
    if before["original_daily"] != explainer.metadata["dataset_sha256"]:
        raise ValueError("model training dataset fingerprint mismatch")
    manifest = json.loads((HOLDOUT.parent / "freeze_manifest.json").read_text())
    independent_hash = next(value for key, value in manifest["heldout"].items() if key.endswith("behavioral_daily.csv"))
    # The frozen CSV is LF in the manifest and may be checked out as CRLF.
    independent_lf = sha256(HOLDOUT.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if independent_hash not in (before["independent_daily"], independent_lf):
        raise ValueError("independent synthetic dataset fingerprint mismatch")

    original = pd.read_csv(DATA)
    backgrounds = {key: select_background(original, *key) for key in BACKGROUNDS}
    used = {key for background in backgrounds.values() for key in background.identifiers}
    donor = select_background(original.loc[[key not in used for key in keys(original)]], 64, 101)
    train = normal_training_rows(original)
    train_features = feature_matrix(train, explainer.metadata["features"])
    frequent = {k for k, n in Counter(activity_context(train_features, exact=True)).items() if n >= 5}
    observed = set(activity_context(train_features, exact=False))
    state = explainer, backgrounds, donor, frequent, observed
    pilot_keys = {(item["child_id"], item["date"]) for item in json.loads(PILOT.read_text())["results"]}
    datasets = {}
    for name, daily, excluded in (("original_test", original, pilot_keys),
                                   ("independent_test", pd.read_csv(HOLDOUT), set())):
        test = scored_test_rows(daily, explainer)
        plan = draw_panel(test, excluded)
        if plan != draw_panel(test.sample(frac=1, random_state=3), excluded):
            raise ValueError("case selection is not reproducible after input reordering")
        challenge, screening = challenge_panel(
            test, excluded | set(keys(test.loc[list(plan)])), explainer, backgrounds[32, 17]
        )
        rows = [evaluate_case(test, index, stratum, state) for index, stratum in (plan | challenge).items()]
        datasets[name] = {
            "eligible_test_rows": len(test), "inventory": distribution(test),
            "challenge_screen": screening, "cases": rows,
            "matched_one_event": matched_one_event(test, train, explainer),
        }
        print(f"{name}: {len(rows)} cases, {sum(len(x['runs']) for x in rows)} runs", flush=True)
    after = {key: sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    if before != after:
        raise ValueError("a protected input changed during evaluation")
    return {
        "hashes_before": before, "hashes_after": after, "threshold": explainer.metadata["threshold"],
        "independent_daily_lf_sha256": independent_lf,
        "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "psutil")},
        "platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
        "selection_seed": SAMPLE_SEED, "train_rows": len(train),
        "explanation_backgrounds": [b.metadata() for b in backgrounds.values()],
        "deletion_background": donor.metadata(), "groups": GROUPS, "datasets": datasets,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path")
    result = evaluate()
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
