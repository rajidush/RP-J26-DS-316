"""Paired one-factor synthetic controls for the frozen detector, replication v1."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .behavioral_shap_independent_synthetic import screen_event, language_event
from .src.behavioral_dataset import (
    C1_COLUMNS, C1_EVENT_COLUMNS, C2_COLUMNS, C2_EVENT_COLUMNS, COMBINED_COLUMNS,
    _c1_day, _c2_day, add_calendar_features, to_descriptive_names,
)
from .src.behavioral_synthetic import make_children

EVENT_SEED = 2026112302
SEEN_SEED = 42
UNSEEN_SEED = 2026112303
MATCH_SEED = 2026112304
BOOTSTRAP_SEED = 2026112305
FIRST_MONDAY = date(2026, 11, 2)
PAIRS = 20
DAYS = 12
ARMS = ("reference", "screen_plus_two", "language_plus_two", "night_40pct",
        "balance_to_screen", "weekend", "unseen_profile", "coverage_loss")
CLUSTERS = ("screen_cluster", "language_cluster", "mixed_cluster")
MIX_CATEGORIES = ("profanity", "bullying", "threat", "hate_identity")


def matched_profiles():
    """Sample original training profiles and pair fresh draws by exact age."""
    rng = np.random.default_rng(MATCH_SEED)
    seen = make_children(SEEN_SEED)
    unseen = make_children(UNSEEN_SEED)
    positions = rng.choice(len(seen), size=PAIRS, replace=False)
    pairs = []
    for i in positions:
        original = seen[i]
        candidates = [j for j, profile in enumerate(unseen) if original.age == profile.age]
        j = int(rng.choice(candidates))
        pairs.append((original, unseen.pop(j)))
    return pairs


def _source_bank(rng, make, source, pair, day_index, count, injected):
    # Structural event content and timing uniforms are reused in every arm.
    content = []
    for i in range(count):
        key = f"P{pair:02d}-{day_index:02d}-{source}-{injected}-{i}"
        content.append(make(rng, key, "", FIRST_MONDAY, injected=injected, night=0))
    return content, rng.random(count)


def _materialize(bank, count, threshold):
    content, uniforms = bank
    return [{**event, "night": bool(u < threshold)}
            for event, u in zip(content[:count], uniforms[:count], strict=True)]


def _normalized_rows(records, event_columns):
    return pd.DataFrame(records, columns=event_columns)


def _aggregate(screen, language, outage):
    # Each input already belongs to one normalized child-day and source.
    result = {}
    for source, records, event_columns, columns, observed, aggregate in (
        ("screen", screen, C1_EVENT_COLUMNS, C1_COLUMNS, "c1_observed", _c1_day),
        ("language", language, C2_EVENT_COLUMNS, C2_COLUMNS, "c2_observed", _c2_day),
    ):
        if outage in (source, "both"):
            result.update(dict.fromkeys(columns, np.nan))
            result[observed] = False
        else:
            result.update(aggregate(_normalized_rows(records, event_columns)))
    result.update(zip(COMBINED_COLUMNS, (
        result["c1_incidents"] + result["c2_flags"],
        result["c1_night_incidents"] + result["c2_night_flags"],
    ), strict=True))
    return result


def generate():
    rows = []
    for pair, (seen, unseen) in enumerate(matched_profiles(), start=1):
        layout, *streams = np.random.SeedSequence(EVENT_SEED, spawn_key=(pair,)).spawn(DAYS + 1)
        layout_rng = np.random.default_rng(layout)
        labels = layout_rng.permutation(["none"] * 9 + list(CLUSTERS))
        outage_days = dict(zip(layout_rng.choice(DAYS, size=3, replace=False),
                               ("screen", "language", "both"), strict=True))
        for i, stream in enumerate(streams):
            base_rng, screen_rng, language_seed, injection_rng = map(np.random.default_rng, stream.spawn(4))
            shock = base_rng.gamma(2, 0.5)
            uniforms = base_rng.random(2)
            # Inverse Poisson CDF gives a paired draw under each profile's rates.
            seen_counts = poisson.ppf(uniforms, [1.8 * seen.c1_rate * shock,
                                                seen.c2_rate * shock]).astype(int)
            unseen_counts = poisson.ppf(uniforms, [1.8 * unseen.c1_rate * shock,
                                                  unseen.c2_rate * shock]).astype(int)
            screen_bank = _source_bank(screen_rng, screen_event, "screen", pair, i,
                                       int(max(seen_counts[0] + 2, unseen_counts[0])), False)
            language_key = int(language_seed.integers(2**31))
            banks = []
            language_bank_size = int(max(seen_counts[1] + 2, unseen_counts[1]))
            for profile in (seen, unseen):
                weights = np.array([profile.mix[name] for name in MIX_CATEGORIES])
                mix = weights / weights.sum()
                banks.append(_source_bank(
                    np.random.default_rng(language_key),
                    lambda rng, *args, **kw: language_event(rng, *args, mix=mix, **kw),
                    "language", pair, i, language_bank_size, False,
                ))
            label = str(labels[i])
            injected_screen = 2 if label in ("screen_cluster", "mixed_cluster") else 0
            injected_language = 2 if label in ("language_cluster", "mixed_cluster") else 0
            screen_injected = _materialize(_source_bank(
                injection_rng, screen_event, "screen", pair, i, injected_screen, True
            ), injected_screen, .10)
            language_injected = _materialize(_source_bank(
                injection_rng, lambda rng, *args, **kw: language_event(rng, *args, mix=None, **kw),
                "language", pair, i, injected_language, True
            ), injected_language, .10)
            for arm in ARMS:
                new = arm == "unseen_profile"
                ns, nl = map(int, unseen_counts if new else seen_counts)
                if arm == "screen_plus_two":
                    ns += 2
                if arm == "language_plus_two":
                    nl += 2
                if arm == "balance_to_screen":
                    moved = min(nl, 2)
                    ns, nl = ns + moved, nl - moved
                # Profile night rate remains profile-specific in the base arm.
                night = .40 if arm == "night_40pct" else min(.40, (unseen if new else seen).night_share)
                screen = _materialize(screen_bank, ns, night) + screen_injected
                language = _materialize(banks[int(new)], nl, night) + language_injected
                day = FIRST_MONDAY + timedelta(days=(i // 5) * 7 + i % 5)
                if arm == "weekend":
                    # Use distinct weekend dates while retaining the same paired day index.
                    day = FIRST_MONDAY + timedelta(days=(i // 2) * 7 + 5 + i % 2)
                outage = outage_days.get(i, "none") if arm == "coverage_loss" else "none"
                rows.append({
                    "pair_id": f"P{pair:02d}", "day_index": i, "arm": arm,
                    "child_id": f"XAI-CR1-NEW-{pair:02d}" if new else seen.child_id,
                    "profile_age": seen.age, "date": day.isoformat(), "outage": outage,
                    "is_anomaly": int(label != "none"), "anomaly_type": label,
                    **_aggregate(screen, language, outage),
                })

    return to_descriptive_names(add_calendar_features(pd.DataFrame(rows)))
