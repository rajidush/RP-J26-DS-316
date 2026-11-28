"""Independent normalized-event simulator for the locked synthetic protocol v1."""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .src.behavioral_dataset import (
    C1_COLUMNS, C1_EVENT_COLUMNS, C2_COLUMNS, C2_EVENT_COLUMNS, COMBINED_COLUMNS,
    add_calendar_features, build_daily_features, to_descriptive_names,
)

SEED = 2026112301
START = date(2026, 10, 5)
DAYS = 28
CHILDREN_PER_SCENARIO = 12
SCENARIOS = {
    "screen_weekend": {"screen_rate": 1.4, "language_rate": 0.65, "weekend": 2.2,
                       "night": 0.22, "anomaly_chance": 0.07, "types": [0.65, 0.10, 0.25],
                       "extra_screen_outage": 0.05, "extra_language_outage": 0.02},
    "language_bursts": {"screen_rate": 0.45, "language_rate": 2.3, "weekend": 0.7,
                        "night": 0.35, "anomaly_chance": 0.16, "types": [0.15, 0.55, 0.30],
                        "extra_screen_outage": 0.02, "extra_language_outage": 0.07},
}
ANOMALIES = ("screen_cluster", "language_cluster", "mixed_cluster")
LANGUAGE_CATEGORIES = ("profanity", "bullying", "threat", "hate_identity")
APPS = (("synthetic-chat", "chat"), ("synthetic-game", "game"), ("synthetic-video", "video"))


def screen_event(rng, key, child, day, *, injected, night):
    frames = int(rng.integers(2, 11) if injected else rng.geometric(0.43))
    confidence = float(rng.uniform(0.72, 0.97) if injected else rng.uniform(0.51, 0.86))
    action = "block" if confidence >= 0.85 else "blur"
    return {
        "child_id": child, "child_id_source": "synthetic_validation", "event_key": key, "date": day,
        "night": bool(rng.random() < night), "certain": True,
        "category": str(rng.choice(("violence", "explicit_visual"), p=(0.55, 0.45))),
        "max_conf": confidence, "mean_conf": confidence - rng.uniform(0, 0.08),
        "frames": frames, "duration_s": frames * rng.uniform(1.1, 3.7), "action": action,
        "escalated": bool(action == "block" and rng.random() < 0.12),
        "dismissed": bool(action == "blur" and rng.random() < 0.09),
    }


def language_event(rng, key, child, day, *, injected, night, mix):
    category = str(rng.choice(("threat", "bullying", "hate_identity")) if injected
                   else rng.choice(LANGUAGE_CATEGORIES, p=mix))
    risk = float(rng.uniform(0.70, 0.99) if injected else rng.uniform(0.48, 0.82))
    rung = "L5" if risk >= 0.96 else "L4" if risk >= 0.85 else "L3" if risk >= 0.67 else "L2"
    app, app_type = APPS[int(rng.integers(len(APPS)))]
    return {
        "child_id": child, "child_id_source": "synthetic_validation", "event_key": key, "date": day,
        "night": bool(rng.random() < night), "is_flag": True, "category": category, "p": risk,
        "rung": rung, "urgent": bool(category == "threat" and rung == "L5"),
        "action": "block" if rung in ("L4", "L5") else "warn",
        "source": str(rng.choice(("audio", "screen", "typed"), p=(0.30, 0.25, 0.45))),
        "app_exe": app, "app_category": app_type,
    }


def generate():
    """Generate latent daily activity first, then redact unavailable channel observations."""
    screen, language, designs = [], [], []
    dates = [START + timedelta(days=i) for i in range(DAYS)]
    for scenario_index, (scenario, cfg) in enumerate(SCENARIOS.items()):
        for profile_index in range(CHILDREN_PER_SCENARIO):
            streams = np.random.SeedSequence(SEED, spawn_key=(scenario_index, profile_index)).spawn(3)
            rng, labels, outages = map(np.random.default_rng, streams)
            child = f"XAI-IV1-{scenario_index + 1}-{profile_index + 1:02d}"
            activity = rng.lognormal(0, 0.42, size=2)
            profile = "quiet" if profile_index % 3 == 0 else "steady" if profile_index % 3 == 1 else "variable"
            activity *= {"quiet": 0.6, "steady": 1.0, "variable": 1.35}[profile]
            mix = rng.dirichlet((3.5, 4.5, 1.4, 2.1))
            burst = rng.gamma(2, 0.5, size=(DAYS + 2) // 3)
            screen_start, language_start = outages.integers(0, DAYS - 2, size=2)
            for i, day in enumerate(dates):
                kind = str(labels.choice(ANOMALIES, p=cfg["types"])) if labels.random() < cfg["anomaly_chance"] else "none"
                factor = burst[i // 3] * (cfg["weekend"] if day.weekday() >= 5 else 1)
                ordinary_screen = int(rng.poisson(cfg["screen_rate"] * activity[0] * factor))
                ordinary_language = int(rng.poisson(cfg["language_rate"] * activity[1] * factor))
                extra_screen = int(rng.integers(1, 5)) if kind in ("screen_cluster", "mixed_cluster") else 0
                extra_language = int(rng.integers(1, 4)) if kind in ("language_cluster", "mixed_cluster") else 0
                for j in range(ordinary_screen + extra_screen):
                    screen.append(screen_event(rng, f"{child}-{i}-s{j}", child, day,
                                               injected=j >= ordinary_screen, night=cfg["night"]))
                for j in range(ordinary_language + extra_language):
                    language.append(language_event(rng, f"{child}-{i}-l{j}", child, day,
                                                   injected=j >= ordinary_language, night=cfg["night"], mix=mix))
                designs.append({
                    "child_id": child, "date": day.isoformat(), "scenario": scenario, "profile": profile,
                    "is_anomaly": int(kind != "none"), "anomaly_type": kind,
                    "screen_outage": bool(screen_start <= i < screen_start + 2
                                          or outages.random() < cfg["extra_screen_outage"]),
                    "language_outage": bool(language_start <= i < language_start + 3
                                            or outages.random() < cfg["extra_language_outage"]),
                })

    coverage = dates[0], dates[-1]
    data = build_daily_features(
        pd.DataFrame(screen, columns=C1_EVENT_COLUMNS), pd.DataFrame(language, columns=C2_EVENT_COLUMNS),
        start=dates[0], end=dates[-1], c1_coverage=coverage, c2_coverage=coverage,
    ).merge(pd.DataFrame(designs), on=["child_id", "date"], how="outer", validate="one_to_one", indicator=True)
    if not data["_merge"].eq("both").all():
        raise ValueError("independent simulator lost a child profile or day")
    data = data.drop(columns="_merge")
    for outage, columns, observed in (
        ("screen_outage", C1_COLUMNS, "c1_observed"), ("language_outage", C2_COLUMNS, "c2_observed")
    ):
        data.loc[data[outage], [column for column in columns if column != observed]] = np.nan
        data.loc[data[outage], observed] = False
    data.loc[data.screen_outage | data.language_outage, COMBINED_COLUMNS] = np.nan
    data["split"] = "independent_synthetic_validation"
    return to_descriptive_names(add_calendar_features(data))
