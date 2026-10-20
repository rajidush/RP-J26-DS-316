"""Function 2 - Behavioral Anomaly Detection, Step 2: synthetic training dataset.

SYNTHETIC DATA. No real child, message or screen is involved. The C1 and C2
sample files only guided the record shapes, value ranges, actions and app
categories; they are far too small to stand for real-world distributions, and
every rate below is an assumption.

Pipeline (one call: build_dataset):

    per-child baseline -> normal C1 / C2 events, day by day
    -> anomaly episodes injected as extra events
    -> C2's own decide() keeps only checks that alert (L1-L5), as C2 does
    -> Step 1 builder (c1_events, c2_events, build_daily_features) with full
       C1 and C2 coverage of all 90 days
    -> calendar and rolling baseline features (earlier dates only)
    -> ground-truth labels joined on (child_id, date)

No daily value is written directly: every count comes from the builder
reading the generated event records. Labels never appear in the records.

Randomness: one SeedSequence(seed). Ages and the episode schedule use spawn
keys (0, *); child i's baseline uses (1, i, 0), its normal events (1, i, 1) and
its injected events (2, i).
A child's normal events therefore depend on no other child.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .behavioral_dataset import (
    BASELINE_COLUMNS,
    CALENDAR_COLUMNS,
    DAILY_COLUMNS,
    FEATURE_CATALOG,
    LABEL_COLUMNS,
    LOCAL_TZ,
    MODEL_FEATURES,
    add_baseline_features,
    add_calendar_features,
    build_daily_features,
    c1_events,
    c2_events,
    parse_timestamp,
    to_descriptive_names,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from component2_hate_speech_detection.src.engine.decide import decide, persona_for_age  # noqa: E402
from component2_hate_speech_detection.src.engine.policy import default_policy  # noqa: E402

GENERATOR_VERSION = "1.0.0"
SEED = 42
N_CHILDREN = 40
N_DAYS = 90
START_DATE = date(2026, 7, 1)
END_DATE = START_DATE + timedelta(days=N_DAYS - 1)          # 2026-09-28

# Chronological split by day index (1-based), the same for every child.
SPLITS = {"train": (1, 56), "validation": (57, 73), "test": (74, 90)}
WARMUP_DAYS = 14          # no episodes here: the rolling baseline needs history

# C2 age groups: 14 children aged 8-10, 14 aged 11-13, 12 aged 14-15.
AGE_GROUPS = [((8, 10), 14), ((11, 13), 14), ((14, 15), 12)]

# Normal C2 category mix (Dirichlet centre). Grooming and self-harm are rare
# on normal days; the C2 sample mix was scripted to cover every category.
NORMAL_MIX = {"profanity": 0.32, "bullying": 0.32, "threat": 0.18, "hate_identity": 0.15,
              "grooming": 0.015, "self_harm": 0.015}
# Candidate-check probability p = 0.3 + 0.7 * Beta(a, b); C2 drops checks under
# its threshold (0.55 / 0.65 / 0.75 by age, 0.50 for grooming and self-harm).
NORMAL_P_BETA = {"profanity": (5, 2), "bullying": (4, 2.5), "threat": (3.5, 3),
                 "hate_identity": (4, 2.5), "grooming": (1.5, 6), "self_harm": (1.5, 6)}
ABSTAIN_RATE = 0.01

# Apps seen in the C2 sample, with C2's app categories.
APPS = {"WhatsApp.exe": "chat", "Discord.exe": "chat", "RobloxPlayerBeta.exe": "game",
        "Minecraft.Windows.exe": "game", "msedge.exe": "browser", "vlc.exe": "video"}
CATEGORY_APPS = {"grooming": ["Discord.exe", "RobloxPlayerBeta.exe"],
                 "bullying": ["WhatsApp.exe", "Discord.exe"],
                 "hate_identity": ["msedge.exe", "Minecraft.Windows.exe"],
                 "threat": ["vlc.exe", "msedge.exe"],
                 "profanity": ["WhatsApp.exe", "Discord.exe", "Minecraft.Windows.exe"],
                 "self_harm": ["WhatsApp.exe", "Discord.exe"]}

# C1, from the C1 sample: block at max_confidence >= 0.80 (inferred, not a
# documented C1 rule), escalation only on blocks, dismissal only on blurs.
C1_BLOCK_AT = 0.80
C1_ESCALATE_RATE = 0.40
C1_DISMISS_RATE = 0.35
C1_MODEL_VERSION = "violence-vit-pretrained-hf"

ANOMALY_TYPES = ["grooming_burst", "late_night_violence", "bullying_run", "self_harm_day",
                 "volume_spike"]
INTENSITIES = ["strong", "moderate", "subtle"]
INTENSITY_P = [0.5, 0.3, 0.2]
CHILDREN_WITHOUT_EPISODES = 6
EPISODES_PER_CHILD = ([1, 2, 3], [0.1, 0.3, 0.6])
MIN_GAP_DAYS = 7
REQUIRED_PER_SPLIT = 2                # per type, in each of train, validation and test

DISCLAIMER = ("Synthetic data. Event shapes follow the Component 1 incident sample and the "
              "Component 2 Verdict.to_record() sample; all rates, mixes and anomaly patterns are "
              "assumptions. Not real children, not an accuracy sample, not a real-world distribution.")


@dataclass
class ChildSpec:
    child_id: str
    index: int
    age: int
    persona: str
    c2_rate: float               # candidate harmful checks per day
    c1_rate: float               # violent-viewing sessions per day
    weekend_mult: float
    night_share: float
    mix: dict
    apps: list


@dataclass
class Episode:
    episode_id: str
    child_id: str
    anomaly_type: str
    intensity: str
    split: str
    start_day: int               # 1-based day index
    end_day: int
    observed_dates: list = field(default_factory=list)

    @property
    def days(self):
        return list(range(self.start_day, self.end_day + 1))


@dataclass
class DatasetResult:
    daily: pd.DataFrame
    episodes: pd.DataFrame
    c1_records: list
    c2_records: list
    metadata: dict


def _rng(seed, *key):
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


def day_date(day_index: int) -> date:
    return START_DATE + timedelta(days=day_index - 1)


def split_of(day_index: int) -> str:
    for name, (lo, hi) in SPLITS.items():
        if lo <= day_index <= hi:
            return name
    raise ValueError(f"day {day_index} is outside the {N_DAYS}-day period")


# ------------------------------------------------------------------ children

def make_children(seed: int = SEED) -> list[ChildSpec]:
    ages = [a for (lo, hi), n in AGE_GROUPS for a in np.linspace(lo, hi, n).round().astype(int)]
    order = _rng(seed, 0, 0).permutation(len(ages))
    children = []
    for i in range(N_CHILDREN):
        rng = _rng(seed, 1, i, 0)
        age = int(ages[order[i]])
        centre = np.array(list(NORMAL_MIX.values()))
        mix = rng.dirichlet(centre * 60)
        apps = sorted(rng.choice(list(APPS), size=int(rng.integers(3, 7)), replace=False).tolist())
        children.append(ChildSpec(
            child_id=f"C{i + 1:03d}", index=i, age=age, persona=persona_for_age(age),
            c2_rate=float(rng.gamma(2.5, 1.5 / 2.5)), c1_rate=float(rng.gamma(2.0, 0.25 / 2.0)),
            weekend_mult=float(rng.uniform(1.2, 1.6)), night_share=float(rng.beta(2, 25)),
            mix=dict(zip(NORMAL_MIX, mix.round(6).tolist())), apps=apps,
        ))
    return children


# ------------------------------------------------------------------ episodes

def _episode_length(rng, anomaly_type) -> int:
    if anomaly_type == "grooming_burst":
        return int(rng.integers(3, 7))
    if anomaly_type == "bullying_run":
        return int(rng.integers(3, 6))
    if anomaly_type == "late_night_violence":
        return int(rng.integers(1, 3))
    if anomaly_type == "self_harm_day":
        return 2 if rng.random() < 0.2 else 1
    return 1


def _window(split):
    lo, hi = SPLITS[split]
    return max(lo, WARMUP_DAYS + 1), hi


def schedule_episodes(children: list[ChildSpec], seed: int = SEED) -> list[Episode]:
    """Place episodes: none in warm-up, none across a split, >= 7 days apart."""
    rng = _rng(seed, 0, 1)
    with_episodes = sorted(rng.choice(N_CHILDREN, N_CHILDREN - CHILDREN_WITHOUT_EPISODES,
                                      replace=False).tolist())
    slots = []
    for i in with_episodes:
        slots += [i] * int(rng.choice(EPISODES_PER_CHILD[0], p=EPISODES_PER_CHILD[1]))
    rng.shuffle(slots)

    required = [(t, s) for s in ("validation", "test", "train") for t in ANOMALY_TYPES
                for _ in range(REQUIRED_PER_SPLIT)]
    widths = {s: _window(s)[1] - _window(s)[0] + 1 for s in SPLITS}
    total = sum(widths.values())
    extra = [(str(rng.choice(ANOMALY_TYPES)),
              str(rng.choice(list(SPLITS), p=[widths[s] / total for s in SPLITS])))
             for _ in range(len(slots) - len(required))]
    wanted = [(t, s, True) for t, s in required] + [(t, s, False) for t, s in extra]

    taken: dict[int, list[tuple[int, int]]] = {}
    episodes = []
    free = list(slots)
    for anomaly_type, split, is_required in wanted:
        length = _episode_length(rng, anomaly_type)
        lo, hi = _window(split)
        # late-night sessions can run past midnight: keep one spare day in the split
        last_start = hi - length + 1 - (1 if anomaly_type == "late_night_violence" else 0)
        placed = False
        for k, child in enumerate(free):
            starts = [d for d in range(lo, last_start + 1)
                      if all(d > e + MIN_GAP_DAYS or d + length - 1 < s - MIN_GAP_DAYS
                             for s, e in taken.get(child, []))]
            if not starts:
                continue
            start = int(rng.choice(starts))
            intensity = (str(rng.choice(["strong", "moderate"], p=[0.6, 0.4])) if is_required
                         else str(rng.choice(INTENSITIES, p=INTENSITY_P)))
            taken.setdefault(child, []).append((start, start + length - 1))
            episodes.append(Episode("", children[child].child_id, anomaly_type, intensity, split,
                                    start, start + length - 1))
            free.pop(k)
            placed = True
            break
        if not placed and is_required:
            raise RuntimeError(f"could not place a required {anomaly_type} episode in {split}")

    episodes.sort(key=lambda e: (e.child_id, e.start_day))
    for n, ep in enumerate(episodes, start=1):
        ep.episode_id = f"E{n:03d}"
    return episodes


# -------------------------------------------------------------------- events

def _local(day: date, minute: float) -> datetime:
    return datetime.combine(day, time(0), tzinfo=LOCAL_TZ) + timedelta(minutes=float(minute))


def _sample_minute(rng, weekend: bool, night_share: float, force_night=None) -> float:
    """Minute of the local day: night is 00:00-06:00 or 22:00-24:00 of the same date."""
    night = rng.random() < night_share if force_night is None else force_night
    if night:
        m = rng.uniform(0, 480)                       # 6 h early morning + 2 h late evening
        return m if m < 360 else m + 960              # 360..480 -> 1320..1440
    return rng.uniform(9 * 60 if weekend else 15 * 60, 22 * 60)


class _Writer:
    """Turns sampled events into C1 incident and C2 Verdict.to_record() rows."""

    def __init__(self, child: ChildSpec):
        self.child = child
        self.c1, self.c2 = [], []
        self.injected: dict[tuple[str, str], str] = {}
        self._c1_n = 0

    def c2_check(self, rng, when: datetime, category, p, *, source=None, app=None,
                 abstained=False, episode_id=None):
        p = round(float(min(max(p, 0.0), 0.9999)), 4)
        d = decide(category, p, self.child.persona, abstained=abstained)
        if not d.alerts:                                 # C2 writes no L0 rows
            return
        if source is None:
            audio_share = 0.6 if category == "threat" else 0.15
            source = "audio" if rng.random() < audio_share else "screen"
        if app is None:
            usual = [a for a in CATEGORY_APPS[category] if a in self.child.apps]
            app = str(rng.choice(usual or CATEGORY_APPS[category]))
        rid = rng.bytes(16).hex()
        self.c2.append({
            "child_id": self.child.child_id, "id": rid,
            "ts": when.astimezone(timezone.utc).isoformat(timespec="milliseconds"),
            "persona": self.child.persona, "top_category": category, "p": p,
            "theta": d.theta, "rung": d.rung, "urgent": d.urgent, "abstained": abstained,
            "recommended_action": d.recommended_action, "source": source,
            "app": {"exe": app, "category": APPS[app]},
        })
        if episode_id and not abstained:
            self.injected[("c2", rid)] = episode_id

    def c1_session(self, rng, start: datetime, n_incidents: int, *, conf_beta=(2, 2),
                   conf_lo=0.5, episode_id=None):
        when = start
        for _ in range(n_incidents):
            if when.date() > END_DATE:                    # nothing outside the 90 days
                break
            max_conf = round(conf_lo + (0.99 - conf_lo) * rng.beta(*conf_beta), 2)
            mean_conf = round(max(0.5, max_conf - rng.uniform(0, 0.08)), 2)
            frames = 1 + int(rng.poisson(2.5))
            duration = float(round(frames * rng.uniform(2, 3)))
            block = max_conf >= C1_BLOCK_AT
            self._c1_n += 1
            iid = (f"c1-{when:%Y%m%d}-{when:%H%M%S}-{self.child.child_id}-{self._c1_n:04d}")
            self.c1.append({
                "child_id": self.child.child_id, "schema_version": "1.0",
                "source": "component1_screen_monitoring", "incident_id": iid,
                "started_at": when.isoformat(timespec="seconds"),
                "ended_at": (when + timedelta(seconds=duration)).isoformat(timespec="seconds"),
                "duration_s": duration, "category": "violence", "max_confidence": max_conf,
                "mean_confidence": mean_conf, "frames_flagged": frames,
                "action": "block" if block else "blur",
                "escalated": bool(block and rng.random() < C1_ESCALATE_RATE),
                "dismissed_by_child": bool(not block and rng.random() < C1_DISMISS_RATE),
                "region_strategy": "tiles", "model_version": C1_MODEL_VERSION,
            })
            if episode_id:
                self.injected[("c1", iid)] = episode_id
            when += timedelta(seconds=duration, minutes=float(rng.uniform(2, 20)))

    def normal_day(self, rng, day: date, mult: float, *, episode_id=None):
        """One day of the child's usual pattern, scaled by mult."""
        child = self.child
        weekend = day.weekday() >= 5
        rate = mult * (child.weekend_mult if weekend else 1.0)
        cats, weights = list(child.mix), np.array(list(child.mix.values()))
        for _ in range(int(rng.poisson(child.c2_rate * rate))):
            category = str(rng.choice(cats, p=weights / weights.sum()))
            a, b = NORMAL_P_BETA[category]
            when = _local(day, _sample_minute(rng, weekend, child.night_share))
            self.c2_check(rng, when, category, 0.3 + 0.7 * rng.beta(a, b),
                          abstained=bool(rng.random() < ABSTAIN_RATE), episode_id=episode_id)
        for _ in range(int(rng.poisson(child.c1_rate * rate))):
            start = _local(day, _sample_minute(rng, weekend, child.night_share))
            self.c1_session(rng, start, 1 + min(int(rng.poisson(0.8)), 3), episode_id=episode_id)


def _inject(writer: _Writer, rng, ep: Episode):
    child, eid, k = writer.child, ep.episode_id, ep.intensity
    for n, day_index in enumerate(ep.days):
        day = day_date(day_index)
        weekend = day.weekday() >= 5
        if ep.anomaly_type == "grooming_burst":
            count = {"strong": rng.integers(2, 4), "moderate": rng.integers(1, 3),
                     "subtle": rng.choice([0, 1], p=[0.3, 0.7])}[k]
            progress = n / max(len(ep.days) - 1, 1)
            for _ in range(int(count)):
                unusual = [a for a in CATEGORY_APPS["grooming"] if a not in child.apps]
                app = str(rng.choice(unusual if unusual and rng.random() < 0.5
                                     else CATEGORY_APPS["grooming"]))
                when = _local(day, _sample_minute(rng, weekend, 0.3))
                writer.c2_check(rng, when, "grooming", 0.5 + 0.4 * progress + rng.normal(0, 0.05),
                                source="screen", app=app, episode_id=eid)
        elif ep.anomaly_type == "late_night_violence":
            n_inc = {"strong": rng.integers(5, 9), "moderate": rng.integers(4, 6), "subtle": 3}[k]
            start = _local(day, rng.choice([rng.uniform(22 * 60, 23 * 60 + 30),
                                            rng.uniform(0, 120)]))
            writer.c1_session(rng, start, int(n_inc), conf_beta=(3, 1.5), conf_lo=0.65,
                              episode_id=eid)
            if rng.random() < 0.5:
                for _ in range(int(rng.integers(1, 3))):
                    when = start + timedelta(minutes=float(rng.uniform(0, 40)))
                    writer.c2_check(rng, when, "threat", rng.uniform(0.6, 0.95), source="audio",
                                    app="vlc.exe", episode_id=eid)
        elif ep.anomaly_type == "bullying_run":
            count = {"strong": rng.integers(2, 5), "moderate": rng.integers(1, 3),
                     "subtle": rng.choice([0, 1, 1])}[k]
            for _ in range(int(count)):
                when = _local(day, _sample_minute(rng, weekend, child.night_share))
                writer.c2_check(rng, when, "bullying", 0.55 + 0.4 * rng.beta(3, 2),
                                source="screen", app=str(rng.choice(CATEGORY_APPS["bullying"])),
                                episode_id=eid)
        elif ep.anomaly_type == "self_harm_day":
            ps = ([rng.uniform(0.85, 0.99)] if k == "subtle"
                  else rng.uniform(0.6, 0.99, size=int(rng.integers(2, 4))).tolist())
            for p in ps:
                when = _local(day, _sample_minute(rng, weekend, 0.4))
                writer.c2_check(rng, when, "self_harm", p, source="screen",
                                app=str(rng.choice(CATEGORY_APPS["self_harm"])), episode_id=eid)
        elif ep.anomaly_type == "volume_spike":
            factor = rng.uniform(1.8, 2.5) if k == "subtle" else rng.uniform(2.5, 4.0)
            writer.normal_day(rng, day, factor - 1.0, episode_id=eid)   # on top of the normal day
        else:
            raise ValueError(f"unknown anomaly type {ep.anomaly_type!r}")


def generate_child_events(child: ChildSpec, episodes: list[Episode], seed: int = SEED) -> _Writer:
    writer = _Writer(child)
    rng = _rng(seed, 1, child.index, 1)
    for day_index in range(1, N_DAYS + 1):
        writer.normal_day(rng, day_date(day_index), float(rng.gamma(4.0, 0.25)))
    inj = _rng(seed, 2, child.index)
    for ep in (e for e in episodes if e.child_id == child.child_id):
        _inject(writer, inj, ep)
    return writer


# ------------------------------------------------------------------- dataset

def _labels(writers, episodes):
    by_id = {e.episode_id: e for e in episodes}
    found: dict[tuple[str, str], str] = {}
    for w in writers:
        for rec in w.c1:
            eid = w.injected.get(("c1", rec["incident_id"]))
            if eid:
                found[(rec["child_id"], parse_timestamp(rec["started_at"], "c1").date().isoformat())] = eid
        for rec in w.c2:
            eid = w.injected.get(("c2", rec["id"]))
            if eid:
                found[(rec["child_id"], parse_timestamp(rec["ts"], "c2").date().isoformat())] = eid
    for (child, d), eid in found.items():
        ep = by_id[eid]
        if ep.child_id != child:
            raise RuntimeError("injected event assigned to the wrong child")
        ep.observed_dates.append(d)
    for ep in episodes:
        ep.observed_dates.sort()
    return found


def build_dataset(seed: int = SEED) -> DatasetResult:
    children = make_children(seed)
    episodes = schedule_episodes(children, seed)
    writers = [generate_child_events(c, episodes, seed) for c in children]
    c1_records = sorted((r for w in writers for r in w.c1),
                        key=lambda r: (r["child_id"], r["started_at"], r["incident_id"]))
    c2_records = sorted((r for w in writers for r in w.c2),
                        key=lambda r: (r["child_id"], r["ts"], r["id"]))

    coverage = (START_DATE, END_DATE)
    daily = build_daily_features(c1_events(c1_records), c2_events(c2_records),
                                 start=START_DATE, end=END_DATE,
                                 c1_coverage=coverage, c2_coverage=coverage)
    if daily["child_id"].nunique() != N_CHILDREN or len(daily) != N_CHILDREN * N_DAYS:
        raise RuntimeError("builder did not return 40 children x 90 days")

    daily = add_baseline_features(add_calendar_features(daily))
    daily["day_index"] = (pd.to_datetime(daily["date"]) - pd.Timestamp(START_DATE)).dt.days + 1
    daily["split"] = daily["day_index"].map(split_of)
    daily["persona"] = daily["child_id"].map({c.child_id: c.persona for c in children})

    found = _labels(writers, episodes)
    keys = list(zip(daily["child_id"], daily["date"]))
    daily["episode_id"] = [found.get(k, "") for k in keys]
    daily["is_anomaly"] = (daily["episode_id"] != "").astype(int)
    by_id = {e.episode_id: e for e in episodes}
    daily["anomaly_type"] = [by_id[e].anomaly_type if e else "none" for e in daily["episode_id"]]

    builder_rest = [c for c in DAILY_COLUMNS if c not in ("child_id", "date")]
    columns = (["child_id", "date", "day_index", "split", "persona"] + builder_rest
               + CALENDAR_COLUMNS + BASELINE_COLUMNS + LABEL_COLUMNS)
    daily = daily[columns].sort_values(["child_id", "date"]).reset_index(drop=True)
    daily = to_descriptive_names(daily)       # dataset names: FEATURE_CATALOG

    episodes_df = pd.DataFrame([{
        "episode_id": e.episode_id, "child_id": e.child_id, "anomaly_type": e.anomaly_type,
        "intensity": e.intensity, "split": e.split,
        "start_date": day_date(e.start_day).isoformat(), "end_date": day_date(e.end_day).isoformat(),
        "planned_days": len(e.days), "observed_days": len(e.observed_dates),
        "observed_dates": ";".join(e.observed_dates),
        "unobserved_dates": ";".join(sorted(set(day_date(d).isoformat() for d in e.days)
                                            - set(e.observed_dates))),
    } for e in episodes])

    metadata = _metadata(seed, children, daily, episodes_df, c1_records, c2_records)
    return DatasetResult(daily, episodes_df, c1_records, c2_records, metadata)


def _metadata(seed, children, daily, episodes_df, c1_records, c2_records) -> dict:
    ready = daily[daily["baseline_ready"]]
    normal = daily[daily["is_anomaly"] == 0]
    return {
        "disclaimer": DISCLAIMER,
        "generator_version": GENERATOR_VERSION, "seed": seed,
        "period": {"start": START_DATE.isoformat(), "end": END_DATE.isoformat(),
                   "days": N_DAYS, "timezone": "Asia/Colombo"},
        "coverage": {"c1": [START_DATE.isoformat(), END_DATE.isoformat()],
                     "c2": [START_DATE.isoformat(), END_DATE.isoformat()]},
        "children": len(children),
        "personas": pd.Series([c.persona for c in children]).value_counts().sort_index().to_dict(),
        "rows": len(daily), "baseline_ready_rows": int(len(ready)),
        "splits": {s: {"days": list(r), "dates": [day_date(r[0]).isoformat(),
                                                  day_date(r[1]).isoformat()],
                       "rows": int((daily["split"] == s).sum()),
                       "anomaly_days": int(daily.loc[daily["split"] == s, "is_anomaly"].sum())}
                   for s, r in SPLITS.items()},
        "anomaly_rate": round(float(daily["is_anomaly"].mean()), 4),
        "anomaly_days_by_type": daily.loc[daily["is_anomaly"] == 1, "anomaly_type"]
                                     .value_counts().sort_index().to_dict(),
        "episodes": int(len(episodes_df)),
        "episodes_by_type": episodes_df["anomaly_type"].value_counts().sort_index().to_dict(),
        "unobserved_episode_days": int((episodes_df["planned_days"]
                                        - episodes_df["observed_days"]).clip(lower=0).sum()),
        "normal_days_with_urgent_flag": round(float((normal["language_urgent_flags"] > 0).mean()), 4),
        "events": {"c1_incidents": len(c1_records), "c2_records": len(c2_records)},
        "c2_policy_version": default_policy().version,
        "label_rule": ("A day is an anomaly when at least one injected event of an episode became "
                       "a record on that local date. Episode days with no surviving injected "
                       "record are labelled 0 and listed in behavioral_episodes.csv."),
        "feature_catalog": [{"builder_name": b, "dataset_name": d, "source_fields": f,
                             "derivation": how} for b, d, f, how in FEATURE_CATALOG],
        "columns": {"model_features": MODEL_FEATURES, "labels": LABEL_COLUMNS,
                    "meta": ["child_id", "date", "day_index", "split", "persona",
                             "child_id_source", "demo_only", "baseline_ready"]},
        "assumptions": {
            "c1_block_at_max_confidence": C1_BLOCK_AT,
            "c1_escalate_rate_on_blocks": C1_ESCALATE_RATE,
            "c1_dismiss_rate_on_blurs": C1_DISMISS_RATE,
            "normal_c2_mix": NORMAL_MIX, "abstain_rate": ABSTAIN_RATE,
            "c1_categories": "violence only (the only live C1 model)",
        },
    }


# --------------------------------------------------------------------- output

CSV_NAME = "behavioral_daily.csv"
EPISODES_NAME = "behavioral_episodes.csv"
METADATA_NAME = "behavioral_daily_metadata.json"
EVENTS_DIR = "synthetic_events"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, records):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=False) + "\n")


def write_outputs(result: DatasetResult, out_dir) -> dict[str, Path]:
    out = Path(out_dir)
    (out / EVENTS_DIR).mkdir(parents=True, exist_ok=True)
    paths = {"daily": out / CSV_NAME, "episodes": out / EPISODES_NAME,
             "c1": out / EVENTS_DIR / "c1_incidents.jsonl",
             "c2": out / EVENTS_DIR / "c2_records.jsonl", "metadata": out / METADATA_NAME}
    result.daily.to_csv(paths["daily"], index=False, float_format="%.10g", lineterminator="\n")
    result.episodes.to_csv(paths["episodes"], index=False, lineterminator="\n")
    _write_jsonl(paths["c1"], result.c1_records)
    _write_jsonl(paths["c2"], result.c2_records)
    meta = dict(result.metadata)
    meta["sha256"] = {k: _sha256(p) for k, p in paths.items() if k != "metadata"}
    paths["metadata"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8",
                                 newline="\n")
    return paths
