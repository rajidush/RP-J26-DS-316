"""Function 2 - Behavioral Anomaly Detection, Step 1: daily feature builder.

Turns Component 1 (screen monitoring) and Component 2 (hate-speech detection)
event records into one row per child per Asia/Colombo calendar day. Nothing
here generates data or trains a model.

    from behavioral_dataset import read_jsonl, c1_events, c2_events, build_daily_features
    c1 = c1_events(read_jsonl("c1.jsonl"), child_id="C001")
    c2 = c2_events(read_jsonl("c2.jsonl"), child_id="C001")
    daily = build_daily_features(c1, c2)

Accepted inputs
---------------
C1, two shapes (detected per record):

* Incident record (``incident_id``, ``started_at``, ``category``,
  ``max_confidence``, ...; the shape of the C1 sample file). One record is one
  incident, as C1 reports it. It is not in a shared contract yet.
* TriggerPayload (docs/interface-contracts/comp1_to_comp3.schema.json). One
  record is one flagged frame. Frames are NOT taken to be one incident per
  ``session_id``: within a session, sorted by time, a new incident starts
  whenever ``context_metadata.duration_visible_ms`` is 0 or goes down (the
  contract defines it as ms since the incident's first flagged frame). If any
  frame in a session lacks ``duration_visible_ms`` the session cannot be split
  reliably; it is kept as one *uncertain* group, and every incident-count column
  for that day is NaN. Incident start = first frame time - its
  duration_visible_ms; duration = the last frame's duration_visible_ms.
  ``action``, ``escalated`` and ``dismissed_by_child`` do not exist in this
  shape, so they are unavailable (NaN), never zero.

C2: ``Verdict.to_record()`` rows (component2_hate_speech_detection/src/engine/
verdict.py, "C4 row"). A record is a *flag* when its rung is not L0 and it did
not abstain. Fine-grained Guardian categories are used, not the coarser
comp2_to_comp3 enum.

Child IDs
---------
Neither C1 nor C2 records carry ``child_id`` or a device ID. Each record is
resolved in this order, and the source is kept in ``child_id_source``:

1. ``child_id`` on the record                         -> "record"
2. ``device_id`` on the record, looked up in device_map -> "device_map"
3. the explicit ``child_id`` argument (the caller states which child the whole
   input belongs to)                                    -> "explicit"
4. ``demo_single_child=True``: the integration placeholder
   ``DEMO_CHILD_ID``; rows are marked ``demo_only=True`` -> "demo_placeholder"

Otherwise ValueError. A device_id missing from device_map is an error, not a
fall-through.

Missing vs zero
---------------
* A source is *observed* on a day inside its coverage range: the explicit
  ``c1_coverage`` / ``c2_coverage`` (start, end) dates if given, else that
  child's first to last event date for that source. Outside it every column of
  that source is NaN and ``c1_observed`` / ``c2_observed`` is False: the data
  does not say whether the monitor was running.
* On an observed day with no events, counts are 0 and confidence statistics are
  0.0 (flagged events always score at least the flag threshold, so 0.0 can only
  mean "no flagged event").
* An optional field missing on any event of a day makes the column built from
  it NaN for that day, rather than a partial count.

Timestamps must be RFC 3339 with a UTC offset (interface CHANGELOG 2026-10-11);
a timestamp that cannot be parsed or has no offset raises ValueError. Dates and
the night flag (local 22:00-06:00) use Asia/Colombo. An incident that crosses
midnight counts on the day it started.

Duplicates: exact repeats are dropped; the same key (C1 ``incident_id``, C1
payload ``session_id`` + ``timestamp``, C2 ``id``) with different content
raises ValueError.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

LOCAL_TZ = ZoneInfo("Asia/Colombo")
NIGHT_START_HOUR = 22  # night = local hour >= 22 or < 6
NIGHT_END_HOUR = 6

# Mirrors integration/comp3_to_comp4.DEVICE_CHILD_ID (one device = one child,
# a placeholder until a device-configuration source exists). Demo use only.
DEMO_CHILD_ID = "device-child-01"

C1_SOURCE = "component1_screen_monitoring"
C1_INCIDENT_SCHEMA_VERSIONS = {"1.0"}
# comp1_to_comp3.schema.json risk_category enum.
C1_CATEGORIES = ["violence", "explicit_visual", "self_harm_imagery", "unknown_flagged"]

# Guardian categories in C2 verdict.py CHILD_SAFE_SUMMARIES, minus "none".
C2_CATEGORIES = ["threat", "hate_identity", "bullying", "sexual_harassment",
                 "self_harm", "grooming", "profanity"]
C2_RUNGS = ["L0", "L1", "L2", "L3", "L4", "L5"]
# C2 engine/decide.py RUNG_TO_ACTION; recommended_action is derived from it.
C2_RUNG_TO_ACTION = {"L0": "none", "L1": "notify", "L2": "blur", "L3": "blur",
                     "L4": "block", "L5": "block"}
C2_SOURCES = ["screen", "audio", "typed"]
C2_REQUIRED = ["id", "ts", "top_category", "p", "rung", "urgent", "abstained"]

KEY_COLUMNS = ["child_id", "date", "child_id_source", "demo_only"]
C1_COLUMNS = (
    ["c1_observed", "c1_incidents", "c1_uncertain_groups", "c1_frames_flagged",
     "c1_total_duration_s", "c1_max_conf", "c1_mean_conf", "c1_night_incidents"]
    + [f"c1_{c}" for c in C1_CATEGORIES]
    + ["c1_blocks", "c1_blurs", "c1_escalated", "c1_dismissed"]
)
C2_COLUMNS = (
    ["c2_observed", "c2_records", "c2_flags"]
    + [f"c2_{c}" for c in C2_CATEGORIES]
    + ["c2_urgent", "c2_l4_plus", "c2_blocks", "c2_max_p", "c2_mean_p",
       "c2_night_flags"]
    + [f"c2_{s}" for s in C2_SOURCES]
    + ["c2_distinct_apps", "c2_distinct_app_categories"]
)
COMBINED_COLUMNS = ["total_events", "night_events"]
DAILY_COLUMNS = KEY_COLUMNS + C1_COLUMNS + C2_COLUMNS + COMBINED_COLUMNS

NAN = float("nan")


# --------------------------------------------------------------------------- io

def read_jsonl(path) -> list[dict]:
    """Read a JSON Lines file. Blank lines are skipped; bad JSON raises ValueError."""
    records = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{Path(path).name} line {n}: invalid JSON ({e.msg})") from None
            if not isinstance(record, dict):
                raise ValueError(f"{Path(path).name} line {n}: expected a JSON object")
            records.append(record)
    return records


# -------------------------------------------------------------------- helpers

def parse_timestamp(value, what: str) -> datetime:
    """RFC 3339 timestamp with a UTC offset -> aware datetime in Asia/Colombo."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what}: timestamp is missing")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"{what}: invalid timestamp {value!r}") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{what}: timestamp {value!r} has no UTC offset")
    return parsed.astimezone(LOCAL_TZ)


def is_night(local: datetime) -> bool:
    return local.hour >= NIGHT_START_HOUR or local.hour < NIGHT_END_HOUR


def _number(value, what: str, *, lo=None, hi=None, optional=False):
    if value is None:
        if optional:
            return NAN
        raise ValueError(f"{what} is missing")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
        raise ValueError(f"{what} must be a number, got {value!r}")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ValueError(f"{what} must be in [{lo}, {hi}], got {value!r}")
    return float(value)


def _bool(value, what: str, *, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{what} must be true/false, got {value!r}")
    return value


def _dedupe(records: list[dict], key, what: str) -> list[dict]:
    """Drop exact repeats; raise if one key carries different content."""
    seen: dict = {}
    out = []
    for record in records:
        k = key(record)
        canon = json.dumps(record, sort_keys=True, default=str)
        if k in seen:
            if seen[k] != canon:
                raise ValueError(f"{what} {k!r} appears twice with different content")
            continue
        seen[k] = canon
        out.append(record)
    return out


def resolve_child_id(record: Mapping, *, child_id=None, device_map=None,
                     demo_single_child=False) -> tuple[str, str]:
    """Return (child_id, child_id_source) for one record; see the module docstring."""
    own = record.get("child_id")
    if own is not None:
        if not isinstance(own, str) or not own.strip():
            raise ValueError(f"record child_id must be a non-empty string, got {own!r}")
        return own.strip(), "record"
    device = record.get("device_id")
    if device is not None and device_map is not None:
        if device not in device_map:
            raise ValueError(f"device_id {device!r} is not in device_map")
        return device_map[device], "device_map"
    if child_id is not None:
        if not isinstance(child_id, str) or not child_id.strip():
            raise ValueError("child_id must be a non-empty string")
        return child_id.strip(), "explicit"
    if demo_single_child:
        return DEMO_CHILD_ID, "demo_placeholder"
    raise ValueError(
        "Cannot resolve child_id: the record has no child_id or mapped device_id. "
        "Pass child_id=..., a device_map, or demo_single_child=True for demo data only."
    )


def _child_kwargs(child_id, device_map, demo_single_child):
    return {"child_id": child_id, "device_map": device_map,
            "demo_single_child": demo_single_child}


# ------------------------------------------------------------------------ C1

C1_EVENT_COLUMNS = ["child_id", "child_id_source", "c1_format", "event_key", "started_at",
                    "date", "night", "certain", "category", "max_conf", "mean_conf",
                    "frames", "duration_s", "action", "escalated", "dismissed"]


def _c1_format(record: Mapping) -> str:
    if "incident_id" in record:
        return "incident"
    if "session_id" in record and "confidence_score" in record:
        return "trigger_payload"
    raise ValueError(
        "C1 record is neither an incident record (incident_id) nor a "
        f"TriggerPayload (session_id + confidence_score): keys {sorted(record)}"
    )


def _c1_category(value, what):
    if value not in C1_CATEGORIES:
        raise ValueError(f"{what}: unknown C1 category {value!r}")
    return value


def _c1_incident(record, child, source) -> dict:
    iid = record.get("incident_id")
    if not isinstance(iid, str) or not iid.strip():
        raise ValueError(f"C1 incident_id must be a non-empty string, got {iid!r}")
    what = f"C1 incident {iid}"
    version = record.get("schema_version")
    if version is not None and version not in C1_INCIDENT_SCHEMA_VERSIONS:
        raise ValueError(f"{what}: unsupported schema_version {version!r}")
    if record.get("source", C1_SOURCE) != C1_SOURCE:
        raise ValueError(f"{what}: source is {record.get('source')!r}, not {C1_SOURCE!r}")

    started = parse_timestamp(record.get("started_at"), f"{what} started_at")
    duration = _number(record.get("duration_s"), f"{what} duration_s", lo=0, optional=True)
    if math.isnan(duration) and record.get("ended_at") is not None:
        ended = parse_timestamp(record["ended_at"], f"{what} ended_at")
        duration = (ended - started).total_seconds()
        if duration < 0:
            raise ValueError(f"{what}: ended_at is before started_at")
    frames = _number(record.get("frames_flagged"), f"{what} frames_flagged", lo=1, optional=True)
    action = record.get("action")
    if action is not None and not isinstance(action, str):
        raise ValueError(f"{what}: action must be a string, got {action!r}")
    return {
        "child_id": child, "child_id_source": source, "c1_format": "incident",
        "event_key": iid, "started_at": started, "date": started.date(),
        "night": is_night(started), "certain": True,
        "category": _c1_category(record.get("category"), what),
        "max_conf": _number(record.get("max_confidence"), f"{what} max_confidence", lo=0, hi=1),
        "mean_conf": _number(record.get("mean_confidence"), f"{what} mean_confidence",
                             lo=0, hi=1, optional=True),
        "frames": frames, "duration_s": duration, "action": action,
        "escalated": _bool(record.get("escalated"), f"{what} escalated", optional=True),
        "dismissed": _bool(record.get("dismissed_by_child"), f"{what} dismissed_by_child",
                           optional=True),
    }


def _c1_frame(record) -> dict:
    sid = record.get("session_id")
    if not isinstance(sid, str) or not sid.strip():
        raise ValueError(f"C1 TriggerPayload session_id must be a non-empty string, got {sid!r}")
    what = f"C1 TriggerPayload {sid}"
    if record.get("source_component") != C1_SOURCE:
        raise ValueError(f"{what}: source_component is {record.get('source_component')!r}")
    meta = record.get("context_metadata") or {}
    if not isinstance(meta, Mapping):
        raise ValueError(f"{what}: context_metadata must be an object")
    dvm = meta.get("duration_visible_ms")
    if dvm is not None:
        dvm = _number(dvm, f"{what} duration_visible_ms", lo=0)
    return {
        "session_id": sid,
        "ts": parse_timestamp(record.get("timestamp"), f"{what} timestamp"),
        "category": _c1_category(record.get("risk_category"), what),
        "conf": _number(record.get("confidence_score"), f"{what} confidence_score", lo=0, hi=1),
        "dvm": dvm,
    }


def _c1_payload_groups(frames: list[dict], child, source) -> list[dict]:
    """Split each session's frames into incidents at duration_visible_ms resets."""
    by_session: dict[str, list[dict]] = {}
    for frame in frames:
        by_session.setdefault(frame["session_id"], []).append(frame)

    events = []
    for sid, session in by_session.items():
        session.sort(key=lambda f: f["ts"])
        certain = all(f["dvm"] is not None for f in session)
        if certain:
            groups, current = [], []
            for frame in session:
                if current and (frame["dvm"] == 0 or frame["dvm"] < current[-1]["dvm"]):
                    groups.append(current)
                    current = []
                current.append(frame)
            groups.append(current)
        else:
            groups = [session]

        for n, group in enumerate(groups):
            categories = {f["category"] for f in group}
            if len(categories) > 1:
                raise ValueError(f"C1 TriggerPayload {sid}: one incident has several "
                                 f"risk_category values {sorted(categories)}")
            first = group[0]
            if certain:
                started = first["ts"] - timedelta(milliseconds=first["dvm"])
                duration = group[-1]["dvm"] / 1000.0
            else:
                started, duration = first["ts"], NAN
            confs = [f["conf"] for f in group]
            events.append({
                "child_id": child, "child_id_source": source, "c1_format": "trigger_payload",
                "event_key": f"{sid}#{n}", "started_at": started, "date": started.date(),
                "night": is_night(started), "certain": certain,
                "category": first["category"], "max_conf": max(confs),
                "mean_conf": sum(confs) / len(confs), "frames": float(len(group)),
                "duration_s": duration, "action": None, "escalated": None, "dismissed": None,
            })
    return events


def c1_events(records: Iterable[Mapping], *, child_id=None, device_map=None,
              demo_single_child=False) -> pd.DataFrame:
    """Validate C1 records (either shape) into one row per incident / payload group."""
    kwargs = _child_kwargs(child_id, device_map, demo_single_child)
    incidents, frames_by_child = [], {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"C1 record must be an object, got {type(record).__name__}")
        record = dict(record)
        child, source = resolve_child_id(record, **kwargs)
        if _c1_format(record) == "incident":
            incidents.append((child, source, record))
        else:
            frames_by_child.setdefault((child, source), []).append(record)

    events = []
    for child, source, record in _dedupe_tagged(incidents, lambda r: r["incident_id"],
                                                "C1 incident_id"):
        events.append(_c1_incident(record, child, source))
    for (child, source), payloads in frames_by_child.items():
        payloads = _dedupe(payloads, lambda r: (r.get("session_id"), r.get("timestamp")),
                           "C1 TriggerPayload (session_id, timestamp)")
        events.extend(_c1_payload_groups([_c1_frame(p) for p in payloads], child, source))
    return pd.DataFrame(events, columns=C1_EVENT_COLUMNS)


def _dedupe_tagged(items, key, what):
    """_dedupe for (child, source, record) tuples; the child is part of the content."""
    wrapped = [{"__child": c, "__source": s, **r} for c, s, r in items]
    kept = _dedupe(wrapped, key, what)
    return [(r.pop("__child"), r.pop("__source"), r) for r in kept]


# ------------------------------------------------------------------------ C2

C2_EVENT_COLUMNS = ["child_id", "child_id_source", "event_key", "ts", "date", "night",
                    "is_flag", "category", "p", "rung", "urgent", "action", "source",
                    "app_exe", "app_category"]


def _c2_event(record, child, source) -> dict:
    rid = record.get("id")
    if not isinstance(rid, str) or not rid.strip():
        raise ValueError(f"C2 record id must be a non-empty string, got {rid!r}")
    what = f"C2 record {rid}"
    missing = [f for f in C2_REQUIRED if f not in record]
    if missing:
        raise ValueError(f"{what}: missing required fields {missing}")
    ts = parse_timestamp(record["ts"], f"{what} ts")
    rung = record["rung"]
    if rung not in C2_RUNGS:
        raise ValueError(f"{what}: unknown rung {rung!r}")
    category = record["top_category"]
    if category not in C2_CATEGORIES + ["none"]:
        raise ValueError(f"{what}: unknown top_category {category!r}")
    abstained = _bool(record["abstained"], f"{what} abstained")
    is_flag = rung != "L0" and not abstained and category != "none"

    action = record.get("recommended_action", C2_RUNG_TO_ACTION[rung])
    if action != C2_RUNG_TO_ACTION[rung]:
        raise ValueError(f"{what}: recommended_action {action!r} does not match rung {rung}")
    src = record.get("source")
    if src is not None and src not in C2_SOURCES:
        raise ValueError(f"{what}: unknown source {src!r}")
    app = record.get("app")
    if app is not None and not isinstance(app, Mapping):
        raise ValueError(f"{what}: app must be an object")
    app = app or {}
    return {
        "child_id": child, "child_id_source": source, "event_key": rid, "ts": ts,
        "date": ts.date(), "night": is_night(ts), "is_flag": is_flag, "category": category,
        "p": _number(record["p"], f"{what} p", lo=0, hi=1), "rung": rung,
        "urgent": _bool(record["urgent"], f"{what} urgent"), "action": action,
        "source": src, "app_exe": app.get("exe") or None,
        "app_category": app.get("category") or None,
    }


def c2_events(records: Iterable[Mapping], *, child_id=None, device_map=None,
              demo_single_child=False) -> pd.DataFrame:
    """Validate C2 Verdict.to_record() rows into one row per record."""
    kwargs = _child_kwargs(child_id, device_map, demo_single_child)
    tagged = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"C2 record must be an object, got {type(record).__name__}")
        record = dict(record)
        tagged.append((*resolve_child_id(record, **kwargs), record))
    events = [_c2_event(r, c, s)
              for c, s, r in _dedupe_tagged(tagged, lambda r: r.get("id"), "C2 id")]
    return pd.DataFrame(events, columns=C2_EVENT_COLUMNS)


# --------------------------------------------------------------------- daily

def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _sum_or_nan(series) -> float:
    return NAN if series.isna().any() else float(series.sum())


def _c1_day(day: pd.DataFrame) -> dict:
    row = dict.fromkeys(C1_COLUMNS[1:], 0.0)
    row["c1_observed"] = True
    if day.empty:
        return row
    certain = bool(day["certain"].all())
    row["c1_uncertain_groups"] = float((~day["certain"]).sum())
    row["c1_frames_flagged"] = _sum_or_nan(day["frames"])
    row["c1_total_duration_s"] = _sum_or_nan(day["duration_s"])
    row["c1_max_conf"] = float(day["max_conf"].max())
    row["c1_mean_conf"] = NAN if day["mean_conf"].isna().any() else float(day["mean_conf"].mean())
    if certain:
        row["c1_incidents"] = float(len(day))
        row["c1_night_incidents"] = float(day["night"].sum())
        for c in C1_CATEGORIES:
            row[f"c1_{c}"] = float((day["category"] == c).sum())
    else:
        for col in ["c1_incidents", "c1_night_incidents"] + [f"c1_{c}" for c in C1_CATEGORIES]:
            row[col] = NAN
    actions = day["action"]
    if actions.isna().any() or not certain:
        row["c1_blocks"] = row["c1_blurs"] = NAN
    else:
        row["c1_blocks"] = float((actions == "block").sum())
        row["c1_blurs"] = float((actions == "blur").sum())
    for col, field in (("c1_escalated", "escalated"), ("c1_dismissed", "dismissed")):
        values = day[field]
        row[col] = NAN if values.isna().any() or not certain else float(values.astype(bool).sum())
    return row


def _c2_day(day: pd.DataFrame) -> dict:
    row = dict.fromkeys(C2_COLUMNS[1:], 0.0)
    row["c2_observed"] = True
    row["c2_records"] = float(len(day))
    flags = day[day["is_flag"]] if not day.empty else day
    if flags.empty:
        return row
    row["c2_flags"] = float(len(flags))
    for c in C2_CATEGORIES:
        row[f"c2_{c}"] = float((flags["category"] == c).sum())
    row["c2_urgent"] = float(flags["urgent"].astype(bool).sum())
    row["c2_l4_plus"] = float(flags["rung"].isin(["L4", "L5"]).sum())
    row["c2_blocks"] = float((flags["action"] == "block").sum())
    row["c2_max_p"] = float(flags["p"].max())
    row["c2_mean_p"] = float(flags["p"].mean())
    row["c2_night_flags"] = float(flags["night"].sum())
    if flags["source"].isna().any():
        for s in C2_SOURCES:
            row[f"c2_{s}"] = NAN
    else:
        for s in C2_SOURCES:
            row[f"c2_{s}"] = float((flags["source"] == s).sum())
    row["c2_distinct_apps"] = (NAN if flags["app_exe"].isna().any()
                               else float(flags["app_exe"].nunique()))
    row["c2_distinct_app_categories"] = (NAN if flags["app_category"].isna().any()
                                         else float(flags["app_category"].nunique()))
    return row


def _coverage(events: pd.DataFrame, child, explicit):
    if explicit is not None:
        start, end = (_as_date(d) for d in explicit)
        if end < start:
            raise ValueError("coverage end is before start")
        return start, end
    mine = events[events["child_id"] == child]
    if mine.empty:
        return None
    return min(mine["date"]), max(mine["date"])


def build_daily_features(c1: pd.DataFrame | None = None, c2: pd.DataFrame | None = None, *,
                         start=None, end=None, c1_coverage=None, c2_coverage=None
                         ) -> pd.DataFrame:
    """One row per child per local date; columns are DAILY_COLUMNS.

    c1 / c2 are the outputs of c1_events / c2_events. Each child's date range is
    start..end if given, else the child's first to last event date over both
    sources. c1_coverage / c2_coverage are (start, end) dates during which the
    monitor is known to have run; see the module docstring.
    """
    c1 = c1 if c1 is not None else pd.DataFrame(columns=C1_EVENT_COLUMNS)
    c2 = c2 if c2 is not None else pd.DataFrame(columns=C2_EVENT_COLUMNS)
    children = sorted(set(c1["child_id"]) | set(c2["child_id"]))
    rows = []
    for child in children:
        c1_child, c2_child = c1[c1["child_id"] == child], c2[c2["child_id"] == child]
        dates = list(c1_child["date"]) + list(c2_child["date"])
        first = _as_date(start) if start is not None else min(dates)
        last = _as_date(end) if end is not None else max(dates)
        if last < first:
            raise ValueError("end is before start")
        cov1 = _coverage(c1, child, c1_coverage)
        cov2 = _coverage(c2, child, c2_coverage)
        id_sources = sorted(set(c1_child["child_id_source"]) | set(c2_child["child_id_source"]))

        day = first
        while day <= last:
            row = {"child_id": child, "date": day.isoformat(),
                   "child_id_source": "+".join(id_sources),
                   "demo_only": "demo_placeholder" in id_sources}
            if cov1 and cov1[0] <= day <= cov1[1]:
                row.update(_c1_day(c1_child[c1_child["date"] == day]))
            else:
                row.update({col: NAN for col in C1_COLUMNS}, c1_observed=False)
            if cov2 and cov2[0] <= day <= cov2[1]:
                row.update(_c2_day(c2_child[c2_child["date"] == day]))
            else:
                row.update({col: NAN for col in C2_COLUMNS}, c2_observed=False)
            row["total_events"] = row["c1_incidents"] + row["c2_flags"]
            row["night_events"] = row["c1_night_incidents"] + row["c2_night_flags"]
            rows.append(row)
            day += timedelta(days=1)
    return pd.DataFrame(rows, columns=DAILY_COLUMNS)


# ------------------------------------------------- model-ready features (Step 2)

# Rolling per-child baselines: each day is compared with the BASELINE_WINDOW
# days before it (never itself or a later day). A baseline needs at least
# BASELINE_MIN_DAYS earlier days; before that the columns are NaN and
# baseline_ready is False.
BASELINE_METRICS = ["total_events", "c2_flags", "c1_incidents", "night_events", "c2_urgent",
                    "c2_bullying", "c2_grooming", "c2_self_harm", "c1_blocks"]
BASELINE_WINDOW = 14
BASELINE_MIN_DAYS = 7
BASELINE_STD_FLOOR = 0.5   # z = (x - mean) / (std + floor): no blow-up on flat histories
BASELINE_COLUMNS = (["baseline_ready"]
                    + [f"{m}_{s}{BASELINE_WINDOW}" for m in BASELINE_METRICS for s in ("mean", "z")])
CALENDAR_COLUMNS = ["is_weekend"]

# Ground truth of the synthetic dataset. Never model input.
LABEL_COLUMNS = ["is_anomaly", "anomaly_type", "episode_id"]

# Proposed model input in builder names (final choice in Step 3). Left out:
# identifiers and meta; columns that are constant under full coverage and the
# live C1 / C2 models (observed flags, non-violence C1 categories,
# c1_uncertain_groups, c2_typed, c2_sexual_harassment); and columns another one
# determines (c1_violence = c1_incidents, c1_blurs = incidents - blocks,
# c2_screen = flags - audio, c2_records, c2_blocks = c2_l4_plus).
MODEL_SOURCE_FEATURES = (
    ["c1_incidents", "c1_frames_flagged", "c1_total_duration_s", "c1_max_conf", "c1_mean_conf",
     "c1_night_incidents", "c1_blocks", "c1_escalated", "c1_dismissed",
     "c2_flags", "c2_threat", "c2_hate_identity", "c2_bullying", "c2_self_harm", "c2_grooming",
     "c2_profanity", "c2_urgent", "c2_l4_plus", "c2_max_p", "c2_mean_p", "c2_night_flags",
     "c2_audio", "c2_distinct_apps", "c2_distinct_app_categories",
     "total_events", "night_events"]
    + CALENDAR_COLUMNS
    + [c for c in BASELINE_COLUMNS if c != "baseline_ready"]
)

# ------------------------------------------------ descriptive dataset names

# The one place builder names map to the names in behavioral_daily.csv and
# the model. The builder keeps its short c1_ / c2_ names (Step 1 API and
# tests); to_descriptive_names() renames at the dataset boundary.
# C1 = Component 1 screen monitoring -> "screen_"; C2 = Component 2 harmful
# language detection (on-screen text and audio) -> "language_".
# (builder name, dataset name, C1 / C2 source fields, how it is derived)
FEATURE_CATALOG = [
    ("child_id", "child_id", "record child_id / device_map / explicit", "key"),
    ("date", "date", "C1 started_at, C2 ts", "key: Asia/Colombo local date"),
    ("child_id_source", "child_id_source", "-", "how child_id was resolved"),
    ("demo_only", "demo_only", "-", "child_id is the demo placeholder"),
    ("c1_observed", "screen_monitor_observed", "coverage", "day inside C1 coverage"),
    ("c1_incidents", "screen_incidents", "incident_id | session_id + duration_visible_ms",
     "count of incidents"),
    ("c1_uncertain_groups", "screen_uncertain_groups", "TriggerPayload session_id",
     "payload sessions that cannot be split into incidents"),
    ("c1_frames_flagged", "screen_flagged_frames", "frames_flagged | payload count", "sum"),
    ("c1_total_duration_s", "screen_flagged_seconds",
     "duration_s | ended_at - started_at | duration_visible_ms", "sum"),
    ("c1_max_conf", "screen_max_confidence", "max_confidence | confidence_score", "max"),
    ("c1_mean_conf", "screen_mean_confidence", "mean_confidence | confidence_score",
     "mean of incident means"),
    ("c1_night_incidents", "screen_night_incidents", "started_at", "count at 22:00-06:00 local"),
    ("c1_violence", "screen_violence_incidents", "category | risk_category = violence", "count"),
    ("c1_explicit_visual", "screen_explicit_visual_incidents", "category = explicit_visual",
     "count"),
    ("c1_self_harm_imagery", "screen_self_harm_imagery_incidents",
     "category = self_harm_imagery", "count"),
    ("c1_unknown_flagged", "screen_unknown_flagged_incidents", "category = unknown_flagged",
     "count"),
    ("c1_blocks", "screen_blocked_incidents", "action = block", "count"),
    ("c1_blurs", "screen_blurred_incidents", "action = blur", "count"),
    ("c1_escalated", "screen_escalated_incidents", "escalated", "count of true"),
    ("c1_dismissed", "screen_dismissed_incidents", "dismissed_by_child", "count of true"),
    ("c2_observed", "language_monitor_observed", "coverage", "day inside C2 coverage"),
    ("c2_records", "language_alert_records", "id", "count of records, incl. abstained"),
    ("c2_flags", "language_flags", "rung != L0 and not abstained", "count of flags"),
    ("c2_threat", "language_threat_flags", "top_category = threat", "count"),
    ("c2_hate_identity", "language_hate_identity_flags", "top_category = hate_identity", "count"),
    ("c2_bullying", "language_bullying_flags", "top_category = bullying", "count"),
    ("c2_sexual_harassment", "language_sexual_harassment_flags",
     "top_category = sexual_harassment", "count"),
    ("c2_self_harm", "language_self_harm_flags", "top_category = self_harm", "count"),
    ("c2_grooming", "language_grooming_flags", "top_category = grooming", "count"),
    ("c2_profanity", "language_profanity_flags", "top_category = profanity", "count"),
    ("c2_urgent", "language_urgent_flags", "urgent", "count of true"),
    ("c2_l4_plus", "language_block_rung_flags", "rung in L4, L5", "count"),
    ("c2_blocks", "language_blocked_flags", "recommended_action = block", "count"),
    ("c2_max_p", "language_max_risk", "p", "max"),
    ("c2_mean_p", "language_mean_risk", "p", "mean"),
    ("c2_night_flags", "language_night_flags", "ts", "count at 22:00-06:00 local"),
    ("c2_screen", "language_screen_text_flags", "source = screen", "count"),
    ("c2_audio", "language_audio_flags", "source = audio", "count"),
    ("c2_typed", "language_typed_flags", "source = typed", "count"),
    ("c2_distinct_apps", "language_distinct_apps", "app.exe", "distinct count"),
    ("c2_distinct_app_categories", "language_distinct_app_categories", "app.category",
     "distinct count"),
    ("total_events", "total_risk_events", "C1 incidents + C2 flags", "sum"),
    ("night_events", "night_risk_events", "C1 night incidents + C2 night flags", "sum"),
    ("is_weekend", "is_weekend", "date", "Saturday or Sunday"),
    ("baseline_ready", "baseline_ready", "date", f">= {BASELINE_MIN_DAYS} earlier days"),
]
_BASE_NAMES = {b: d for b, d, _, _ in FEATURE_CATALOG}
for _m in BASELINE_METRICS:
    FEATURE_CATALOG.append((f"{_m}_mean{BASELINE_WINDOW}", f"{_BASE_NAMES[_m]}_prev{BASELINE_WINDOW}d_mean",
                            _m, f"mean of the previous {BASELINE_WINDOW} days"))
    FEATURE_CATALOG.append((f"{_m}_z{BASELINE_WINDOW}", f"{_BASE_NAMES[_m]}_prev{BASELINE_WINDOW}d_z",
                            _m, f"(x - prev mean) / (prev std + {BASELINE_STD_FLOOR})"))
del _m

DESCRIPTIVE_NAMES = {b: d for b, d, _, _ in FEATURE_CATALOG}
if len(set(DESCRIPTIVE_NAMES.values())) != len(DESCRIPTIVE_NAMES):
    raise RuntimeError("FEATURE_CATALOG maps two builder columns to one name")

# Model input in dataset (descriptive) names: the columns of behavioral_daily.csv.
MODEL_FEATURES = [DESCRIPTIVE_NAMES[c] for c in MODEL_SOURCE_FEATURES]
DATASET_BASELINE_COLUMNS = [DESCRIPTIVE_NAMES[c] for c in BASELINE_COLUMNS]


def to_descriptive_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename builder columns to dataset names; other columns are left as they are."""
    renamed = df.rename(columns=DESCRIPTIVE_NAMES)
    if renamed.columns.duplicated().any():
        raise ValueError("renaming produced duplicate column names")
    return renamed


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add is_weekend (Saturday / Sunday of the local date)."""
    out = daily.copy()
    out["is_weekend"] = pd.to_datetime(out["date"]).dt.dayofweek >= 5
    return out


def add_baseline_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add rolling per-child baselines that use only earlier dates.

    Each child's rows must be consecutive dates (as build_daily_features gives).
    For metric m: m_mean14 = mean of the previous 14 days, m_z14 =
    (m - m_mean14) / (std of the previous 14 days + 0.5). Days with fewer than
    7 earlier values get NaN and baseline_ready = False.
    """
    out = daily.sort_values(["child_id", "date"]).reset_index(drop=True)
    dates = pd.to_datetime(out["date"])
    gaps = dates.groupby(out["child_id"]).diff().dropna()
    if (gaps != pd.Timedelta(days=1)).any():
        raise ValueError("each child's rows must be consecutive dates with no gaps or repeats")

    grouped = out.groupby("child_id", sort=False)
    ready = grouped["total_events"].transform(
        lambda s: s.shift(1).rolling(BASELINE_WINDOW, min_periods=1).count())
    out["baseline_ready"] = ready >= BASELINE_MIN_DAYS
    for m in BASELINE_METRICS:
        prev = grouped[m].transform(lambda s: s.shift(1))
        roll = prev.groupby(out["child_id"]).rolling(BASELINE_WINDOW, min_periods=BASELINE_MIN_DAYS)
        mean = roll.mean().reset_index(level=0, drop=True)
        std = roll.std().reset_index(level=0, drop=True)
        out[f"{m}_mean{BASELINE_WINDOW}"] = mean
        out[f"{m}_z{BASELINE_WINDOW}"] = (out[m] - mean) / (std + BASELINE_STD_FLOOR)
    return out
