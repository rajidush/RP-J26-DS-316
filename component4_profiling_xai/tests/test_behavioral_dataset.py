import json
import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.behavioral_dataset import (
    DAILY_COLUMNS,
    DEMO_CHILD_ID,
    build_daily_features,
    c1_events,
    c2_events,
    parse_timestamp,
    read_jsonl,
)

MOCK = Path(__file__).resolve().parents[1] / "mock_inputs"
C1_SAMPLE = MOCK / "c1_sample_incidents.jsonl"
C2_SAMPLE = MOCK / "c2_sample_records_sanitized.jsonl"
REPO = Path(__file__).resolve().parents[2]


def incident(iid="c1-1", started="2026-10-19T16:00:00+05:30", **extra):
    record = {"schema_version": "1.0", "source": "component1_screen_monitoring",
              "incident_id": iid, "started_at": started, "duration_s": 10.0,
              "category": "violence", "max_confidence": 0.9, "mean_confidence": 0.8,
              "frames_flagged": 3, "action": "block", "escalated": False,
              "dismissed_by_child": False}
    record.update(extra)
    return {k: v for k, v in record.items() if v is not None}


def payload(sid, ts, dvm=None, conf=0.8):
    meta = {} if dvm is None else {"duration_visible_ms": dvm}
    return {"session_id": sid, "timestamp": ts, "source_component": "component1_screen_monitoring",
            "content_type": "image", "risk_category": "violence", "confidence_score": conf,
            "context_metadata": meta}


def verdict(rid="v1", ts="2026-10-19T10:00:00+00:00", **extra):
    record = {"id": rid, "ts": ts, "top_category": "bullying", "p": 0.7, "rung": "L3",
              "urgent": False, "abstained": False, "recommended_action": "blur",
              "source": "screen", "app": {"exe": "WhatsApp.exe", "category": "chat"}}
    record.update(extra)
    return {k: v for k, v in record.items() if v is not None}


def by_date(daily):
    return daily.set_index("date")


@pytest.fixture(scope="module")
def sample_daily():
    c1 = c1_events(read_jsonl(C1_SAMPLE), child_id="C-SAMPLE")
    c2 = c2_events(read_jsonl(C2_SAMPLE), child_id="C-SAMPLE")
    return build_daily_features(c1, c2)


# ----------------------------------------------------------- real samples

def test_sample_files_shape_and_columns(sample_daily):
    assert list(sample_daily.columns) == DAILY_COLUMNS
    assert list(sample_daily["date"]) == [f"2026-10-{d}" for d in range(13, 20)]
    assert set(sample_daily["child_id_source"]) == {"explicit"}
    assert not sample_daily["demo_only"].any()


def test_sample_c2_counts_in_colombo_days(sample_daily):
    d = by_date(sample_daily)
    assert d["c2_flags"].to_dict() == {"2026-10-13": 2, "2026-10-14": 3, "2026-10-15": 4,
                                       "2026-10-16": 5, "2026-10-17": 3, "2026-10-18": 1,
                                       "2026-10-19": 5}
    assert sample_daily["c2_flags"].sum() == 23
    assert sample_daily["c2_urgent"].sum() == 9          # C2 README: 9 urgent
    assert sample_daily["c2_grooming"].sum() == 7
    assert sample_daily["c2_audio"].sum() == 6 and sample_daily["c2_screen"].sum() == 17
    # 16:35 UTC = 22:05 Colombo: the two self-harm rows on 16 Oct are night-time.
    assert d.loc["2026-10-16", "c2_night_flags"] == 2
    assert d.loc["2026-10-16", "c2_self_harm"] == 2
    assert sample_daily["c2_night_flags"].sum() == 2


def test_sample_c1_incidents(sample_daily):
    d = by_date(sample_daily).loc["2026-10-19"]
    assert d["c1_observed"]
    assert d["c1_incidents"] == 15 and d["c1_violence"] == 15
    assert (d["c1_blocks"], d["c1_blurs"]) == (8, 7)
    assert (d["c1_escalated"], d["c1_dismissed"]) == (4, 3)
    assert d["c1_frames_flagged"] == 55 and d["c1_total_duration_s"] == 167
    assert d["c1_max_conf"] == 0.97 and d["c1_night_incidents"] == 0
    assert d["total_events"] == 15 + 5


def test_sample_c1_unobserved_days_are_nan_not_zero(sample_daily):
    before = sample_daily[sample_daily["date"] < "2026-10-19"]
    assert not before["c1_observed"].any()
    assert before["c1_incidents"].isna().all() and before["total_events"].isna().all()
    assert before["c2_observed"].all()


def test_explicit_coverage_turns_empty_days_into_zero():
    c1 = c1_events(read_jsonl(C1_SAMPLE), child_id="C-SAMPLE")
    daily = build_daily_features(c1, c1_coverage=("2026-10-17", "2026-10-19"),
                                 start="2026-10-16", end="2026-10-19")
    d = by_date(daily)
    assert math.isnan(d.loc["2026-10-16", "c1_incidents"])
    assert d.loc["2026-10-17", "c1_incidents"] == 0 and d.loc["2026-10-17", "c1_max_conf"] == 0
    assert d.loc["2026-10-19", "c1_incidents"] == 15
    assert not daily["c2_observed"].any() and daily["c2_flags"].isna().all()


def test_sanitized_mock_has_no_text_or_title_hash():
    text = C2_SAMPLE.read_text(encoding="utf-8")
    for field in ("evidence", "redacted_snippet", "explanation", "child_safe_summary",
                  "title_hash", "fingerprint", "persona"):
        assert field not in text
    assert all(set(r) <= {"id", "ts", "top_category", "p", "rung", "urgent", "abstained",
                          "recommended_action", "source", "app"} for r in read_jsonl(C2_SAMPLE))


# ------------------------------------------------------------- child ids

def test_child_id_is_required():
    with pytest.raises(ValueError, match="Cannot resolve child_id"):
        c1_events([incident()])
    with pytest.raises(ValueError, match="Cannot resolve child_id"):
        c2_events([verdict()])


def test_demo_fallback_is_marked():
    daily = build_daily_features(c2=c2_events([verdict()], demo_single_child=True))
    assert daily.loc[0, "child_id"] == DEMO_CHILD_ID
    assert daily.loc[0, "child_id_source"] == "demo_placeholder" and daily.loc[0, "demo_only"]


def test_demo_child_id_matches_integration_placeholder():
    source = (REPO / "integration" / "comp3_to_comp4.py").read_text(encoding="utf-8")
    assert re.search(r'DEVICE_CHILD_ID\s*=\s*"([^"]+)"', source).group(1) == DEMO_CHILD_ID


def test_record_child_id_and_device_map():
    events = c2_events([verdict("a", child_id="C007"), verdict("b", device_id="laptop-1")],
                       device_map={"laptop-1": "C008"}, child_id="C999")
    assert dict(zip(events["event_key"], events["child_id"])) == {"a": "C007", "b": "C008"}
    assert set(events["child_id_source"]) == {"record", "device_map"}
    with pytest.raises(ValueError, match="not in device_map"):
        c2_events([verdict(device_id="laptop-2")], device_map={"laptop-1": "C008"})


# ------------------------------------------------------------ timestamps

@pytest.mark.parametrize("bad", ["2026-10-19T16:00:00", "yesterday", "", None])
def test_invalid_or_naive_timestamps_raise(bad):
    with pytest.raises(ValueError, match="timestamp"):
        c1_events([incident(started=bad)], child_id="C1")
    with pytest.raises(ValueError, match=r"timestamp|missing required fields \['ts'\]"):
        c2_events([verdict(ts=bad)], child_id="C1")


def test_utc_and_colombo_offsets_give_same_local_day():
    assert parse_timestamp("2026-10-18T19:00:00Z", "t").date().isoformat() == "2026-10-19"
    assert parse_timestamp("2026-10-19T00:30:00+05:30", "t") == parse_timestamp(
        "2026-10-18T19:00:00+00:00", "t")


@pytest.mark.parametrize("local, night", [("21:59:59", False), ("22:00:00", True),
                                          ("05:59:59", True), ("06:00:00", False)])
def test_night_boundaries(local, night):
    events = c2_events([verdict(ts=f"2026-10-19T{local}+05:30")], child_id="C1")
    assert bool(events.loc[0, "night"]) is night


def test_incident_crossing_midnight_counts_on_start_day():
    daily = build_daily_features(c1_events(
        [incident(started="2026-10-19T23:59:50+05:30", duration_s=30.0)], child_id="C1"))
    assert list(daily["date"]) == ["2026-10-19"]
    assert daily.loc[0, "c1_night_incidents"] == 1


# ------------------------------------------------------------- duplicates

def test_exact_duplicates_dropped_conflicts_raise():
    assert len(c1_events([incident(), incident()], child_id="C1")) == 1
    assert len(c2_events([verdict(), verdict()], child_id="C1")) == 1
    with pytest.raises(ValueError, match="different content"):
        c1_events([incident(), incident(max_confidence=0.5)], child_id="C1")
    with pytest.raises(ValueError, match="different content"):
        c2_events([verdict(), verdict(p=0.9)], child_id="C1")
    with pytest.raises(ValueError, match="different content"):
        c1_events([payload("s", "2026-10-19T10:00:00Z", 0, 0.8),
                   payload("s", "2026-10-19T10:00:00Z", 0, 0.9)], child_id="C1")


# ------------------------------------------------------ missing vs zero

def test_missing_c1_action_and_escalation_are_nan_not_zero():
    daily = build_daily_features(c1_events(
        [incident("a", action=None, escalated=None),
         incident("b", started="2026-10-20T16:00:00+05:30")], child_id="C1"))
    d = by_date(daily)
    assert math.isnan(d.loc["2026-10-19", "c1_blocks"])
    assert math.isnan(d.loc["2026-10-19", "c1_escalated"])
    assert d.loc["2026-10-19", "c1_dismissed"] == 0          # present and false
    assert d.loc["2026-10-19", "c1_incidents"] == 1
    assert d.loc["2026-10-20", "c1_blocks"] == 1


def test_missing_c2_app_and_source_are_nan():
    daily = build_daily_features(c2=c2_events([verdict(app=None, source=None)], child_id="C1"))
    assert math.isnan(daily.loc[0, "c2_distinct_apps"])
    assert math.isnan(daily.loc[0, "c2_screen"])
    assert daily.loc[0, "c2_flags"] == 1


# ------------------------------------------------------ C1 TriggerPayload

def test_trigger_payloads_split_at_duration_reset():
    frames = [payload("s1", "2026-10-19T10:00:02Z", 2000, 0.6),
              payload("s1", "2026-10-19T10:00:00Z", 0, 0.7),
              payload("s1", "2026-10-19T10:00:01Z", 1000, 0.9),
              payload("s1", "2026-10-19T10:05:00Z", 0, 0.8),
              payload("s1", "2026-10-19T10:05:01Z", 1000, 0.8)]
    events = c1_events(frames, child_id="C1")
    assert len(events) == 2 and events["certain"].all()
    assert sorted(events["frames"]) == [2, 3]
    assert sorted(events["duration_s"]) == [1.0, 2.0]
    daily = build_daily_features(events)
    assert daily.loc[0, "c1_incidents"] == 2 and daily.loc[0, "c1_max_conf"] == 0.9
    assert math.isnan(daily.loc[0, "c1_blocks"]) and math.isnan(daily.loc[0, "c1_escalated"])


def test_payload_start_backdated_by_duration_visible():
    events = c1_events([payload("s1", "2026-10-19T16:30:05Z", 5000)], child_id="C1")
    # 16:30:00 UTC = 22:00:00 Colombo: started at night, though the frame came 5 s later.
    assert events.loc[0, "started_at"].strftime("%H:%M:%S") == "22:00:00"
    assert bool(events.loc[0, "night"])


def test_payloads_without_duration_are_uncertain():
    frames = [payload("s1", "2026-10-19T10:00:00Z"), payload("s1", "2026-10-19T10:00:01Z", 1000)]
    daily = build_daily_features(c1_events(frames, child_id="C1"))
    assert daily.loc[0, "c1_uncertain_groups"] == 1
    assert math.isnan(daily.loc[0, "c1_incidents"]) and math.isnan(daily.loc[0, "c1_violence"])
    assert daily.loc[0, "c1_frames_flagged"] == 2


def test_unknown_c1_shape_raises():
    with pytest.raises(ValueError, match="neither an incident record"):
        c1_events([{"foo": 1}], child_id="C1")


# ------------------------------------------------------------- C2 rules

def test_c2_l0_and_abstained_are_records_not_flags():
    daily = build_daily_features(c2=c2_events(
        [verdict("a", rung="L0", recommended_action="none"),
         verdict("b", abstained=True), verdict("c")], child_id="C1"))
    assert daily.loc[0, "c2_records"] == 3 and daily.loc[0, "c2_flags"] == 1


def test_c2_invalid_values_raise():
    with pytest.raises(ValueError, match="does not match rung"):
        c2_events([verdict(recommended_action="block")], child_id="C1")
    with pytest.raises(ValueError, match="unknown top_category"):
        c2_events([verdict(top_category="spam")], child_id="C1")
    with pytest.raises(ValueError, match="missing required"):
        c2_events([verdict(urgent=None)], child_id="C1")
    with pytest.raises(ValueError, match="must be in"):
        c2_events([verdict(p=1.5)], child_id="C1")


def test_read_jsonl_reports_bad_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(verdict()) + "\n\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        read_jsonl(path)
