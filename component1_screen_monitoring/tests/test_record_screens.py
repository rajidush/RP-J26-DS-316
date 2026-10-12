"""Tests for src/record_screens.py with a fake capture (no screen access)."""
import csv
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.capture import RawFrame
from component1_screen_monitoring.src.record_screens import record


def _capture(monitor_index=1):
    return RawFrame(image=Image.new("RGB", (64, 40), "blue"), timestamp="2026-10-10T10:00:00+00:00")


def test_record_saves_frames_and_logs(tmp_path):
    n = record("violence", 0, 3, "browser1", 1, countdown=0, out_dir=tmp_path / "rec",
               csv_path=tmp_path / "rec.csv", capture_fn=_capture, sleep_fn=lambda s: None)
    assert n == 3
    assert sorted(p.name for p in (tmp_path / "rec" / "violence").iterdir()) == \
        ["browser1_0.png", "browser1_1.png", "browser1_2.png"]
    rows = list(csv.DictReader(open(tmp_path / "rec.csv")))
    assert [r["filename"] for r in rows] == [f"violence/browser1_{i}.png" for i in range(3)]
    assert {r["label"] for r in rows} == {"violence"}


def test_record_refuses_to_overwrite_a_session(tmp_path):
    kw = dict(countdown=0, out_dir=tmp_path, csv_path=tmp_path / "r.csv",
              capture_fn=_capture, sleep_fn=lambda s: None)
    record("safe", 0, 1, "s1", 1, **kw)
    with pytest.raises(FileExistsError):
        record("safe", 0, 1, "s1", 1, **kw)


@pytest.mark.parametrize("label, session", [("gore", "s1"), ("safe", "bad_name")])
def test_record_validates_label_and_session(tmp_path, label, session):
    with pytest.raises(ValueError):
        record(label, 0, 1, session, 1, countdown=0, out_dir=tmp_path,
               csv_path=tmp_path / "r.csv", capture_fn=_capture, sleep_fn=lambda s: None)
