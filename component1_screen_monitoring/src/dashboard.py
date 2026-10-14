"""
dashboard.py — Local Streamlit demo dashboard for Component 1 (Function 1).

    streamlit run component1_screen_monitoring/src/dashboard.py

Live tab: samples the screen, runs the fine-tuned ViolenceDetector, and shows
status, p(violent) over time, and the schema-valid TriggerPayloads emitted.
Test tab: classify an uploaded image and show the payload it would produce.

Reuses capture_frame, ViolenceDetector.predict, build_payload and
validate_payload (via build_payload). Frames stay in memory only — nothing is
written to disk — and at most dashboard_state.MAX_READINGS readings are kept.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component1_screen_monitoring.src.capture import RawFrame, capture_frame
from component1_screen_monitoring.src.component1 import IncidentTracker, build_payload
from component1_screen_monitoring.src.config import CAPTURE_INTERVAL_S, REGION_STRATEGY, THRESHOLD
from component1_screen_monitoring.src.dashboard_state import (
    MAX_READINGS,
    ReadingHistory,
    make_thumbnail,
    process_frame,
)
from component1_screen_monitoring.src.detector import ViolenceDetector
from component1_screen_monitoring.src.regions import STRATEGIES

PAYLOADS_SHOWN = 20

st.set_page_config(page_title="Component 1 — Screen Monitoring", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------------------
# Resources + session state
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading fine-tuned violence detector …")
def load_detector() -> ViolenceDetector:
    return ViolenceDetector()


def list_monitors() -> dict[str, int]:
    """Not cached: the display list is empty while the screen is asleep/locked."""
    import mss

    with mss.MSS() as sct:
        mons = sct.monitors
    options = {f"Monitor {i} ({m['width']}×{m['height']})": i for i, m in enumerate(mons) if i > 0}
    if options:
        options[f"All monitors ({mons[0]['width']}×{mons[0]['height']})"] = 0
    return options


ss = st.session_state
ss.setdefault("running", False)
ss.setdefault("history", ReadingHistory())
ss.setdefault("tracker", IncidentTracker())
ss.setdefault("thumbnail", None)
ss.setdefault("prev", None)          # small grayscale copy of the previous frame (motion)
ss.setdefault("error", None)

detector = load_detector()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Live monitoring")
    start, stop = st.columns(2)
    if start.button("▶ Start", disabled=ss.running, use_container_width=True, type="primary"):
        ss.running, ss.error = True, None
    if stop.button("■ Stop", disabled=not ss.running, use_container_width=True):
        ss.running = False

    interval = st.number_input("Capture interval (s)", min_value=0.5, max_value=30.0,
                               value=float(CAPTURE_INTERVAL_S), step=0.5)
    threshold = st.slider("Threshold — flag when p(violent) ≥", 0.0, 1.0, float(THRESHOLD), 0.01)
    strategy = st.selectbox("Region strategy", STRATEGIES, index=STRATEGIES.index(REGION_STRATEGY),
                            help="Which crops of each frame are classified: full frame only, "
                                 "+ 2×2 tiles and centre, + the moving region, or both.")
    monitors = list_monitors()
    if monitors:
        monitor_index = monitors[st.selectbox("Monitor", list(monitors))]
    else:
        st.warning("No display found — is the screen asleep or locked? Wake it and rerun.")
        monitor_index = 1


    st.caption(f"Model on **{detector.device}** · keeps the last {MAX_READINGS} readings · "
               "frames are never written to disk")

detector.threshold = threshold          # one shared detector; slider sets its threshold


def _status_badge(flagged: bool) -> None:
    color, text = ("#d62728", "FLAGGED") if flagged else ("#2ca02c", "SAFE")
    st.markdown(f"<div style='font-size:3rem;font-weight:800;color:{color};"
                f"line-height:1.1'>{text}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

st.title("🛡️ Component 1 — Screen Monitoring")
live_tab, test_tab = st.tabs(["Live monitoring", "Test an image"])


@st.fragment(run_every=interval if ss.running else None)
def live_panel() -> None:
    history: ReadingHistory = ss.history

    if ss.running:
        try:
            reading, ss.thumbnail, ss.prev = process_frame(
                detector, lambda: capture_frame(monitor_index=monitor_index), ss.tracker,
                strategy, ss.prev)
            history.add(reading)
            ss.error = None
        except Exception as exc:          # show it, keep the app alive
            ss.error = f"{type(exc).__name__}: {exc}"

    if ss.error:
        st.error(ss.error)

    latest = history.latest
    if latest is None:
        st.info("Press **▶ Start** in the sidebar to begin monitoring.")
        return

    frame_col, info_col = st.columns([3, 2])
    with frame_col:
        st.image(ss.thumbnail, caption=f"{latest.frame_reference} · {latest.timestamp}"
                 + (" · blurred (flagged)" if latest.flagged else "")
                 + (f" · red box = highest-scoring region ({latest.region})"
                    if latest.region != "full" else ""))
        if ss.thumbnail is not None and ss.thumbnail.convert("L").getextrema() == (0, 0):
            st.warning("Captured frame is completely black — grant **Screen Recording** "
                       "permission to your terminal / VS Code (System Settings → Privacy & "
                       "Security), then restart Streamlit.")
    with info_col:
        _status_badge(latest.flagged)
        st.metric("p(violent)", f"{latest.p_violent:.4f}", help="max over all crops")
        st.caption(f"Highest-scoring region: **{latest.region}** · {latest.n_crops} crops")
        a, b = st.columns(2)
        a.metric("Inference", f"{latest.inference_ms:.0f} ms")
        b.metric("Total (capture + inference)", f"{latest.total_ms:.0f} ms")
        st.caption("● running" if ss.running else "■ stopped")

    st.subheader("p(violent) over time")
    chart = pd.DataFrame(history.chart_rows(threshold))
    st.line_chart(chart, x="seconds", y=["p_violent", "threshold"], color=["#1f77b4", "#d62728"],
                  x_label="seconds since start", y_label="p(violent)", height=260)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames processed", history.frames_processed)
    c2.metric("Payloads emitted", history.payloads_emitted)
    c3.metric("Mean latency", f"{history.mean_latency_ms:.0f} ms")
    c4.metric("Crops per frame", f"{history.mean_crops:.1f}")

    st.subheader("Payload log (newest first)")
    payloads = history.payloads_newest_first()
    if not payloads:
        st.caption("No payloads yet — a TriggerPayload is emitted only for flagged frames.")
    for i, p in enumerate(payloads[:PAYLOADS_SHOWN]):
        st.json(p, expanded=(i == 0))
    if len(payloads) > PAYLOADS_SHOWN:
        st.caption(f"… {len(payloads) - PAYLOADS_SHOWN} older payloads kept in memory, not shown.")


with live_tab:
    live_panel()


with test_tab:
    upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp"])
    if upload is not None:
        image = Image.open(upload)                      # in memory only
        detection = detector.predict(image)
        payload = build_payload(RawFrame(image=image, frame_reference=f"upload_{upload.name}"),
                                detection)

        img_col, info_col = st.columns([3, 2])
        with img_col:
            st.image(make_thumbnail(image, blur=detection.flagged),
                     caption=upload.name + (" · blurred (flagged)" if detection.flagged else ""))
        with info_col:
            _status_badge(detection.flagged)
            st.metric("p(violent)", f"{detection.confidence:.4f}")
            st.metric("Inference", f"{detection.inference_ms:.0f} ms")
        st.subheader("Payload that would be emitted")
        if payload is None:
            st.info("No payload — safe.")
        else:
            st.json(payload)
