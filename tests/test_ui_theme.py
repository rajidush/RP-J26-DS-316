import plotly.graph_objects as go

from ui import services, theme


def test_every_pattern_has_the_specified_style():
    assert set(theme.PATTERN_STYLES) == set(services.PATTERN_ORDER)
    expected = {
        "Calm / reflective": ("Calm", theme.COLORS["green"], "smile"),
        "Withdrawn / unclear": ("Withdrawn", theme.COLORS["blue_pattern"], "neutral"),
        "Defensive": ("Defensive", theme.COLORS["red"], "frown"),
        "Distressed": ("Distressed", theme.COLORS["amber"], "sad"),
    }
    for pattern, (label, color, face) in expected.items():
        style = theme.pattern_style(pattern)
        assert (style.label, style.color, style.face) == (label, color, face)


def test_unknown_pattern_gets_fallback_style():
    assert theme.pattern_style("Not enough data") is theme.UNKNOWN_STYLE


def test_faces_are_distinct_svgs():
    svgs = {face: theme.face_svg(face, "#000") for face in ("smile", "neutral", "frown", "sad")}
    assert all(s.startswith("<svg") for s in svgs.values())
    assert len(set(svgs.values())) == 4


TIMELINE = [
    {"date": "2026-08-03", "pattern": "Calm / reflective", "sessions": 6},
    {"date": "2026-08-17", "pattern": "Calm / reflective", "sessions": 7},
    {"date": "2026-08-31", "pattern": "Withdrawn / unclear", "sessions": 5},
    {"date": "2026-09-14", "pattern": "Defensive", "sessions": 6},
]


def test_timeline_figure_axes_and_legend():
    fig = theme.pattern_timeline_figure(TIMELINE)
    assert isinstance(fig, go.Figure)
    assert list(fig.layout.yaxis.ticktext) == ["Calm", "Withdrawn", "Defensive", "Distressed"]
    assert list(fig.layout.yaxis.tickvals) == [0, 1, 2, 3]
    legend = [t.name for t in fig.data if t.mode == "markers"]
    assert legend == ["Calm", "Withdrawn", "Defensive", "Distressed"]  # even with no Distressed point
    assert fig.layout.legend.orientation == "h" and fig.layout.legend.xanchor == "center"
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert not fig.layout.title.text


def test_timeline_segments_take_the_colour_of_the_new_pattern():
    fig = theme.pattern_timeline_figure(TIMELINE)
    segments = [t for t in fig.data if t.mode == "lines"]
    assert len(segments) == len(TIMELINE) - 1
    assert [s.line.color for s in segments] == [
        theme.COLORS["green"], theme.COLORS["blue_pattern"], theme.COLORS["red"]]
    assert list(segments[2].y) == [1, 2]


def test_timeline_figure_ignores_non_patterns_and_handles_empty():
    fig = theme.pattern_timeline_figure([{"date": "2026-08-03", "pattern": "Not enough data",
                                          "sessions": 1}])
    assert not [t for t in fig.data if t.mode == "lines"]
    assert theme.pattern_timeline_figure([])


def test_bar_figure_colours_follow_categories():
    fig = theme.bar_figure({"low": 1, "moderate": 2, "high": 3}, theme.RISK_COLORS)
    assert list(fig.data[0].marker.color) == [theme.COLORS["green"], theme.COLORS["amber"],
                                              theme.COLORS["red"]]


def test_plural():
    assert theme.plural(0, "session") == "0 sessions"
    assert theme.plural(1, "session") == "1 session"
    assert theme.plural(2, "session") == "2 sessions"
    assert theme.plural(1, "child", "children") == "1 child"
    assert theme.plural(3, "child", "children") == "3 children"


def test_not_enough_data_style_is_neutral_grey():
    style = theme.NOT_ENOUGH_DATA_STYLE
    assert style.label == "Not enough data yet"
    assert style.face == "neutral"
    assert style not in theme.PATTERN_STYLES.values()


def test_timeline_hover_pluralises_sessions():
    fig = theme.pattern_timeline_figure([{"date": "2026-08-03", "pattern": "Defensive",
                                          "sessions": 1}])
    defensive = next(t for t in fig.data if t.name == "Defensive")
    assert defensive.customdata[0][1] == "1 session"


# --- Function 2 ------------------------------------------------------------------------

def _anomaly_days():
    import pandas as pd
    return pd.DataFrame({"date": pd.date_range("2026-07-01", periods=4).date,
                         "anomaly_score": [-13.0, -12.0, -12.9, -11.0],
                         "margin": [-0.5, 0.5, -0.4, 1.5],
                         "label": ["Normal", "Unusual", "Normal", "Unusual"]})


def test_anomaly_figure_has_threshold_line_and_marks_unusual_days():
    fig = theme.anomaly_history_figure(_anomaly_days(), threshold=-12.5)
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(lines) == 1 and lines[0].y0 == lines[0].y1 == -12.5
    by_name = {t.name: t for t in fig.data if t.name}
    assert set(by_name) == {"Normal day", "Unusual day"}
    assert list(by_name["Unusual day"].y) == [-12.0, -11.0]
    assert by_name["Unusual day"].marker.color == theme.ANOMALY_STYLES["Unusual"].color


def test_anomaly_figure_without_unusual_days():
    days = _anomaly_days().assign(label="Normal")
    fig = theme.anomaly_history_figure(days, threshold=-10.0)
    unusual = next(t for t in fig.data if t.name == "Unusual day")
    assert list(unusual.y) == [None]


def test_normal_status_is_not_shown_as_safe_green():
    assert set(theme.ANOMALY_STYLES) == {"Normal", "Unusual"}
    assert theme.ANOMALY_STYLES["Normal"].color != theme.COLORS["green"]
