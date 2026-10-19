"""
Shared look and feel for every page: colours, the pattern -> colour/face
mapping, the injected CSS, small layout helpers and the Plotly figures.
Pages import from here so they stay consistent; no data logic lives here.
"""
import base64
from dataclasses import dataclass
from html import escape

import plotly.graph_objects as go
import streamlit as st

from ui.services import PATTERN_ORDER

# --- Palette ------------------------------------------------------------------------

COLORS = {
    "page_bg": "#F5F8FD",
    "sidebar_bg": "#E9F0FB",
    "card_bg": "#FFFFFF",
    "border": "#E3EAF5",
    "navy": "#0F1F4B",
    "text": "#2B3A5C",
    "muted": "#5B6B8C",
    "blue": "#2F6FDE",
    "active_bg": "#DCE7FA",
    "grid": "#E6ECF5",
    "green": "#5DB55B",
    "blue_pattern": "#3B7BEA",
    "red": "#EF4F5E",
    "amber": "#F7B52C",
}
FONT_FAMILY = "Inter, -apple-system, 'Segoe UI', sans-serif"


@dataclass(frozen=True)
class PatternStyle:
    label: str        # short name used on the chart axis and legend
    color: str        # line / marker / face colour
    background: str   # badge fill
    text: str         # badge text colour
    face: str         # smile | neutral | frown | sad


PATTERN_STYLES = {
    "Calm / reflective": PatternStyle("Calm", COLORS["green"], "#E5F4E4", "#1F5A1E", "smile"),
    "Withdrawn / unclear": PatternStyle("Withdrawn", COLORS["blue_pattern"], "#E3EDFD", "#163C85", "neutral"),
    "Defensive": PatternStyle("Defensive", COLORS["red"], "#FDE3E5", "#6E0F17", "frown"),
    "Distressed": PatternStyle("Distressed", COLORS["amber"], "#FEF2D6", "#6B4800", "sad"),
}
UNKNOWN_STYLE = PatternStyle("Unknown", "#8A97B1", "#EEF1F6", COLORS["navy"], "neutral")
# Shown instead of a pattern when a child has too few sessions.
NOT_ENOUGH_DATA_STYLE = PatternStyle("Not enough data yet", "#9AA6BD", "#F0F3F8", "#3D4A66",
                                     "neutral")

# Chart colours for the Component 3 enums, matching the pattern colours.
EMOTION_COLORS = {"calm": COLORS["green"], "unclear": COLORS["blue_pattern"],
                  "defensive": COLORS["red"], "distressed": COLORS["amber"]}
RISK_COLORS = {"low": COLORS["green"], "moderate": COLORS["amber"], "high": COLORS["red"]}


def plural(count, word, plural_word=None):
    """'1 session', '2 sessions', '0 sessions'."""
    return f"{count} {word if count == 1 else (plural_word or word + 's')}"


def pattern_style(pattern):
    """Style for a Function 1 pattern name; UNKNOWN_STYLE for anything else."""
    return PATTERN_STYLES.get(pattern, UNKNOWN_STYLE)


# --- Faces (inline SVG) ----------------------------------------------------------------

_MOUTHS = {
    "smile": '<path d="M19 33 Q28 42 37 33" />',
    "neutral": '<path d="M20 36 L36 36" />',
    "frown": '<path d="M19 39 Q28 30 37 39" />',
    "sad": '<path d="M20 40 Q28 33 36 40" />',
}
_EXTRAS = {
    # worried eyebrows + a tear for "sad", to tell it apart from "frown"
    "sad": '<path d="M15 17 L23 20 M41 17 L33 20" /><path d="M39 27 q2.5 4 0 6 q-2.5 -2 0 -6z" '
           'fill="#fff" stroke="none" />',
}


def svg_img(svg, alt=""):
    """Inline SVG as an <img> data URI (st.html strips raw <svg> elements)."""
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{data}" alt="{escape(alt)}" />'


def face_svg(face, color, size=92):
    mouth = _MOUTHS.get(face, _MOUTHS["neutral"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 56 56">'
        f'<circle cx="28" cy="28" r="28" fill="{color}" />'
        f'<circle cx="20" cy="23" r="3.2" fill="#fff" /><circle cx="36" cy="23" r="3.2" fill="#fff" />'
        f'<g fill="none" stroke="#fff" stroke-width="3.2" stroke-linecap="round">'
        f'{mouth}{_EXTRAS.get(face, "")}</g></svg>'
    )


# --- CSS ------------------------------------------------------------------------------

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, [class*="css"] {{ font-family: {FONT_FAMILY}; }}
.stApp {{ background: {COLORS["page_bg"]}; color: {COLORS["text"]}; }}

/* Hide Streamlit chrome; keep the header itself so the sidebar toggle still works. */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stAppDeployButton"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}

[data-testid="stMainBlockContainer"], .block-container {{ padding: 2.2rem 2.5rem 3rem; max-width: 1240px; margin-left: 0; }}
h1, h2, h3 {{ color: {COLORS["navy"]}; letter-spacing: -0.01em; }}

/* Page header */
.c4-title {{ color: {COLORS["navy"]}; font-size: 2.4rem; font-weight: 800; margin: 0; line-height: 1.2; }}
.c4-subtitle {{ color: {COLORS["muted"]}; font-size: 1.2rem; margin: .4rem 0 1.4rem; }}

/* Cards: any container created with key="card-..." */
[class*="st-key-card-"] {{
    background: {COLORS["card_bg"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 12px;
    box-shadow: 0 1px 2px rgba(15, 31, 75, .04), 0 6px 18px rgba(15, 31, 75, .05);
    padding: 1.4rem 2rem 1.5rem;
    margin-bottom: .6rem;
}}
.c4-card-title {{ color: {COLORS["navy"]}; font-size: 1.35rem; font-weight: 700; margin: 0 0 .2rem; }}
.c4-card-text {{ color: {COLORS["muted"]}; margin: 0; }}

/* Error cards: containers created with key="card-error-..." */
[class*="st-key-card-error-"] {{
    background: #FFF5F5; border: 1px solid #F3B7BD; border-left: 6px solid #D93A49;
}}
.c4-error-title {{ color: #A3192A; font-size: 1.3rem; font-weight: 700; margin: 0 0 .5rem; }}
.c4-error-label {{ color: #6E0F17; font-weight: 700; margin: .9rem 0 .2rem; }}
.c4-error-text {{ color: #4A1D24; margin: 0; }}
.c4-error-text code {{ background: #FCE4E7; padding: .05rem .3rem; border-radius: 4px; }}
.c4-error-list {{ color: #4A1D24; margin: .2rem 0 0 1.1rem; padding: 0; }}

/* Sidebar model status line */
.c4-status {{ display: flex; align-items: center; gap: .5rem; font-size: .85rem;
             margin: 1.4rem .6rem 0; color: {COLORS["text"]}; }}
.c4-status-dot {{ width: .6rem; height: .6rem; border-radius: 50%; flex: none; }}

/* Pattern / status badge */
.c4-badge {{
    display: flex; align-items: center; gap: 2.4rem;
    max-width: 640px; padding: 1rem 2.6rem; border-radius: 10px; border: 1px solid;
}}
.c4-badge-label {{ font-size: 2.3rem; font-weight: 800; line-height: 1.1; }}
.c4-badge-sub {{ font-size: .95rem; opacity: .85; margin-top: .25rem; }}

.c4-note {{ color: {COLORS["muted"]}; font-size: .85rem; margin: .4rem 0 0; }}
.c4-tag {{
    display: inline-block; font-size: .78rem; font-weight: 600; padding: .15rem .6rem;
    border-radius: 999px; background: {COLORS["active_bg"]}; color: {COLORS["blue"]};
}}

/* Inputs: visible border inside white cards, like the mockup's dropdown */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="textarea"],
[data-testid="stNumberInputContainer"] {{
    border: 1px solid #D3DCEA !important; border-radius: 8px !important; background: #fff !important;
}}
.st-key-card-select-child [data-baseweb="select"] > div {{ min-height: 3.2rem; font-size: 1.25rem; color: {COLORS["navy"]}; }}

/* Metric tiles */
[data-testid="stMetric"] {{
    background: {COLORS["card_bg"]}; border: 1px solid {COLORS["border"]}; border-radius: 12px;
    padding: 1rem 1.25rem; box-shadow: 0 6px 18px rgba(15, 31, 75, .05);
}}
[data-testid="stMetricValue"] {{ color: {COLORS["navy"]}; font-weight: 700; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: {COLORS["sidebar_bg"]}; border-right: 1px solid {COLORS["border"]}; }}
.c4-brand {{
    display: flex; align-items: center; gap: .9rem; padding: .2rem .4rem 1.3rem;
    margin-bottom: 1rem; border-bottom: 1px solid {COLORS["border"]};
}}
.c4-brand-name {{ color: {COLORS["navy"]}; font-weight: 700; font-size: 1.15rem; line-height: 1.25; }}
.c4-nav-section {{
    color: {COLORS["muted"]}; font-size: .72rem; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; margin: 1.6rem 0 .3rem .6rem;
}}
[class*="st-key-nav-"] a {{ border-radius: 10px; padding: .6rem .8rem; }}
[class*="st-key-nav-"] a p {{ color: {COLORS["text"]}; font-size: 1.08rem; }}
[class*="st-key-nav-"] a span {{ color: {COLORS["text"]}; }}
[class*="st-key-nav-"] [data-testid="stIconMaterial"] {{ font-size: 1.6rem; margin-right: .35rem; }}
[class*="st-key-navsm-"] [data-testid="stIconMaterial"] {{ font-size: 1.15rem; }}
[class*="st-key-navsm-"] a {{ border-radius: 8px; padding: .3rem .8rem; }}
[class*="st-key-navsm-"] a p, [class*="st-key-navsm-"] a span {{ color: {COLORS["muted"]}; font-size: .9rem; }}
"""

_PEOPLE_ICON = (
    f'<svg xmlns="http://www.w3.org/2000/svg" width="50" height="46" viewBox="0 0 50 46" fill="{COLORS["blue"]}">'
    '<circle cx="13" cy="9" r="7"/><circle cx="37" cy="9" r="7"/><circle cx="25" cy="22" r="7.5"/>'
    '<path d="M1 30c0-7 5-11 12-11 3 0 5.6.8 7.4 2.2C16 23.6 13.6 27.8 13.6 33H1z"/>'
    '<path d="M49 30c0-7-5-11-12-11-3 0-5.6.8-7.4 2.2C34 23.6 36.4 27.8 36.4 33H49z"/>'
    '<path d="M11 46c0-8 6-14 14-14s14 6 14 14z"/></svg>'
)


def apply_style():
    """Inject the shared CSS. Called once per run from app.py."""
    st.html(f"<style>{CSS}</style>")


def active_nav_css(key):
    """Highlight the sidebar link wrapped in container `key` (the current page)."""
    st.html(f"""<style>
    .st-key-{key} a {{ background: {COLORS["active_bg"]}; }}
    .st-key-{key} a p, .st-key-{key} a span {{ color: {COLORS["blue"]} !important; font-weight: 600; }}
    </style>""")


def sidebar_brand(name_lines=("Child Behavior", "Monitoring")):
    st.sidebar.html(
        f'<div class="c4-brand">{svg_img(_PEOPLE_ICON)}<div class="c4-brand-name">'
        f'{"<br>".join(escape(line) for line in name_lines)}</div></div>'
    )


def sidebar_model_status(name, loaded):
    """Sidebar line like "Function 1 model: loaded" with a green or red dot."""
    color, word = ("#2E9E44", "loaded") if loaded else ("#D93A49", "not loaded")
    st.sidebar.html(
        f'<div class="c4-status"><span class="c4-status-dot" style="background:{color}"></span>'
        f'<span>{escape(name)}: <b style="color:{color}">{word}</b></span></div>')


def sidebar_section(title):
    st.sidebar.html(f'<div class="c4-nav-section">{escape(title)}</div>')


# --- Layout helpers -------------------------------------------------------------------

def page_header(title, subtitle=None):
    html = f'<h1 class="c4-title">{escape(title)}</h1>'
    if subtitle:
        html += f'<p class="c4-subtitle">{escape(subtitle)}</p>'
    st.html(html)


def card(key, title=None, text=None):
    """A white rounded card. Use as `with card("name", "Title"):`."""
    box = st.container(key=f"card-{key}")
    if title or text:
        html = f'<p class="c4-card-title">{escape(title)}</p>' if title else ""
        if text:
            html += f'<p class="c4-card-text">{escape(text)}</p>'
        box.html(html)
    return box


def _code(text):
    """Escape text, rendering `backticked` parts as <code>."""
    parts = escape(text).split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def error_card(key, title, reason, fixes):
    """Red card for a failure that blocks a result (e.g. a model that won't load).
    reason: one line; fixes: list of steps (`backticks` render as code)."""
    with st.container(key=f"card-error-{key}"):
        steps = "".join(f"<li>{_code(f)}</li>" for f in fixes)
        st.html(f'<p class="c4-error-title">{escape(title)}</p>'
                f'<p class="c4-error-label">Reason</p><p class="c4-error-text">{_code(reason)}</p>'
                f'<p class="c4-error-label">How to fix</p><ul class="c4-error-list">{steps}</ul>')


def note(text):
    st.html(f'<p class="c4-note">{escape(text)}</p>')


def badge(style, subtitle=None):
    """Coloured badge with a face on the left and the label in large bold text."""
    sub = f'<div class="c4-badge-sub">{escape(subtitle)}</div>' if subtitle else ""
    st.html(
        f'<div class="c4-badge" style="background:{style.background};border-color:{style.color}33;'
        f'color:{style.text}">{svg_img(face_svg(style.face, style.color), alt=style.face)}'
        f'<div><div class="c4-badge-label">{escape(style.label)}</div>{sub}</div></div>'
    )


# --- Plotly figures -------------------------------------------------------------------

def _base_layout(fig, height):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=COLORS["text"], size=14),
        hoverlabel=dict(font_family=FONT_FAMILY),
    )
    return fig


def pattern_timeline_figure(timeline):
    """Line chart of Function 1 patterns over the trend windows.

    timeline: [{"date": "YYYY-MM-DD", "pattern": <PATTERN_ORDER name>, "sessions": n}, ...].
    y is categorical, bottom-to-top in PATTERN_ORDER; each segment takes the
    colour of the pattern it moves into; the legend lists all four patterns.
    """
    order = {p: i for i, p in enumerate(PATTERN_ORDER)}
    points = [t for t in timeline if t["pattern"] in order]
    fig = go.Figure()

    for prev, cur in zip(points, points[1:]):
        fig.add_trace(go.Scatter(
            x=[prev["date"], cur["date"]], y=[order[prev["pattern"]], order[cur["pattern"]]],
            mode="lines", line=dict(color=pattern_style(cur["pattern"]).color, width=3),
            hoverinfo="skip", showlegend=False))

    for pattern in PATTERN_ORDER:
        style = pattern_style(pattern)
        mine = [t for t in points if t["pattern"] == pattern]
        fig.add_trace(go.Scatter(
            x=[t["date"] for t in mine] or [None], y=[order[pattern]] * len(mine) or [None],
            mode="markers", name=style.label,
            marker=dict(color=style.color, size=12, line=dict(color="#fff", width=1.5)),
            customdata=[[pattern, plural(t["sessions"], "session")] for t in mine] or None,
            hovertemplate="<b>%{customdata[0]}</b><br>2 weeks from %{x|%Y-%m-%d}"
                          "<br>%{customdata[1]}<extra></extra>"))

    axis = dict(showgrid=True, gridcolor=COLORS["grid"], griddash="dash", zeroline=False,
                linecolor="#C9D3E3", showline=True, ticks="")
    fig.update_xaxes(**axis, type="date", tickformat="%Y-%m-%d", tickfont=dict(size=15))
    fig.update_yaxes(**axis, tickmode="array", tickvals=list(range(len(PATTERN_ORDER))),
                     ticktext=[pattern_style(p).label for p in PATTERN_ORDER],
                     range=[-0.4, len(PATTERN_ORDER) - 0.6], fixedrange=True,
                     tickfont=dict(size=16, color=COLORS["text"]))
    fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18,
                                  yanchor="top", font=dict(size=15), itemclick=False,
                                  itemdoubleclick=False))
    return _base_layout(fig, height=420)


def bar_figure(counts, colors, height=300):
    """Vertical bar chart of {category: count}, one colour per category."""
    labels = list(counts)
    fig = go.Figure(go.Bar(
        x=labels, y=[counts[k] for k in labels],
        marker=dict(color=[colors.get(k, COLORS["blue"]) for k in labels],
                    cornerradius=6),
        hovertemplate="%{x}: %{y}<extra></extra>"))
    fig.update_xaxes(showgrid=False, linecolor="#C9D3E3")
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"], griddash="dash", zeroline=False,
                     rangemode="tozero", tickformat="d",
                     dtick=1 if max(counts.values(), default=0) <= 10 else None)
    return _base_layout(fig, height)


PLOTLY_CONFIG = {"displayModeBar": False}
