"""Session-scoped assessment context for the parent dashboard."""

from collections import Counter
from contextlib import closing
from hashlib import sha256
from uuid import uuid4

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from db.child_profiles import AVATARS, ChildProfileRepository, ProfileStorageError
from ui import activity_summary, safety_summary, services, theme


SELECTED_PROFILE = "dashboard_selected_profile_id"
PROFILE_WIDGET = "dashboard_profile_widget"
DEMO_WIDGET = "dashboard_demo_mode"
_PROFILE_SNAPSHOT = "dashboard_profile_snapshot"
_HAS_ACTIVE_PROFILES = "dashboard_has_active_profiles"
_PROFILE_UI_KEYS = "dashboard_profile_ui_keys"
F1_DEMO_CHILD = "f1_demo_child"
F2_DEMO_CHILD = "f2_child"
OVERVIEW_DEMO_RESULTS = "overview_demo_results"
_F2_DEMO_CONTEXT = "dashboard_f2_demo_context"
F2_SAFETY_DATE = "dashboard_f2_safety_date"
_SAFETY_RESULT = "dashboard_safety_result"
ACTIVITY_PERIOD = "dashboard_activity_period"
_ACTIVITY_RESULT = "dashboard_activity_result"


def _profile_labels(profiles):
    names = Counter(profile.display_name for profile in profiles)
    occurrences = Counter()
    labels = {}
    for profile in profiles:
        name = profile.display_name
        occurrences[name] += 1
        avatar = AVATARS.get(profile.avatar, "").split(" ", 1)[0]
        label = f"{avatar} {name}" if avatar else name
        if names[name] > 1:
            label += f" · Profile {occurrences[name]}"
        labels[profile.id] = label
    # A display name can itself contain a suffix used for disambiguation.
    if len(set(labels.values())) != len(labels):
        labels = {key: f"{index}. {label}" for index, (key, label) in enumerate(labels.items(), 1)}
    return labels


def _select_profile():
    st.session_state[SELECTED_PROFILE] = st.session_state[PROFILE_WIDGET]


def clear_behavioral_dates():
    for key in list(st.session_state.keys()):
        if key.startswith("f2_day_"):
            del st.session_state[key]


def _change_mode():
    for key in (F1_DEMO_CHILD, F2_DEMO_CHILD, OVERVIEW_DEMO_RESULTS, _F2_DEMO_CONTEXT,
                F2_SAFETY_DATE, _SAFETY_RESULT, ACTIVITY_PERIOD, _ACTIVITY_RESULT):
        st.session_state.pop(key, None)
    clear_behavioral_dates()


def demo_mode_active():
    return st.session_state.get(DEMO_WIDGET, False)


def behavioral_demo_picker(label, ids, key, *, index=None, **kwargs):
    """Retain the Function 02 identity across page-specific widgets."""
    def select(child):
        st.session_state[_F2_DEMO_CONTEXT] = child
        st.session_state.pop(F2_SAFETY_DATE, None)
        st.session_state.pop(_SAFETY_RESULT, None)
        st.session_state.pop(_ACTIVITY_RESULT, None)
        clear_behavioral_dates()
        st.session_state.get(OVERVIEW_DEMO_RESULTS, {}).pop("behavioral", None)

    child = st.session_state.get(_F2_DEMO_CONTEXT)
    if child not in ids:
        default = None if index is None or not ids else ids[index]
        if child != default:
            select(default)
        child = default
    if st.session_state.get(key) != child:
        st.session_state[key] = child

    return st.selectbox(label, ids, index=None, key=key,
                        on_change=lambda: select(st.session_state[key]), **kwargs)


def behavioral_date_picker(label, dates, key):
    stored_date = st.session_state.get(F2_SAFETY_DATE)
    default_date = stored_date if stored_date in dates else dates[0]
    through = st.selectbox(label, dates, index=dates.index(default_date), key=key, format_func=str)
    st.session_state[F2_SAFETY_DATE] = through
    return through


def _assessment_date(rows):
    through = st.session_state.get(F2_SAFETY_DATE)
    return through if through in set(rows["date"]) else rows["date"].max()


def activity_demo_summary(source, child):
    rows = safety_summary.records_for(source, child)
    if rows.empty:
        return None
    period = st.session_state.get(ACTIVITY_PERIOD, activity_summary.DEFAULT_PERIOD)
    days = activity_summary.PERIODS[period]
    through = _assessment_date(rows)
    cached = st.session_state.get(_ACTIVITY_RESULT)
    if cached and cached[:2] == (through, days) and cached[2].equals(rows):
        return cached[3]
    result = activity_summary.summarize(rows, through, days)
    st.session_state[_ACTIVITY_RESULT] = (through, days, rows.copy(), result)
    return result


def artifact_revisions(directory, names):
    versions = []
    for name in names:
        path = (directory / name).absolute()
        try:
            versions.append((path, sha256(path.read_bytes()).digest()))
        except OSError:
            versions.append((path, None))
    return tuple(versions)


def safety_demo_summary(source, child, through=None):
    rows = safety_summary.records_for(source, child)
    if rows.empty:
        return safety_summary.SafetySummary()
    through = through or _assessment_date(rows)
    rows = rows[rows["date"] <= through]
    context = (source["source"], child, through, demo_mode_active(),
               artifact_revisions(services.f2_artifact_dir(), services.F2_ARTIFACT_FILES),
               services.F2_CONTEXT_DAYS, services.F2_TOP_FEATURES)
    cached = st.session_state.get(_SAFETY_RESULT)
    if cached and cached[0] == context and cached[1].equals(rows):
        return cached[2]
    result = safety_summary.summarize(rows)
    if (result.behavioral_message != safety_summary.BEHAVIOR_FAILED
            and result.urgent_message != safety_summary.URGENT_FAILED):
        st.session_state[_SAFETY_RESULT] = (context, rows.copy(), result)
    else:
        st.session_state.pop(_SAFETY_RESULT, None)
    return result


def has_active_profiles():
    """Return None if profile storage could not be read."""
    return st.session_state.get(_HAS_ACTIVE_PROFILES)


def selected_profile():
    """Return the revalidated local profile only in local mode."""
    if demo_mode_active():
        return None
    return st.session_state.get(_PROFILE_SNAPSHOT)


def selected_profile_label(*, include_avatar=False):
    profile = selected_profile()
    if profile is None:
        return None
    avatar = AVATARS.get(profile.avatar, "").split(" ", 1)[0] if include_avatar else ""
    return f"{avatar} {profile.display_name}" if avatar else profile.display_name


def profile_ui_key(profile_id):
    keys = st.session_state.setdefault(_PROFILE_UI_KEYS, {})
    if profile_id not in keys:
        keys[profile_id] = uuid4().hex
    return keys[profile_id]


def render_profile_selector():
    with st.sidebar:
        try:
            with closing(ChildProfileRepository()) as repository:
                profiles = repository.list_profiles()
        except (OSError, ProfileStorageError, SQLAlchemyError):
            profiles = []
            st.session_state[_HAS_ACTIVE_PROFILES] = None
            st.warning("Local child profiles are temporarily unavailable.")
        else:
            st.session_state[_HAS_ACTIVE_PROFILES] = bool(profiles)
            if not profiles:
                st.caption("Add a child profile on the Child Profiles page.")

        labels = _profile_labels(profiles)
        selected = st.session_state.get(SELECTED_PROFILE)
        if not isinstance(selected, str) or selected not in labels:
            selected = None
        st.session_state[SELECTED_PROFILE] = selected

        if st.session_state.get(PROFILE_WIDGET) != selected:
            st.session_state[PROFILE_WIDGET] = selected
        st.selectbox(
            "Child profile", list(labels), index=None, placeholder="Select a child profile",
            key=PROFILE_WIDGET, format_func=labels.__getitem__,
            on_change=_select_profile, disabled=demo_mode_active() or not profiles,
        )
        st.session_state[_PROFILE_SNAPSHOT] = next(
            (profile for profile in profiles if profile.id == selected), None,
        )


def render_demo_control():
    with st.sidebar:
        if st.toggle("Demo Mode", key=DEMO_WIDGET, on_change=_change_mode):
            st.caption("Demo Mode uses fictional data. Each assessment has its own demo child selector.")


def show_assessment_context(title):
    """Stop before scoring unless the parent explicitly enabled Demo Mode."""
    if demo_mode_active():
        return

    theme.page_header(title)
    label = selected_profile_label()
    if label is None:
        st.info("Select a child profile in the sidebar, or turn on Demo Mode to explore fictional data.")
    else:
        with theme.card("selected-profile", label):
            st.info("Monitoring data has not been connected for this child.")
    st.stop()
