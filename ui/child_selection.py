"""Session-scoped assessment context for the parent dashboard."""

from collections import Counter
from contextlib import closing

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from db.child_profiles import AVATARS, ChildProfileRepository, ProfileStorageError
from ui import theme


SELECTED_PROFILE = "dashboard_selected_profile_id"
PROFILE_WIDGET = "dashboard_profile_widget"
DEMO_WIDGET = "dashboard_demo_mode"
_PROFILE_SNAPSHOT = "dashboard_profile_snapshot"
_HAS_ACTIVE_PROFILES = "dashboard_has_active_profiles"
F1_DEMO_CHILD = "f1_demo_child"
F2_DEMO_CHILD = "f2_child"


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
    for key in (F1_DEMO_CHILD, F2_DEMO_CHILD):
        st.session_state.pop(key, None)
    clear_behavioral_dates()


def demo_mode_active():
    return st.session_state.get(DEMO_WIDGET, False)


def has_active_profiles():
    """Return None if profile storage could not be read."""
    return st.session_state.get(_HAS_ACTIVE_PROFILES)


def selected_profile_label(*, include_avatar=False):
    if demo_mode_active():
        return None
    profile = st.session_state.get(_PROFILE_SNAPSHOT)
    if profile is None:
        return None
    avatar = AVATARS.get(profile.avatar, "").split(" ", 1)[0] if include_avatar else ""
    return f"{avatar} {profile.display_name}" if avatar else profile.display_name


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
