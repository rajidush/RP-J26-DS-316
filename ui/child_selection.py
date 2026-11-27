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
OVERVIEW_DEMO_RESULTS = "overview_demo_results"
_F2_DEMO_CONTEXT = "dashboard_f2_demo_context"
F2_SAFETY_DATE = "dashboard_f2_safety_date"
_SAFETY_RESULT = "dashboard_safety_result"


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
                F2_SAFETY_DATE, _SAFETY_RESULT):
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


def safety_demo_summary(source, child, through=None):
    from ui import safety_summary, services

    rows = safety_summary.records_for(source, child)
    through = through or st.session_state.get(F2_SAFETY_DATE)
    if through is not None:
        rows = rows[rows["date"] <= through]
    artifacts = tuple(services.f2_artifact_dir() / name for name in services.F2_ARTIFACT_FILES)
    versions = []
    for path in artifacts:
        try:
            stat = path.stat()
            versions.append((path, stat.st_mtime_ns, stat.st_size))
        except OSError:
            versions.append((path, None, None))
    cached = st.session_state.get(_SAFETY_RESULT)
    if cached and cached[0] == versions and cached[1].equals(rows):
        return cached[2]
    result = safety_summary.summarize(rows)
    if result.behavioral_message != safety_summary.BEHAVIOR_FAILED:
        st.session_state[_SAFETY_RESULT] = (versions, rows.copy(), result)
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
