from contextlib import closing
from uuid import uuid4

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from db.child_profiles import (
    AGE_GROUPS, AVATARS, ChildProfileRepository, ProfileError, ProfileStorageError,
)
from ui import theme


def profile_fields(profile=None):
    name = st.text_input("Display name", value=profile.display_name if profile else "")
    age_group = st.selectbox(
        "Age group (optional)", [None, *AGE_GROUPS],
        index=([None, *AGE_GROUPS].index(profile.age_group) if profile else 0),
        format_func=lambda value: value or "Prefer not to say",
    )
    avatar = st.selectbox(
        "Avatar (optional)", [None, *AVATARS],
        index=([None, *AVATARS].index(profile.avatar) if profile else 0),
        format_func=lambda value: AVATARS.get(value, "No avatar"),
    )
    return name, age_group, avatar


def save(action, success, *, new_profile=False):
    try:
        action()
    except ProfileError as exc:
        st.error(str(exc))
    except (OSError, SQLAlchemyError):
        st.error("The profile could not be saved. Please try again.")
    else:
        if new_profile:
            st.session_state["profiles_submission"] = str(uuid4())
        st.session_state["profiles_feedback"] = success
        st.rerun()


def add_profile(repository):
    if "profiles_submission" not in st.session_state:
        st.session_state["profiles_submission"] = str(uuid4())
    token = st.session_state["profiles_submission"]
    with st.expander("Add Child Profile"):
        with st.form(f"add-profile-{token}"):
            name, age_group, avatar = profile_fields()
            if st.form_submit_button("Save new profile", type="primary"):
                save(
                    lambda: repository.create(
                        name, submission_token=token, age_group=age_group, avatar=avatar,
                    ),
                    "Child profile added.",
                    new_profile=True,
                )


def profile_card(repository, profile):
    key = profile.id
    title = " ".join(filter(None, (AVATARS.get(profile.avatar), profile.display_name)))
    details = "Archived" if profile.archived else "Active"
    if profile.age_group:
        details = f"{details} · Age group {profile.age_group}"
    with theme.card(f"profile-{key}", title, details):
        if profile.archived:
            if st.button("Restore Profile", key=f"restore-{key}"):
                save(lambda: repository.restore(key), "Child profile restored.")
            return
        st.caption("Monitoring data has not been connected.")
        with st.expander("Edit Profile"):
            with st.form(f"edit-profile-{key}"):
                name, age_group, avatar = profile_fields(profile)
                if st.form_submit_button("Save changes"):
                    save(
                        lambda: repository.edit(key, name, age_group=age_group, avatar=avatar),
                        "Child profile updated.",
                    )
        with st.expander("Archive Profile"):
            with st.form(f"archive-profile-{key}", clear_on_submit=True):
                confirmed = st.checkbox("I confirm that I want to archive this profile.")
                if st.form_submit_button("Archive profile"):
                    if confirmed:
                        save(lambda: repository.archive(key), "Child profile archived.")
                    else:
                        st.error("Confirm that you want to archive this profile.")


def render_profiles(repository):
    profiles = repository.list_all()
    feedback = st.session_state.pop("profiles_feedback", None)
    if feedback:
        st.success(feedback)
    add_profile(repository)
    st.subheader("Active profiles")
    active = [profile for profile in profiles if not profile.archived]
    if not active:
        st.info("No active child profiles. Add a profile to get started.")
    for profile in active:
        profile_card(repository, profile)

    with st.expander("Archived profiles"):
        archived = [profile for profile in profiles if profile.archived]
        if not archived:
            st.caption("No archived profiles.")
        for profile in archived:
            profile_card(repository, profile)


theme.page_header("Child Profiles", "Manage the child profiles saved on this device.")
try:
    with closing(ChildProfileRepository()) as repository:
        render_profiles(repository)
except (OSError, ProfileStorageError, SQLAlchemyError):
    st.error(
        "Child profiles are unavailable. Check that local profile storage is accessible and try again."
    )
