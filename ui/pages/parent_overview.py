import streamlit as st

from ui import child_selection, theme


theme.page_header("Overview", "See which child profile is selected and whether monitoring is connected.")

has_profiles = child_selection.has_active_profiles()
child = child_selection.selected_profile_label(include_avatar=True)
if child_selection.demo_mode_active():
    st.info("Demo Mode is on. The assessment pages use fictional data.")
elif child:
    theme.card("overview-child", child, "Monitoring data has not been connected for this child.")
elif has_profiles is None:
    st.info("Child profiles are temporarily unavailable.")
elif has_profiles:
    st.info("Select a child profile in the sidebar.")

if has_profiles is False:
    st.info("There are no active child profiles.")
    st.page_link("pages/child_profile.py", label="Go to Child Profiles", icon=":material/person:")
