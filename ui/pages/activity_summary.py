import streamlit as st

from ui import child_selection, theme


theme.page_header("Activity Summary")
child = child_selection.selected_profile_label(include_avatar=True)
if child:
    theme.card("activity-child", child, "Activity data has not been connected for this child.")
elif child_selection.demo_mode_active():
    st.info("Activity summaries are unavailable in Demo Mode.")
else:
    st.info("Select a child profile in the sidebar. Activity data is not connected yet.")
