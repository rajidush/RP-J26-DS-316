import streamlit as st

from ui import child_selection, theme


theme.page_header("Safety Notifications")
child = child_selection.selected_profile_label(include_avatar=True)
if child:
    theme.card("safety-child", child, "Safety notifications have not been connected for this child.")
elif child_selection.demo_mode_active():
    st.info("Safety notifications are unavailable in Demo Mode.")
else:
    st.info("Select a child profile in the sidebar. Child-specific notifications are not connected yet.")
