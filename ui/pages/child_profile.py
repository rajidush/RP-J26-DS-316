import streamlit as st

from ui import theme

theme.page_header("Child Profile",
                  "One place to see how your child is doing across every part of the system.")

with theme.card("profile-soon", "Coming soon",
                "This page is not built yet. Once Component 3 sessions carry a child_id, it "
                "will bring together for one child:"):
    st.markdown(
        "- Current emotional pattern and trend (Function 1)\n"
        "- Recent sessions, and any escalated for parent review (Component 3)\n"
        "- A plain-language explanation of what drove each result"
    )
