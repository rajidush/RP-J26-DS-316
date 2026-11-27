"""
Component 4 (Profiling / XAI) UI.

    streamlit run ui/app.py        # from the repo root, so .streamlit/config.toml applies

Reads Component 3 session records from the shared database (DATABASE_URL,
default sqlite:///project.db) and runs the Component 4 functions on them.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui import child_selection, services, theme  # noqa: E402

st.set_page_config(page_title="Parent Dashboard", page_icon=":material/groups:",
                   layout="wide")


@st.cache_resource
def _startup():
    return services.startup()


database_url = _startup()
theme.apply_style()

PAGES_DIR = Path(__file__).resolve().parent / "dashboard_pages"

# (file, title, icon, key) -- key also names the sidebar link's container.
# Keep scripts outside Streamlit's automatically discovered pages directory.
MAIN_PAGES = [
    ("parent_overview.py", "Overview", ":material/home:", "overview"),
    ("child_profile.py", "Child Profiles", ":material/person:", "child-profile"),
    ("emotional_pattern.py", "Emotional Pattern", ":material/bar_chart:", "emotional-pattern"),
    ("behavioral_anomaly.py", "Behavioral Anomaly", ":material/monitoring:", "behavioral-anomaly"),
    ("safety_notifications.py", "Safety Notifications", ":material/notifications:", "safety-notifications"),
    ("activity_summary.py", "Activity Summary", ":material/insights:", "activity-summary"),
]


def make_pages(specs, default_first=False):
    return [(st.Page(PAGES_DIR / f, title=t, icon=i, url_path=k,
                     default=default_first and n == 0), k)
            for n, (f, t, i, k) in enumerate(specs)]


main = make_pages(MAIN_PAGES, default_first=True)

# Built-in navigation is hidden; the sidebar is drawn below so the brand sits
# above the links.
current = st.navigation([p for p, _ in main], position="hidden")

theme.sidebar_brand()
child_selection.render_profile_selector()
for page, key in main:
    with st.sidebar.container(key=f"nav-{key}"):
        st.page_link(page)
child_selection.render_demo_control()

active_key = next(k for p, k in main if p.url_path == current.url_path)
theme.active_nav_css(f"nav-{active_key}")

current.run()
