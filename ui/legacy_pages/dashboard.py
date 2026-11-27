import streamlit as st

from ui import services, theme

theme.page_header("Dashboard", "Overview of Component 3 sessions stored in the database.")

kpis = services.dashboard_kpis()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total sessions", kpis["total_sessions"])
c2.metric("Escalated", kpis["escalated"])
c3.metric("High risk", kpis["high_risk"])
c4.metric("Self-regulation shown", f"{kpis['self_regulation_pct']}%")

if kpis["total_sessions"] == 0:
    with theme.card("empty", "No sessions yet",
                    "Add some on the Add record page (Data & Admin)."):
        pass
    st.stop()

left, right = st.columns(2)
with left:
    with theme.card("by-risk", "Sessions by risk level"):
        st.plotly_chart(theme.bar_figure(services.risk_level_counts(), theme.RISK_COLORS),
                        config=theme.PLOTLY_CONFIG, width="stretch")
with right:
    with theme.card("by-emotion", "Sessions by emotional state"):
        st.plotly_chart(theme.bar_figure(services.emotional_state_counts(), theme.EMOTION_COLORS),
                        config=theme.PLOTLY_CONFIG, width="stretch")
