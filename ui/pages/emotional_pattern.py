import streamlit as st

from ui import child_selection, services, theme
from ui.parent_guidance_view import (
    PARENT_PATTERN_LABELS,
    emotional_change_view,
    parent_guidance_view,
    parent_xai_view,
)

MODEL_UNAVAILABLE = "The emotional wellbeing summary is temporarily unavailable."


def show_model_error():
    st.info(MODEL_UNAVAILABLE)
    st.stop()


child_selection.show_assessment_context("Emotional Pattern")

f1_status = services.function1_status()
if not f1_status["available"]:
    show_model_error()

source = services.child_session_source(demo_only=True)
child = st.selectbox("Select Child", source["child_ids"], key=child_selection.F1_DEMO_CHILD)
if source["source"] == "demo":
    st.caption("SYNTHETIC DEMO DATA · These sessions are fictional.")

try:
    report = services.emotional_pattern_report(services.sessions_for_child(source, child), child)
except services.ModelUnavailableError:
    show_model_error()
except ValueError:
    st.info("The recorded sessions could not be used to describe an emotional pattern.")
    st.stop()

min_sessions = theme.plural(report["min_sessions"], "session")

guidance = parent_guidance_view(report)
change = emotional_change_view(report, guidance)
ai = parent_xai_view(report)
if not report["enough_data"]:
    pattern_note = "There are too few recorded sessions to describe a pattern yet."
elif report["current_since"]:
    pattern_note = "This describes how your child responded in the latest period with enough sessions."
else:
    pattern_note = "This describes how your child responded across the sessions recorded so far."

with theme.card("emotional-wellbeing", "Your Child's Emotional Wellbeing"):
    st.markdown("#### Current Emotional Pattern")
    st.markdown(f"**{guidance['pattern']}**")
    st.write(pattern_note)

    st.markdown("#### Has Anything Changed?")
    st.write(" ".join(filter(None, (change["change"], change["trend_summary"]))))

    st.markdown("#### What Can You Do?")
    recommendation = guidance["recommendation"]
    if guidance["timing"]:
        recommendation = f"{guidance['timing']}, {recommendation[0].lower()}{recommendation[1:]}"
    st.write(recommendation)

    st.markdown("#### Try Asking")
    st.write(f"“{guidance['question']}”")


with st.expander("View Emotional History", expanded=False, icon=":material/timeline:"):
    if report["timeline"]:
        st.plotly_chart(theme.pattern_timeline_figure(report["timeline"], labels=PARENT_PATTERN_LABELS),
                        config=theme.PLOTLY_CONFIG, width="stretch")
        st.caption(change["history_summary"])
    else:
        st.write(f"The chart will appear when a two-week period has at least {min_sessions}.")

with st.expander("Why Did the AI Identify This Pattern?", expanded=False,
                 icon=":material/lightbulb:"):
    st.write(ai["explanation"])
    st.caption("These observations describe recorded interactions and cannot diagnose your child.")
