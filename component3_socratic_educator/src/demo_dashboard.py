"""
Component 3 viva demo page: what the child sees, what the system decides, and the fine-tuned INQUIRE adapter
as a SHADOW candidate.

The dialogue is the real, unchanged FSMController.run (each reply replays the session from the start; the
INTERCEPT model line is stored after the first call, so a replay does not change it). INQUIRE still sends
template lines only. The adapter's lines (src/shadow.py) are shown beside them, labelled "not sent to the
child", together with the measured limitations and what PP2 will fix. Nothing is saved.

    streamlit run component3_socratic_educator/src/demo_dashboard.py

Shadow output has three sources, always labelled on the page:
    live             both GGUFs in LM Studio, temperature 0, now
    as served        the same prompts run on the GGUFs earlier (evidence/inquire/inquire_served_<date>/)
    as evaluated     the Colab run behind the blind comparison (evidence/inquire/inquire_comparison_2026-10-10/)
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component3_socratic_educator.src import config, emotion_rules                           # noqa: E402
from component3_socratic_educator.src.assessor import (                                       # noqa: E402
    CHANGE_TALK, DISTRESS_SEVERITY, EXPRESSED_EMOTION, STANCE, RulesAssessor,
)
from component3_socratic_educator.src.fsm_controller import DialogueState, FSMController, Turn  # noqa: E402
from component3_socratic_educator.src.model_client import LM_STUDIO_URL, call_local_model     # noqa: E402
from component3_socratic_educator.src import shadow                                          # noqa: E402
from component3_socratic_educator.src.schemas import RiskCategory, TriggerPayload             # noqa: E402
from integration.comp3_to_comp4 import to_emotional_pattern_row                               # noqa: E402
from integration.end_to_end_pipeline import RecordingGenerator, build_trigger_from            # noqa: E402

st.set_page_config(page_title="Socratic Educator demo", layout="wide")

SOURCES = {
    "component1": ("Component 1: screen monitoring", "component1_screen_monitoring", "image",
                   ["explicit_visual", "violence", "self_harm_imagery", "unknown_flagged"]),
    "component2": ("Component 2: text detection", "component2_hate_speech_detection", "text_message",
                   ["hate_speech", "cyberbullying", "grooming_language", "self_harm_language"]),
}
# What each upstream score means, and the lowest score that detector ever sends (read on 20 Oct from main:
# component1 config.THRESHOLD; component2 packs/policy/decision.json v1.3.0). Component 3 reads only the number.
SCORE_MEANING = {
    "component1": "Component 1's score is the vision model's probability that the frame is violent. "
                  "It sends a trigger from 0.50.",
    "component2": "Component 2's score is the fused probability of its top category. It sends a trigger only above "
                  "the child's age threshold: 0.55 (ages 8-10), 0.65 (11-13), 0.75 (14-15). Self-harm and "
                  "grooming use 0.50 at every age.",
}
CRITICAL_C2 = {"self_harm_language", "grooming_language"}


def risk_rule(conf):
    """(risk_level, rule) Component 3 gives a trigger score: the cut-offs in FSMController._evaluate."""
    if conf >= 0.85:
        return "high", "≥ 0.85 → high"
    if conf >= 0.5:
        return "moderate", "≥ 0.50 → moderate"
    return "low", "< 0.50 → low"


def lowest_sent(src, category):
    """The lowest confidence this detector sends Component 3 for this category (youngest age band for C2)."""
    return 0.5 if src == "component1" or category in CRITICAL_C2 else 0.55


CHILDREN = {
    "reflective": ["I was just watching videos", "I guess I should close it and tell my mum next time"],
    "defensive": ["It wasn't me", "whatever", "no big deal", "It was nothing, my friend sent it"],
    "distressed": ["It was really scary", "I feel sick", "so scared", "I can't stop crying about it"],
    "withdrawn": ["idk", "nothing", "leave me alone", "I don't want to talk"],
}
STATES = ["INTERCEPT", "INQUIRE", "EVALUATE", "CONTRACT"]


# ------------------------------------------------------------------ engine (shared data, no layout)
class NeedReply(Exception):
    """Raised by the reply callback when the replies so far are used up."""


@st.cache_data(ttl=5, show_spinner=False)
def lm_studio_status():
    try:
        with urllib.request.urlopen(LM_STUDIO_URL.split("/v1/")[0] + "/v1/models", timeout=0.7) as r:
            ids = [m["id"] for m in json.loads(r.read())["data"]]
        return True, (config.LOCAL_MODEL_ID if config.LOCAL_MODEL_ID in ids else f"up, but {config.LOCAL_MODEL_ID} not listed")
    except Exception:
        return False, "not reachable"


@st.cache_data(ttl=5, show_spinner=False)
def lm_studio_ids():
    """Model IDs LM Studio lists right now, or [] when it is not reachable."""
    try:
        with urllib.request.urlopen(LM_STUDIO_URL.split("/v1/")[0] + "/v1/models", timeout=0.7) as r:
            return [m["id"] for m in json.loads(r.read())["data"]]
    except Exception:
        return []


def ss():
    s = st.session_state
    s.setdefault("trigger", None)
    s.setdefault("replies", [])
    s.setdefault("memo", {})
    s.setdefault("live", lm_studio_status()[0])
    return s


def replay():
    """Run the real FSM with the replies so far. Returns everything the layouts show."""
    s = ss()
    memo = s.memo

    def model_call(prompt, **options):
        if "result" not in memo:
            if not s.live:
                memo["result"] = ("error", "offline mode: no model called", 0.0)
            else:
                start = time.perf_counter()
                try:
                    raw, _ = call_local_model(prompt, **options)
                    memo["result"] = ("ok", raw, time.perf_counter() - start)
                except Exception as e:  # shown in the under-the-hood view
                    memo["result"] = ("error", repr(e), time.perf_counter() - start)
        kind, value, _ = memo["result"]
        if kind == "error":
            raise ConnectionError(value)
        return value, memo["result"][2]

    gen = RecordingGenerator(model_call=model_call)
    asked = []

    def respond(line):
        if len(asked) < len(s.replies):
            asked.append(line)
            return s.replies[len(asked) - 1]
        raise NeedReply(line)

    out = {"done": False, "record": None, "transcript": None, "pending": None}
    try:
        out["record"], out["transcript"] = FSMController(generator=gen).run(s.trigger, respond)
        out["done"] = True
    except NeedReply as e:
        out["pending"] = e.args[0]

    turns = []   # one per system line the child saw
    for i, (state, line, inquire) in enumerate(gen.lines):
        reply = s.replies[i] if i < len(asked) else None
        easier = inquire.get("easier", False)
        turns.append({"state": state, "text": line.text, "source": line.source, "reason": line.reason,
                      "model_output": line.model_output, "reply": reply, "move": inquire.get("move"),
                      "easier": easier, "next": next_action(state, easier, reply) if reply is not None else None})
    out["turns"] = turns
    out["closing"] = out["transcript"].turns[-1].system_text if out["done"] else None
    out["memo"] = memo.get("result")
    if out["done"]:
        out["current"] = "DONE"
        out["visited"] = {t["state"] for t in turns} | {"EVALUATE", "CONTRACT"}
    else:
        out["current"] = turns[-1]["state"] if turns else "INTERCEPT"
        out["visited"] = {t["state"] for t in turns}
    return out


def next_action(state, easier, reply):
    """What the controller does after this reply, by the same rules (src/fsm_controller.py)."""
    if emotion_rules.is_refusal(reply):
        return "refusal → stop"
    if state == "INTERCEPT":
        return "→ the first Socratic move"
    stance = RulesAssessor().assess([Turn(DialogueState.INQUIRE, "", reply)]).replies[0].labels[STANCE]
    if stance == CHANGE_TALK:
        return "safer plan (change talk) → stop"
    if emotion_rules.is_dont_know(reply) and not easier:
        return "“I don't know” → the same move, easier form"
    return f"→ the next move (at most {config.MAX_INQUIRE_MOVES} questions)"


def explain(run):
    """The EVALUATE 'why' lines, from the assessor's labels and evidence (the FSM's default rules arm)."""
    r = run["record"]
    conf = ss().trigger.confidence_score
    assessment = RulesAssessor().assess(run["transcript"].turns)
    phrases = "; ".join(
        f"“{a.reply}”: " + " / ".join(a.labels[part] + (f" ({', '.join(a.evidence[part])})" if a.evidence[part] else "")
                                       for part in (EXPRESSED_EMOTION, DISTRESS_SEVERITY, STANCE))
        for a in assessment.replies) or "no reply"
    replies = [a.reply for a in assessment.replies]
    selfreg = [m.group(0) for rep in replies for m in [emotion_rules.SELF_REGULATION.search(rep.lower())] if m]
    why = r.dialogue_summary.split("escalated: ")[-1] if r.escalation_flag else "not escalated"
    rule = risk_rule(conf)[1]
    return [("risk_level", f"confidence {conf:.2f} {rule}"),
            ("emotional_state", f"{r.emotional_state.value} because {phrases}"),
            ("self_regulation_shown", f"{r.self_regulation_shown}" + (f" because {selfreg}" if selfreg else "")),
            ("escalation_flag", f"{r.escalation_flag} ({why})")]


def fsm_dot(run, size="7,1.6"):
    lines = [f'digraph {{ rankdir=LR; size="{size}"; node [shape=box, style="rounded,filled", fontname=Helvetica];']
    for state in STATES:
        fill = "#2563eb" if state == run["current"] else "#bfdbfe" if state in run["visited"] else "#f1f5f9"
        font = "white" if state == run["current"] else "black"
        lines.append(f'{state} [fillcolor="{fill}", fontcolor="{font}"];')
    moves, m = " → ".join(config.SOCRATIC_MOVES), config.MAX_INQUIRE_MOVES
    lines += ['INTERCEPT -> INQUIRE [label="reply"];',
              'INTERCEPT -> EVALUATE [label="refusal", style=dashed];',
              f'INQUIRE -> INQUIRE [label="{moves}\\n(easier form after I don\'t know)"];',
              f'INQUIRE -> EVALUATE [label="refusal, change talk\\nor {m} questions"];',
              'EVALUATE -> CONTRACT [label="record built"];', "}"]
    return "\n".join(lines)


def badge(t):
    if t["source"] == "model":
        return "🟢 model"
    if t["source"] == "fallback":
        return f"🟠 fallback: {t['reason']}"
    return "⚪ template"


# ------------------------------------------------------------------ shared widgets (not layout)
def trigger_form(key):
    s = ss()
    locked = s.trigger is not None
    src = st.radio("Upstream detector", list(SOURCES), format_func=lambda k: SOURCES[k][0], key=f"{key}_src",
                   disabled=locked, horizontal=True)
    label, component, content_type, cats = SOURCES[src]
    fixture = build_trigger_from(src)
    cat = st.selectbox("risk_category (allowed by this detector's contract)", cats,
                       index=cats.index(fixture.risk_category.value), key=f"{key}_cat_{src}", disabled=locked)
    conf = st.slider("confidence_score", 0.0, 1.0, float(fixture.confidence_score), 0.01, key=f"{key}_conf_{src}",
                     disabled=locked)
    level, rule = risk_rule(conf)
    st.caption(f"Component 3 reads this as **{level}** ({rule}). "
               + ("High escalates to a parent whatever the child says. "
                  if level == "high" else
                  "Only a distressed child is escalated at this level. ")
               + SCORE_MEANING[src])
    unsent = None
    if not locked and conf < lowest_sent(src, cat):
        unsent = (f"{label} would not send this. Its lowest trigger for {cat} is {lowest_sent(src, cat):.2f}, "
                  "so Component 3 is never woken and no dialogue starts. Raise the confidence to start one.")
        st.warning(unsent)
    if not locked and st.button("Start session", type="primary", key=f"{key}_start", disabled=unsent is not None):
        t = TriggerPayload.new(component, content_type, RiskCategory(cat), conf)
        t.validate()
        s.trigger, s.replies, s.memo = t, [], {}
        st.rerun()
    if locked:
        st.caption("Trigger payload, validated against its inbound schema ✓")
        st.json(s.trigger.to_dict(), expanded=False)
    return unsent


def reply_controls(run, key):
    s = ss()
    if run["done"]:
        st.success("Session finished.")
        return
    with st.form(f"{key}_reply", clear_on_submit=True):
        text = st.text_input("Reply as the child", key=f"{key}_text")
        if st.form_submit_button("Send reply") and text.strip():
            s.replies.append(text)
            st.rerun()
    c1, c2 = st.columns([1, 1])
    st.session_state.setdefault(f"{key}_persona", "distressed")
    persona = c1.selectbox("Scripted child", list(CHILDREN), key=f"{key}_persona")
    script = CHILDREN[persona]
    nxt = script[min(len(s.replies), len(script) - 1)]
    if c2.button(f"Send next scripted reply: “{nxt}”", key=f"{key}_scripted"):
        s.replies.append(nxt)
        st.rerun()


def top_bar(key):
    s = ss()
    up, model = lm_studio_status()
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.markdown(f"{'🟢' if up else '🔴'} LM Studio at `{LM_STUDIO_URL.split('/v1/')[0]}`: {model}")
    s.live = c2.toggle("Live model", value=s.live and up, disabled=not up or s.trigger is not None, key=f"{key}_live")
    if c3.button("New session", key=f"{key}_new"):
        s.trigger, s.replies, s.memo = None, [], {}
        st.rerun()


LIMITATIONS = [
    "INQUIRE asks the Socratic moves with template lines per age band; the model writes only INTERCEPT "
    "until model-written INQUIRE lines (ticket 05).",
    "Triggers are built from the Component 1/2 mock fixtures, not live detections.",
    "EVALUATE's labels come from phrase rules (the rules arm); implicit feelings with no fixed wording and "
    "off-topic replies are not detected.",
    "The adapter is a first iteration and is not in the child-facing loop (see the adapter section below).",
]


# ------------------------------------------------------------------ the fine-tuned INQUIRE adapter (shadow only)
SOURCE_LABELS = {"live": "live, LM Studio, now", "served": "as served: the GGUFs, earlier run",
                 "evaluated": "as evaluated: Colab run behind the blind comparison"}


@st.cache_data(show_spinner=False)
def held_out_cases():
    return shadow.load_test_cases()


@st.cache_data(show_spinner=False)
def evaluated():
    return shadow.evaluated_figures()


@st.cache_data(show_spinner=False)
def replay_sets():
    sets = {"evaluated": shadow.load_replay(shadow.EVALUATED_DIR)}
    served = shadow.served_dir()
    if served:
        sets["served"] = shadow.load_replay(served)
    return sets


def auto_models():
    """Adapted and base LM Studio model IDs: the user's pick in the evidence section if any, else a guess by name."""
    ids = lm_studio_ids()
    adapted = st.session_state.get("adapted_id") or next((i for i in ids if "adapted" in i.lower()), None)
    base = st.session_state.get("base_id") or next(
        (i for i in ids if "base" in i.lower() and "adapted" not in i.lower()), None)
    return adapted, base, bool(adapted and base and adapted != base and adapted in ids and base in ids)


def pick_model_ids():
    """Override widgets for the two IDs (shown in the evidence section)."""
    ids = lm_studio_ids() or ["(LM Studio not reachable)"]
    c1, c2 = st.columns(2)
    c1.selectbox("Adapted model (LM Studio ID)", ids, key="adapted_id",
                 index=next((n for n, i in enumerate(ids) if "adapted" in i.lower()), 0))
    c2.selectbox("Base model, same f16 conversion (LM Studio ID)", ids, key="base_id",
                 index=next((n for n, i in enumerate(ids) if "base" in i.lower() and "adapted" not in i.lower()), 0))


def live_cached(prompt, adapted_id, base_id):
    cache = st.session_state.setdefault("shadow_cache", {})
    key = (prompt, adapted_id, base_id)
    if key not in cache:
        cache[key] = shadow.live_lines(prompt, call_local_model, base_id, adapted_id)
    return cache[key]


def shadow_column(col, title, result):
    """One model's line, with the heuristic limitation flags. `result` has response/latency_seconds/error."""
    top = evaluated()["adapted"]["top_closing"]
    with col:
        st.markdown(f"**{title}**")
        if result is None:
            st.caption("no output for this input in this mode")
            return
        if result.get("error"):
            st.error(f"call failed: {result['error']}")
            return
        text = result["response"]
        st.markdown(f"> {text}")
        f = shadow.flags(text)
        marks = []
        if f["trusted_adult"]:
            marks.append("🟠 mentions a trusted adult")
        if f["closing"].strip() == top.strip():
            marks.append("🟠 the repeated closing question")
        st.caption(" · ".join(marks) if marks else "⚪ no limitation flag fired (heuristic string match)")
        if result.get("latency_seconds") is not None:
            st.caption(f"latency {result['latency_seconds']:.2f} s")


def metrics_strip():
    """Measured results from the saved blind comparison. Quality, format and speed stay separate."""
    fig = evaluated()
    ctl, ada = fig["control"], fig["adapted"]
    st.caption(f"Held-out test split: {fig['sample']}. Saved results from the 10 Oct comparison (base vs adapted, "
               "Colab fp32, greedy). Not recomputed on this page.")
    q, f, sp, lim = st.columns(4)
    with q:
        st.markdown("**Behaviour quality** (blind rubric)")
        st.metric("Rubric overall", f"{ada['rubric_overall']:.1%}",
                  f"{ada['rubric_overall'] - ctl['rubric_overall']:+.1%} vs base")
        st.metric("Handles the reply type", f"{ada['handles_reply_type']:.1%}",
                  f"{ada['handles_reply_type'] - ctl['handles_reply_type']:+.1%} vs base")
    with f:
        st.markdown("**Format** (shared format check)")
        st.metric("Pass rate", f"{ada['format_pass_rate']:.1%}",
                  f"{ada['format_pass_rate'] - ctl['format_pass_rate']:+.1%} vs base")
        st.metric("Stated-emotion violations", f"{ada['stated_emotion_violations']} / {ada['n']}",
                  f"base {ctl['stated_emotion_violations']} / {ctl['n']}", delta_color="off")
    with sp:
        st.markdown("**Speed** (separate, not quality)")
        st.metric("Mean latency", f"{ada['latency_mean']:.2f} s", f"base {ctl['latency_mean']:.2f} s",
                  delta_color="off")
        st.caption("Measured on a Colab T4, not on the demo laptop.")
    with lim:
        st.markdown("**Limitations the adapter introduced**")
        st.metric("Replies mentioning a trusted adult", f"{ada['trusted_adult']} / {ada['n']}",
                  f"base {ctl['trusted_adult']} / {ctl['n']}", delta_color="off")
        st.metric("Most repeated closing line", f"{ada['top_closing_count']} / {ada['n']}",
                  f"base {ctl['top_closing_count']}×", delta_color="off")
    st.caption("Not measured: JSON validity, grammar compliance and controller correctness, because the adapter is "
               "not wired into the FSM.")


def pp2_panel():
    st.markdown("**PP2: what will be fixed, and how**")
    for problem, fix in shadow.PP2_FIXES:
        st.markdown(f"- **{problem}.** {fix}")
    st.caption("Plan: ticket 07, after PP1. The 10 Oct dataset, adapter and comparison stay unchanged as the baseline.")


def session_shadow(run):
    """Under the dialogue: per INQUIRE turn, the candidate lines beside the template line the child got."""
    s = ss()
    adapted_id, base_id, ready = auto_models()
    header = held_out_cases()[0]["header"]
    inquire = [(i, t) for i, t in enumerate(run["turns"])
               if t["state"] == "INQUIRE" and i > 0 and run["turns"][i - 1]["reply"] is not None]
    if not inquire:
        return
    st.markdown("**Candidate adapter lines (not sent to the child)**")
    for i, t in inquire:
        prev = run["turns"][i - 1]
        with st.container(border=True):
            st.caption(f"After the child said “{prev['reply']}”, the child got the template line above.")
            if ready:
                prompt = shadow.session_prompt(header, s.trigger.risk_category.value, prev["text"], prev["reply"])
                res = live_cached(prompt, adapted_id, base_id)
                c1, c2 = st.columns(2)
                shadow_column(c1, "Base model", res["control"])
                shadow_column(c2, "Adapted model", res["adapted"])
            else:
                st.info("Live shadow needs both GGUFs loaded in LM Studio. The section at the bottom has the replay.")


def held_out_section(models):
    cases = held_out_cases()
    sets = replay_sets()
    adapted_id, base_id, ready = models
    modes = (["live"] if ready else []) + [m for m in ("served", "evaluated") if m in sets]
    mode = st.radio("Source of the model lines", modes, format_func=lambda m: SOURCE_LABELS[m], horizontal=True,
                    key="held_mode")
    labels = [f"{c['example_id']} · {c['risk_category']} · {c['reply_type']}" for c in cases]
    chosen = st.selectbox("Held-out case (never used in training or checkpoint choice)", labels, key="held_case")
    c = cases[labels.index(chosen)]
    st.markdown(f"**Previous educator message:** {c['previous']}")
    typed = st.text_input("Or type a different child reply (live mode only)", key="held_typed", disabled=mode != "live")
    child = typed.strip() if (typed.strip() and mode == "live") else c["child"]
    st.markdown(f"**Child's reply:** {child}")
    if mode == "live":
        prompt = shadow.build_prompt(c["header"], c["context"], c["previous"], child)
        res = live_cached(prompt, adapted_id, base_id)
    else:
        res = {arm: sets[mode][arm].get(c["example_id"]) for arm in ("control", "adapted")}
    b, a = st.columns(2)
    shadow_column(b, "Base model (gemma-3-1b-it)", res["control"])
    shadow_column(a, "Adapted model (+ INQUIRE LoRA)", res["adapted"])
    st.caption(f"Source: {SOURCE_LABELS[mode]}. Both models get the identical prompt at temperature 0.")


def adapter_evidence():
    """The one collapsed section below the three columns: the adapter's held-out outputs, results and PP2 plan."""
    with st.expander("Fine-tuned INQUIRE adapter: a candidate, not in the loop (outputs, results, PP2 plan)"):
        st.info("The child only ever receives template lines: INQUIRE's grammar is exactly the template list, whatever "
                "a model writes. The adapter is a first measured iteration.")
        pick_model_ids()
        held_out_section(auto_models())
        st.markdown("**Measured results**")
        metrics_strip()
        pp2_panel()
        st.markdown("**Other limitations**")
        st.markdown("\n".join(f"- {x}" for x in LIMITATIONS))


# ------------------------------------------------------------------ the page: Input | Process | Output
def page():
    s = ss()
    st.title("Socratic Educator: input → process → output")
    top_bar("c")
    cin, cproc, cout = st.columns([1, 1.7, 1.1], gap="large")

    with cin:
        st.subheader("① Input")
        unsent = trigger_form("c")
    if s.trigger is None:
        cproc.info("Nothing was sent, so Component 3 is not running. Raise the confidence to the detector's "
                   "lowest trigger." if unsent else "Choose a trigger and start a session.")
        cout.info("The record appears when the dialogue ends.")
        return adapter_evidence()

    run = replay()
    with cproc:
        st.subheader("② Process")
        st.graphviz_chart(fsm_dot(run, size="6,1.2"))
        for t in run["turns"]:
            with st.chat_message("assistant"):
                move = f" · {t['move']}{' (easier)' if t['easier'] else ''}" if t["move"] else ""
                st.markdown(f"**{t['state']}**{move} · {badge(t)}  \n{t['text']}")
            if t["reply"] is not None:
                with st.chat_message("user"):
                    st.markdown(t["reply"])
                    st.caption(t["next"])
        if run["done"]:
            with st.chat_message("assistant"):
                st.markdown(f"**CONTRACT**  \n{run['closing']}")
        reply_controls(run, "c")
        session_shadow(run)

    with cout:
        st.subheader("③ Output")
        if not run["done"]:
            st.info("The record for Component 4 appears when the dialogue reaches CONTRACT.")
        else:
            r = run["record"]
            if r.escalation_flag:
                st.error("Escalated: a parent is asked to review this session.")
            st.json(r.to_dict())
            st.markdown("**Why**")
            for name, text in explain(run):
                st.markdown(f"- `{name}`: {text}")
    adapter_evidence()


page()
