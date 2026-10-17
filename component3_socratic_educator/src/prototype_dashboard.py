"""
PROTOTYPE - throwaway, not production. Delete after the layout is picked.

Question: what should the Component 3 PP1 demo page look like?
Three variants of the demo page, switchable via ?variant=A|B|C (floating bar at
the bottom, or the left/right arrow keys), on this throwaway Streamlit page:

    A  Three columns   Input | Process | Output side by side (the grilled design)
    B  Numbered steps  1 Trigger -> 2 Dialogue -> 3 Evaluation -> 4 Hand-off
    C  Child vs system What the child sees, next to a log of what the system decides

All variants drive the real, unchanged FSMController.run: each reply replays the
session from the start with the replies so far, and the INTERCEPT model line is
stored after the first call so a replay does not change it. Nothing is saved.

    streamlit run component3_socratic_educator/src/prototype_dashboard.py
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component3_socratic_educator.src import config, emotion_rules                           # noqa: E402
from component3_socratic_educator.src.fsm_controller import _SELF_REGULATION_PATTERN, FSMController  # noqa: E402
from component3_socratic_educator.src.grammar_decoder import delivery_violation               # noqa: E402
from component3_socratic_educator.src.model_client import LM_STUDIO_URL, call_local_model     # noqa: E402
from component3_socratic_educator.src.schemas import RiskCategory, TriggerPayload             # noqa: E402
from integration.comp3_to_comp4 import to_emotional_pattern_row                               # noqa: E402
from integration.end_to_end_pipeline import RecordingGenerator, build_trigger_from            # noqa: E402

st.set_page_config(page_title="Socratic Educator demo (prototype)", layout="wide")

VARIANTS = {"A": "Three columns", "B": "Numbered steps", "C": "Child vs system"}
SOURCES = {
    "component1": ("Component 1: screen monitoring", "component1_screen_monitoring", "image",
                   ["explicit_visual", "violence", "self_harm_imagery", "unknown_flagged"]),
    "component2": ("Component 2: text detection", "component2_hate_speech_detection", "text_message",
                   ["hate_speech", "cyberbullying", "grooming_language", "self_harm_language"]),
}
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
    for i, (state, line) in enumerate(gen.lines):
        reply = s.replies[i] if i < len(asked) else None
        turns.append({"state": state, "text": line.text, "source": line.source, "reason": line.reason,
                      "model_output": line.model_output, "reply": reply,
                      "reply_type": emotion_rules.classify_reply(reply) if reply else None,
                      "refusal": emotion_rules.is_refusal(reply) if reply else False,
                      "complete": FSMController()._is_complete(reply) if reply else False})
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


def explain(run):
    """The EVALUATE 'why' lines, from the same rules the FSM used."""
    r = run["record"]
    conf = ss().trigger.confidence_score
    replies = ss().replies[:len([t for t in run["turns"] if t["reply"] is not None])]
    verdict = emotion_rules.classify(replies)
    phrases = "; ".join(f"{k}: {', '.join(repr(p) for p in v)}" for k, v in verdict.hits.items()) or "no phrase fired"
    selfreg = [m.group(0) for rep in replies for m in [_SELF_REGULATION_PATTERN.search(rep.lower())] if m]
    why = r.dialogue_summary.split("escalated: ")[-1] if r.escalation_flag else "not escalated"
    rule = "≥ 0.85 → high" if conf >= 0.85 else "≥ 0.50 → moderate" if conf >= 0.5 else "< 0.50 → low"
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
    n, m = config.MIN_RESPONSE_WORDS_FOR_COMPLETENESS, config.MAX_INQUIRE_ATTEMPTS
    lines += ['INTERCEPT -> INQUIRE [label="reply"];',
              'INTERCEPT -> EVALUATE [label="refusal", style=dashed];',
              f'INQUIRE -> INQUIRE [label="< {n} words, no feeling/denial/plan,\\nattempts < {m}"];',
              f'INQUIRE -> EVALUATE [label="complete, refusal\\nor {m} attempts"];',
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
    if not locked and st.button("Start session", type="primary", key=f"{key}_start"):
        t = TriggerPayload.new(component, content_type, RiskCategory(cat), conf)
        t.validate()
        s.trigger, s.replies, s.memo = t, [], {}
        st.rerun()
    if locked:
        st.caption("Trigger payload, validated against its inbound schema ✓")
        st.json(s.trigger.to_dict(), expanded=False)


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


def under_the_hood(t, memo):
    kind, value, secs = memo if memo else ("-", "-", 0.0)
    st.markdown(f"**Model call:** {kind}, {secs:.2f} s")
    st.code(value if kind == "ok" else f"no model line ({value})", language="json")
    st.markdown(f"**Grammar + post-check on the model line:** "
                f"{delivery_violation(t['model_output']) or 'passed ✓' if t['model_output'] else 'no model line'}")
    st.markdown(f"**Delivered:** {t['source']}" + (f" (reason: {t['reason']})" if t["reason"] else ""))


def top_bar(key):
    s = ss()
    up, model = lm_studio_status()
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.markdown(f"{'🟢' if up else '🔴'} LM Studio at `{LM_STUDIO_URL.split('/v1/')[0]}`: {model}")
    s.live = c2.toggle("Live model", value=s.live and up, disabled=not up or s.trigger is not None, key=f"{key}_live")
    if c3.button("New session", key=f"{key}_new"):
        s.trigger, s.replies, s.memo = None, [], {}
        st.rerun()


def limitations():
    with st.expander("Limitations (said openly)"):
        st.markdown("- INQUIRE lines are templates picked by reply type; the model writes only INTERCEPT until after PP1.\n"
                    "- Triggers are built from the Component 1/2 mock fixtures, not live detections.\n"
                    "- EVALUATE is rule-based (phrase lists); off-topic replies are not detected.")


# ------------------------------------------------------------------ Variant A: three columns
def variant_a():
    st.title("Socratic Educator: input → process → output")
    top_bar("a")
    cin, cproc, cout = st.columns([1, 1.7, 1.1], gap="large")
    with cin:
        st.subheader("① Input: trigger")
        trigger_form("a")
    if ss().trigger is None:
        cproc.info("Choose a trigger and start a session.")
        return limitations()
    run = replay()
    with cproc:
        st.subheader("② Process: FSM dialogue")
        st.graphviz_chart(fsm_dot(run))
        for i, t in enumerate(run["turns"]):
            with st.chat_message("assistant"):
                st.markdown(f"**{t['state']}** · {badge(t)}  \n{t['text']}")
                if t["state"] == "INTERCEPT":
                    with st.expander("Under the hood"):
                        under_the_hood(t, run["memo"])
            if t["reply"] is not None:
                with st.chat_message("user"):
                    st.markdown(t["reply"])
                    st.caption(f"reply type: {t['reply_type']} · "
                               + ("refusal → stop" if t["refusal"] else "complete" if t["complete"] else "incomplete"))
        if run["done"]:
            with st.chat_message("assistant"):
                st.markdown(f"**CONTRACT**  \n{run['closing']}")
        reply_controls(run, "a")
    with cout:
        st.subheader("③ Output: record for Component 4")
        if not run["done"]:
            st.info("Appears when the dialogue reaches CONTRACT.")
        else:
            r = run["record"]
            if r.escalation_flag:
                st.error("Escalated: a parent is asked to review this session.")
            st.caption("comp3_to_comp4 schema ✓")
            st.json(r.to_dict())
            st.markdown("**Why**")
            for name, text in explain(run):
                st.markdown(f"- `{name}`: {text}")
            with st.expander("Component 4 input row"):
                st.json(to_emotional_pattern_row(r))
    limitations()


# ------------------------------------------------------------------ Variant B: numbered vertical steps
def variant_b():
    st.title("Socratic Educator: one session, step by step")
    top_bar("b")
    s = ss()
    run = replay() if s.trigger else None

    def step(n, title, status):
        icon = {"done": "✅", "active": "🔵", "waiting": "⏳"}[status]
        st.markdown(f"### {icon} Step {n} · {title}")

    step(1, "A detector raises a trigger", "done" if s.trigger else "active")
    with st.container(border=True):
        trigger_form("b")
    st.markdown("<div style='text-align:center;font-size:28px'>↓</div>", unsafe_allow_html=True)

    step(2, "The FSM talks with the child", "waiting" if not run else "done" if run["done"] else "active")
    with st.container(border=True):
        if run:
            st.graphviz_chart(fsm_dot(run, size="9,1.2"))
            rows = []
            for t in run["turns"]:
                rows.append({"state": t["state"], "system line": t["text"], "line source": badge(t),
                             "child reply": t["reply"] or "(waiting)", "reply type": t["reply_type"] or "",
                             "decision": "" if t["reply"] is None else
                             "refusal → stop" if t["refusal"] else "complete → stop" if t["complete"] else "ask again"})
            st.dataframe(rows, hide_index=True, width="stretch")
            if run["turns"]:
                with st.expander("Under the hood: the INTERCEPT model call"):
                    under_the_hood(run["turns"][0], run["memo"])
            reply_controls(run, "b")
        else:
            st.caption("Waiting for a trigger.")
    st.markdown("<div style='text-align:center;font-size:28px'>↓</div>", unsafe_allow_html=True)

    done = bool(run and run["done"])
    step(3, "EVALUATE explains its decision", "done" if done else "waiting")
    with st.container(border=True):
        if done:
            cols = st.columns(4)
            for col, (name, text) in zip(cols, explain(run)):
                col.metric(name, text.split(" ")[0] if name != "risk_level" else run["record"].risk_level.value)
                col.caption(text)
        else:
            st.caption("Runs after the dialogue.")
    st.markdown("<div style='text-align:center;font-size:28px'>↓</div>", unsafe_allow_html=True)

    step(4, "CONTRACT closes and hands off to Component 4", "done" if done else "waiting")
    with st.container(border=True):
        if done:
            st.markdown(f"**The child hears:** {run['closing']}")
            (st.error if run["record"].escalation_flag else st.success)(
                "Escalated to a parent" if run["record"].escalation_flag else "Not escalated")
            c1, c2 = st.columns(2)
            c1.caption("Record (comp3_to_comp4 schema ✓)")
            c1.json(run["record"].to_dict())
            c2.caption("Component 4 input row")
            c2.json(to_emotional_pattern_row(run["record"]))
        else:
            st.caption("Waiting for CONTRACT.")
    limitations()


# ------------------------------------------------------------------ Variant C: child's screen vs system log
def variant_c():
    s = ss()
    st.title("What the child sees, and what the system decides")
    top_bar("c")
    with st.expander("Input: the trigger a detector raised on the child's device", expanded=s.trigger is None):
        trigger_form("c")
    if s.trigger is None:
        return limitations()
    run = replay()
    child, system = st.columns([1, 1.3], gap="large")
    with child:
        st.markdown("#### 📱 The child's screen")
        bubbles = []
        for t in run["turns"]:
            bubbles.append(f"<div style='background:#eef2ff;padding:10px 14px;border-radius:14px;margin:8px 40px 8px 0'>{t['text']}</div>")
            if t["reply"] is not None:
                bubbles.append(f"<div style='background:#dcfce7;padding:10px 14px;border-radius:14px;margin:8px 0 8px 40px;text-align:right'>{t['reply']}</div>")
        if run["done"]:
            bubbles.append(f"<div style='background:#eef2ff;padding:10px 14px;border-radius:14px;margin:8px 40px 8px 0'>{run['closing']}</div>")
        st.markdown("<div style='border:10px solid #1f2937;border-radius:28px;padding:16px;min-height:420px;"
                    "background:white;color:#111;font-size:17px'>" + "".join(bubbles) + "</div>",
                    unsafe_allow_html=True)
        reply_controls(run, "c")
    with system:
        st.markdown("#### 🧠 System log")
        st.graphviz_chart(fsm_dot(run, size="6,1.4"))
        log = [f"Trigger received from {s.trigger.source_component}: {s.trigger.risk_category.value}, "
               f"confidence {s.trigger.confidence_score:.2f}; inbound schema ✓"]
        prev_type = None
        for t in run["turns"]:
            if t["state"] == "INTERCEPT":
                kind, value, secs = run["memo"] or ("-", "-", 0)
                log.append(f"INTERCEPT: model call {kind} ({secs:.2f} s) → {badge(t)}; the category is never named")
            else:
                log.append(f"INQUIRE: line picked for reply type '{prev_type}' → {badge(t)}")
            if t["reply"] is not None:
                prev_type = t["reply_type"]
                log.append(f"Child replied; reply type {t['reply_type']}; "
                           + ("refusal → no more questions" if t["refusal"] else
                              "complete reply → stop asking" if t["complete"] else "not complete → ask again"))
        if run["done"]:
            for name, text in explain(run):
                log.append(f"EVALUATE {name}: {text}")
            log.append(f"CONTRACT: {'escalated to a parent' if run['record'].escalation_flag else 'closed normally'}; "
                       "record validated against comp3_to_comp4 ✓ and sent to Component 4")
        for i, entry in enumerate(log, 1):
            st.markdown(f"`{i:02d}` {entry}")
        if run["turns"]:
            with st.expander("Under the hood: INTERCEPT"):
                under_the_hood(run["turns"][0], run["memo"])
    if run["done"]:
        with st.container(border=True):
            st.markdown("**Output to Component 4**")
            c1, c2 = st.columns(2)
            c1.json(run["record"].to_dict())
            c2.json(to_emotional_pattern_row(run["record"]))
    limitations()


# ------------------------------------------------------------------ floating variant switcher (prototype only)
def switcher(current):
    keys = list(VARIANTS)
    i = keys.index(current)
    st.markdown("""<style>
      .st-key-proto_switcher {position:fixed; bottom:18px; left:50%; transform:translateX(-50%); z-index:9999;
        background:#111827; color:white; border-radius:999px; padding:4px 14px; box-shadow:0 4px 18px rgba(0,0,0,.35);
        width:auto !important;}
      .st-key-proto_switcher p {color:white; margin:0; white-space:nowrap;}
      .st-key-proto_switcher button {background:transparent; color:white; border:none;}
    </style>""", unsafe_allow_html=True)
    with st.container(key="proto_switcher", horizontal=True, vertical_alignment="center"):
        if st.button("◀", key="proto_prev"):
            st.query_params["variant"] = keys[(i - 1) % len(keys)]
            st.rerun()
        st.markdown(f"**PROTOTYPE** · {current} ({VARIANTS[current]})")
        if st.button("▶", key="proto_next"):
            st.query_params["variant"] = keys[(i + 1) % len(keys)]
            st.rerun()
    # Left/right arrow keys click the buttons (ignored while typing in a field).
    components.html("""<script>
      const doc = window.parent.document;
      if (!doc.__protoKeys) { doc.__protoKeys = true;
        doc.addEventListener('keydown', e => {
          const t = e.target; if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
          const label = e.key === 'ArrowLeft' ? '◀' : e.key === 'ArrowRight' ? '▶' : null; if (!label) return;
          const b = [...doc.querySelectorAll('.st-key-proto_switcher button')].find(x => x.innerText.trim() === label);
          if (b) b.click();
        });
      }</script>""", height=0)


variant = st.query_params.get("variant", "A")
variant = variant if variant in VARIANTS else "A"
{"A": variant_a, "B": variant_b, "C": variant_c}[variant]()
switcher(variant)
