"""
src/shadow.py: the fine-tuned INQUIRE adapter shown as a candidate beside the template lines.
No model and no LM Studio: the model call is faked, and the evidence files are read from disk.

The important guarantee is the first group: nothing the adapter writes can reach the child.
"""
import ast
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from component3_socratic_educator.src import model_client, shadow  # noqa: E402
from component3_socratic_educator.src.fsm_controller import FSMController  # noqa: E402
from component3_socratic_educator.src.grammar_decoder import GrammarConstrainedGenerator, INQUIRE_TEMPLATES  # noqa: E402
from component3_socratic_educator.src.schemas import RiskCategory, TriggerPayload  # noqa: E402

SRC = PROJECT_ROOT / "src"


# ---------------------------------------------------------------- the adapter cannot reach the child
def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names |= {a.name for a in node.names}
    return names


def test_shadow_module_does_not_import_the_controller_or_the_generator():
    names = _imports(SRC / "shadow.py")
    assert not {n for n in names if any(x in n for x in ("fsm_controller", "grammar_decoder", "FSMController",
                                                         "GrammarConstrainedGenerator", "model_client"))}


def test_controller_and_generator_do_not_know_about_the_shadow_module():
    for name in ("fsm_controller.py", "grammar_decoder.py"):
        assert "shadow" not in (SRC / name).read_text(encoding="utf-8").lower()


TEMPLATE_LINES = {line for bands in INQUIRE_TEMPLATES.values() for pair in bands.values() for line in pair}


def test_inquire_grammar_is_still_exactly_the_template_lines():
    gen = GrammarConstrainedGenerator(model_call=lambda *a, **k: ("I am the adapter and I say something else.", 0.1))
    for move in INQUIRE_TEMPLATES:
        assert gen.generate("INQUIRE", RiskCategory.CYBERBULLYING, move=move) in TEMPLATE_LINES


def test_a_session_gives_the_child_only_template_lines_even_when_shadow_lines_are_fetched():
    shadow_text = "It's okay to feel annoyed. Would you rather take a short break or talk to a parent?"
    calls = []

    def fake_call(prompt, **kw):
        calls.append(kw.get("model"))
        return shadow_text, 0.01

    shadow.live_lines("any prompt", fake_call, "base-id", "adapted-id")      # the dashboard does this per turn
    assert calls == ["base-id", "adapted-id"]

    trigger = TriggerPayload.new("component2_hate_speech_detection", "text_message", RiskCategory.CYBERBULLYING, 0.7)
    seen = []

    def respond(line):
        seen.append(line)
        return "idk" if len(seen) < 3 else "I will tell my mum"

    _record, transcript = FSMController().run(trigger, respond)
    inquire = [t.system_text for t in transcript.turns if t.state.name == "INQUIRE"]
    assert inquire and all(line in TEMPLATE_LINES for line in inquire)
    assert shadow_text not in [t.system_text for t in transcript.turns]


# ---------------------------------------------------------------- live shadow calls
def test_live_lines_asks_both_models_the_same_prompt_at_temperature_zero():
    sent = []

    def fake_call(prompt, **kw):
        sent.append((prompt, kw))
        return f"  reply from {kw['model']}  ", 0.5

    out = shadow.live_lines("P", fake_call, "base-id", "adapted-id")
    assert [s[0] for s in sent] == ["P", "P"]
    assert all(s[1]["temperature"] == 0 for s in sent)
    assert out["control"]["response"] == "reply from base-id" and out["adapted"]["response"] == "reply from adapted-id"
    assert out["adapted"]["error"] is None


def test_a_failed_model_call_is_reported_not_raised():
    def fake_call(prompt, **kw):
        if kw["model"] == "adapted-id":
            raise ConnectionError("LM Studio is off")
        return "fine", 0.2

    out = shadow.live_lines("P", fake_call, "base-id", "adapted-id")
    assert out["control"]["response"] == "fine"
    assert out["adapted"]["response"] == "" and "LM Studio is off" in out["adapted"]["error"]


def test_call_local_model_uses_the_given_model_id_and_defaults_to_the_configured_one(monkeypatch):
    import io
    sent = {}

    def fake_urlopen(req, timeout):
        sent["body"] = json.loads(req.data)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "x"}}]}).encode())

    monkeypatch.setattr(model_client.urllib.request, "urlopen", fake_urlopen)
    model_client.call_local_model("p", model="adapted-id")
    assert sent["body"]["model"] == "adapted-id"
    model_client.call_local_model("p")
    assert sent["body"]["model"] == model_client.config.LOCAL_MODEL_ID


# ---------------------------------------------------------------- held-out cases and prompts
def test_held_out_cases_are_the_test_split():
    cases = shadow.load_test_cases()
    assert len(cases) == 50
    assert len({c["scenario_id"] for c in cases}) == 10
    manifest = json.loads((shadow.TEST_FILE.parent / "manifest.json").read_text(encoding="utf-8"))
    train_scenarios = {json.loads(line)["scenario_id"]
                       for line in (shadow.TEST_FILE.parent / "train.jsonl").read_text(encoding="utf-8").splitlines()}
    assert manifest["counts"]["test"]["rows"] == 50
    assert not {c["scenario_id"] for c in cases} & train_scenarios


def test_build_prompt_reproduces_the_training_prompt_exactly():
    for c in shadow.load_test_cases():
        assert shadow.build_prompt(c["header"], c["context"], c["previous"], c["child"]) == c["prompt"]


def test_session_prompt_has_the_training_layout_and_never_names_internal_ids():
    header = shadow.load_test_cases()[0]["header"]
    p = shadow.session_prompt(header, "self_harm_language", "Hi there.", "idk")
    parts = shadow.split_prompt(p)
    assert parts["previous"] == "Hi there." and parts["child"] == "idk"
    assert "self harm language" in parts["context"] and "_" not in parts["context"]


# ---------------------------------------------------------------- flags and the measured limitations
def test_trusted_adult_word_list_matches_the_one_the_comparison_used():
    import inquire_finetune as ft
    assert shadow.TRUSTED_ADULT.pattern == ft.TRUSTED_ADULT.pattern
    assert shadow.TRUSTED_ADULT.flags == ft.TRUSTED_ADULT.flags


def test_flags_and_closing_sentence():
    f = shadow.flags("It's okay to feel annoyed. Would you rather take a short break or talk to a parent?")
    assert f["trusted_adult"] and f["closing"].startswith("Would you rather")
    assert not shadow.flags("That sounds hard. What happened next?")["trusted_adult"]


def test_the_page_figures_equal_the_saved_comparison():
    fig = shadow.evaluated_figures()
    summary = json.loads((shadow.EVALUATED_DIR / "summary.json").read_text(encoding="utf-8"))
    for arm in ("control", "adapted"):
        saved = summary["arms"][arm]
        assert fig[arm]["trusted_adult"] == saved["trusted_adult"]["mentions"]
        assert fig[arm]["top_closing_count"] == saved["most_repeated_closing"]["count"]
        assert fig[arm]["rubric_overall"] == saved["rubric"]["overall"]
    assert (fig["adapted"]["trusted_adult"], fig["adapted"]["top_closing_count"]) == (43, 16)


# ---------------------------------------------------------------- replay
def test_replay_covers_every_held_out_case_for_both_arms():
    replay = shadow.load_replay(shadow.EVALUATED_DIR)
    ids = {c["example_id"] for c in shadow.load_test_cases()}
    assert set(replay["control"]) == ids and set(replay["adapted"]) == ids
    assert all(v["response"] for v in replay["adapted"].values())


def test_served_dir_picks_the_newest_complete_folder(tmp_path):
    assert shadow.served_dir(tmp_path) is None
    for name, files in (("inquire_served_2026-10-18", ("raw_outputs_control.jsonl", "raw_outputs_adapted.jsonl")),
                        ("inquire_served_2026-10-19", ("raw_outputs_control.jsonl",))):   # incomplete: ignored
        (tmp_path / name).mkdir()
        for f in files:
            (tmp_path / name / f).write_text("", encoding="utf-8")
    assert shadow.served_dir(tmp_path).name == "inquire_served_2026-10-18"
