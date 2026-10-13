"""
model_client builds the LM Studio request. urlopen is faked, so these run
without a server; the live plumbing is covered by the live_model tests.
"""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from component3_socratic_educator.src import config, model_client


def _capture(monkeypatch, content="hi"):
    sent = {}

    def fake_urlopen(req, timeout):
        sent["body"], sent["timeout"] = json.loads(req.data), timeout
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())
    monkeypatch.setattr(model_client.urllib.request, "urlopen", fake_urlopen)
    return sent


def test_plain_call_sends_the_configured_model_and_no_response_format(monkeypatch):
    sent = _capture(monkeypatch, "Hello.")
    text, seconds = model_client.call_local_model("Say hello.")
    assert text == "Hello." and seconds >= 0
    assert sent["body"]["model"] == config.LOCAL_MODEL_ID
    assert sent["body"]["messages"] == [{"role": "user", "content": "Say hello."}]
    assert "response_format" not in sent["body"]
    assert sent["timeout"] == 30.0


def test_response_format_is_passed_through_unchanged(monkeypatch):
    sent = _capture(monkeypatch, "{}")
    fmt = {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {"type": "object"}}}
    model_client.call_local_model("p", response_format=fmt, timeout=5)
    assert sent["body"]["response_format"] == fmt and sent["timeout"] == 5
