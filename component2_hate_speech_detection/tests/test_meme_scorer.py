"""Meme scorer runtime (SO3, FR5, NFR8) and its wiring as an evidence-only signal (I-1).

A tiny ONNX graph and tokenizer stand in for the exported model, so this runs
fast and offline; real-model parity is in tools/export_meme_onnx.py.
"""
import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers

from component2_hate_speech_detection.src.agent.runtime import Agent, AgentConfig
from component2_hate_speech_detection.src.engine.analyzer import Analyzer
from component2_hate_speech_detection.src.engine.fusion import Signal
from component2_hate_speech_detection.src.engine.meme_scorer import MemeScorer, preprocess
from component2_hate_speech_detection.src.readers.screen import ScreenReader

MEAN, STD = [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]


def fake_model_dir(tmp, logit=2.0, temperature=2.0, calibrated=False):
    """pixels, ids, mask -> a constant logit (pixels * 0 + logit)."""
    nodes = [
        helper.make_node("ReduceMean", ["pixel_values"], ["m"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Constant", [], ["zero"], value=helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])),
        helper.make_node("Constant", [], ["c"], value=helper.make_tensor("c", TensorProto.FLOAT, [], [logit])),
        helper.make_node("Mul", ["m", "zero"], ["z"]),
        helper.make_node("Add", ["z", "c"], ["logit"]),
    ]
    graph = helper.make_graph(nodes, "fake_meme", [
        helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["n", 3, 8, 8]),
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["n", "s"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["n", "s"]),
    ], [helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["n"])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # readable by any onnxruntime the app supports
    onnx.save(model, str(tmp / "model.onnx"))
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(tmp / "tokenizer.json"))
    (tmp / "meme.json").write_text(json.dumps({
        "temperature": temperature, "max_length": 77, "calibrated": calibrated,
        "image": {"size": 8, "mean": MEAN, "std": STD}}), encoding="utf-8")
    return tmp


def test_preprocess_floors_the_long_side_and_centre_crops_like_clip():
    image = Image.linear_gradient("L").convert("RGB").resize((362, 400))
    expected = image.resize((224, 247), Image.BICUBIC).crop((0, 11, 224, 235))
    expected = (np.asarray(expected, dtype=np.float32) / 255.0 - 0.5) / 0.25
    assert np.allclose(preprocess(image, 224, MEAN, STD), expected.transpose(2, 0, 1), atol=1e-6)


def test_score_is_temperature_scaled_and_carries_the_gate_verdict(tmp_path):
    scorer = MemeScorer(root=fake_model_dir(tmp_path, logit=2.0, temperature=2.0, calibrated=False))
    signal = scorer.score(Image.new("RGB", (40, 30), "white"), "hello there")
    assert signal.name == "vision" and signal.score == pytest.approx(1 / (1 + np.exp(-1.0)), abs=1e-4)
    assert signal.calibrated is False and signal.producer.endswith("meme-clip-b32-align")


def test_a_model_that_passed_the_gate_may_fuse(tmp_path):
    scorer = MemeScorer(root=fake_model_dir(tmp_path, calibrated=True))
    assert scorer.score(Image.new("RGB", (8, 8)), "").calibrated is True


def test_missing_model_returns_no_signal_and_never_raises(tmp_path):
    scorer = MemeScorer(root=tmp_path / "absent")
    assert not scorer.available
    assert scorer.score(Image.new("RGB", (8, 8)), "x") is None and scorer.last_error
    assert scorer.score(None, "x") is None


class OneLineOcr:
    name = "fake"

    def read(self, image):
        return "we learned about history today", [
            {"box": [0.1, 0.1, 0.9, 0.2], "text": "we learned about history today", "conf": 0.9}]


class FixedMeme:
    """Stands in for MemeScorer: a confident, uncalibrated meme score."""

    name = "fake-meme"
    available = True

    def __init__(self, score=0.9, calibrated=False):
        self.p, self.calibrated = score, calibrated
        self.captions = []

    def probability(self, image, caption=""):
        self.captions.append(caption)
        return self.p


def agent_with(lexicon_scorer, meme):
    return Agent(config=AgentConfig(age=10, audio=False, screen=False), analyzer=Analyzer(scorer=lexicon_scorer),
                 screen_reader=ScreenReader(OneLineOcr()), meme_scorer=meme)


def test_dropped_meme_is_scored_with_its_ocr_text_as_the_caption(lexicon_scorer):
    meme = FixedMeme()
    verdict = agent_with(lexicon_scorer, meme).check_image(Image.new("RGB", (400, 300), "white"))
    assert meme.captions == ["we learned about history today"]
    vision = [e for e in verdict.evidence if e.source == "vision"]
    assert len(vision) == 1 and vision[0].score == 0.9 and vision[0].rule_id == "uncalibrated:evidence_only"
    assert "meme_ms" in verdict.latency_ms


def test_uncalibrated_meme_score_cannot_raise_a_benign_verdict(lexicon_scorer):
    verdict = agent_with(lexicon_scorer, FixedMeme(score=0.99)).check_image(Image.new("RGB", (400, 300)))
    assert verdict.rung == "L0" and verdict.p < 0.35


def test_meme_evidence_never_reaches_the_c3_payload_as_a_rule(lexicon_scorer):
    class AbuseOcr(OneLineOcr):
        def read(self, image):
            return "you should kys", [{"box": [0.1, 0.1, 0.5, 0.2], "text": "you should kys", "conf": 0.9}]

    agent = Agent(config=AgentConfig(age=10, audio=False, screen=False), analyzer=Analyzer(scorer=lexicon_scorer),
                  screen_reader=ScreenReader(AbuseOcr()), meme_scorer=FixedMeme())
    verdict = agent.check_image(Image.new("RGB", (400, 300)))
    payload = verdict.to_trigger_payload()
    assert payload is not None
    assert "uncalibrated:evidence_only" not in payload["context_metadata"]["matched_families"]
    assert "not calibrated" in verdict.explanation


def test_no_meme_model_still_checks_the_image_text(lexicon_scorer, tmp_path):
    agent = agent_with(lexicon_scorer, MemeScorer(root=tmp_path / "absent"))
    verdict = agent.check_image(Image.new("RGB", (400, 300)))
    assert not [e for e in verdict.evidence if e.source == "vision"]
    assert agent.snapshot()["sources"]["meme"] == "off"
