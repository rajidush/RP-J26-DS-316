"""Step 9, RQ3: does C2 fit an ordinary laptop, offline?

    python -m component2_hate_speech_detection.evaluation.resources

One process loads the whole C2 agent (keyword layer, four text models, OCR,
meme model, speech-to-text) and runs every input path while it measures:

    memory    resident set size after each phase and the peak (budget: about 2 GB, NFR2)
    speed     typed message (288 child-context texts, no cache hits), a chat
              screenshot read whole and then after one new message, a meme
              image, a spoken sentence (synthesised offline)
    network   internet connections the process holds after each phase (must be 0, NFR4)
    torch     whether PyTorch is in the process after each phase (the shipped engine must not need it)

Writes `results/resources.json` and `results/resources.md`.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import psutil
from PIL import Image

from ..src.agent.runtime import Agent, AgentConfig
from ..src.engine.analyzer import Observation, ScoreCache
from .grooming_set import dev_cases as grooming_dev
from .screen_text import CONTEXT, STYLES, TARGET
from .step9 import child_items

RESULTS = Path(__file__).resolve().parent / "results"
PROC = psutil.Process()


def mb() -> int:
    return round(PROC.memory_info().rss / 2**20)


def connections() -> int:
    return len(PROC.net_connections(kind="inet"))


def pct(values: List[float], q: float) -> float:
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 1)


def main() -> None:
    phases: Dict[str, dict] = {}
    peak = mb()

    def mark(name: str, **extra) -> None:
        nonlocal peak
        peak = max(peak, mb())
        phases[name] = {"rss_mb": mb(), "connections": connections(), "torch_loaded": "torch" in sys.modules, **extra}
        print(f"{name:28s} {phases[name]}", flush=True)

    mark("python started")
    agent = Agent(config=AgentConfig(age=10, audio=False, screen=False))
    agent.analyzer.scorer.warm_up()
    mark("text models loaded")

    # Typed messages: every text scored for real (empty cache each time).
    typed_ms, grooming_topic_ms = [], []
    on_topic = agent.analyzer.scorer.grooming_gate.on_topic
    for item in child_items():
        agent.analyzer.cache = ScoreCache()
        t = time.perf_counter()
        agent.analyzer.analyze_text(item.text, 10)
        ms = (time.perf_counter() - t) * 1000
        typed_ms.append(ms)
        if on_topic(item.text):
            grooming_topic_ms.append(ms)
    mark("typed messages", n=len(typed_ms), median_ms=pct(typed_ms, 0.5), p95_ms=pct(typed_ms, 0.95),
         grooming_topic_median_ms=pct(grooming_topic_ms, 0.5), grooming_topic_n=len(grooming_topic_ms))

    # A chat screenshot: first read is the whole frame, then one new message is a partial read.
    render = STYLES["chat-dark-22"][1]
    before, after = CONTEXT["design"]
    who, when = TARGET["design"]
    reader = agent.screen_reader
    reader.reset()
    t = time.perf_counter()
    first = reader.read(render(before + after))
    agent.analyzer.analyze(Observation(first.spans, source="screen"), 10)
    full_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    nxt = reader.read(render(before + after + [(who, "nobody likes you, just leave the group", when)]))
    agent.analyzer.analyze(Observation(nxt.spans, source="screen"), 10)
    partial_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    same = reader.read(render(before + after + [(who, "nobody likes you, just leave the group", when)]))
    unchanged_ms = (time.perf_counter() - t) * 1000
    mark("chat screen", full_frame_ms=round(full_ms), partial_ms=round(partial_ms), partial_mode=nxt.mode,
         unchanged_ms=round(unchanged_ms, 1), unchanged_mode=same.mode)

    # Images: memes through the app's own path (OCR + text + meme model, evidence only).
    from ..meme_model.embed import IMAGE_ROOT, MANIFEST_DIR, read_manifest
    image_ms = []
    rows = read_manifest(MANIFEST_DIR / "val.jsonl")[:30] if (MANIFEST_DIR / "val.jsonl").exists() else []
    for r in rows:
        with Image.open(IMAGE_ROOT / r["img"]) as img:
            image = img.convert("RGB")
        t = time.perf_counter()
        agent.check_image(image)
        image_ms.append((time.perf_counter() - t) * 1000)
    if image_ms:
        mark("meme images", n=len(image_ms), median_ms=pct(image_ms, 0.5), p95_ms=pct(image_ms, 0.95))

    # Voice: sentences synthesised offline, cut by the voice detector, transcribed, scored.
    from . import tts
    voice_ms = []
    if tts.available():
        # The speech model loads on first use; time that once, apart from the sentences.
        import numpy as np
        t = time.perf_counter()
        agent.speech_reader.asr.transcribe(np.zeros(16000, dtype=np.float32))
        mark("speech model loaded", load_ms=round((time.perf_counter() - t) * 1000))
        with tempfile.TemporaryDirectory() as tmp:
            for i, text in enumerate(["nobody likes you just leave the group", "want to play minecraft after school",
                                      grooming_dev()[0].text]):
                audio = tts.load_wav(tts.speak_to_wav(text, Path(tmp) / f"{i}.wav"))
                t = time.perf_counter()
                agent.check_audio(audio)
                voice_ms.append({"seconds_of_speech": round(len(audio) / 16000, 1),
                                 "ms": round((time.perf_counter() - t) * 1000)})
        mark("voice clips", clips=voice_ms)

    report = {
        "machine": {"cpu": platform.processor(), "cores": psutil.cpu_count(logical=False),
                    "logical": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
                    "os": platform.platform(), "python": platform.python_version()},
        "engine": agent.analyzer.scorer.name, "policy_version": agent.analyzer.policy.version,
        "phases": phases, "peak_rss_mb": peak, "budget_mb": 2048,
        "max_connections": max(p["connections"] for p in phases.values()),
        "torch_imported": any(p["torch_loaded"] for p in phases.values()),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "resources.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    typed, screen = phases["typed messages"], phases["chat screen"]
    lines = [
        "# Step 9 results (RQ3): resources", "",
        f"`evaluation/resources.py` on {report['machine']['cpu']} ({report['machine']['cores']} cores, "
        f"{report['machine']['ram_gb']} GB RAM), CPU only. Engine `{report['engine']}`.", "",
        "| What | Result |", "|---|---|",
        f"| Peak memory, every C2 model loaded | {peak:,} MB (budget about 2,048 MB) |",
        f"| Typed message ({typed['n']} texts) | {typed['median_ms']} ms median, {typed['p95_ms']} ms p95 |",
        f"| ... on a grooming topic ({typed['grooming_topic_n']}) | {typed['grooming_topic_median_ms']} ms median |",
        f"| Chat screenshot, whole frame read | {screen['full_frame_ms']:,} ms |",
        f"| Chat screenshot, one new message ({screen['partial_mode']} read) | {screen['partial_ms']:,} ms |",
        f"| Chat screenshot, nothing changed | {screen['unchanged_ms']} ms ({screen['unchanged_mode']}) |",
    ]
    if image_ms:
        img = phases["meme images"]
        lines.append(f"| Meme image ({img['n']}) | {img['median_ms']} ms median, {img['p95_ms']} ms p95 |")
    if "speech model loaded" in phases:
        lines.append(f"| Speech model load (once, on first speech) | {phases['speech model loaded']['load_ms']:,} ms |")
    for clip in voice_ms:
        lines.append(f"| Spoken sentence, {clip['seconds_of_speech']} s of speech | {clip['ms']:,} ms |")
    lines += [f"| Internet connections held by the process | {report['max_connections']} (after every phase) |",
              f"| PyTorch imported | {'yes' if report['torch_imported'] else 'no'} |"]
    (RESULTS / "resources.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
