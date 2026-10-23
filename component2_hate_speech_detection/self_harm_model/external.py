"""External test sets for the self-harm specialist (never trained on; the data builder drops their texts).

    openai_moderation   Markov et al., AAAI 2023 (MIT): 1,680 texts. SH = self-harm (label 1); violence
                        without self-harm (V or V2, not SH) and everything else are label 0.
    xstest              Röttger et al., NAACL 2024 (CC BY 4.0): the 250 safe prompts only, which use violent
                        homonyms and figurative language ("How do I kill a Python process?"). All label 0.
    hatecheck           Röttger et al., ACL 2021 (CC BY 4.0): 3,728 cases, none of them self-harm. All label 0;
                        C2 labelled 23 of them self_harm at age 12 on 19 Oct 2026.

Each row: {"text", "label", "group"}. Files are cached in data/self_harm_v2/external/ (git-ignored).
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import urllib.request
from pathlib import Path

from .data import OUT

SOURCES = {
    "openai_moderation": "https://raw.githubusercontent.com/openai/moderation-api-release/main/data/samples-1680.jsonl.gz",
    "xstest": "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
    "hatecheck": "https://raw.githubusercontent.com/paul-rottger/hatecheck-data/main/test_suite_cases.csv",
}


def _fetch(name: str, cache: Path) -> bytes:
    path = cache / SOURCES[name].rsplit("/", 1)[-1]
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCES[name], timeout=120) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def openai_moderation(raw: bytes) -> list[dict]:
    rows = []
    for line in gzip.decompress(raw).decode("utf-8").splitlines():
        r = json.loads(line)
        if r.get("SH") == 1:
            group, label = "self_harm", 1
        elif r.get("V") == 1 or r.get("V2") == 1:
            group, label = "violence", 0
        else:
            group, label = "other", 0
        rows.append({"text": r["prompt"], "label": label, "group": group})
    return rows


def xstest(raw: bytes) -> list[dict]:
    return [{"text": r["prompt"], "label": 0, "group": r["type"]}
            for r in csv.DictReader(io.StringIO(raw.decode("utf-8"))) if r.get("label") == "safe"]


def hatecheck(raw: bytes) -> list[dict]:
    return [{"text": r["test_case"].strip(), "label": 0, "group": r["functionality"]}
            for r in csv.DictReader(io.StringIO(raw.decode("utf-8")))]


def load_all(cache: Path = OUT / "external") -> dict[str, list[dict]]:
    parsers = {"openai_moderation": openai_moderation, "xstest": xstest, "hatecheck": hatecheck}
    return {name: parse(_fetch(name, cache)) for name, parse in parsers.items()}
