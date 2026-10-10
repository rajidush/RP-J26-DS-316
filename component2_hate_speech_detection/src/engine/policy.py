"""Policy pack loading (Guardian PROJECT_PLAN §6.7, story GS-011).

The lexicon, framing rules and decision parameters are data, not code: they
live in `packs/policy/*.json` so the same rules can be loaded by this Python
reference engine now and by the C# engine later, and so a rule change is a
reviewable diff to one file.

`manifest.json` records a SHA-256 for every file. Loading verifies each one
and refuses a pack that does not match, so an edited or corrupted rule file
cannot silently change what the engine flags. Ed25519 signing of the
manifest is the next step (Guardian FR-P-03); the hash check is the part the
engine depends on.

Rebuild the manifest after editing a pack file:
    python -m component2_hate_speech_detection.tools.build_policy_manifest
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parents[2] / "packs" / "policy"
PACK_FILES = ("lexicon.json", "framing.json", "decision.json")


class PolicyIntegrityError(RuntimeError):
    """A pack file is missing or does not match its manifest hash."""


@dataclass(frozen=True)
class Policy:
    lexicon: dict
    framing: dict
    decision: dict
    version: str  # manifest version, recorded on every Verdict (FR-C2-14)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy(pack_dir: Path = PACK_DIR) -> Policy:
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        raise PolicyIntegrityError(f"policy manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})

    loaded = {}
    for name in PACK_FILES:
        path = pack_dir / name
        expected = files.get(name, "")
        if not path.exists():
            raise PolicyIntegrityError(f"policy file missing: {name}")
        actual = "sha256:" + sha256_file(path)
        if actual != expected:
            raise PolicyIntegrityError(f"{name} does not match its manifest hash (tampered or not rebuilt)")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))

    return Policy(
        lexicon=loaded["lexicon.json"],
        framing=loaded["framing.json"],
        decision=loaded["decision.json"],
        version=str(manifest.get("version", "0.0.0")),
    )


@lru_cache(maxsize=1)
def default_policy() -> Policy:
    return load_policy()
