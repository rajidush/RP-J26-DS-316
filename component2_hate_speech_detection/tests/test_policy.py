"""Policy pack: loads, matches its manifest, and refuses tampered rules (Guardian GS-011)."""
import json
import shutil

import pytest

from component2_hate_speech_detection.src.engine.policy import (
    PACK_DIR,
    PolicyIntegrityError,
    default_policy,
    load_policy,
)


def test_policy_pack_loads_and_matches_manifest():
    policy = default_policy()
    assert policy.version
    assert {f["id"] for f in policy.lexicon["families"]} >= {"sexual:coercion", "self_harm:intent"}


def test_tampered_policy_pack_is_rejected(tmp_path):
    pack = tmp_path / "policy"
    shutil.copytree(PACK_DIR, pack)
    lexicon = json.loads((pack / "lexicon.json").read_text(encoding="utf-8"))
    lexicon["phrases"]["high"].remove("kys")  # someone quietly weakens the rules
    (pack / "lexicon.json").write_text(json.dumps(lexicon), encoding="utf-8")
    with pytest.raises(PolicyIntegrityError):
        load_policy(pack)


def test_missing_manifest_is_rejected(tmp_path):
    pack = tmp_path / "policy"
    shutil.copytree(PACK_DIR, pack)
    (pack / "manifest.json").unlink()
    with pytest.raises(PolicyIntegrityError):
        load_policy(pack)
