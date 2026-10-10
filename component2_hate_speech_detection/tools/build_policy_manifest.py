"""Rebuild packs/policy/manifest.json after editing a policy pack file.

    python -m component2_hate_speech_detection.tools.build_policy_manifest [--version X.Y.Z]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from component2_hate_speech_detection.src.engine.policy import PACK_DIR, PACK_FILES, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", help="pack version (default: keep the current one)")
    args = parser.parse_args()

    manifest_path = PACK_DIR / "manifest.json"
    current = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest = {
        "pack": "c2-policy",
        "version": args.version or current.get("version", "1.0.0"),
        "min_engine": "1.0.0",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": {name: "sha256:" + sha256_file(PACK_DIR / name) for name in PACK_FILES},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
