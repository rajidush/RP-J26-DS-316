"""Facts about the machine a run was recorded on, shared by the evidence scripts."""
import json
from pathlib import Path

LM_STUDIO_VERSION_FILE = Path.home() / ".lmstudio" / ".internal" / "historical-version-info.json"


def lm_studio_version() -> str:
    """LM Studio's own record of its last app version; not read from the running server."""
    try:
        info = json.loads(LM_STUDIO_VERSION_FILE.read_text(encoding="utf-8"))
        target = next(t for t in info["targetHistories"] if t["target"] == "lmstudio")
        return (f"{target['lastRecordedAppVersion']} build {target['lastRecordedAppBuildVersion']} "
                f"(last recorded app version in {LM_STUDIO_VERSION_FILE.name}, not read from the server)")
    except (OSError, ValueError, KeyError, StopIteration):
        return "not recorded"
