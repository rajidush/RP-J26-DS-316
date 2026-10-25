"""
"As served" INQUIRE outputs: the 50 held-out test prompts run through the two f16 GGUFs in LM Studio.

The Oct 10 comparison (evidence/inquire/inquire_comparison_2026-10-10/) came from Colab (fp32, transformers). This
records the same prompts on the files the demo actually serves, so the two sets can be compared and the demo's
replay mode can show either. Same prompts, one user message each, temperature 0, same token limit, no repair.

Needs LM Studio running with both models available. Run from component3_socratic_educator/:
    python -m evidence.capture_inquire_served [--base gemma3-1b-it-base] [--adapted gemma3-1b-inquire-adapted]

Writes evidence/inquire/inquire_served_<date>/ : raw_outputs_control.jsonl, raw_outputs_adapted.jsonl (same fields as the
Colab files) and run_record.json. Never overwrites a recorded run; preserved baseline outputs stay untouched.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from component3_socratic_educator.evidence.run_info import lm_studio_version              # noqa: E402
from component3_socratic_educator.src import shadow                                        # noqa: E402
from component3_socratic_educator.src.model_client import LM_STUDIO_URL, call_local_model  # noqa: E402

MAX_TOKENS = 128          # the Colab run used max_new_tokens=128
TEMPERATURE = 0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_file(model_id):
    """LM Studio's file for a model ID, when it sits under ~/.lmstudio/models/<publisher>/<id>/. None otherwise."""
    for p in (Path.home() / ".lmstudio" / "models").glob(f"*/{model_id}/*.gguf"):
        return p
    return None


def run_arm(cases, arm, model_id):
    rows = []
    for c in cases:
        try:
            text, secs = call_local_model(c["prompt"], timeout=120, max_tokens=MAX_TOKENS,
                                          temperature=TEMPERATURE, model=model_id)
        except Exception as e:
            raise SystemExit(f"{arm}: call failed on {c['example_id']}: {e!r}. Nothing written.") from e
        rows.append({"example_id": c["example_id"], "arm": arm, "model_id": model_id, "response": text.strip(),
                     "latency_seconds": round(secs, 3)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gemma3-1b-it-base")
    ap.add_argument("--adapted", default="gemma3-1b-inquire-adapted")
    args = ap.parse_args(argv)

    out_dir = ROOT / "evidence" / "inquire" / f"inquire_served_{datetime.now().strftime('%Y-%m-%d')}"
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; recorded runs are never overwritten.")

    cases = shadow.load_test_cases()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    # One warm-up call per model so load time is not counted as latency in the rows.
    for model_id in (args.base, args.adapted):
        call_local_model("Say hello.", timeout=180, max_tokens=8, temperature=TEMPERATURE, model=model_id)
    arms = {"control": run_arm(cases, "control", args.base), "adapted": run_arm(cases, "adapted", args.adapted)}

    out_dir.mkdir(parents=True)
    for arm, rows in arms.items():
        (out_dir / f"raw_outputs_{arm}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")

    files = {}
    for arm, model_id in (("control", args.base), ("adapted", args.adapted)):
        p = model_file(model_id)
        files[arm] = {"model_id": model_id, "file": p.name if p else "not recorded",
                      "sha256": sha256(p) if p else "not recorded"}
    record = {
        "started_at": started, "runtime_seconds": round(time.perf_counter() - t0, 1),
        "prompts": {"source": str(shadow.TEST_FILE.relative_to(ROOT)), "count": len(cases),
                    "form": "the full training-format prompt as one user message, no system message"},
        "generation_settings": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                                "other": "server defaults, not set by this script (top_p, top_k, repeat penalty)"},
        "runtime": {"server": LM_STUDIO_URL, "lm_studio": lm_studio_version(), "python": sys.version.split()[0],
                    "machine": platform.platform(), "gguf_precision": "f16", "gpu": "not recorded"},
        "models": files,
        "conversion_record": "see evidence/inquire/gguf_conversion_<date>/conversion_record.json (not read by this script)",
        "comparable_to": "evidence/inquire/inquire_comparison_2026-10-10/ (Colab, fp32, greedy, max_new_tokens 128)",
    }
    (out_dir / "run_record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    for arm, rows in arms.items():
        t = shadow.tally([r["response"] for r in rows])
        print(f"{arm:<8} {len(rows)} rows, trusted adult {t['trusted_adult']}/{t['n']}, "
              f"top closing {t['top_closing_count']}x: {t['top_closing']!r}")
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
