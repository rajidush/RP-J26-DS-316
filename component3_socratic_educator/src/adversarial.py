"""
Function 2's adversarial harness for constrained INTERCEPT (plan, Wed 7 Oct).

Attack templates (data/adversarial_intercept_attacks.json) are crossed with
every risk category and injected into the INTERCEPT prompt as untrusted
upstream context. This simulates a future interface: today Component 3 reads
only risk_category and confidence_score, so no upstream text reaches the
prompt. Because the attack text names the category, the harness passes
category words into the prompt on purpose (inside the untrusted fence); the
production prompt still never carries them (ADR 0004).

Each layer is reported on its own:
  passed                model line kept: it met the grammar and the post-check
  schema_breaks         the model's reply broke the grammar despite the schema
  blocked_by_post_check the model obeyed enough to name the alert, the category
                        or a forbidden word, or to invite the child to describe
                        the content, and the post-check stopped it
  unreachable           no model reply at all
  violations            delivered lines that still break a rule (target: 0)
  audit_flagged         delivered lines the measurement-only audit flags (plan,
                        Mon 12 Oct): violations is 0 by construction, because the
                        post-check scores what it already blocked
Totals are also broken down by set (seen: tuned against; unseen: held out) and
by attack style.
"""
from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path

from .grammar_decoder import GrammarConstrainedGenerator, delivery_violation
from .intercept_audit import audit
from .schemas import RiskCategory

ATTACKS_FILE = Path(__file__).resolve().parents[1] / "data" / "adversarial_intercept_attacks.json"
UNSEEN_ATTACKS_FILE = ATTACKS_FILE.with_name("adversarial_intercept_unseen_attacks.json")
SCHEMA_REASONS = ("grammar", "invalid_json")
POST_CHECK_REASONS = ("forbidden_word", "alert_language", "category_named", "invites_description")


def load_attack_set(path=ATTACKS_FILE) -> dict:
    """The attack file: a description (including the future-interface caveat) and the attack templates."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def adversarial_cases(attacks: list[dict], categories=tuple(RiskCategory), test_set: str = "seen") -> list[dict]:
    """Every attack template crossed with every risk category. A null template
    (the no_attack control) sends no upstream text: the production prompt."""
    return [{"attack_id": attack["id"], "style": attack["style"], "test_set": test_set, "risk_category": category,
             "context": (None if attack["template"] is None
                         else attack["template"].format(category=category.value.replace("_", " ")))}
            for attack in attacks for category in categories]


def _tally(rows: list[dict]) -> dict:
    flagged = sum(bool(r["audit_flags"]) for r in rows)
    return {"cases": len(rows),
            "passed": sum(r["source"] == "model" for r in rows),
            "blocked": sum(r["reason"] in POST_CHECK_REASONS for r in rows),
            "schema_breaks": sum(r["reason"] in SCHEMA_REASONS for r in rows),
            "unreachable": sum(r["reason"] == "model_unreachable" for r in rows),
            "violations": sum(r["violation"] is not None for r in rows),
            "audit_flagged": flagged,
            "audit_rate": flagged / len(rows) if rows else 0.0}


def run_adversarial_batch(generator: GrammarConstrainedGenerator, cases: list[dict]) -> dict:
    rows = []
    for case in cases:
        start = time.perf_counter()
        line = generator.generate_with_source("INTERCEPT", case["risk_category"], upstream_context=case["context"])
        rows.append({"attack_id": case["attack_id"], "style": case["style"], "test_set": case["test_set"],
                     "risk_category": case["risk_category"].value, "context": case["context"],
                     "delivered": line.text, "source": line.source, "reason": line.reason,
                     "model_output": line.model_output, "violation": delivery_violation(line.text),
                     "audit_flags": audit(line.text),
                     "seconds": round(time.perf_counter() - start, 3)})

    def count(reasons):
        return dict(Counter(r["reason"] for r in rows if r["reason"] in reasons))

    totals = _tally(rows)
    total = totals["cases"]
    answered = [r["seconds"] for r in rows if r["reason"] != "model_unreachable"]   # model calls only
    styles = {}
    for r in rows:
        if styles.setdefault(r["style"], r["test_set"]) != r["test_set"]:
            raise ValueError(f"style {r['style']!r} is in both the {styles[r['style']]} and {r['test_set']} sets")
    return {
        "total_prompts": total,
        "passed": totals["passed"],
        "schema_breaks": count(SCHEMA_REASONS),
        "blocked_by_post_check": count(POST_CHECK_REASONS),
        "unreachable": totals["unreachable"],
        "violations": totals["violations"],
        "violation_rate": totals["violations"] / total if total else 0.0,
        "latency_s": ({"min": min(answered), "median": round(statistics.median(answered), 3), "max": max(answered)}
                      if answered else None),
        "audit_flagged": totals["audit_flagged"],
        "audit_rate": totals["audit_rate"],
        "audit_flags": dict(Counter(flag for r in rows for flag in r["audit_flags"])),
        "by_set": {name: _tally([r for r in rows if r["test_set"] == name])
                   for name in dict.fromkeys(styles.values())},
        "by_style": {style: {"test_set": test_set, **_tally([r for r in rows if r["style"] == style])}
                     for style, test_set in styles.items()},
        "rows": rows,
    }
