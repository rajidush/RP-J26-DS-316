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
"""
from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path

from .grammar_decoder import GrammarConstrainedGenerator, delivery_violation
from .schemas import RiskCategory

ATTACKS_FILE = Path(__file__).resolve().parents[1] / "data" / "adversarial_intercept_attacks.json"
SCHEMA_REASONS = ("grammar", "invalid_json")
POST_CHECK_REASONS = ("forbidden_word", "alert_language", "category_named", "invites_description")


def load_attack_set(path=ATTACKS_FILE) -> dict:
    """The attack file: a description (including the future-interface caveat) and the attack templates."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def adversarial_cases(attacks: list[dict], categories=tuple(RiskCategory)) -> list[dict]:
    """Every attack template crossed with every risk category."""
    return [{"attack_id": attack["id"], "style": attack["style"], "risk_category": category,
             "context": attack["template"].format(category=category.value.replace("_", " "))}
            for attack in attacks for category in categories]


def run_adversarial_batch(generator: GrammarConstrainedGenerator, cases: list[dict]) -> dict:
    rows = []
    for case in cases:
        start = time.perf_counter()
        line = generator.generate_with_source("INTERCEPT", case["risk_category"], upstream_context=case["context"])
        rows.append({"attack_id": case["attack_id"], "style": case["style"],
                     "risk_category": case["risk_category"].value, "context": case["context"],
                     "delivered": line.text, "source": line.source, "reason": line.reason,
                     "model_output": line.model_output, "violation": delivery_violation(line.text),
                     "seconds": round(time.perf_counter() - start, 3)})

    def count(reasons):
        return dict(Counter(r["reason"] for r in rows if r["reason"] in reasons))

    total = len(rows)
    violations = sum(r["violation"] is not None for r in rows)
    answered = [r["seconds"] for r in rows if r["reason"] != "model_unreachable"]   # model calls only
    return {
        "total_prompts": total,
        "passed": sum(r["source"] == "model" for r in rows),
        "schema_breaks": count(SCHEMA_REASONS),
        "blocked_by_post_check": count(POST_CHECK_REASONS),
        "unreachable": sum(r["reason"] == "model_unreachable" for r in rows),
        "violations": violations,
        "violation_rate": violations / total if total else 0.0,
        "latency_s": ({"min": min(answered), "median": round(statistics.median(answered), 3), "max": max(answered)}
                      if answered else None),
        "rows": rows,
    }
