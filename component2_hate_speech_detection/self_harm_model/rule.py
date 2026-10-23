"""The acceptance rule for self-harm v2, written on 23 Oct 2026 before the real training run.

v2 replaces the shipped specialist only if every gated line holds, at each of ages 9, 12 and 15, in the
full engine (engine_eval.py; Colab fp32 first, then confirmed on the laptop's ONNX):

    R1  other-directed held-out lines read as self_harm: at most half the shipped count (age 12),
        and at most 1 of the 5 known film lines
    R2  self-harm sets (dev + held-out) caught as self_harm: not lower than shipped
    R3  self-harm lines in the other-directed held-out set caught: at most 1 fewer than shipped
    R4  benign self-harm-set lines alerted: not higher than shipped
    R5  general 107 cases: no harmful case loses its alert
Then on the laptop: ONNX matches torch (<= 1e-4), the INT8 gate changes no decision (else ship fp32),
and Step 9 is rerun with every changed number listed.

Reported, not gated: other-directed lines still alerted (as threat or anything else). Some of today's
alerts on real threats exist only because the threat was misread as self-harm, so they may drop at
ages 12 and 15; that is a threat-rule gap to fix separately, not a reason to keep the mislabel.
v2_public_only is an ablation and is never shipped.
"""
from __future__ import annotations

AGES = (9, 12, 15)


def _n(fraction: str) -> int:
    return int(fraction.split("/")[0])


def check(report: dict) -> dict:
    """PASS / FAIL per line from an engine_level.json report."""
    s, t = report["shipped"], report["tuned"]
    lines = {}
    od = lambda r, age, key: _n(r["other_directed_heldout"][f"age_{age}"][key])  # noqa: E731
    shipped_od, tuned_od = od(s, 12, "other_read_as_self_harm"), od(t, 12, "other_read_as_self_harm")
    known = _n(t["other_directed_dev"]["age_12"]["other_read_as_self_harm"])
    lines["R1"] = {"ok": tuned_od <= shipped_od / 2 and known <= 1,
                   "detail": f"held-out {shipped_od} -> {tuned_od} (need <= {shipped_od / 2:g}); known film lines {known}/5"}

    def sh(r, age, key):
        return sum(_n(r[part][f"age_{age}"][key]) for part in ("self_harm_dev", "self_harm_heldout"))

    r2 = {age: (sh(s, age, "self_harm_caught"), sh(t, age, "self_harm_caught")) for age in AGES}
    lines["R2"] = {"ok": all(b >= a for a, b in r2.values()), "detail": {f"age_{k}": f"{a} -> {b} of 40" for k, (a, b) in r2.items()}}
    r3 = {age: (od(s, age, "self_harm_caught"), od(t, age, "self_harm_caught")) for age in AGES}
    lines["R3"] = {"ok": all(b >= a - 1 for a, b in r3.values()), "detail": {f"age_{k}": f"{a} -> {b} of 24" for k, (a, b) in r3.items()}}
    r4 = {age: (sh(s, age, "benign_false_alerts"), sh(t, age, "benign_false_alerts")) for age in AGES}
    lines["R4"] = {"ok": all(b <= a for a, b in r4.values()), "detail": {f"age_{k}": f"{a} -> {b} of 33" for k, (a, b) in r4.items()}}
    lost = [c for c in report["general_107_cases"]["changed"]
            if c["harmful"] and c["shipped"][0] != "L0" and c["tuned"][0] == "L0"]
    lines["R5"] = {"ok": not lost, "detail": f"{len(lost)} harmful case(s) lost their alert", "lost": lost}
    lines["verdict"] = "PASS" if all(v["ok"] for v in lines.values() if isinstance(v, dict)) else "FAIL"
    return lines
