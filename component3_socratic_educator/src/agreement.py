"""
Agreement between independent raters on the items they both rated (spec story 48):
percent agreement, prevalence (so a high agreement on a rare value is visible)
and Krippendorff's alpha for nominal labels. A missing rating (a skip) is None
and never counts as a judgement.

Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability. Annenberg
School for Communication, University of Pennsylvania.
"""
from __future__ import annotations

from collections import Counter
from typing import Hashable, Optional, Sequence

Value = Optional[Hashable]


def percent_agreement(pairs: Sequence[tuple[Value, Value]]) -> Optional[float]:
    """The share of pairs with the same value, over pairs where neither rating is missing."""
    rated = [(a, b) for a, b in pairs if a is not None and b is not None]
    return sum(a == b for a, b in rated) / len(rated) if rated else None


def prevalence(pairs: Sequence[tuple[Value, Value]]) -> dict:
    """Each value's share of all ratings in pairs where neither rating is missing."""
    values = Counter(v for a, b in pairs if a is not None and b is not None for v in (a, b))
    total = sum(values.values())
    return {v: n / total for v, n in sorted(values.items(), key=lambda kv: str(kv[0]))}


def krippendorff_alpha_nominal(units: Sequence[Sequence[Value]]) -> Optional[float]:
    """Alpha over units (one list of ratings per item, any number of raters, None = missing).
    Only units with two or more ratings are pairable. None when nothing is pairable or every
    rating has one value, since then disagreement expected by chance is zero."""
    coincidence: Counter = Counter()
    for unit in units:
        rated = [v for v in unit if v is not None]
        if len(rated) < 2:
            continue
        for i, a in enumerate(rated):
            for j, b in enumerate(rated):
                if i != j:
                    coincidence[a, b] += 1 / (len(rated) - 1)
    n = sum(coincidence.values())
    totals: Counter = Counter()
    for (a, _), weight in coincidence.items():
        totals[a] += weight
    if n <= 1:
        return None
    expected = (n * n - sum(t * t for t in totals.values())) / (n * (n - 1))
    if expected == 0:
        return None
    observed = sum(w for (a, b), w in coincidence.items() if a != b) / n
    return 1 - observed / expected
