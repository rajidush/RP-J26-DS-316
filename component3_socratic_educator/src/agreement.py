"""
Agreement between independent raters on the items they both rated (spec story 48):
percent agreement, prevalence (so a high agreement on a rare value is visible),
Krippendorff's alpha for nominal labels and ordinal scores, and Gwet's AC1 for pass/fail items,
whose rare failures make kappa low or undefined despite high agreement. A
missing rating (a skip) is None and never counts as a judgement.

Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability. Annenberg
School for Communication, University of Pennsylvania.
Gwet, K. L. (2008). Computing inter-rater reliability and its variance in the
presence of high agreement. British Journal of Mathematical and Statistical
Psychology, 61(1), 29-48.
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


def _coincidences(units: Sequence[Sequence[Value]]) -> tuple[Counter, Counter, float]:
    """Krippendorff's coincidence matrix o_ck over the pairable units (two or more ratings),
    each value's total n_c, and n, the number of pairable values."""
    coincidence: Counter = Counter()
    for unit in units:
        rated = [v for v in unit if v is not None]
        if len(rated) < 2:
            continue
        for i, a in enumerate(rated):
            for j, b in enumerate(rated):
                if i != j:
                    coincidence[a, b] += 1 / (len(rated) - 1)
    totals: Counter = Counter()
    for (a, _), weight in coincidence.items():
        totals[a] += weight
    return coincidence, totals, sum(coincidence.values())


def krippendorff_alpha_nominal(units: Sequence[Sequence[Value]]) -> Optional[float]:
    """Alpha over units (one list of ratings per item, any number of raters, None = missing).
    Only units with two or more ratings are pairable. None when nothing is pairable or every
    rating has one value, since then disagreement expected by chance is zero."""
    coincidence, totals, n = _coincidences(units)
    if n <= 1:
        return None
    expected = (n * n - sum(t * t for t in totals.values())) / (n * (n - 1))
    if expected == 0:
        return None
    observed = sum(w for (a, b), w in coincidence.items() if a != b) / n
    return 1 - observed / expected


def gwet_ac1(pairs: Sequence[tuple[Value, Value]], categories: Sequence[Hashable]) -> Optional[float]:
    """AC1 for two raters over the pairs where neither rating is missing, on the item's q possible
    categories (Gwet 2008): chance agreement is the sum of pi_k(1 - pi_k) over q - 1, where pi_k is
    category k's share of both raters' ratings. None when no pair is rated."""
    rated = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not rated:
        return None
    observed = sum(a == b for a, b in rated) / len(rated)
    shares = Counter(v for pair in rated for v in pair)
    expected = sum(shares[k] / (2 * len(rated)) * (1 - shares[k] / (2 * len(rated))) for k in categories)
    expected /= len(categories) - 1
    return (observed - expected) / (1 - expected)


def krippendorff_alpha_ordinal(units: Sequence[Sequence[Value]], order: Sequence[Hashable]) -> Optional[float]:
    """Alpha with the ordinal metric, for scores such as 0/1/2 whose values are ranked by `order`:
    the distance between c and k grows with how many pairable values lie between them,
    delta^2 = (sum of n_g for g from c to k - (n_c + n_k) / 2)^2 (Krippendorff 2011).
    None when nothing is pairable or every rating has one value."""
    coincidence, totals, n = _coincidences(units)
    if n <= 1:
        return None
    rank = {v: i for i, v in enumerate(order)}

    def delta2(c, k):
        low, high = sorted((rank[c], rank[k]))
        return (sum(totals[g] for g in order[low:high + 1]) - (totals[c] + totals[k]) / 2) ** 2

    expected = sum(totals[c] * totals[k] * delta2(c, k) for c in totals for k in totals)
    if expected == 0:
        return None
    observed = sum(weight * delta2(c, k) for (c, k), weight in coincidence.items())
    return 1 - (n - 1) * observed / expected
