"""
dataset_checks.py — Overlap checks between the screen TRAIN and TEST data.

URL level (used by scrape_screenshots.py before it scrapes a train list, and by
check_leakage.py):

    url_overlap(train_urls, test_urls)        → URLs present in both (normalised)
    domain_overlap(train_urls, test_urls)     → websites present in both
    assert_disjoint_url_lists(train_file, test_file)
        raises ValueError on ANY shared URL or shared website (registrable domain)

Normalisation: case-insensitive scheme/host, "www." dropped, trailing "/",
query-less fragments ("#…") and default ports ignored — so
"https://www.X.com/a/" and "http://x.com/a" count as the same URL.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

#: Two-label public suffixes we need to handle (UK / AU / NZ / JP sites in the lists).
_TWO_LABEL_SUFFIXES = {"co.uk", "org.uk", "gov.uk", "ac.uk", "com.au", "co.nz", "co.jp"}


def read_url_list(path: Path) -> list[str]:
    """Non-empty, non-comment lines of a urls file (same rules as the scraper)."""
    out = []
    for line in Path(path).read_text().splitlines():
        url = line.split("#", 1)[0].strip()
        if url:
            out.append(url)
    return out


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower().removeprefix("www.")
    port = f":{parts.port}" if parts.port and parts.port not in (80, 443) else ""
    path = parts.path.rstrip("/") or "/"
    query = f"?{parts.query}" if parts.query else ""
    return f"{host}{port}{path}{query}"


def site_of(url: str) -> str:
    """Registrable domain ('docs.djangoproject.com' → 'djangoproject.com', 'bbc.co.uk' stays)."""
    host = (urlsplit(url.strip()).hostname or "").lower().removeprefix("www.")
    labels = host.split(".")
    n = 3 if ".".join(labels[-2:]) in _TWO_LABEL_SUFFIXES else 2
    return ".".join(labels[-n:])


def url_overlap(train_urls, test_urls) -> set[str]:
    return {normalize_url(u) for u in train_urls} & {normalize_url(u) for u in test_urls}


def domain_overlap(train_urls, test_urls) -> set[str]:
    return {site_of(u) for u in train_urls} & {site_of(u) for u in test_urls}


def assert_disjoint_url_lists(train_file: Path, test_file: Path) -> None:
    """Raise ValueError if the two URL files share any URL or any website."""
    train, test = read_url_list(train_file), read_url_list(test_file)
    urls, sites = url_overlap(train, test), domain_overlap(train, test)
    if urls or sites:
        raise ValueError(
            f"{Path(train_file).name} overlaps the test list {Path(test_file).name}: "
            f"{len(urls)} shared URL(s) {sorted(urls)[:10]}, "
            f"{len(sites)} shared website(s) {sorted(sites)[:10]}")
