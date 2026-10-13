"""
scrape_screenshots.py — Screenshot ordinary (safe) web pages for the screen-style dataset.

For each URL in data/screens/urls.txt this loads the page in headless Chromium
(Playwright) and saves a 1280x800 viewport screenshot plus up to 2 scrolled
screenshots:

    data/screens/scraped_pages/<sourceid>_<n>.png      n = 0 (top), 1, 2

These are the SAFE class and the backgrounds for make_composites.py. Every
attempt (saved or skipped) is logged to data/screens/sources.csv.

Etiquette (enforced here)
-------------------------
* robots.txt is checked for every host with our own user agent; disallowed,
  unauthorised (401/403) or unreachable robots.txt → the URL is skipped.
* A clearly identified user agent, a 2 s delay before every page load, no
  logins, no form filling, no link following — only the listed URLs.
* Only ordinary pages are listed. Violent / graphic content is NEVER scraped;
  violent images come only from data/raw/violence (RLVS frames).

Usage
-----
    pip install playwright && playwright install chromium
    python component1_screen_monitoring/src/scrape_screenshots.py            # skip already-captured URLs
    python component1_screen_monitoring/src/scrape_screenshots.py --overwrite
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

try:
    from .config import SCREENS_DIR
except ImportError:
    from config import SCREENS_DIR

USER_AGENT = ("RP-J26-DS-316-research-screenshotter/0.1 "
              "(university research project; non-commercial; "
              "+https://github.com/rajidush/RP-J26-DS-316)")
VIEWPORT = {"width": 1280, "height": 800}
DELAY_S = 2.0
PAGE_TIMEOUT_MS = 30_000
SCROLLED_SHOTS = 2

URLS_FILE = SCREENS_DIR / "urls.txt"
OUT_DIR = SCREENS_DIR / "scraped_pages"
SOURCES_CSV = SCREENS_DIR / "sources.csv"
CSV_FIELDS = ["filename", "url", "timestamp", "status"]

#: Button labels tried (once each, briefly) to dismiss cookie banners.
COOKIE_BUTTON_TEXT = re.compile(
    r"^\s*(accept( all)?( cookies)?|allow all( cookies)?|i agree|agree|got it|ok|okay|"
    r"i accept|yes,? i agree|continue)\s*$", re.I)


# ---------------------------------------------------------------------------
# Helpers (no browser needed — unit-testable)
# ---------------------------------------------------------------------------

def read_urls(path: Path) -> list[str]:
    """Non-empty, non-comment lines; duplicates removed, order kept."""
    seen, urls = set(), []
    for line in path.read_text().splitlines():
        url = line.split("#", 1)[0].strip()
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def source_id(url: str) -> str:
    """
    Stable id for a URL, e.g. 'en-wikipedia-org-3fa2c1' (host + 6-char hash).

    Independent of the URL's position in urls.txt, and has no underscores, so
    '<sourceid>_<n>' splits cleanly for group-aware train/test splits.
    """
    host = urlparse(url).netloc.lower().removeprefix("www.")
    digest = hashlib.sha1(url.encode()).hexdigest()[:6]
    return re.sub(r"[^a-z0-9]+", "-", host).strip("-") + "-" + digest


class RobotsCache:
    """robots.txt per host, fetched once with our user agent (conservative on errors)."""

    def __init__(self, user_agent: str = USER_AGENT) -> None:
        self.user_agent = user_agent
        self._parsers: dict[str, RobotFileParser | str] = {}

    def check(self, url: str) -> str:
        """'allowed', or the reason the URL must be skipped."""
        parts = urlparse(url)
        root = f"{parts.scheme}://{parts.netloc}"
        if root not in self._parsers:
            self._parsers[root] = self._fetch(root)
        rp = self._parsers[root]
        if isinstance(rp, str):
            return rp
        return "allowed" if rp.can_fetch(self.user_agent, url) else "robots_disallowed"

    def _fetch(self, root: str) -> RobotFileParser | str:
        rp = RobotFileParser()
        req = urllib.request.Request(root + "/robots.txt", headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                rp.parse(resp.read().decode("utf-8", "replace").splitlines())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return f"robots_http_{e.code}"         # treat as disallowed
            if 400 <= e.code < 500:
                rp.parse([])                          # no robots.txt → allowed
            else:
                return f"robots_http_{e.code}"
        except Exception as e:                        # timeout, DNS, TLS …
            return f"robots_unreachable: {type(e).__name__}"
        return rp


def append_log(rows: list[dict], path: Path = SOURCES_CSV) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Browser part
# ---------------------------------------------------------------------------

def _dismiss_cookie_banner(page) -> bool:
    """Click an obvious 'accept' button in the page or its iframes. Best effort."""
    for frame in page.frames[:6]:
        try:
            btn = frame.get_by_role("button", name=COOKIE_BUTTON_TEXT).first
            if btn.is_visible(timeout=500):
                btn.click(timeout=1500)
                page.wait_for_timeout(800)
                return True
        except Exception:
            continue
    return False


def capture_page(page, url: str, sid: str, out_dir: Path) -> list[dict]:
    """Load *url* and save top + scrolled screenshots. Returns log rows."""
    page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
    try:
        page.wait_for_load_state("networkidle", timeout=8000)
    except Exception:
        pass                                          # busy pages never go idle; fine
    banner = _dismiss_cookie_banner(page)
    page.wait_for_timeout(1000)

    rows, last_y = [], None
    for n in range(SCROLLED_SHOTS + 1):
        if n:
            page.mouse.wheel(0, VIEWPORT["height"] * 0.9)
            page.wait_for_timeout(1200)
        y = page.evaluate("window.scrollY")
        if y == last_y:                               # page too short to scroll further
            break
        last_y = y
        name = f"{sid}_{n}.png"
        page.screenshot(path=str(out_dir / name))     # viewport only, 1280x800
        rows.append({"filename": name, "url": url, "timestamp": _now(),
                     "status": "ok" + ("; cookie_banner_clicked" if banner and n == 0 else "")})
    return rows


def scrape(urls: list[str], out_dir: Path = OUT_DIR, log_path: Path = SOURCES_CSV,
           overwrite: bool = False) -> None:
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    robots = RobotsCache()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport=VIEWPORT, user_agent=USER_AGENT,
                                      device_scale_factor=1, java_script_enabled=True)
        page = context.new_page()
        for i, url in enumerate(urls):
            sid = source_id(url)
            if not overwrite and (out_dir / f"{sid}_0.png").exists():
                print(f"[{i + 1}/{len(urls)}] {sid}: already captured — skipping")
                continue
            verdict = robots.check(url)
            if verdict != "allowed":
                print(f"[{i + 1}/{len(urls)}] {sid}: SKIP ({verdict}) {url}")
                append_log([{"filename": "", "url": url, "timestamp": _now(), "status": verdict}],
                           log_path)
                continue
            time.sleep(DELAY_S)
            try:
                rows = capture_page(page, url, sid, out_dir)
                print(f"[{i + 1}/{len(urls)}] {sid}: {len(rows)} screenshot(s)  {url}")
            except Exception as e:
                status = "timeout" if "Timeout" in type(e).__name__ else f"error: {type(e).__name__}"
                print(f"[{i + 1}/{len(urls)}] {sid}: {status} {url}")
                rows = [{"filename": "", "url": url, "timestamp": _now(), "status": status}]
            append_log(rows, log_path)
        browser.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--urls", type=Path, default=URLS_FILE)
    ap.add_argument("--overwrite", action="store_true", help="re-capture URLs already on disk")
    args = ap.parse_args()
    scrape(read_urls(args.urls), overwrite=args.overwrite)
    print(f"Log → {SOURCES_CSV}")


if __name__ == "__main__":
    main()
