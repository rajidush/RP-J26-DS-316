"""
Human review of the meme worksheet in the browser (CODEBOOK section 6, steps 3-4).

Serves a page on 127.0.0.1 only. Images are read from data/hateful_memes/img and never
leave the machine; every save goes straight into the worksheet CSV.

    # Verification pass: see the AI draft, accept or correct it -> final_* in annotation_dev_seen.csv
    python -m component2_hate_speech_detection.annotation.review --annotator IT23209152

    # Blind pass: a label-stratified sample with the AI draft hidden, saved to its own sheet.
    # Do this BEFORE the verification pass, so the draft cannot anchor your judgement.
    python -m component2_hate_speech_detection.annotation.review --annotator IT23209152 --blind
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import threading
import webbrowser
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from component2_hate_speech_detection.annotation.agreement import DIMENSIONS
from component2_hate_speech_detection.annotation.build_labels import (
    ANNOTATION_SPLIT,
    COMPONENT_ROOT,
    DEFAULT_OUT_DIR,
    WORKSHEET_COLUMNS,
)

IMG_DIR = COMPONENT_ROOT / "data" / "hateful_memes" / "img"
UI_PATH = Path(__file__).with_name("review_ui.html")
IMG_NAME = re.compile(r"^\d{5}\.png$")
BLIND_SAMPLE_SIZE = 100  # 20 % of dev_seen, the codebook's minimum for a reliability subset
BLIND_SEED = 316
MAX_BODY = 16_384


def read_sheet(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_sheet(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WORKSHEET_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)  # a crash mid-write never leaves a half-written worksheet


def blind_sample(rows: list[dict], size: int = BLIND_SAMPLE_SIZE, seed: int = BLIND_SEED) -> list[dict]:
    """Equal hateful / not_hateful draw with the AI draft and any human work stripped."""
    rng = random.Random(seed)
    picked = []
    for label in ("hateful", "not_hateful"):
        pool = sorted((r for r in rows if r["hateful"] == label), key=lambda r: int(r["id"]))
        picked += rng.sample(pool, min(size // 2, len(pool)))
    blank = {c: "" for c in WORKSHEET_COLUMNS if c.startswith(("draft_", "final_"))}
    blank.update(drafted_by="", annotator="", date="", notes="")
    return [dict(r, **blank) for r in sorted(picked, key=lambda r: int(r["id"]))]


def validate(update: dict) -> list[str]:
    """Allowed values per CODEBOOK; severity 0 if and only if there is no harm locus."""
    errors = []
    for dim, (allowed, _) in DIMENSIONS.items():
        value = update.get(f"final_{dim}", "")
        if value not in allowed:
            errors.append(f"{dim}: choose one of {', '.join(allowed)}")
    sev, locus = update.get("final_child_severity"), update.get("final_harm_locus")
    if sev in DIMENSIONS["child_severity"][0] and locus in DIMENSIONS["harm_locus"][0]:
        if (sev == "0") != (locus == "none"):
            errors.append("severity 0 goes with harm locus 'none' (and only with it)")
    return errors


class Store:
    """The worksheet in memory, saved to disk after every change."""

    def __init__(self, path: Path, annotator: str, blind: bool):
        self.path, self.annotator, self.blind = path, annotator, blind
        self.rows = read_sheet(path)
        self.by_id = {r["id"]: r for r in self.rows}
        self.lock = threading.Lock()

    def public_rows(self) -> list[dict]:
        hidden = ("draft_", "drafted_by") if self.blind else ()
        return [{k: v for k, v in r.items() if not (hidden and k.startswith(hidden))} for r in self.rows]

    def save(self, update: dict) -> Optional[list[str]]:
        row = self.by_id.get(str(update.get("id", "")))
        if row is None:
            return ["unknown meme id"]
        errors = validate(update)
        if errors:
            return errors
        with self.lock:
            for dim in DIMENSIONS:
                row[f"final_{dim}"] = update[f"final_{dim}"]
            row["notes"] = str(update.get("notes", ""))[:500]
            row["annotator"], row["date"] = self.annotator, date.today().isoformat()
            write_sheet(self.path, self.rows)
        return None


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(HTTPStatus.OK, UI_PATH.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(HTTPStatus.OK, {"annotator": store.annotator, "blind": store.blind,
                                           "rows": store.public_rows()})
            elif self.path.startswith("/img/") and IMG_NAME.match(self.path[5:]):
                image = IMG_DIR / self.path[5:]
                if image.exists():
                    self._send(HTTPStatus.OK, image.read_bytes(), "image/png")
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "image not found"})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            if self.path != "/api/save" or not 0 < length <= MAX_BODY:
                self._json(HTTPStatus.BAD_REQUEST, {"errors": ["bad request"]})
                return
            try:
                update = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                self._json(HTTPStatus.BAD_REQUEST, {"errors": ["invalid JSON"]})
                return
            errors = store.save(update)
            self._json(HTTPStatus.BAD_REQUEST if errors else HTTPStatus.OK, {"errors": errors or []})

    return Handler


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotator", required=True, help="who is labelling, e.g. a student ID")
    parser.add_argument("--blind", action="store_true", help="reliability pass: hide the AI draft")
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    main_sheet = args.labels_dir / f"annotation_{ANNOTATION_SPLIT}.csv"
    path = main_sheet
    if args.blind:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", args.annotator)
        path = args.labels_dir / f"blind_{ANNOTATION_SPLIT}_{safe}.csv"
        if not path.exists():  # created once; later runs continue the same sample
            write_sheet(path, blind_sample(read_sheet(main_sheet)))

    store = Store(path, args.annotator, args.blind)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{'BLIND ' if args.blind else ''}review of {len(store.rows)} memes -> {path}\nopen {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
