"""Training data for self-harm specialist v2: replay, other-directed threats, counterfactual pairs.

    python -m component2_hate_speech_detection.self_harm_model.data            # downloads, builds, prints counts

Label 1 = the writer's own suicidality or self-injury; 0 = anything else.

Sources
-------
    replay   Twitter suicidal intention (laxmimerit, GitHub; one of sentinet's six training sources), both
             labels. It keeps what the model already knows (a guard against forgetting); the model saw these
             rows, so its validation score on them is not a test. Licence not stated: used, never shared.
    threat   Civil Comments (Borkan et al., WWW 2019; CC0), train split, threat >= 0.5, at most 40 words,
             not self-directed. Threats aimed at someone else: a class sentinet never saw as a negative.
    pairs    C2-written counterfactual pairs (Kaushik et al., ICLR 2020): one first-person sentence aimed at
             "myself" (1) or at "you / him / them" (0), plus film commands, pleas, protective lines and
             harmless "myself" sentences (0).

Every candidate row is dropped when it matches a test text (C2's self-harm, other-directed and general
sets, and the external test sets passed in): the same normalised text, or token Jaccard >= 0.8.
Validation is 15 % of each source; a pair's two sentences always land in the same split.
Raw rows are written to data/self_harm_v2/ (git-ignored) and never printed.
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

C2 = Path(__file__).resolve().parents[1]
OUT = C2 / "data" / "self_harm_v2"
SEED = 316
VAL_SHARE = 0.15
REPLAY_URL = "https://raw.githubusercontent.com/laxmimerit/twitter-suicidal-intention-dataset/master/twitter-suicidal_data.csv"
THREAT_MIN, THREAT_MAX_WORDS, THREAT_ROWS = 0.5, 40, 3000
JACCARD_DROP = 0.8
SELF_DIRECTED = re.compile(r"\b(?:myself|kill\s+me|end\s+my\s+life|my\s+own\s+life|suicid\w*|self[\s-]?harm)\b", re.I)

# --- C2-written counterfactual pairs -------------------------------------------------------------------
OPENERS = ["i'm going to", "i'm gonna", "i will", "i'll", "i swear i'll", "i want to", "i'm about to",
           "i promise i'll", "i really want to", "i'm ready to"]
# Acts that read the same aimed at myself or at someone else ("beat myself up" is an idiom, so it is left out).
ACTS = ["kill", "hurt", "shoot", "stab", "strangle", "drown", "cut", "burn", "poison", "hang", "end"]
OTHERS = ["you", "him", "her", "them", "that guy", "your brother", "the whole team", "every one of you",
          "the teacher", "the new kid", "my neighbour", "those idiots"]
TAILS = ["", " tonight", " tomorrow", " after school", " if you tell anyone", " when i find you",
         " for what you did", " right now", " one day", " before the weekend", " this time"]
SELF_TAILS = ["", " tonight", " tomorrow", " after school", " if this keeps happening", " right now",
              " one day", " before the weekend", " this time", " when everyone is asleep"]

FILM = ["drop the {w} or i'll {a}", "one more move and i'll {a}", "if you come closer i'll {a}",
        "put your hands up or i'll {a}", "back off or i'll {a}", "tell them i'm coming to {a}",
        "you have ten seconds before i {a}", "i'm warning you, i'll {a}"]
WEAPONS = ["gun", "knife", "weapon", "bat", "phone"]
FILM_ACTS = ["shoot", "fire", "pull the trigger", "kill her", "kill him", "blow this place up",
             "end this", "take you all down", "shoot the hostage", "make you regret it"]
PLEAS = ["please don't {v} me", "stop, you're going to {v} me", "don't {v} me, i'll do whatever you want",
         "he said he would {v} me", "i'm scared they'll {v} me", "somebody help, he's trying to {v} me",
         "i begged him not to {v} me", "they keep saying they'll {v} me"]
PLEA_VERBS = ["hurt", "kill", "shoot", "hit", "stab", "choke", "beat"]
PROTECT = ["nobody {v}s my {f}", "i won't let anyone {v} my {f}", "anyone who tries to {v} my {f} will regret it",
           "i'd stand in front of anyone who wanted to {v} my {f}", "i'll make sure no one can {v} my {f}"]
PROTECT_VERBS = ["hurt", "touch", "threaten", "bully", "attack"]
FAMILY = ["family", "little brother", "little sister", "mum", "dad", "best friend", "dog", "cousin"]
# "myself" without harm, so the model cannot learn the shortcut "myself -> 1".
BENIGN_SELF = {"i made myself {}": ["a sandwich", "a cup of tea", "some toast", "a playlist", "breakfast"],
               "i taught myself {}": ["guitar", "to code", "to skateboard", "some french", "to juggle"],
               "i bought myself {}": ["new shoes", "a game", "a hoodie", "some sweets", "a book"],
               "i'm proud of myself for {}": ["finishing the race", "passing maths", "trying out", "speaking up"],
               "i told myself to {}": ["keep going", "stay calm", "study harder", "be brave", "go to bed early"],
               "i pushed myself {}": ["at training today", "to finish the run", "in the match", "to read more"]}


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower().replace("’", "'")))


def tokens(text: str) -> frozenset:
    return frozenset(norm(text).split())


def pair_rows(rng: random.Random, n_pairs: int = 900) -> list[dict]:
    """Counterfactual pairs: the same opener + act + tail, aimed at myself (1) or at someone else (0)."""
    combos = list(itertools.product(OPENERS, ACTS, TAILS))
    rng.shuffle(combos)
    rows = []
    for i, (opener, act, tail) in enumerate(combos[:n_pairs]):
        family = f"pair{i}"
        target = rng.choice(OTHERS)
        # "end you" is film English; for the self side the natural phrasing is "end my life".
        self_obj = "my life" if act == "end" else "myself"
        self_tail = rng.choice(SELF_TAILS)
        rows.append({"text": f"{opener} {act} {self_obj}{self_tail}", "label": 1, "source": "pairs",
                     "kind": "self", "family": family})
        rows.append({"text": f"{opener} {act} {target}{tail}", "label": 0, "source": "pairs",
                     "kind": "other", "family": family})
    return rows


def other_rows(rng: random.Random) -> list[dict]:
    """Film commands, pleas and protective lines: violent words, first person, not self-harm."""
    rows = []
    film = [t.format(w=w, a=a) for t in FILM for w in WEAPONS for a in FILM_ACTS if "{w}" in t or w == WEAPONS[0]]
    pleas = [t.format(v=v) for t in PLEAS for v in PLEA_VERBS]
    protect = [t.format(v=v, f=f) for t in PROTECT for v in PROTECT_VERBS for f in FAMILY]
    benign_self = [t.format(x) for t, fills in BENIGN_SELF.items() for x in fills]
    for kind, texts, cap in (("film", film, 300), ("plea", pleas, 56), ("protective", protect, 150),
                             ("benign_self", benign_self, 60)):
        texts = sorted(set(texts))
        rng.shuffle(texts)
        rows += [{"text": t, "label": 0, "source": "pairs", "kind": kind, "family": f"{kind}{i}"}
                 for i, t in enumerate(texts[:cap])]
    return rows


# --- Downloaded sources --------------------------------------------------------------------------------
def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def replay_rows(raw: Optional[bytes] = None) -> list[dict]:
    raw = raw if raw is not None else _get(REPLAY_URL)
    rows, seen = [], set()
    for r in csv.DictReader(io.StringIO(raw.decode("utf-8", errors="replace"))):
        text = re.sub(r"https?://\S+", "", r.get("tweet") or "")
        text = re.sub(r"@\w+", "@user", text).strip()
        key = norm(text)
        if len(key.split()) < 3 or key in seen or str(r.get("intention")).strip() not in ("0", "1"):
            continue
        seen.add(key)
        rows.append({"text": text, "label": int(str(r["intention"]).strip()), "source": "replay", "kind": "tweet",
                     "family": f"tw{len(rows)}"})
    return rows


def threat_rows(rng: random.Random, frame=None) -> list[dict]:
    """Civil Comments threats aimed at someone else. `frame` (a DataFrame with text, threat) skips the download."""
    if frame is None:
        from datasets import load_dataset

        frame = load_dataset("google/civil_comments", split="train").to_pandas()
    keep = frame[frame["threat"] >= THREAT_MIN]
    texts = []
    for text in keep["text"]:
        text = " ".join(str(text).split())
        if 3 <= len(text.split()) <= THREAT_MAX_WORDS and not SELF_DIRECTED.search(text):
            texts.append(text)
    texts = sorted(set(texts))
    rng.shuffle(texts)
    return [{"text": t, "label": 0, "source": "threat", "kind": "civil_comments", "family": f"cc{i}"}
            for i, t in enumerate(texts[:THREAT_ROWS])]


# --- Decontamination and split -------------------------------------------------------------------------
def c2_test_texts() -> list[str]:
    from ..evaluation.dev_set import all_cases as dev_cases
    from ..evaluation.heldout_set import all_cases as heldout_cases
    from ..evaluation.other_directed_set import dev_cases as od_dev, heldout_cases as od_heldout
    from ..evaluation.self_harm_set import dev_cases as sh_dev, heldout_cases as sh_heldout

    return [c.text for c in dev_cases() + heldout_cases() + sh_dev() + sh_heldout() + od_dev() + od_heldout()]


def decontaminate(rows: list[dict], test_texts: Iterable[str]) -> tuple[list[dict], Counter]:
    """Drop rows equal to, or with token Jaccard >= 0.8 against, any test text.

    Prefix filtering (Chaudhuri et al., ICDE 2006): with tokens in one global order, rarest first, two sets
    with Jaccard >= t always share a token within each one's first |S| - ceil(t * |S|) + 1 tokens, so only
    those are indexed and probed, and no near copy is missed.
    """
    test_texts = list(test_texts)
    tests = [tokens(t) for t in test_texts if norm(t)]
    exact = {norm(t) for t in test_texts}
    df = Counter(w for t in tests for w in t)

    def prefix(s: frozenset) -> list[str]:
        ordered = sorted(s, key=lambda w: (df.get(w, 0), w))
        return ordered[:len(s) - math.ceil(JACCARD_DROP * len(s)) + 1]

    index = defaultdict(list)
    for i, t in enumerate(tests):
        for w in prefix(t):
            index[w].append(i)
    kept, dropped = [], Counter()
    for r in rows:
        key, toks = norm(r["text"]), tokens(r["text"])
        if key in exact:
            dropped[f"{r['source']}:exact"] += 1
            continue
        candidates = {i for w in prefix(toks) for i in index.get(w, ())} if toks else set()
        if any(len(toks & tests[i]) / len(toks | tests[i]) >= JACCARD_DROP for i in candidates):
            dropped[f"{r['source']}:near"] += 1
            continue
        kept.append(r)
    return kept, dropped


def split(rows: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """15 % of each source to validation, by family, so a pair never straddles train and val."""
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    train, val = [], []
    for source, items in sorted(by_source.items()):
        families = sorted({r["family"] for r in items})
        rng.shuffle(families)
        held = set(families[:max(1, round(len(families) * VAL_SHARE))])
        for r in items:
            (val if r["family"] in held else train).append(r)
    rng.shuffle(train)
    return train, val


def summarise(rows: list[dict]) -> dict:
    out = defaultdict(lambda: {"rows": 0, "label_1": 0})
    for r in rows:
        k = f"{r['source']}/{r['kind']}"
        out[k]["rows"] += 1
        out[k]["label_1"] += r["label"]
    return {"rows": len(rows), "label_1": sum(r["label"] for r in rows), "by_source": dict(sorted(out.items()))}


def build(extra_test_texts: Iterable[str] = (), out: Path = OUT, replay_raw: Optional[bytes] = None,
          threat_frame=None) -> dict:
    rng = random.Random(SEED)
    candidates = replay_rows(replay_raw) + threat_rows(rng, threat_frame) + pair_rows(rng) + other_rows(rng)
    seen, unique = set(), []
    for r in candidates:  # one copy of each text; the first source wins
        if norm(r["text"]) not in seen:
            seen.add(norm(r["text"]))
            unique.append(r)
    test_texts = c2_test_texts() + list(extra_test_texts)
    kept, dropped = decontaminate(unique, test_texts)
    train, val = split(kept, rng)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    report = {"seed": SEED, "candidates": len(candidates), "after_dedupe": len(unique),
              "test_texts_checked": len(test_texts), "dropped_as_test_overlap": dict(sorted(dropped.items())),
              "train": summarise(train), "val": summarise(val)}
    (out / "data_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def load(name: str, root: Path = OUT) -> list[dict]:
    with (root / f"{name}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    from .external import load_all

    extra = [t for rows in load_all().values() for t in (r["text"] for r in rows)]
    print(json.dumps(build(extra, args.out), indent=2))


if __name__ == "__main__":
    main()
