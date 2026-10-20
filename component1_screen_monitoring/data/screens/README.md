# Screen-style dataset (Component 1)

**Purpose.** The violence classifier (ViT fine-tuned on Real Life Violence
Situations video frames) scores ≈0.001 on a violent clip playing inside a
browser window: it has only ever seen full-frame video stills, never screen
layouts. This small dataset of *screen-style* images measures that gap
(whole-screen vs. region inference, `src/eval_screens.py`) and can later be
used for fine-tuning.

**Images are not committed or redistributed** — only the code, this README,
the URL lists and the CSV source logs are in git. Anyone can rebuild the
dataset with the commands below.

## Train vs test

There are two strictly separate sets. **The TEST set is frozen**: it is not
modified, rebuilt or reused for training, and `results/screen_eval.*` stays
comparable.

| | TEST (evaluation) | TRAIN (fine-tuning) |
|---|---|---|
| URL list | `urls.txt` (41 URLs) | `urls_train.txt` (192 URLs, 174 websites) |
| Page screenshots | `scraped_pages/` → `sources.csv` | `train_pages/` → `train_sources.csv` |
| Composites | `composite/<class>/` → `composites.csv` | `train_composites/composites/<class>/` |
| Tile crops | — (cut at eval time) | `train_composites/tiles/<class>/` |
| Index | `composites.csv` | `screen_train.csv` (with the train/val split) |
| RLVS frames | `data/raw/` (300 frames) | RLVS **train** split only (`results/splits.csv`), never near a test frame |

`python src/check_leakage.py` verifies the two share nothing (exit code 1
otherwise; report in `results/screen_train_leakage.json`):

- **URLs** — no shared URL and no shared website (registrable domain, so
  `docs.x.com` vs `x.com` counts as shared), in the lists and in what was
  actually scraped. The scraper itself refuses to scrape a non-test list that
  overlaps `urls.txt` (`src/dataset_checks.py`), or to write into the test
  folders.
- **Pages** — no shared page id, no byte-identical screenshot, every training
  background comes from `train_pages/`.
- **RLVS** — no shared frame (by name and by content), no shared approximate
  video group, no training frame flagged `near_test`, every training frame from
  the RLVS train split.
- **Train/val** — no page, website or video group in both.

### RLVS video groups (approximate)

The RLVS frames used here carry **no video id** (files are numbered per class
in shuffled order, and `results/splits.csv` was a frame-level split).
`src/rlvs_groups.py` approximates source videos by linking same-class frames
whose ViT CLS features (cosine ≥ 0.97) or 24×24 grayscale thumbnails
(cosine ≥ 0.85) are near-identical; `video_group` = connected component →
`rlvs_frames.csv` (11 073 frames, 5 335 groups). Grouped frames were checked
visually to be the same video, but two frames of one video that look
different can end up in different groups — the grouping is an approximation.

To compensate, a frame is **`near_test`** (never used for training) if it is
one of the 300 test frames, in the same group as one, or loosely similar to
any test frame (CLS ≥ 0.93 or thumbnail ≥ 0.75, any class). This is
deliberately over-inclusive: it removes 5 291 of 11 073 frames. Left for
training: 1 941 violence frames (1 727 groups) and 2 082 non-violence
frames (1 307 groups) from the RLVS train split.

> **Note on the TEST set.** The 300 RLVS frames in `data/raw/` (used by the
> test composites) come from all three RLVS splits: 203 train, 48 val,
> 49 test (278 / 68 / 70 of the 416 composites). So the ViT checkpoint
> already saw ~68 % of the test frames as full-frame images during its
> original training. The screen test still measures the *layout* gap (the
> frames are small and embedded in a page), but this should be stated when
> reporting results.

### Building the training set

```bash
# 1. scrape (robots.txt, 2 s delay, logged); refuses if urls_train.txt overlaps urls.txt
python src/scrape_screenshots.py --urls data/screens/urls_train.txt \
    --out data/screens/train_pages --log data/screens/train_sources.csv
# 2. manually review train_pages/ (bot walls, violent imagery) — log removals
# 3. approximate RLVS video groups + near_test flags (ViT features cached in .cache/)
python src/rlvs_groups.py
# 4. composites + tiles + screen_train.csv
python src/make_composites.py --pages data/screens/train_pages --frames-split train \
    --out data/screens/train_composites --n 750 --seed 7
# 5. leakage check
python src/check_leakage.py
```

`make_composites.py` in training mode:

1. **Splits before compositing**: websites 85/15 and video groups 85/15 (per
   class) into train/val, then pairs train pages only with train frames and
   val pages only with val frames. No page, website or video group can end up
   in both splits.
2. Builds `--n` composites per class with the same layouts and generator as
   the test set (`video_player`, `article_image`, `feed_post`), each frame
   used once.
3. Cuts the same 5 tiles as region inference (`src/regions.py`: 2×2 grid at
   60 % + centre), saved at half size. Tile label = the share of the pasted
   frame inside the tile: ≥ 60 % → violence, ≤ 2 % → safe, in between →
   skipped (ambiguous). Every tile of a safe composite is safe.

**`screen_train.csv`** columns: `path` (relative to `data/screens/`),
`label` (1 = violence, 0 = safe), `group` (`<page id>|<video group>`),
`size_bucket` (pasted frame width / page width: `under30`, `30-45`,
`45plus` — the same buckets as `screen_eval`), `is_tile`, `split`
(train/val), plus `page_id`, `site`, `video_group`, `source_frame`, `layout`,
`tile`. When training, keep composites and tiles of the same `group` together
(the `split` column already does this).

Review of `train_pages/`: 448 screenshots from 169 URLs (23 URLs skipped by
robots.txt or errors). 7 removed (bot-check / access-denied / 403 / 404 /
CAPTCHA pages and one blank page), logged as `removed_after_review` in
`train_sources.csv`; no violent imagery found. 441 kept.

## Sources (TEST set)

| Folder | Class | How it is made | Log |
|---|---|---|---|
| `scraped_pages/<sourceid>_<n>.png` | safe | Headless-Chromium screenshots (1280×800) of the ordinary pages in `urls.txt`: top of page + up to 2 scrolled views | `sources.csv` |
| `composite/violence/<sourceid>_<n>.png` | violence | An RLVS **violence** frame from `data/raw/violence/` pasted into a scraped page | `composites.csv` |
| `composite/non_violence/<sourceid>_<n>.png` | safe | An RLVS **non-violence** frame pasted the same way (hard negatives: a video is on screen, but it is not violent) | `composites.csv` |
| `recorded/<label>/<session>_<n>.png` | violence / safe | Real screen recordings (`src/record_screens.py`) while RLVS clips play in a browser or media player at different window sizes, or during ordinary use | `recorded.csv` |

`<sourceid>` (e.g. `en-wikipedia-org-3fa2c1`, host + hash of the URL) is shared
by every screenshot and composite derived from the same page, so train/test
splits can be **grouped by page** later (no page appears in both).

## Method

1. `python src/scrape_screenshots.py` — reads `urls.txt`, checks robots.txt,
   loads each page (2 s delay), clicks an obvious cookie "Accept" button if
   present, saves the viewport + 2 scrolled screenshots, logs every attempt
   (`filename, url, timestamp, status`) to `sources.csv`.
2. **Manually review `scraped_pages/`** and delete any screenshot that
   happens to contain violent or disturbing imagery (news and video-site home
   pages change daily). Note deletions here.
3. `python src/make_composites.py [--per-background 1] [--seed 42]` — for
   every page screenshot and each class, pastes one RLVS frame as a
   `video_player` (16:9, 20–60 % of the page width, letterboxed or filled,
   fake control bar 60 % of the time), `article_image` (4:3 / 3:2, 25–45 %)
   or `feed_post` (social-feed card). Reproducible for a given seed. Each
   composite's background, source frame, layout and box are logged in
   `composites.csv`.
4. `python src/record_screens.py --label violence|safe --interval 1 --max N`
   — optional real recordings (the only tool that writes screen frames to disk).
5. `python src/eval_screens.py` → `results/screen_eval.{json,md}` and
   `results/screen_eval_predictions.csv`.

All commands from `component1_screen_monitoring/`.

## Counts

TEST set:

| Folder | Images |
|---|---|
| scraped_pages | 104 |
| composite/violence | 208 |
| composite/non_violence | 208 |
| recorded/violence | 0 |
| recorded/safe | 0 |

TRAIN set (`screen_train.csv`, seed 7; 161 pages from 144 websites, 1 500
frames from 1 267 video groups):

| | split | under30 | 30-45 | 45plus | total |
|---|---|---|---|---|---|
| composites, violence | train | 257 | 307 | 74 | 638 |
| composites, safe | train | 259 | 309 | 70 | 638 |
| composites, violence | val | 32 | 67 | 13 | 112 |
| composites, safe | val | 39 | 59 | 14 | 112 |
| tiles, violence | train | 587 | 715 | 164 | 1 466 |
| tiles, safe | train | 1 511 | 1 578 | 350 | 3 439 |
| tiles, violence | val | 82 | 160 | 27 | 269 |
| tiles, safe | val | 216 | 299 | 70 | 585 |

Val = 14.9 % of composites (22 websites, 26 pages, 192 video groups).
Size mix matches the test set (train 39 / 49 / 11 %, test 37 / 49 / 14 %).
Tile size bucket = the bucket of the composite it was cut from.

## Ethics and licensing

- **No violent or graphic content is scraped from the web.** Only the
  ordinary pages in `urls.txt` / `urls_train.txt` are visited. Every violent
  image in this dataset comes from RLVS frames (`data/raw/violence/` for the
  test set, the RLVS train split for the training set).
- **robots.txt is respected**: checked for every host with the scraper's own
  user agent; disallowed URLs — and hosts whose robots.txt returns 401/403 or
  cannot be fetched — are skipped and logged.
- **Polite, identified access**: user agent
  `RP-J26-DS-316-research-screenshotter/0.1 (university research project; non-commercial; +https://github.com/rajidush/RP-J26-DS-316)`,
  2 s between page loads, only the listed URLs (no crawling), no logins, no
  forms, no paywalled or personal content.
- **No redistribution**: page screenshots contain third-party copyrighted
  material; they are used only for local, non-commercial research
  evaluation and are kept out of git. Only URLs and logs are shared.
- **Screen recordings** must not contain personal data (close email, chats,
  accounts before recording).
- **RLVS**: M. M. Soliman, M. H. Kamal, M. A. El-Massih Nashed, Y. M. Mostafa,
  B. S. Chawky and D. Khattab, "Violence Recognition from Videos using Deep
  Learning Techniques," *Proc. 9th Int. Conf. on Intelligent Computing and
  Information Systems (ICICIS)*, Cairo, 2019, pp. 80–85. Kaggle: "Real Life
  Violence Situations Dataset". Use under the licence stated on the Kaggle
  dataset page — _confirm and record it here_.
  - _To confirm_: the frames in `data/raw/` are 416×416, i.e. they appear to be
    a re-exported version of RLVS (e.g. via Roboflow) rather than frames
    extracted directly from the original videos — record the exact source
    and its licence here.
