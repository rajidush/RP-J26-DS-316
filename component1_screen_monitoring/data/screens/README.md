# Screen-style dataset (Component 1)

**Purpose.** The violence classifier (ViT fine-tuned on Real Life Violence
Situations video frames) scores ≈0.001 on a violent clip playing inside a
browser window: it has only ever seen full-frame video stills, never screen
layouts. This small dataset of *screen-style* images measures that gap
(whole-screen vs. region inference, `src/eval_screens.py`) and can later be
used for fine-tuning.

**Images are not committed or redistributed** — only the code, this README,
`urls.txt` and the CSV source logs are in git. Anyone can rebuild the dataset
with the commands below.

## Sources

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

Fill in after building (`ls <folder> | wc -l`, or the counts printed by
`eval_screens.py`):

| Folder | Images |
|---|---|
| scraped_pages | _TBD_ |
| composite/violence | _TBD_ |
| composite/non_violence | _TBD_ |
| recorded/violence | _TBD_ |
| recorded/safe | _TBD_ |

## Ethics and licensing

- **No violent or graphic content is scraped from the web.** Only the
  ordinary pages in `urls.txt` are visited. Every violent image in this
  dataset comes from the RLVS frames already in `data/raw/violence/`.
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
