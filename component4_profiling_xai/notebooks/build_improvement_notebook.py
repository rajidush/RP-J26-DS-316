"""Build and execute the self-contained Step 4A research notebook."""
from pathlib import Path
import contextlib
import io
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md('''# Function 2 — Step 4A model improvement
Research only: production artifacts and Step 3 remain unchanged. Fresh seed 20261020 was generated and hashed before this notebook was developed. Never tune against the old test partition or the fresh holdout. F2 is the sole primary selection objective; ties prefer recall, then fewer false alerts.

Behavioral ground truth is the generator's observed `volume_spike`, `late_night_violence`, and `bullying_run` episodes. Grooming/self-harm episode days are excluded from behavioral metrics (not relabeled normal). All injected days are excluded from novelty-detector fitting. This is an operational definition, not independently annotated behavior. Urgent outputs are evaluated separately against event-level synthetic rule eligibility, not against behavioral labels.

The saved model's fixed configuration is refitted inside each fold. It retains legacy content-risk features as a comparator. Proposed detectors use activity counts and causal within-child changes, with no severity/confidence, urgent, grooming or self-harm inputs. Counts are flagged activity, not total device usage.''')
code('''from pathlib import Path
import hashlib, json, sys, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
ROOT = Path.cwd()
if not (ROOT / 'component4_profiling_xai').exists():
    ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from component4_profiling_xai.src.behavioral_anomaly import feature_matrix
C = ROOT / 'component4_profiling_xai'
OUT = C / 'outputs' / 'step4a_improvement'
OUT.mkdir(parents=True, exist_ok=True)
HOLD = C / 'data' / 'step4a_heldout_seed20261020'
manifest = json.loads((HOLD / 'freeze_manifest.json').read_text())
def verify_frozen():
    for name, digest in {**manifest['heldout'], **manifest['protected']}.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
verify_frozen()
# Read only original development days; original test rows never enter development.
dev = pd.read_csv(C / 'data' / 'behavioral_daily.csv')
dev = dev.loc[dev.day_index <= 73].copy()
# Read configuration keys only; historical metrics are not used for selection.
legacy_config = json.loads((C / 'models' / 'behavioral_anomaly_metadata.json').read_text())
legacy = {k: legacy_config[k] for k in ['features', 'params', 'threshold_quantile']}
del legacy_config
BEHAVIOR = ['volume_spike', 'late_night_violence', 'bullying_run']
HIGH = ['grooming_burst', 'self_harm_day']
FOLDS = [(35, 36, 49), (49, 50, 63), (63, 64, 73)]
QUANTILES = [0.90, 0.95, 0.97, 0.99]
print('Python', platform.python_version(), 'sklearn', sklearn.__version__)
print('Forward folds:', FOLDS, 'primary objective: mean fold F2')''')
code('''def activity_features(d):
    d = d.sort_values(['child_id', 'day_index']).copy()
    x = pd.DataFrame(index=d.index)
    # Remove grooming/self-harm counts from language volume. No labels used.
    language = (d.language_flags - d.language_grooming_flags - d.language_self_harm_flags).clip(lower=0)
    values = {'screen_count': d.screen_incidents,
              'screen_seconds': d.screen_flagged_seconds,
              'screen_night': d.screen_night_incidents,
              'language_count': language}
    for name, value in values.items():
        v = np.log1p(value.astype(float))
        x[name + '_log'] = v
        groups = v.groupby(d.child_id)
        mean = groups.transform(lambda s: s.shift(1).rolling(14, min_periods=7).mean())
        std = groups.transform(lambda s: s.shift(1).rolling(14, min_periods=7).std())
        x[name + '_change'] = (v - mean) / std.clip(lower=0.25)
        recent = groups.transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
        x[name + '_recent_change'] = (recent - mean) / std.clip(lower=0.25)
    return x.reindex(d.index)

def metrics(d, pred):
    y = d.anomaly_type.isin(BEHAVIOR).astype(int).to_numpy()
    pred = np.asarray(pred, dtype=bool)
    fp = int(((y == 0) & pred).sum())
    episodes = d.loc[y == 1, 'episode_id'].nunique()
    detected = d.loc[(y == 1) & pred, 'episode_id'].nunique()
    return dict(f2=fbeta_score(y, pred, beta=2, zero_division=0),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0), false_alerts=fp,
        false_alerts_per_child=fp/d.child_id.nunique(),
        false_alerts_per_100_child_days=100*fp/len(d),
        episode_detection=detected/episodes if episodes else np.nan,
        episodes=episodes, detected_episodes=detected, rows=len(d),
        positive_days=int(y.sum()), alerts=int(pred.sum()))

def fit_score(name, train, target, x):
    normal = train.loc[train.is_anomaly == 0]
    if name == 'legacy_svm':
        model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), OneClassSVM(**legacy['params']))
        a, b = feature_matrix(normal, legacy['features']), feature_matrix(target, legacy['features'])
    elif name == 'activity_iforest':
        model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
            IsolationForest(n_estimators=200, max_samples=256, random_state=1729, n_jobs=1))
        a, b = x.loc[normal.index], x.loc[target.index]
    elif name == 'activity_svm':
        model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), OneClassSVM(nu=.05, gamma='scale'))
        a, b = x.loc[normal.index], x.loc[target.index]
    else:
        # Maximum absolute standardized log-count deviation, with a 3-day persistence channel.
        cols = [c for c in x if c.endswith('_change')]
        score = x[cols].abs().max(axis=1).fillna(0)
        return score.loc[normal.index].to_numpy(), score.loc[target.index].to_numpy()
    model.fit(a)
    return -model.score_samples(a), -model.score_samples(b)

NAMES = ['legacy_svm', 'activity_iforest', 'activity_svm', 'statistical_change']
x = activity_features(dev)
rows = []
for fold, (end, lo, hi) in enumerate(FOLDS, 1):
    train = dev.loc[(dev.day_index >= 15) & (dev.day_index <= end)]
    val = dev.loc[dev.day_index.between(lo, hi) & ~dev.anomaly_type.isin(HIGH)]
    assert train.day_index.max() < val.day_index.min()
    for name in NAMES:
        ts, vs = fit_score(name, train, val, x)
        qs = [legacy['threshold_quantile']] if name == 'legacy_svm' else QUANTILES
        for q in qs:
            threshold = float(np.quantile(ts, q))
            rows.append(dict(fold=fold, detector=name, quantile=q, threshold=threshold,
                             **metrics(val, vs > threshold)))
fold_results = pd.DataFrame(rows)
ranking = (fold_results.groupby(['detector', 'quantile'], as_index=False)
    [['f2', 'recall', 'precision', 'f1', 'false_alerts_per_child', 'episode_detection']].mean()
    .sort_values(['f2', 'recall', 'false_alerts_per_child'], ascending=[False, False, True]))
best_each = ranking.drop_duplicates('detector')
selected = best_each.iloc[0]
fold_results.to_csv(OUT / 'forward_fold_metrics.csv', index=False)
ranking.to_csv(OUT / 'validation_ranking.csv', index=False)
# Freeze all final comparator thresholds before opening fresh held-out data.
final_train = dev.loc[dev.day_index >= 15]
locked = []
for row in best_each.itertuples():
    ts, _ = fit_score(row.detector, final_train, final_train, x)
    locked.append(dict(detector=row.detector, quantile=float(row.quantile),
                       threshold=float(np.quantile(ts, row.quantile))))
lock = dict(selected=str(selected.detector), objective='mean_forward_fold_F2',
            folds=FOLDS, candidates=locked, heldout_seed=manifest['seed'])
(OUT / 'selection_lock.json').write_text(json.dumps(lock, indent=2))
lock_digest = hashlib.sha256((OUT / 'selection_lock.json').read_bytes()).hexdigest()
print(best_each.to_string(index=False))
print('Selected and locked before holdout:', lock['selected'], lock_digest)''')
md('''## Final evaluation — first held-out access
All candidate definitions, quantiles and final thresholds are locked above. Below, each candidate is reported for comparison; the winner is not changed after evaluation. Histories use only prior observed days, including earlier evaluation days, as in sequential deployment. Fresh synthetic children are independent draws but have the same generator and calendar; this is not a real-world or calendar-shift test.''')
code('''verify_frozen()
assert hashlib.sha256((OUT / 'selection_lock.json').read_bytes()).hexdigest() == lock_digest
held = pd.read_csv(HOLD / 'behavioral_daily.csv')
# Distinct indexes keep development and heldout histories and identities separate.
held.index = held.index + len(dev) + 10000
hx = activity_features(held)
combined_x = pd.concat([x, hx])
test = held.loc[(held.day_index >= 15) & ~held.anomaly_type.isin(HIGH)]
final_rows = []
predictions = []
for candidate in locked:
    name = candidate['detector']
    _, scores = fit_score(name, final_train, test, combined_x)
    pred = scores > candidate['threshold']
    final_rows.append(dict(detector=name, selected=name == lock['selected'], **metrics(test, pred)))
    p = test[['child_id','date','day_index','anomaly_type','episode_id']].copy()
    p['detector'], p['score'], p['alert'] = name, scores, pred
    predictions.append(p)
final = pd.DataFrame(final_rows)
final.to_csv(OUT / 'heldout_behavioral_metrics.csv', index=False)
pd.concat(predictions).to_csv(OUT / 'heldout_predictions.csv', index=False)
print(final.to_string(index=False))
# Episode denominators include planned behavioral episodes with no surviving injected event.
episodes = pd.read_csv(HOLD / 'behavioral_episodes.csv')
episode_rows = []
for name, p in zip([c['detector'] for c in locked], predictions):
    for e in episodes.loc[episodes.anomaly_type.isin(BEHAVIOR)].itertuples():
        if e.end_date < held.loc[held.day_index == 15, 'date'].iloc[0]: continue
        match = p.loc[(p.child_id == e.child_id) & p.date.between(e.start_date,e.end_date) & p.alert]
        delay = (pd.Timestamp(match.date.min()) - pd.Timestamp(e.start_date)).days if len(match) else np.nan
        episode_rows.append(dict(detector=name, episode_id=e.episode_id, anomaly_type=e.anomaly_type,
            intensity=e.intensity, observed_days=e.observed_days, detected=bool(len(match)), delay_days=delay))
episode_table = pd.DataFrame(episode_rows)
episode_table.to_csv(OUT / 'heldout_episode_detection.csv', index=False)
print('Planned-episode detection and mean delay among detected:')
print(episode_table.groupby('detector').agg(detection=('detected','mean'), mean_delay_days=('delay_days','mean')).to_string())
print('Selected model by episode type and intensity:')
print(episode_table.loc[episode_table.detector == lock['selected']].groupby(['anomaly_type','intensity']).detected.agg(['mean','count']).to_string())''')
code('''# Independent rule output. No threshold fitting; this is a wiring/policy test.
def high_risk_rule(d):
    return ((d.language_urgent_flags > 0) | (d.language_grooming_flags > 0)
            | (d.language_self_harm_flags > 0) | (d.screen_self_harm_imagery_incidents > 0))
event_truth = set()
from component4_profiling_xai.src.behavioral_dataset import parse_timestamp
for line in (HOLD / 'synthetic_events' / 'c2_records.jsonl').read_text().splitlines():
    r = json.loads(line)
    if r['rung'] != 'L0' and not r['abstained'] and (r['urgent'] or r['top_category'] in ['grooming', 'self_harm']):
        event_truth.add((r['child_id'], parse_timestamp(r['ts'], 'ts').date().isoformat()))
for line in (HOLD / 'synthetic_events' / 'c1_incidents.jsonl').read_text().splitlines():
    r = json.loads(line)
    if r['category'] == 'self_harm_imagery':
        event_truth.add((r['child_id'], parse_timestamp(r['started_at'], 'start').date().isoformat()))
y_rule = np.array([(r.child_id, r.date) in event_truth for r in held.itertuples()])
rp = high_risk_rule(held).to_numpy()
rule_result = dict(precision=precision_score(y_rule,rp,zero_division=0), recall=recall_score(y_rule,rp,zero_division=0),
    f1=f1_score(y_rule,rp,zero_division=0), alerts=int(rp.sum()), eligible_days=int(y_rule.sum()),
    false_alerts_per_child=float((rp & ~y_rule).sum()/held.child_id.nunique()),
    limitation='Agreement with synthetic event policy eligibility only; no independent clinical/safety ground truth.')
(OUT / 'heldout_rule_metrics.json').write_text(json.dumps(rule_result,indent=2))
print(json.dumps(rule_result,indent=2))
print('Behavioral and urgent alerts remain separate outputs. No combined accuracy claimed.')''')
md('''## Verification and limitations
The tests below check causal history, child isolation, label independence, rule behavior, and artifact preservation. Model selection averages three folds of unequal duration; fold values are included, and validation is selection evidence rather than an unbiased performance estimate. Episode recall counts one detection anywhere in the episode; late detections are detailed separately. False alerts per child cover the evaluated window, not a daily rate. Planned episodes with no surviving injected event are included in the separate episode table.

Synthetic injections define both distributions and labels. No real-world generalization, upstream risk classifier accuracy, clinical assessment, or safety effectiveness is established. The generator provides flagged events rather than comprehensive activity; urgent event volume can still affect upstream screen durations. Removing grooming/self-harm days from behavioral evaluation narrows the target. Cold-start days 1–14 are excluded from behavioral evaluation but included for urgent rules. No production replacement is made; review and approval remain required after validation.''')
code('''# Future perturbations cannot change earlier features.
base = activity_features(dev)
changed = dev.copy()
changed.loc[changed.day_index > 50, 'screen_incidents'] += 100000
pd.testing.assert_frame_equal(base.loc[dev.day_index <= 50], activity_features(changed).loc[dev.day_index <= 50])
# Child histories cannot contaminate each other.
child = dev.child_id.iloc[0]
changed = dev.copy()
changed.loc[changed.child_id == child, 'screen_incidents'] += 100000
pd.testing.assert_frame_equal(base.loc[dev.child_id != child], activity_features(changed).loc[dev.child_id != child])
changed = dev.copy()
changed['is_anomaly'], changed['anomaly_type'], changed['episode_id'] = 999, 'changed', 'changed'
pd.testing.assert_frame_equal(base, activity_features(changed))
toy = held.iloc[:4].copy()
for c in ['language_urgent_flags','language_grooming_flags','language_self_harm_flags','screen_self_harm_imagery_incidents']: toy[c] = 0
toy.loc[toy.index[1], 'language_grooming_flags'] = 1
toy.loc[toy.index[2], 'language_urgent_flags'] = 1
toy.loc[toy.index[3], 'screen_self_harm_imagery_incidents'] = 1
assert high_risk_rule(toy).tolist() == [False, True, True, True]
assert set(dev.day_index).isdisjoint(range(74,91))
assert not any(any(t in c for t in ['urgent', 'grooming', 'self_harm', 'risk', 'anomaly', 'episode']) for c in base.columns)
verify_frozen()
assert hashlib.sha256((OUT / 'selection_lock.json').read_bytes()).hexdigest() == lock_digest
verification = {'checks_passed': ['future causality', 'child isolation', 'label independence', 'rule fixtures',
    'old test excluded', 'behavioral feature separation', 'original and heldout hashes unchanged', 'selection lock unchanged'],
    'production_replaced': False}
(OUT / 'verification.json').write_text(json.dumps(verification,indent=2))
print(json.dumps(verification,indent=2))''')
md('''## Recorded run outcome (2026-10-20)
Forward-time validation selected activity SVM at the 0.95 normal-training-score quantile, with mean fold F2 **0.4469** (legacy configuration **0.3760**). On the fresh heldout, F2 was **0.3414** versus **0.2504**. Recall improved from **26.27%** to **40.68%**, while precision was **20.78%** versus **21.09%**, F1 **0.2751** versus **0.2340**, and false alerts per child **4.575** versus **2.900** over days 15–90.

Observed-episode detection was **31/47 (65.96%)**, versus **19/47 (40.43%)**. Including planned episodes with no surviving injected event gives **31/52 (59.62%)**, versus **19/52 (36.54%)**. Selected-model volume-spike detection was only **3/16** planned episodes; this remains a serious limitation. The rule output matched **201/201** synthetic eligible child-days with no extra alerts; these are rule-consistency results, not independently measured high-risk classification accuracy.

The eight notebook verification checks passed. The existing anomaly, feature-builder and synthetic-generator regression tests passed **64/64** (`regression_tests.xml`). The first pytest invocation had four temporary-directory setup errors; rerunning with an explicit writable temporary directory resolved them. The notebook was executed cell-by-cell in a shared Python namespace, including all assertions, and validated with nbformat (nbclient is unavailable).

The result supports further investigation, with an increased alert burden and low absolute recall. It does not justify production replacement. No changes were made after heldout evaluation to candidates, thresholds or the selected winner. This heldout has now been consumed and must not become a tuning set.''')

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}})
path = ROOT / 'component4_profiling_xai/notebooks/Function_2_Model_Improvement.ipynb'
ns = {'__name__':'__main__'}
import os
os.chdir(ROOT)
for i, cell in enumerate(nb.cells):
    if cell.cell_type != 'code': continue
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(cell.source, f'{path.name}:cell{i}', 'exec'), ns)
    cell.execution_count = sum(c.cell_type == 'code' for c in nb.cells[:i+1])
    cell.outputs = [nbf.v4.new_output('stream', name='stdout', text=stream.getvalue())]
    print(stream.getvalue(), flush=True)
    nbf.write(nb, path)
nbf.validate(nb)
nbf.write(nb,path)
print(path)
