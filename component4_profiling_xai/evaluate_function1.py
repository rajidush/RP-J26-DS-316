"""Reproduce the Function 1 research evaluation from the repository root."""

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
    silhouette_samples, silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from component4_profiling_xai.src import emotional_pattern as f1


COMPONENT = Path(__file__).resolve().parent
SEEDS = (42, 0, 1, 2, 3, 4, 5, 6, 7, 8)
SPLIT_SEED = 316
METRICS = {
    "inertia": "Within-cluster sum of squares",
    "silhouette": "Silhouette score",
    "davies_bouldin": "Davies–Bouldin index",
    "calinski_harabasz": "Calinski–Harabasz index",
}
FEATURE_LABELS = ("Calm", "Guarded", "Distressed", "Unclear", "Self-regulation\nflag", "Mean turns")
COLORS = ("#aa4b46", "#277d7b", "#5a63a5", "#946a2d")


def artifact_hashes(directory):
    return {file.name: sha256(file.read_bytes()).hexdigest()
            for file in sorted(directory.glob("emotional_pattern_*")) if file.is_file()}


def distributions(frame):
    q1, median, q3 = frame.quantile([0.25, 0.5, 0.75]).to_numpy()
    iqr = q3 - q1
    outliers = (frame < (q1 - 1.5 * iqr)) | (frame > (q3 + 1.5 * iqr))
    return {
        feature: {"min": float(frame[feature].min()), "q1": float(q1[i]), "median": float(median[i]),
                  "q3": float(q3[i]), "max": float(frame[feature].max()),
                  "mean": float(frame[feature].mean()), "sd": (
                      float(frame[feature].std()) if len(frame) > 1 else None),
                  "tukey_outliers": int(outliers[feature].sum())}
        for i, feature in enumerate(frame)
    }


def prepare(raw, settings):
    if raw.empty:
        raise ValueError("Evaluation requires session records")
    timestamps = pd.to_datetime(raw["timestamp"], utc=True, format="mixed", errors="coerce")
    turns = pd.to_numeric(raw["dialogue_turns"], errors="coerce")
    invalid_turns = turns.isna() | ~np.isfinite(turns) | turns.lt(1) | turns.mod(1).ne(0)
    quality = {
        "raw_sessions": len(raw), "missing_by_column": raw.isna().sum().to_dict(),
        "duplicate_rows": int(raw.duplicated().sum()),
        "repeated_session_ids": int(raw["session_id"].duplicated().sum()),
        "invalid_timestamps": int(timestamps.isna().sum()),
        "invalid_turn_counts": int(invalid_turns.sum()),
        "invalid_emotions": int((~raw["emotional_state"].astype(str).str.strip().str.lower()
                                 .isin(f1.EMOTIONS)).sum()),
        "invalid_self_regulation_flags": int((~raw["self_regulation_shown"].astype(str)
                                               .str.strip().str.lower().isin(["true", "false"])).sum()),
    }
    if raw[["child_id", "session_id"]].isna().any().any():
        raise ValueError("Evaluation requires child and session identifiers")
    sessions = f1.clean_sessions(raw)
    starts = sessions.groupby("child_id")["timestamp"].transform("min").dt.normalize()
    sessions["window"] = (sessions["timestamp"] - starts).dt.days // settings["window_days"]
    counts = sessions.groupby(["child_id", "window"]).size()
    reliable = counts[counts >= settings["min_sessions"]]
    selected = sessions.set_index(["child_id", "window"]).loc[reliable.index].reset_index()
    if selected.empty:
        raise ValueError("Evaluation requires at least one reliable window")
    window_rows = selected.rename(columns={"child_id": "original_child_id"})
    window_rows["child_id"] = window_rows.groupby(["original_child_id", "window"]).ngroup()
    windows = f1.build_features(window_rows)
    windows.index = pd.MultiIndex.from_frame(
        window_rows.sort_values("child_id").drop_duplicates("child_id")[["original_child_id", "window"]],
        names=["child_id", "window"],
    )
    children = f1.build_features(sessions)
    possible_bins = int((sessions.groupby("child_id")["window"].max() + 1).sum())
    quality.update(
        valid_sessions=len(sessions), children=len(children),
        session_labels=sessions["emotional_state"].value_counts().to_dict(),
        turns=distributions(sessions[["dialogue_turns"]])["dialogue_turns"],
        sessions_per_child=distributions(sessions.groupby("child_id").size().to_frame("count"))["count"],
        reliable_windows=len(windows), nonempty_windows=len(counts),
        sparse_windows=len(counts) - len(windows), empty_windows=possible_bins - len(counts),
        children_with_reliable_windows=int(reliable.index.get_level_values("child_id").nunique()),
        windows_per_child=distributions(reliable.groupby(level="child_id").size()
                                       .reindex(children.index, fill_value=0).to_frame("count"))["count"],
    )
    return children, windows, quality


def cluster_metrics(scaled, labels, centers):
    sizes = np.bincount(labels, minlength=len(centers))
    inertia = float(np.square(scaled - centers[labels]).sum())
    metrics = {"inertia": inertia, "sizes": sizes.tolist(),
               "silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    if 2 <= np.count_nonzero(sizes) < len(scaled):
        metrics.update(silhouette=float(silhouette_score(scaled, labels)),
                       davies_bouldin=float(davies_bouldin_score(scaled, labels)),
                       calinski_harabasz=float(calinski_harabasz_score(scaled, labels)))
    return metrics


def fit_experiment(scaled, k, seed):
    if not 2 <= k < len(scaled) or len(np.unique(scaled, axis=0)) < k:
        raise ValueError("k requires at least k distinct vectors and more than k samples")
    return KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=seed).fit(scaled)


def summarize(values):
    return {"min": float(np.min(values)), "mean": float(np.mean(values)),
            "median": float(np.median(values)), "max": float(np.max(values)),
            "sd": float(np.std(values))}


def stability(predictions, reference, k):
    pairwise = [adjusted_rand_score(a, b) for a, b in combinations(predictions, 2)]
    aligned_sizes = []
    for labels in predictions:
        contingency = np.zeros((k, k), dtype=int)
        np.add.at(contingency, (reference, labels), 1)
        rows, columns = linear_sum_assignment(-contingency)
        aligned_sizes.append(np.bincount(labels, minlength=k)[columns[np.argsort(rows)]])
    return {
        "pairwise_ari": {"pairs": len(pairwise), **summarize(pairwise)},
        "aligned_cluster_sizes": [summarize(column) for column in np.asarray(aligned_sizes).T],
    }


def sweep(scaled, saved_labels, saved_k):
    results = []
    distinct = len(np.unique(scaled, axis=0))
    for k in range(2, 9):
        if k >= len(scaled) or k > distinct:
            results.append({"k": k, "skipped": "Insufficient samples or distinct vectors"})
            continue
        runs = [fit_experiment(scaled, k, seed) for seed in SEEDS]
        reference = runs[0]
        metrics = cluster_metrics(scaled, reference.labels_, reference.cluster_centers_)
        cross_tab = np.zeros((saved_k, k), dtype=int)
        np.add.at(cross_tab, (saved_labels, reference.labels_), 1)
        results.append({
            "k": k, **metrics, "stability": stability([run.labels_ for run in runs], reference.labels_, k),
            "ari_vs_saved": float(adjusted_rand_score(saved_labels, reference.labels_)) if k == saved_k else None,
            "saved_by_experimental_cluster": cross_tab.tolist(),
        })
    return results


def grouped_split(groups):
    children = np.unique(groups)
    if len(children) < 8:
        raise ValueError("Grouped stability requires at least eight children")
    rng = np.random.default_rng(SPLIT_SEED)
    evaluation = np.sort(rng.choice(children, size=max(2, len(children) // 4), replace=False))
    pool = np.setdiff1d(children, evaluation)
    samples = [np.sort(rng.choice(pool, size=int(0.8 * len(pool)), replace=False)) for _ in SEEDS]
    return evaluation, samples


def grouped_stability(frame, groups, saved, scaler):
    evaluation, samples = grouped_split(groups)
    eval_mask = np.isin(groups, evaluation)
    common = frame.loc[eval_mask]
    reference = saved.predict(scaler.transform(common))
    predictions = []
    for sample in samples:
        train = frame.loc[np.isin(groups, sample)]
        train_scaler = StandardScaler().fit(train)
        experiment = fit_experiment(train_scaler.transform(train), saved.n_clusters, SEEDS[0])
        predictions.append(experiment.predict(train_scaler.transform(common)))
    return {
        "evaluation_children": evaluation.tolist(), "evaluation_vectors": len(common),
        "fit_children_per_run": len(samples[0]), "training_child_ids": [s.tolist() for s in samples],
        "ari_vs_saved": summarize([adjusted_rand_score(reference, labels) for labels in predictions]),
        **stability(predictions, reference, saved.n_clusters),
    }


def profiles(frame, labels, names):
    return {
        cluster: {"pattern": name, "count": int((labels == cluster).sum()),
                  "proportion": float((labels == cluster).mean()),
                  "features": distributions(frame.loc[labels == cluster]) if (labels == cluster).any() else {}}
        for cluster, name in names.items()
    }


def evaluate_dataset(raw, model, scaler, metadata, experiments):
    children, windows, quality = prepare(raw, metadata["trend_settings"])
    results = {"data_quality": quality}
    for unit, frame in (("children", children), ("windows", windows)):
        frame = frame[metadata["features"]]
        shares = frame[[f"p_{emotion}" for emotion in f1.EMOTIONS]]
        proportions = shares.assign(selfreg_rate=frame["selfreg_rate"])
        valid = (np.isfinite(frame).all(axis=1) & proportions.ge(0).all(axis=1)
                 & proportions.le(1).all(axis=1) & np.isclose(shares.sum(axis=1), 1)
                 & frame["mean_turns"].ge(1))
        if not valid.all():
            raise ValueError(f"{unit} contain invalid aggregate feature values")
        scaled = scaler.transform(frame)
        labels = model.predict(scaled)
        metrics = cluster_metrics(scaled, labels, model.cluster_centers_)
        result = {"vectors": len(frame), "distinct_vectors": len(np.unique(scaled, axis=0)),
                  "invalid_feature_vectors": int((~valid).sum()),
                  "saved": metrics, "feature_distributions": distributions(frame),
                  "profiles": profiles(frame, labels, metadata["cluster_names"])}
        if metrics["silhouette"] is not None:
            samples = silhouette_samples(scaled, labels)
            result["silhouette_by_cluster"] = {
                int(cluster): {"mean": float(samples[labels == cluster].mean()),
                               "negative_count": int((samples[labels == cluster] < 0).sum())}
                for cluster in np.unique(labels)}
        if experiments:
            result["sweep"] = sweep(scaled, labels, model.n_clusters)
            result["grouped_stability"] = grouped_stability(
                frame, frame.index.get_level_values("child_id").to_numpy(), model, scaler)
        results[unit] = result
    return results, windows


def write_plots(output, results, windows, model, scaler, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"figure.dpi": 145, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    training = results["training"]
    for key, title in METRICS.items():
        fig, ax = plt.subplots(figsize=(7.0, 4.3), layout="constrained")
        for unit, color in (("children", "#255c89"), ("windows", "#a65724")):
            rows = [row for row in training[unit]["sweep"] if not row.get("skipped")]
            ax.plot([r["k"] for r in rows], [r[key] for r in rows], "o-", color=color,
                    label=f"Experimental · {unit}")
            ax.scatter(model.n_clusters, training[unit]["saved"][key], s=105, marker="D",
                       facecolors="none", edgecolors=color, zorder=3)
        ax.set(xlabel="Number of clusters (k)", ylabel=title, title=f"{title} by cluster count")
        ax.set_xticks(range(2, 9))
        ax.grid(axis="y", alpha=0.22)
        ax.legend(title="Open diamond = saved model", frameon=False)
        save(fig, "elbow" if key == "inertia" else f"{key}_vs_k")

    names = metadata["cluster_names"]
    ids = list(sorted(names))
    names_list = [names[c] for c in ids]
    fig, ax = plt.subplots(figsize=(7.4, 4.0), layout="constrained")
    for unit, offset, color in (("children", -0.18, "#255c89"), ("windows", 0.18, "#a65724")):
        profiles_by_id = training[unit]["profiles"]
        bars = ax.bar(np.arange(len(ids)) + offset, [profiles_by_id[c]["proportion"] for c in ids],
                      width=0.34, color=color, label=unit.capitalize())
        ax.bar_label(bars, labels=[str(profiles_by_id[c]["count"]) for c in ids], padding=3)
    ax.set(xticks=range(len(ids)), xticklabels=names_list, ylim=(0, 0.57),
           ylabel="Share of evaluated vectors", title="Saved-model cluster sizes")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False)
    save(fig, "cluster_sizes")

    standardized = scaler.transform(windows[metadata["features"]])
    labels = model.predict(standardized)
    pca = PCA(n_components=2, svd_solver="full")
    projected = pca.fit_transform(standardized)
    variance = pca.explained_variance_ratio_
    results["window_pca_explained_variance_ratio"] = variance.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 4.9), layout="constrained")
    for cluster, color in zip(ids, COLORS):
        mask = labels == cluster
        ax.scatter(*projected[mask].T, color=color, alpha=0.72, s=35, label=names[cluster])
    ax.set(xlabel=f"PC1 ({variance[0]:.1%})", ylabel=f"PC2 ({variance[1]:.1%})",
           title="Saved assignments in a two-component projection")
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.16)
    save(fig, "saved_assignments_pca")

    window_profiles = training["windows"]["profiles"]
    profiled_ids = [c for c in ids if window_profiles[c]["count"]]
    means = pd.DataFrame([[window_profiles[c]["features"][f]["mean"] for f in metadata["features"]]
                          for c in profiled_ids], columns=metadata["features"])
    scaled_means = scaler.transform(means)
    fig, ax = plt.subplots(figsize=(8.2, 3.8), layout="constrained")
    limit = np.abs(scaled_means).max()
    image = ax.imshow(scaled_means, cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit)
    for i, j in np.ndindex(means.shape):
        value = means.iloc[i, j]
        ax.text(j, i, f"{value:.1%}" if j < 5 else f"{value:.2f}", ha="center", va="center",
                color="white" if abs(scaled_means[i, j]) > limit * 0.63 else "#1c2632")
    ax.set(xticks=range(len(FEATURE_LABELS)), xticklabels=FEATURE_LABELS,
           yticks=range(len(profiled_ids)), yticklabels=[names[c] for c in profiled_ids],
           title="Mean features of assigned windows")
    fig.colorbar(image, ax=ax, label="Standardized mean (saved scaler)")
    save(fig, "window_feature_profiles")


def run(output, demo):
    artifacts = f1.ARTIFACT_DIR.resolve()
    output = output.resolve()
    if output == artifacts or artifacts in output.parents:
        raise ValueError("Research outputs must be outside the production artifact directory")
    before = artifact_hashes(artifacts)
    model, scaler, metadata = f1.load_artifacts(artifacts)
    if sklearn.__version__ != metadata["sklearn_version"]:
        raise ValueError(f"Saved artifacts require scikit-learn {metadata['sklearn_version']}")
    training_path = COMPONENT / "data" / "educator_sessions.csv"
    training, windows = evaluate_dataset(pd.read_csv(training_path), model, scaler, metadata, True)
    supplemental, _ = evaluate_dataset(demo, model, scaler, metadata, False)
    results = {
        "training_data_sha256": sha256(training_path.read_bytes()).hexdigest(),
        "artifact_sha256": before, "model_parameters": model.get_params(),
        "stored_inertia": float(model.inertia_), "fit_iterations": int(model.n_iter_),
        "scaler_parameters": scaler.get_params(), "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(), "metadata": metadata,
        "experiment_seeds": SEEDS, "grouped_split_seed": SPLIT_SEED,
        "training": training, "supplemental_dashboard_demo": supplemental,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_plots(output, results, windows, model, scaler, metadata)
    if artifact_hashes(artifacts) != before:
        raise RuntimeError("Production artifact hashes changed during evaluation")
    (output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    return results


def main():
    from ui.services import demo_sessions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=COMPONENT / "outputs" / "function1_kmeans")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.out, demo_sessions())
    print(f"Evaluation results written to {args.out}")


if __name__ == "__main__":
    main()
