"""Archived research interface for the saved Function 02 experiments."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component4_profiling_xai.src.behavioral_profile import BUNDLE

OUTPUT = BUNDLE.parent

st.set_page_config(page_title="Function 02 Research Evaluation", layout="wide")
st.title("Function 02 Research Evaluation")
if not (OUTPUT / "evaluation.json").is_file():
    st.info("Train the models to produce the local evaluation artifacts.")
    st.stop()

summary = json.loads((OUTPUT / "evaluation.json").read_text(encoding="utf-8"))
models = pd.read_csv(OUTPUT / "model_comparison.csv")
st.warning("Synthetic benchmark previously used in this project. No real-world safety or child outcome evaluation.")
st.write("Selected model", summary["selected"]["algorithm"], summary["selected"]["design"])
st.dataframe(pd.DataFrame(summary["splits"]).T)

st.header("1. Model Comparison")
st.dataframe(models[models["design"].isin(["global", "personalized", "frozen_legacy"])], hide_index=True)
st.caption("Confusion matrix cells are TN, FP, FN and TP. FPR is FP divided by all labeled normal scored days.")
st.header("2. Personalization Effectiveness")
svm = models[models["algorithm"].eq("one_class_svm")].set_index("design")
st.bar_chart(svm.loc[["global", "personalized", "frozen_legacy"], ["f1", "fpr"]])

st.header("3. Temporal Change Detection")
st.dataframe(pd.DataFrame({name: data["test"] for name, data in summary["temporal"].items()}).T)
st.caption("Detection requires a new change point inside a labeled episode. Delay is measured in calendar days "
           "from episode start among detected episodes. False alert rate uses scored normal days.")

st.write("Validation-selected repeat threshold for priority review", summary["priority_repeated"])
st.header("4. XAI Evaluation")
st.json(summary["xai_means"], expanded=True)
st.bar_chart(pd.Series(summary["global_shap"], name="Mean absolute SHAP"))
st.caption("SHAP reconstruction error measures additivity. LIME fidelity is weighted R² on held-out donor "
           "perturbations. Stability measures agreement on the largest absolute group under a changed reference "
           "and donor partition. Deletion is the signed score drop after replacing that group.")
st.dataframe(pd.read_csv(OUTPUT / "xai_evaluation.csv"), hide_index=True)
st.header("5. Multimodal Ablation")
st.dataframe(models[models["design"].ne("frozen_legacy")], hide_index=True)
st.write(summary["emotional_ablation"])
st.dataframe(pd.read_csv(OUTPUT / "training_correlations.csv"), hide_index=True)
st.header("6. On-Device Performance")
st.json(summary["resources"], expanded=True)
st.caption("Windows host measurements. Model times cover warm one-row scoring including preprocessing. "
           "The XAI timing covers both explanations and their shared scoring. Mobile and SUS were unevaluated.")

# Keep the original panels above as the historical baseline for this study.
optimization = OUTPUT.with_name("function02_optimization")
if (optimization / "evaluation.json").is_file():
    with st.expander("Optimization and sealed synthetic holdout", expanded=True):
        study = json.loads((optimization / "evaluation.json").read_text(encoding="utf-8"))
        lock = json.loads((optimization / "selection_lock.json").read_text(encoding="utf-8"))
        st.warning("Fresh random profiles and events from the same generator. This cohort has now been used for evaluation.")
        st.write(f"Validation selected **{study['selected']['design']}** using "
                 f"**{study['selected']['algorithm']}**. The deployed parent model is unchanged.")
        st.write(f"{study['test_rows']:,} scored child-days · {study['test_positive']} labeled anomaly days · "
                 f"{lock['candidate_count']} validation candidates")
        for title, table in (
            ("Holdout model and ablation results", "test_comparison.csv"),
            ("Confusion matrices", "confusion_matrices.csv"),
            ("Window and feature ablation validation", "window_and_ablation_validation.csv"),
            ("False negatives by scenario", "false_negatives_by_scenario.csv"),
            ("False-positive feature context", "false_positive_context.csv"),
        ):
            st.markdown(f"**{title}**")
            st.dataframe(pd.read_csv(optimization / table), hide_index=True)
        st.image(str(optimization / "test_precision_recall.png"))
        st.markdown("**Paired child-bootstrap differences versus optimized global**")
        st.json(study["intervals"], expanded=False)
        st.markdown("**Held-out temporal change detection**")
        st.dataframe(pd.DataFrame(study["temporal"]).T)
        st.caption("Delay starts at the planned episode onset. Detection requires a new alert on an observed labeled episode day.")
        st.markdown("**Grouped SHAP and LIME over five training donor samples**")
        st.json(study["xai_means"], expanded=False)
