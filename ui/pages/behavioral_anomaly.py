import streamlit as st

from ui import services, theme

theme.page_header("Behavioral Anomaly (Function 2)",
                  "Check whether a child's daily device and lifestyle habits look unusual.")


def show_model_error(status):
    """No result of any kind without the trained model -- only this card."""
    theme.error_card(
        "f2", "Function 2 model could not be loaded. No behavioral anomaly result is shown.",
        status["reason"],
        ["Put `behavioral_anomaly_ocsvm_variation.pkl`, `behavioral_scaler_variation.pkl` and "
         "`behavioral_features_variation.pkl` in `component4_profiling_xai/models/`.",
         "Install the scikit-learn version the model was trained with: "
         "`pip install scikit-learn==1.6.1`.",
         "Then restart the app."])
    st.stop()


status = services.behavioral_anomaly_status()
if not status["available"]:
    show_model_error(status)

sample = services.behavioral_sample_input()

with theme.card("f2-input", "Daily Behavior",
                "Pre-filled with the example record from component4_profiling_xai/mock_inputs."):
    with st.form("f2_form", border=False):
        cols = st.columns(3)
        values = {}
        for i, name in enumerate(status["features"]):
            default = sample.get(name, 0)
            step = 1 if isinstance(default, int) else 0.1
            values[name] = cols[i % 3].number_input(services.feature_label(name), value=default,
                                                    step=step, min_value=0 * step)
        submitted = st.form_submit_button("Check behavior", type="primary")

if submitted:
    try:
        result = services.predict_behavioral_anomaly(values)
    except services.ModelUnavailableError as exc:
        show_model_error(exc.status)
    except (ValueError, TypeError) as exc:
        st.error(str(exc))
    else:
        with theme.card("f2-result", "Result"):
            theme.badge(theme.ANOMALY_STYLES[result["status"]],
                        subtitle=f"Anomaly score {result['anomaly_score']:.2f} (higher = more unusual)")
            theme.note("The score is a relative ranking from the One-Class SVM, not a "
                       "probability, percentage or clinical risk score.")
