"""
Input:  X_train.csv, X_test.csv, y_train.csv, y_test.csv
Output: Interactive Streamlit web app
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
import json
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(
    page_title="Exoplanet Detection Dashboard",
    page_icon="🪐",
    layout="wide"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: radial-gradient(ellipse at top left, #141B36 0%, #0B1026 55%);
        color: #F5F3EC;
    }

    /* Headings */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #F5F3EC !important;
        letter-spacing: -0.01em;
    }
    h1 { font-weight: 700 !important; }

    /* Numbers / metrics get the mono telemetry look */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #F5A623 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #9AA3C7 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #141B36;
        border: 1px solid #2A3563;
        border-radius: 10px;
        padding: 14px 16px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #2A3563;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #9AA3C7;
        font-weight: 500;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #141B36 !important;
        color: #F5A623 !important;
        border-bottom: 2px solid #F5A623 !important;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] > div > div {
        background: #2A3563 !important;
    }
    .stSlider [role="slider"] {
        background-color: #F5A623 !important;
        border: 2px solid #0B1026 !important;
    }

    /* Success / error result boxes, restyled to match theme */
    div[data-testid="stAlertContainer"] {
        border-radius: 10px;
        border-width: 1px;
        border-style: solid;
    }

    /* DataFrame */
    [data-testid="stDataFrame"] {
        border: 1px solid #2A3563;
        border-radius: 8px;
    }

    /* Sidebar-free layout caption */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #9AA3C7 !important;
    }

    /* Divider */
    hr {
        border-color: #2A3563 !important;
    }

    /* Radio buttons */
    .stRadio [role="radiogroup"] label {
        color: #F5F3EC;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#141B36",
    "axes.facecolor": "#141B36",
    "axes.edgecolor": "#2A3563",
    "axes.labelcolor": "#F5F3EC",
    "text.color": "#F5F3EC",
    "xtick.color": "#9AA3C7",
    "ytick.color": "#9AA3C7",
    "grid.color": "#2A3563",
    "font.family": "sans-serif",
})

FLAG_COLS = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]

FEATURE_LABELS = {
    "koi_period": "Orbital Period (days)",
    "koi_impact": "Impact Parameter",
    "koi_duration": "Transit Duration (hours)",
    "koi_depth": "Transit Depth (ppm)",
    "koi_prad": "Planet Radius (Earth radii)",
    "koi_teq": "Equilibrium Temperature (K)",
    "koi_insol": "Insolation Flux (Earth = 1)",
    "koi_model_snr": "Signal-to-Noise Ratio",
    "koi_steff": "Stellar Temperature (K)",
    "koi_slogg": "Stellar Surface Gravity (log g)",
    "koi_srad": "Stellar Radius (Solar radii)",
    "koi_kepmag": "Kepler Magnitude (brightness)",
}

@st.cache_data
def load_data():
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").squeeze()
    y_test = pd.read_csv("y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


@st.cache_resource
def train_model(X_train, y_train):
    X_train_nf = X_train.drop(columns=FLAG_COLS)
    imputer = SimpleImputer(strategy="median")
    X_train_i = pd.DataFrame(imputer.fit_transform(X_train_nf), columns=X_train_nf.columns)

    model = GradientBoostingClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_i, y_train)

    explainer = shap.TreeExplainer(model)
    return model, imputer, explainer, list(X_train_nf.columns)


@st.cache_data
def get_slider_bounds(X_train, feature_cols):
    """Use 1st-99th percentile instead of raw min/max to avoid outlier-driven
    sliders (e.g. koi_prad has some artifact rows in the hundreds of
    thousands of Earth radii that would make the slider unusable)."""
    bounds = {}
    for col in feature_cols:
        lo = float(X_train[col].quantile(0.01))
        hi = float(X_train[col].quantile(0.99))
        default = float(X_train[col].median())
        bounds[col] = (lo, hi, default)
    return bounds


X_train, X_test, y_train, y_test = load_data()
model, imputer, explainer, feature_cols = train_model(X_train, y_train)
slider_bounds = get_slider_bounds(X_train, feature_cols)

X_test_nf = X_test.drop(columns=FLAG_COLS)
X_test_i = pd.DataFrame(imputer.transform(X_test_nf), columns=feature_cols)
y_test_proba = model.predict_proba(X_test_i)[:, 1]
y_test_pred = (y_test_proba >= 0.5).astype(int)

YOUTUBE_VIDEO_ID = "7OY_TNHtXsM"
YOUTUBE_URL = f"https://youtu.be/7OY_TNHtXsM"
YOUTUBE_THUMBNAIL = f"https://img.youtube.com/vi/7OY_TNHtXsM/maxresdefault.jpg"
TRANSIT_SVG = f"""
<div style="background:#141B36; border:1px solid #2A3563; border-radius:14px;
            padding:22px 28px; margin-bottom:6px;">
  <div style="display:flex; align-items:center; gap:22px; flex-wrap:wrap;">
    <div style="flex:0 0 auto;">
      <svg width="220" height="90" viewBox="0 0 220 90">
        <line x1="0" y1="70" x2="220" y2="70" stroke="#2A3563" stroke-width="1"/>
        <path d="M0,20 L70,20 Q80,20 85,45 Q90,60 100,60 Q110,60 115,45 Q120,20 130,20 L220,20"
              fill="none" stroke="#F5A623" stroke-width="2.5" stroke-linecap="round">
          <animate attributeName="stroke" values="#F5A623;#4FD1E8;#F5A623" dur="4s" repeatCount="indefinite"/>
        </path>
        <circle cx="100" cy="59" r="3.5" fill="#F5A623">
          <animate attributeName="cy" values="59;58;59" dur="2s" repeatCount="indefinite"/>
        </circle>
      </svg>
    </div>
    <div style="flex:1 1 320px;">
      <div style="font-family:'JetBrains Mono', monospace; color:#F5A623; font-size:0.78rem;
                  letter-spacing:0.08em; text-transform:uppercase; margin-bottom:4px;">
        NASA Kepler Objects of Interest &middot; Binary Classifier
      </div>
      <h1 style="margin:0 0 8px 0; font-size:1.9rem;">Exoplanet Detection via Transit Photometry</h1>
      <div style="color:#C7CCE8; font-size:0.95rem; line-height:1.5;">
        A dip in starlight, timed and shaped correctly, is how every transiting exoplanet
        Kepler ever found was first flagged. This model predicts <b style="color:#F5A623;">CONFIRMED</b>
        vs <b style="color:#E8617A;">FALSE POSITIVE</b> from that signal alone &mdash;
        trained on raw transit. See <i>Model Details</i>. Built by Jarin Binta Yeasin, Student of University of Dhaka as an Individual Project on NASA Kepler Exoplanet Archive Data
      </div>
    </div>
  </div>
      <a href="https://youtu.be/7OY_TNHtXsM" target="_blank" style="flex:0 0 auto; text-decoration:none;">
      <div style="position:relative; width:220px; border-radius:10px; overflow:hidden;
                  border:1px solid #2A3563;">
        <img src="Thumbnail.png" style="width:100%; display:block;" alt="Video explanation thumbnail"/>
        <div style="position:absolute; top:0; left:0; right:0; bottom:0;
                    background:rgba(11,16,38,0.25); display:flex;
                    align-items:center; justify-content:center;">
          <div style="width:52px; height:52px; border-radius:50%; background:rgba(11,16,38,0.75);
                      display:flex; align-items:center; justify-content:center;
                      border:2px solid #F5A623;">
            <div style="width:0; height:0; margin-left:4px;
                        border-top:10px solid transparent; border-bottom:10px solid transparent;
                        border-left:16px solid #F5A623;"></div>
          </div>
        </div>
      </div>
      <div style="color:#9AA3C7; font-size:0.72rem; text-align:center; margin-top:6px;
                  text-transform:uppercase; letter-spacing:0.05em;">
        ▶ Watch: Project Walkthrough
      </div>
    </a>
</div>
"""
st.markdown(TRANSIT_SVG, unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs([
    "Try a Prediction", "Browse Test Examples",
    "Model Details", "Evaluation"
])

with tab1:
    st.subheader("Enter transit and star measurements")
    st.caption("Sliders default to the median value from the training set. "
               "Try adjusting Planet Radius or Signal-to-Noise to see how the prediction changes.")

    col1, col2 = st.columns(2)
    user_input = {}

    left_features = feature_cols[:6]
    right_features = feature_cols[6:]

    with col1:
        for feat in left_features:
            lo, hi, default = slider_bounds[feat]
            user_input[feat] = st.slider(
                FEATURE_LABELS.get(feat, feat), min_value=lo, max_value=hi, value=default
            )

    with col2:
        for feat in right_features:
            lo, hi, default = slider_bounds[feat]
            user_input[feat] = st.slider(
                FEATURE_LABELS.get(feat, feat), min_value=lo, max_value=hi, value=default
            )

    input_df = pd.DataFrame([user_input])[feature_cols]
    proba = model.predict_proba(input_df)[0, 1]
    prediction = "CONFIRMED" if proba >= 0.5 else "FALSE POSITIVE"

    st.markdown("---")
    result_col1, result_col2 = st.columns([1, 2])

    with result_col1:
        badge_color = "#F5A623" if prediction == "CONFIRMED" else "#E8617A"
        badge_glow = "rgba(245,166,35,0.15)" if prediction == "CONFIRMED" else "rgba(232,97,122,0.15)"
        st.markdown(f"""
        <div style="background:{badge_glow}; border:1px solid {badge_color};
                    border-radius:12px; padding:20px; text-align:center; margin-bottom:14px;">
            <div style="color:#9AA3C7; font-size:0.75rem; text-transform:uppercase;
                        letter-spacing:0.08em; margin-bottom:6px;">Prediction</div>
            <div style="color:{badge_color}; font-size:1.6rem; font-weight:700;
                        font-family:'JetBrains Mono', monospace;">{prediction}</div>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Probability of being a real exoplanet", f"{proba:.1%}")

    with result_col2:
        st.markdown("**Why the model made this prediction:**")
        shap_vals = explainer.shap_values(input_df)[0]
        contribution_df = pd.DataFrame({
            "feature": [FEATURE_LABELS.get(f, f) for f in feature_cols],
            "shap_value": shap_vals
        }).sort_values("shap_value", key=abs, ascending=True).tail(6)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = ["#F5A623" if v > 0 else "#E8617A" for v in contribution_df["shap_value"]]
        ax.barh(contribution_df["feature"], contribution_df["shap_value"], color=colors)
        ax.axvline(0, color="#9AA3C7", linewidth=0.8)
        ax.set_xlabel("← pushes FALSE POSITIVE   |   pushes CONFIRMED →")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab2:
    st.subheader("Explore real Kepler candidates from the test set")

    filter_choice = st.radio(
        "Show:",
        ["All", "Correct predictions", "Model mistakes (false negatives)", "Model mistakes (false positives)"],
        horizontal=True
    )

    display_df = X_test_i.copy()
    display_df["actual"] = y_test.reset_index(drop=True).map({0: "FALSE POSITIVE", 1: "CONFIRMED"})
    display_df["predicted"] = np.where(y_test_pred == 1, "CONFIRMED", "FALSE POSITIVE")
    display_df["confidence"] = y_test_proba

    if filter_choice == "Correct predictions":
        display_df = display_df[display_df["actual"] == display_df["predicted"]]
    elif filter_choice == "Model mistakes (false negatives)":
        display_df = display_df[(display_df["actual"] == "CONFIRMED") & (display_df["predicted"] == "FALSE POSITIVE")]
    elif filter_choice == "Model mistakes (false positives)":
        display_df = display_df[(display_df["actual"] == "FALSE POSITIVE") & (display_df["predicted"] == "CONFIRMED")]

    st.caption(f"Showing {len(display_df)} candidates")

    rename_map = {**FEATURE_LABELS, "actual": "Actual", "predicted": "Predicted", "confidence": "P(CONFIRMED)"}
    st.dataframe(
        display_df.rename(columns=rename_map).round(3),
        width="stretch",
        height=400
    )

with tab3:
    st.subheader("Model performance on held-out test set")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_test_pred):.1%}")
    m2.metric("Precision", f"{precision_score(y_test, y_test_pred):.1%}")
    m3.metric("Recall", f"{recall_score(y_test, y_test_pred):.1%}")
    m4.metric("F1 Score", f"{f1_score(y_test, y_test_pred):.3f}")
    m5.metric("ROC-AUC", f"{roc_auc_score(y_test, y_test_proba):.3f}")

    st.markdown("---")
    st.markdown("""
    **Why this model excludes pipeline diagnostic flags:**

    NASA's Kepler pipeline computes four diagnostic flags (`koi_fpflag_*`) that
    represent its own automated verdict on whether a signal looks planet-shaped.
    A model trained including these flags reaches ~99% accuracy, but that's
    largely because it's learning to copy an existing decision.

    This dashboard's model was deliberately trained without those flags,
    using only raw physical measurements (orbital period, transit depth,
    planet radius, signal-to-noise, host star properties, etc). It scores
    lower (~91-92% accuracy).

    **Global feature importance (SHAP):**
    """)

    shap_values_all = explainer.shap_values(X_test_i)
    mean_abs_shap = pd.DataFrame({
        "feature": [FEATURE_LABELS.get(f, f) for f in feature_cols],
        "importance": np.abs(shap_values_all).mean(axis=0)
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(mean_abs_shap["feature"], mean_abs_shap["importance"], color="#4FD1E8")
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    **Data source:** [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) --
    Kepler Objects of Interest (KOI) cumulative table.

    **Model:** Gradient Boosting Classifier (scikit-learn), trained on 80% of the data, evaluated on 20% held-out test set.
    """)

with tab4:
    st.subheader("Cross-validation, hyperparameter tuning & benchmark comparison")

    RESULTS_FILE = "evaluation_results.json"

    if not os.path.exists(RESULTS_FILE):
        st.warning(
            f"`{RESULTS_FILE}` not found. Run `07_evaluation.py` once "
            "in this same folder to generate it, this tab reads pre-computed "
            "results rather than re-running the ~4-5 minute analysis on every "
            "dashboard visit."
        )
    else:
        with open(RESULTS_FILE) as f:
            results = json.load(f)

        st.markdown("#### 5-fold cross-validation: default vs. tuned hyperparameters")
        st.caption(
            "Replaces a single 80/20 split with 5-fold stratified cross-validation, "
            "reporting mean ± std across folds."
        )

        cv_default = results["cross_validation"]["default_hyperparameters"]
        cv_tuned = results["cross_validation"]["tuned_hyperparameters"]

        metric_display = {
            "accuracy": "Accuracy", "f1": "F1 Score",
            "roc_auc": "ROC-AUC", "average_precision": "PR-AUC"
        }

        cols = st.columns(4)
        for col, (key, label) in zip(cols, metric_display.items()):
            d = cv_default[key]
            t = cv_tuned[key]
            delta = t["mean"] - d["mean"]
            col.metric(
                label,
                f"{t['mean']:.3f}",
                delta=f"{delta:+.3f} vs default",
            )
            col.caption(f"±{t['std']:.3f} std (tuned)  ·  default: {d['mean']:.3f}±{d['std']:.3f}")

        st.markdown("---")
        st.markdown("#### Hyperparameter tuning (Optuna)")
        st.caption(
            f"{results['hyperparameter_search']['n_trials']} trials, Bayesian "
            "optimization (TPE sampler), optimized for F1 score via 3-fold CV, "
            "then validated with full 5-fold CV above."
        )

        best_params = results["best_hyperparameters"]
        param_cols = st.columns(len(best_params))
        for col, (param, value) in zip(param_cols, best_params.items()):
            display_val = f"{value:.4f}" if isinstance(value, float) else str(value)
            col.metric(param, display_val)

        fig, ax = plt.subplots(figsize=(9, 3.5))
        trial_values = results["hyperparameter_search"]["trial_values"]
        best_so_far = np.maximum.accumulate(trial_values)
        ax.plot(range(1, len(trial_values) + 1), trial_values, "o", color="#4FD1E8",
                alpha=0.6, label="Trial F1 score")
        ax.plot(range(1, len(best_so_far) + 1), best_so_far, color="#F5A623",
                linewidth=2, label="Best so far")
        ax.set_xlabel("Trial number")
        ax.set_ylabel("F1 score (3-fold CV)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.markdown("#### Benchmark against Kepler's Robovetter (Thompson et al. 2018, DR25)")

        rv = results["robovetter_comparison"]
        bench_col1, bench_col2 = st.columns(2)

        with bench_col1:
            st.markdown("**This project (out-of-sample, cross-validated)**")
            st.metric("Recall (~ completeness analog)", f"{rv['this_model']['recall']:.1%}")
            st.metric("Precision (~ reliability analog)", f"{rv['this_model']['precision']:.1%}")

        with bench_col2:
            st.markdown("**NASA's Robovetter (published, DR25)**")
            st.metric("Completeness", f">{rv['robovetter_published']['completeness']:.0%}")
            st.caption(rv["robovetter_published"]["completeness_note"])
            st.metric("Reliability (inferred)", f"~{rv['robovetter_published']['reliability']:.0%}")
            st.caption(rv["robovetter_published"]["reliability_note"])

        st.info(
            "**This is not a like-for-like comparison.** The Robovetter operates on "
            "raw threshold-crossing events before any KOI designation exists, uses "
            "injection/recovery testing on real and simulated data, and has access "
            "to far more diagnostic information than this project's tabular feature "
            "set. This model lands in a broadly comparable range on these specific "
            "figures."
        )