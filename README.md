# Exoplanet Detection via Transit Photometry

Predicting whether a Kepler-observed transit signal is a real confirmed exoplanet or a false positive, using NASA's Kepler Objects of Interest (KOI) catalog.

**Jarin Binta Yeasin** | Department of Mass Communication & Journalism | University of Dhaka

**[Live Dashboard](https://exoplanetdetectionviatransitphotometry-ffkjcb3vnewuwawcrynykx.streamlit.app/)** &nbsp;·&nbsp; **[Dataset Source](https://exoplanetarchive.ipac.caltech.edu/)** &nbsp;·&nbsp; Built with scikit-learn, SHAP, Optuna, Streamlit

---

## Motivation

When a planet passes in front of its host star, it blocks a small, periodic fraction of the star's light, a **transit**. NASA's Kepler space telescope stared at hundreds of thousands of stars for four years recording exactly this kind of brightness data, and it remains one of the primary methods used to discover exoplanets today.

Not every dip is a planet, though. Instrument noise, cosmic ray hits, and eclipsing binary stars can all produce transit-like signals. Kepler's own automated pipeline flags candidates, but final classification into `CONFIRMED` or `FALSE POSITIVE` requires further analysis. This project asks: **how well can a model learn to make that distinction from the underlying transit and stellar physics alone** without leaning on the pipeline's own pre-computed diagnostic judgment?

### Related methods

The initial detection of a periodic dip in a light curve is itself a well-established signal-processing problem, most classically addressed by the **Box Least Squares (BLS)** algorithm (Kovács, Zucker & Mazeh, 2002), which searches a light curve for the box-shaped periodic dimming characteristic of a transit. Several features used in this project — `koi_period`, `koi_duration`, `koi_depth` are essentially the *outputs* of that kind of detection process, already extracted from Kepler's raw light curves by its processing pipeline.

This project sits one layer downstream of that detection step: rather than searching raw light curves for candidate signals, it addresses the **classification** question, given a candidate that's already been flagged, is it a genuine transiting planet or a false positive? This is the same broad task Kepler's own automated vetting pipeline and the Robovetter system (Thompson et al., 2018) were built to perform, though using a substantially simpler feature set and model here (see [Benchmark](#benchmark-against-nasas-robovetter) below).

---

## Dataset

- **Source:** [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) — Kepler Objects of Interest (KOI) cumulative table
- **Raw size:** 9,564 candidates, 49 columns
- **After removing `CANDIDATE` rows** (unresolved, not a real class — see below): 7,586 rows
- **After removing rows with missing derived-physics values:** 7,327 rows
- **Final class balance:** 62.5% `FALSE POSITIVE`, 37.5% `CONFIRMED`

### Why `CANDIDATE` was dropped

The dataset has three disposition labels: `CONFIRMED`, `FALSE POSITIVE`, and `CANDIDATE`. `CANDIDATE` doesn't represent a physical class, it means the object hasn't been reviewed to a final verdict yet. Including it as a third label would ask the model to predict "still undecided," which isn't a coherent target. This project trains a **binary classifier**: `CONFIRMED` vs `FALSE POSITIVE`, mirroring how real vetting pipelines treat the category.

### Missing data pattern

259 rows were missing key derived-physics columns (planet radius, equilibrium temperature, etc.). Checking *why* mattered: 257 of those 259 were `FALSE POSITIVE` rows signals so clearly not transit-shaped that the pipeline never bothered computing derived properties for them. Dropping these barely shifted the class balance (63.8% → 62.5%), confirming it was safe, and these were also the *easiest* false positives, so their removal is unlikely to have hurt the model's ability to catch subtler cases.

---

## Methodology

| Step | What happened |
|---|---|
| 1 | Data cleaning, column-by-column physical interpretation, feature selection, stratified 80/20 train/test split |
| 2 | Baseline logistic regression |
| 3 | Random Forest & Gradient Boosting, **with and without** pipeline diagnostic flags |
| 4 | Evaluation with domain-specific cost framing (false negatives vs. false positives), precision-recall/threshold analysis |
| 5 | SHAP-based interpretability |
| 6 | Interactive Streamlit dashboard |
| 7 | 5-fold cross-validation, Optuna hyperparameter tuning, PR-AUC, benchmark against NASA's Robovetter |

### Features used

**Included**:
`koi_period`, `koi_impact`, `koi_duration`, `koi_depth`, `koi_prad`, `koi_teq`, `koi_insol`, `koi_model_snr`, `koi_steff`, `koi_slogg`, `koi_srad`, `koi_kepmag`

**Deliberately excluded:**
- `koi_pdisposition`, `koi_score` — the pipeline's own prediction/confidence. Including these would be **data leakage**: the model would learn to copy an existing verdict rather than learn from physics.
- `kepid`, `kepoi_name`, `kepler_name` identifiers with no physical meaning
- `ra`, `dec` — sky coordinates, physically irrelevant to whether a signal is a real planet
- `koi_tce_delivname` internal pipeline metadata

---

## Key finding: the 99% that wasn't quite real

A first-pass model (logistic regression, all features including `koi_fpflag_*`) scored **~99% accuracy**. That number is real, but misleading on its own.

The four `koi_fpflag_*` columns are Kepler's **own automated diagnostic verdict** on whether a signal looks transit-shaped, comes from the wrong star, or matches a known false-positive pattern. A model trained with these included dominates almost entirely on them which means it's largely **replicating an existing decision**.

Retraining the same model architecture **without** those four flags dropped accuracy to a still-strong but far more honest **~91-92%**. That gap is itself the most scientifically interesting number in this project: it quantifies how much of the "easy" signal in this dataset comes from a pre-existing pipeline judgment versus how much is recoverable from raw physics alone.

| Model | Features | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | All (incl. flags) | 0.99 | 0.99 | 0.997 |
| Random Forest | All (incl. flags) | 0.99 | 0.99 | 0.999 |
| Gradient Boosting | All (incl. flags) | 0.99 | 0.99 | 1.000 |
| Random Forest | Flags removed | 0.91 | 0.88 | 0.972 |
| Gradient Boosting | Flags removed (single split) | 0.91 | 0.89 | 0.974 |
| Gradient Boosting | **Flags removed, 5-fold CV, default params** | **0.922 ± 0.006** | **0.898 ± 0.008** | **0.974 ± 0.005** |
| Gradient Boosting | **Flags removed, 5-fold CV, tuned params** | **0.929 ± 0.007** | **0.906 ± 0.009** | **0.976 ± 0.004** |

The single-split numbers (first five rows) are from initial development; the cross-validated rows are the results actually trusted for this project's conclusions — see below.

> **Why this matters beyond this one dataset:** this finding is a concrete instance of a broader, well-documented ML phenomenon sometimes called **shortcut learning** where a model achieves high performance by exploiting an incidental, easy-to-find signal correlated with the label, rather than learning the causal or physically meaningful features the task actually intends to test (Geirhos et al., 2020, *"Shortcut Learning in Deep Neural Networks"*). Here, the `koi_fpflag_*` columns are literally a prior model's output stored as a feature, making the shortcut unusually direct, but the same failure mode shows up in less obvious forms constantly in applied ML, and checking for it explicitly.

---

## Evaluation: cross-validation, tuning, and benchmarking

A single 80/20 train/test split can over- or under-state performance depending on which rows happen to land in the test set. To report numbers that are actually trustworthy, the flags-removed Gradient Boosting model was re-evaluated with:

### 5-fold stratified cross-validation

This reports mean ± standard deviation across 5 folds, the low standard deviations (all under 1%) indicate the single-split results weren't a lucky or unlucky draw.

### Hyperparameter tuning (Optuna)

Default hyperparameters (`n_estimators=200`, otherwise sklearn defaults) are rarely optimal. A 12-trial Bayesian optimization search (Optuna's TPE sampler) over `n_estimators`, `max_depth`, `learning_rate`, and `subsample` optimizing for F1 via 3-fold CV during search:

```
n_estimators:   290
max_depth:      6
learning_rate:  0.287
subsample:      0.855
```

This produced a modest, honestly-reported improvement (F1: 0.898 → 0.906): the default configuration was already reasonably well-suited to this problem.

### Precision-Recall AUC

Given the dataset's imbalance (62.5% / 37.5%), **PR-AUC (Average Precision) is reported alongside ROC-AUC**, since ROC-AUC can look optimistic on imbalanced data, the large number of true negatives inflates the false-positive-rate denominator, making the metric less sensitive to minority-class performance than PR-AUC. The tuned model's PR-AUC: **0.956 ± 0.005**.

### Benchmark against NASA's Robovetter

Kepler's own production vetting system, the **Robovetter** (Thompson et al., 2018, DR25 catalog paper), reports:
- **Completeness > 85%** for orbital periods under 100 days (fraction of real transit signals correctly recovered)
- **Reliability ~98%** (inferred from a related audit finding only ~2% of 306 borderline pipeline-rejected signals were later judged by human review to be real)

This project's tuned model, evaluated with proper out-of-sample cross-validated predictions:
- **Recall (completeness analog): 91.5%**
- **Precision (reliability analog): 89.7%**

**This is not a like-for-like comparison**, and that caveat matters: the Robovetter operates on raw threshold-crossing events before any KOI designation exists, uses injection/recovery testing on real and simulated data, and has access to far more diagnostic information than this project's tabular feature set. This model lands in a broadly comparable range on completeness, and somewhat behind on reliability/precision read as "operating in the right neighborhood" of a real NASA production system, not a claim of matching or exceeding it.

*Full results, including per-fold score distributions and the hyperparameter search trajectory, are available in the dashboard's "Evaluation" tab and in `evaluation_and_tuning.py`.*

---

## What actually drives the predictions

Using [SHAP](https://github.com/shap/shap) (SHapley Additive exPlanations) rather than the model's built-in feature importances, since SHAP captures direction and interaction effects that raw importance scores miss. SHAP's underlying method traces back to Shapley values from cooperative game theory (Shapley, 1953), a way of fairly attributing a shared outcome among contributing "players", adapted into a practical, model-agnostic feature attribution method by Lundberg & Lee (2017), which is the specific implementation used here:

1. **Planet radius (`koi_prad`)** and **signal-to-noise ratio (`koi_model_snr`)** together account for roughly 60% of the model's decision-making.
2. Planet radius has a **non-monotonic** relationship with the prediction: moderate radii (roughly Earth-to-Neptune sized) push toward `CONFIRMED`, while both very large and very small radii push toward `FALSE POSITIVE`. This matches physical intuition implausibly large "planets" are often eclipsing binary stars in disguise, and near-zero radii are often noise artifacts. 
3. Very short orbital periods (under ~1 day) are disproportionately associated with `FALSE POSITIVE`, consistent with such signals more often being eclipsing binaries than planets.

---

## Evaluation beyond accuracy

Accuracy treats every mistake as equally costly, which isn't true here:

- **A false negative** (missing a real planet) discards a genuine discovery outright.
- **A false positive** (wrongly flagging a non-planet) wastes expensive follow-up telescope time, but the mistake is typically caught during that follow-up.

Because these costs aren't symmetric, the project explores **threshold tuning** rather than defaulting to the standard 0.5 cutoff: lowering the decision threshold from 0.5 to 0.3 recovers 25 additional real planets at the cost of 34 additional false alarms.

One notable individual error surfaced during this analysis: a Jupiter-sized candidate (`koi_prad` ≈ 16.7) with an extremely clean signal (`koi_model_snr` ≈ 2590) was still misclassified as `FALSE POSITIVE` with high model confidence. This is flagged as a specific limitation worth further investigation: very large planets may produce transit shapes the model has learned to associate with eclipsing binaries.

---

## Interactive dashboard

**[Live app](https://exoplanetdetectionviatransitphotometry-ffkjcb3vnewuwawcrynykx.streamlit.app/)**

The Streamlit dashboard has four tabs:

- **Try a Prediction** — adjust transit/stellar measurements via sliders and see a live prediction, with a SHAP-based explanation of *why* the model decided what it decided
- **Browse Test Examples** — real test-set candidates, filterable by correct predictions vs. specific error types
- **Model Details** — overall performance and the flags-included-vs-excluded comparison in plain language
- **Rigorous Evaluation** — cross-validation results, hyperparameter search progress, and the Robovetter benchmark comparison, read from pre-computed results.

---

## Limitations & future work

- **Tabular features only.** This project uses pre-extracted features (period, depth, radius, etc.), not raw light-curve time series. A natural extension is working directly with flux measurements via the `lightkurve` package, which would allow CNN- or transformer-based approaches on the actual light curves rather than hand-engineered summary statistics.
- **The large-planet false negative** noted above suggests the model may underperform on the tail end of the planet-radius distribution — worth a dedicated error analysis.
- **Uncertainty columns** (`_err1`/`_err2` pairs for each measurement) were excluded from this version for simplicity; incorporating measurement uncertainty directly could improve performance, especially for borderline cases.
- **`koi_model_snr` may carry residual pipeline judgment.** It's derived from a fitted transit model rather than a raw measurement, so it's not entirely free of the same "pre-computed diagnostic" concern that motivated removing the flag columns.
- **The Robovetter benchmark is not apples-to-apples** — a fairer comparison would require replicating this model's evaluation on the exact same injection/recovery test set NASA used.

---

## Repository structure

```
01_preprocessing.py         # Data cleaning, feature selection, train/test split
02_baseline_model.py        # Logistic regression baseline
03_improved_models.py       # Random Forest & Gradient Boosting, with/without flags
04_evaluation_context.py    # Precision-recall analysis, threshold tuning
05_interpretability.py      # SHAP global + individual prediction explanations
06_evaluation.py    # 5-fold CV, Optuna tuning, PR-AUC, Robovetter benchmark
07_dashboard.py             # Interactive Streamlit dashboard
requirements.txt
README.md                   # This file
```

### Running locally

```bash
pip install -r requirements.txt

# Download koi_cumulative.csv from the NASA Exoplanet Archive first, then:
python 01_preprocessing.py
python 02_baseline_model.py
python 03_improved_models.py
python 04_evaluation_context.py
python 05_interpretability.py
python 06_evaluation.py   # takes ~4-5 min; generates results JSON for the dashboard
streamlit run 07_dashboard.py
```

---

## References

- Kovács, G., Zucker, S., & Mazeh, T. (2002). A box-fitting algorithm in the search for periodic transits. *Astronomy & Astrophysics*, 391(1), 369-377.
- Thompson, S. E., et al. (2018). Planetary Candidates Observed by Kepler. VIII. A Fully Automated Catalog With Measured Completeness and Reliability Based on Data Release 25. *The Astrophysical Journal Supplement Series*, 235(2), 38.
- Shapley, L. S. (1953). A value for n-person games. *Contributions to the Theory of Games*, 2(28), 307-317.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665-673.

## Data attribution

Data from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/), operated by the California Institute of Technology under contract with NASA. This project makes use of the Kepler Objects of Interest (KOI) cumulative table.

---

## Author

**Jarin Binta Yeasin**
| Undergraduate student, Department of Mass Communication and Journalism
| University of Dhaka

- 📧 jarinyeasin@gmail.com
- 🔗 [LinkedIn](www.linkedin.com/in/jarin-binta-yeasin-b61b88278)
- 🐙 [GitHub](https://github.com/jarinyeasin)

---
