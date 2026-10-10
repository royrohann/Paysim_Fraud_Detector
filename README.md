# PaySim Fraud Detector 🚨

A fraud-detection project built on the PaySim synthetic mobile-money dataset. It trains XGBoost, sets the decision threshold by business cost, explains decisions with SHAP, replays a realistic fraud-tactic change through an ADWIN drift monitor, and serves everything in a Streamlit app.

It also includes a **signal audit** that answers the question a near-perfect score should raise: *what is this number actually made of?*

**[→ Live demo](https://paysimfrauddetector-bahcxzay2gufai7aemcplh.streamlit.app/)** — click **Score Transaction** on the first tab. The default inputs are a typical PaySim fraud, and the app flags it.

---

## Results (test set = final 20% of simulated time)

| | Value |
|---|---|
| Test rows (TRANSFER + CASH_OUT) | 552,504 — 4,258 frauds |
| No-skill PR-AUC baseline (test fraud rate) | 0.0077 |
| XGBoost PR-AUC | 1.0000 (rounded) |
| At the tuned threshold (0.970) | 4,256 of 4,258 frauds caught, 4 false alarms — recall 0.9995, precision 0.9991 |
| At the default 0.5 threshold | 4,257 caught, 7 false alarms |

Model selection and threshold tuning use a validation slice (last 10% of the training period). The test set is touched once.

**An honest note on the threshold:** the validation slice holds only 268 frauds and the model separates all of them, so the "cost-optimal" threshold (0.970) is poorly pinned down. On the test set it cost $1,040 (2 missed frauds × $500 + 4 false alarms × $10) versus $570 at 0.5. With so few validation frauds, a cost-tuned threshold isn't reliable — it needs a bigger validation window or cross-validation over time.

## Signal audit — why the score is so high

| | Test PR-AUC |
|---|---|
| A. Full model (11 features, incl. balances *after* the transaction) | 1.0000 |
| B. Decision-time features only | 0.9871 |
| C. Decision-time, without the sender's balance | 0.5908 |
| One rule: `amount == oldbalanceOrg` | recall 0.975 at precision 0.9998 |

- PaySim's fraud script **empties the sender's account**: 98% of frauds move the exact balance, versus ~0% of legitimate transactions. One `if` statement catches 97.5% of test fraud.
- `newbalanceOrig`, `newbalanceDest` and everything derived from them exist only **after** a transaction posts. A real-time system scoring a payment before it clears wouldn't have them (model B shows what's left without them).
- The balance-error features work the opposite way from the usual story: it is **legitimate** PaySim rows whose books don't balance (~90% have a non-zero `errorBalanceOrig`), while fraud, by draining the exact balance, balances perfectly.

Treat the headline metrics as a property of the simulator, not as evidence the model would perform like this on real payments.

## Drift monitor

Stage 12 replays all 552,504 test transactions in time order. At the 70% mark, fraudsters change tactics: they take 50–90% of the balance instead of all of it, and every dependent field is recomputed so each row stays consistent.

- Fraud recall on the post-change rows drops from **99.9% to 14.0%**.
- ADWIN (delta = 0.002) raises its first alert **3,839 transactions** after the change; the no-drift stream raises none.
- Caveat: the replay assumes labels arrive instantly. Real chargebacks arrive days later, so monitoring input distributions (e.g. the share of transfers that empty an account) would catch this shift sooner than error-based monitoring.

---

## What the app does

- **Score a Transaction** — enter raw fields; engineered features are computed and the model scores it. A chart compares the transaction with typical legitimate and fraud rows.
- **Model Performance** — test metrics, PR curves, confusion matrices at 0.5 vs the tuned threshold, SHAP importance, and the signal audit.
- **Drift Monitor** — the Stage 12 replay with the ADWIN alert marked.

## Pipeline

| Notebook | Stages |
|---|---|
| `Fraud_Detection_Stg01_02` | Load `data.csv`, inspect, save `data/paysim_raw.parquet` |
| `Fraud_Detection_Stg03_04` | Filter to TRANSFER/CASH_OUT, EDA |
| `Fraud_Detection_Stg05_06` | Feature engineering, time-aware split |
| `Fraud_Detection_Stg07_08` | LR baseline, XGBoost (scale_pos_weight), XGBoost + SMOTE |
| `Fraud_Detection_Stg09_10` | Validation-based model selection, cost-tuned threshold, test evaluation, saves `models/best_model.joblib` |
| `Fraud_Detection_Stg11_12` | SHAP explanations, drift replay with ADWIN |
| `Fraud_Detection_Stg15_Signal_Audit` | Rule baseline and feature-set ablation (run before Stage 13) |
| `Fraud_Detection_Stg13_14` | Builds `models/streamlit_bundle.joblib`, demo sample, `requirements.txt` |

Run notebooks from the repo root so the relative paths (`data/`, `models/`, `reports/figures/`) resolve.

### October 2026 fixes

- **Shipped model could never flag fraud.** Early stopping on PR-AUC stopped at the first tree (validation PR-AUC was already 1.0), so the bundled model only output 0.48–0.53 while the threshold was 0.763. Early stopping now watches logloss.
- **Model–threshold mismatch.** The threshold was tuned on the SMOTE model but the bundle always shipped the scale_pos_weight model. Stage 10 now saves the chosen model and Stage 13 ships exactly that, with a check that it reproduces Stage 10's recall.
- **Test-set leakage in decisions.** Model selection and threshold tuning moved from the test set to a validation slice; LR and SMOTE models no longer train on the validation rows.
- **App metrics showed 0.** Stage 13 guessed metric key names; Stage 10 now writes one `APP_METRICS` dict.
- **Drift demo didn't drift.** Negating `errorBalanceOrig` changed nothing (fraud's value is ≈ 0). Replaced with the partial-drain scenario above.
- **Figures missing on Linux.** `Reports/Figures` renamed to `reports/figures` to match the code (Streamlit Cloud is case-sensitive).
- **Version pins.** `xgboost` and `scikit-learn` pinned to the versions that built the bundle.
- Removed the notebook cell that patched SHAP's installed source (not needed with shap ≥ 0.46).

## Run it locally

```bash
git clone https://github.com/royrohann/Paysim_Fraud_Detector
cd Paysim_Fraud_Detector
pip install -r requirements.txt
streamlit run app.py
```

The app loads `models/streamlit_bundle.joblib`; no retraining needed. To rebuild from scratch, download the dataset, save it as `data.csv` in the repo root, and run the notebooks in the order above. Training notebooks also need `shap>=0.46 imbalanced-learn river seaborn "matplotlib<3.10"`.

## Tech stack

| Area | Tools |
|---|---|
| Data | pandas, NumPy, PyArrow |
| Modelling | scikit-learn, XGBoost, imbalanced-learn |
| Explainability | SHAP |
| Drift detection | River (ADWIN) |
| Visualisation | matplotlib, seaborn, Plotly |
| App | Streamlit |

## Dataset

[PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) — E.A. Lopez-Rojas, A. Elmir, S. Axelsson. *PaySim: A Financial Mobile Money Simulator for Fraud Detection.* EMSS 2016. 6,362,620 transactions, 8,213 frauds (0.129% overall; 0.30% within TRANSFER/CASH_OUT, where all fraud occurs).

## Caveats

- **PaySim is synthetic** and its fraud follows a fixed script. Real fraud is varied and adapts.
- **Labels arrive late** in production; this project assumes they're immediate.
- **The threshold is under-determined** by a validation slice with 268 frauds (see Results).
