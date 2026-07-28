# PaySim Fraud Detector 🚨

A real-time fraud detection system built on the PaySim synthetic financial dataset. It trains an XGBoost model, explains its decisions with SHAP, monitors for concept drift using ADWIN, and ships everything as an interactive Streamlit app.

**[→ Live demo](https://YOUR-APP-URL.streamlit.app)** — open it and click Score Transaction on the first screen. You should get a fraud flag within seconds.

---

## Why I built this

Most fraud detection portfolios stop at a confusion matrix. This one goes further and asks the more realistic question: *what happens after deployment?*

Fraudsters don't stay still. They adapt to whatever defenses exist, which means a model trained today can quietly become useless six months from now — without throwing a single error or raising any alarm. This project builds the monitoring layer that catches that decay before it becomes a real problem.

The starting point for this dataset is also a good story. PaySim includes a column called `isFlaggedFraud` — a hardcoded business rule that flags any transfer over 200,000. Sounds reasonable. In practice it fires **16 times** across 6.36 million transactions and catches fewer than 0.2% of real frauds. The entire project is the answer to the question that raises.

---

## What the app does

Three tabs, three audiences.

**Score a Transaction** — enter a transaction's raw fields (amount, account balances, transaction type) and the app computes the engineered features automatically, scores it with the model, and tells you whether it's flagged as fraud. A feature-deviation chart shows which inputs pushed the decision in each direction — a simplified explainability view for anyone who doesn't want to read a SHAP paper.

**Model Performance** — the metrics panel. PR-AUC as the headline number (not accuracy — more on that below), precision-recall curve, confusion matrix at the tuned threshold, and SHAP global feature importance. Everything you'd want to see if you were reviewing this model before putting it into production.

**Drift Monitor** — this is the part most projects skip. It replays 20,000 test transactions in temporal order through an ADWIN drift detector, shows the rolling error rate as a line chart, and marks exactly where the detector fired. Drift was simulated by flipping the strongest fraud signal feature at the 70% mark — you can see the error rate climb and the alert land.

---

## Results

| Metric | 
  PR-AUC
  ROC-AUC
  F1 (fraud class)
  Precision
  Recall
  Decision threshold

The decision threshold was tuned using a cost function — a missed fraud costs $500, a false alarm costs $10 — rather than defaulting to 0.5. At 0.5 the model performs differently; at the tuned threshold it reflects something closer to what a real business would actually care about.

**Why PR-AUC and not accuracy?** PaySim has a 0.129% fraud rate. A model that predicts "legitimate" for every single transaction scores 99.871% accuracy while catching zero fraud. Accuracy is the wrong metric here. PR-AUC focuses only on the positive class and can't be inflated by the model correctly ignoring millions of obvious legitimate transactions.

---

## How it works

The pipeline runs in 15 stages across seven notebooks. Here's the short version.

**Data** — PaySim has 6.36 million simulated mobile money transactions over 30 days. Fraud only appears in two transaction types: TRANSFER and CASH_OUT. Everything else is irrelevant noise. Filtering to just those two types removes about 5 million rows before the model ever sees the data.

**Feature engineering** — the most useful features aren't in the raw data. They're derived from it. The key insight: for a legitimate transaction, the accounting has to balance. If you send $50,000, the sender's balance should drop by $50,000 and the receiver's should rise by $50,000. When that accounting doesn't add up — `errorBalanceOrig` and `errorBalanceDest` are non-zero — something went wrong. These two engineered features consistently rank at the top of the SHAP importance chart.

**Class imbalance** — at 0.129% fraud, the class imbalance is severe. Three approaches were tested: logistic regression with balanced class weights (baseline), XGBoost with `scale_pos_weight` (primary model), and XGBoost with SMOTE in a pipeline (comparison). SMOTE was applied strictly after the train/test split — a common mistake is applying it before, which leaks synthetic data into the test set and produces unrealistically high scores.

**Evaluation** — the train/test split is time-aware. Train on earlier transactions, test on later ones. This mirrors how the model would actually be used: trained on the past, predicting the future.

**Explainability** — SHAP values for both global feature importance and per-prediction explanations. The balance-error features dominate the global chart, which matches the domain intuition: legitimate transactions balance their books, fraudulent ones don't.

**Drift monitoring** — ADWIN (Adaptive Windowing) monitors the model's binary error stream in real time. It maintains a variable-length window and signals drift when two sub-windows within it diverge beyond what statistical chance would explain. It doesn't need a fixed window size or a manually tuned threshold — it figures out both automatically.

---

## Project structure

```
├── app.py                              # Streamlit app — run this
├── requirements.txt                    # deployment dependencies
│
├── models/
│   └── streamlit_bundle.joblib         # model + metadata in one file
│
├── data/
│   └── sample/
│       └── demo_transactions.parquet   # 500 pre-scored rows for the demo
│
├── reports/
│   └── figures/                        # plots generated by the notebooks
│
└── notebooks/
    ├── fraud_detection_stage1_2.py     # environment, data loading, inspection
    ├── fraud_detection_stage3_4.py     # cleaning and EDA
    ├── fraud_detection_stage5_6.py     # feature engineering, train/test split
    ├── fraud_detection_stage7_8.py     # imbalance handling, model training
    ├── fraud_detection_stage9_10.py    # evaluation, threshold tuning
    ├── fraud_detection_stage11_12.py   # SHAP explainability, drift monitoring
    └── fraud_detection_stage13_14.py   # deployment bundle, Streamlit export
```

The raw PaySim CSV is not in this repository — it's 470 MB and doesn't belong in git. Download it from Kaggle (link below) and point `DATA_PATH` in Stage 1 to it.

---

## Run it locally

```bash
git clone https://github.com/YOUR_USERNAME/paysim-fraud-detector
cd paysim-fraud-detector
pip install -r requirements.txt
streamlit run app.py
```

The app loads from `models/streamlit_bundle.joblib` — no retraining needed. It should be running at `http://localhost:8501` in about ten seconds.

To reproduce the full pipeline from scratch, download the dataset and run the notebooks in order (Stages 1 through 14).

---

## Tech stack

| Area | Tools |
|---|---|
| Data | pandas, NumPy, PyArrow |
| Modelling | scikit-learn, XGBoost, imbalanced-learn |
| Explainability | SHAP |
| Drift detection | River (ADWIN) |
| Visualisation | matplotlib, seaborn, Plotly |
| App | Streamlit |
| Persistence | joblib |

---

## Dataset

[PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) — E.A. Lopez-Rojas, A. Elmir, S. Axelsson. *PaySim: A Financial Mobile Money Simulator for Fraud Detection.* EMSS 2016.

6,362,620 transactions. 8,213 frauds. 0.129% positive class.

---

## Honest caveats

**PaySim is synthetic.** The fraud in this dataset follows a clean, predictable script — fraudsters transfer a balance to a mule account and cash out. Real fraud is messier, more varied, and actively tries to evade detection. The strong results here are partly a function of how structured the simulation is. Treat the metrics as a learning outcome, not a production benchmark.

**The drift simulation is illustrative.** Injecting drift by flipping a feature value is a reproducible way to test a detector, but it's not the same as the slow, organic drift that happens in real systems. The real challenge is detecting drift that nobody designed and nobody is expecting.

**Labels arrive late in production.** Customers often report fraud days after it happens. This project assumes labels are available immediately, which wouldn't be true in a live system. Late labels make online drift detection significantly harder and are worth acknowledging in any production design.

---
