# app.py — PaySim Fraud Detector
# ─────────────────────────────────────────────────────────────────────────────
# Run locally  :  streamlit run app.py
# Needs (relative to repo root):
#   models/streamlit_bundle.joblib          (built by Stage 13)
#   data/sample/demo_transactions.parquet   (built by Stage 13)
#   reports/figures/*.png                   (optional — missing images are skipped)
# ─────────────────────────────────────────────────────────────────────────────
# Oct 2026 fixes:
#   • The bundle now holds the SAME model the threshold was tuned on (it used to
#     ship a one-tree model that could never score above the threshold).
#   • Metrics come from one dict written by Stage 10 (they used to show 0).
#   • The drift tab shows the real Stage 12 ADWIN replay.
#   • New "Signal audit" section: what the near-perfect score depends on.
#   • Default form values are a real PaySim fraud pattern (exact balance drain).

import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title='PaySim Fraud Detector',
    page_icon='\U0001F6A8',
    layout='wide',
    initial_sidebar_state='expanded',
)

BUNDLE_PATH    = 'models/streamlit_bundle.joblib'
DEMO_DATA_PATH = 'data/sample/demo_transactions.parquet'
FIG_DIR        = 'reports/figures'


@st.cache_resource
def load_bundle():
    if not os.path.exists(BUNDLE_PATH):
        st.error('Bundle not found: ' + BUNDLE_PATH + ' — run Stage 13 first.')
        st.stop()
    return joblib.load(BUNDLE_PATH)


@st.cache_data
def load_demo():
    if os.path.exists(DEMO_DATA_PATH):
        return pd.read_parquet(DEMO_DATA_PATH)
    return None


def fig_path(name):
    path = os.path.join(FIG_DIR, name)
    return path if os.path.exists(path) else None


bundle  = load_bundle()
demo_df = load_demo()

model         = bundle['model']
MODEL_NAME    = bundle.get('model_name', 'XGBoost')
FEATURE_COLS  = bundle['feature_cols']
THRESHOLD     = float(bundle['threshold'])
metrics       = bundle['metrics']
train_stats   = bundle.get('train_stats', {})
shap_ranking  = bundle.get('shap_ranking', [])
shap_mean_abs = bundle.get('shap_mean_abs', {})
drift         = bundle.get('drift')
ablation      = bundle.get('ablation')

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('\U0001F6A8 PaySim Fraud Detector')
    st.caption(MODEL_NAME + ' + ADWIN drift monitor')
    st.divider()
    st.markdown('**Test-set snapshot** (final 20% of time)')
    st.metric('PR-AUC', f"{metrics['pr_auc']:.4f}",
              help='Primary metric. No-skill baseline = test fraud rate '
                   f"({metrics['fraud_rate_pct'] / 100:.4f}).")
    st.metric('Recall', f"{metrics['recall']:.2%}",
              help='Share of real frauds caught at the tuned threshold.')
    st.metric('Threshold', f'{THRESHOLD:.3f}',
              help='Cost-tuned on the validation set: missed fraud $'
                   + str(bundle.get('fn_cost', 500)) + ', false alarm $'
                   + str(bundle.get('fp_cost', 10)) + '.')
    st.divider()
    st.caption('Built    : ' + bundle.get('created_at', 'N/A')[:10])
    st.caption('XGBoost  : ' + bundle.get('xgboost_version', 'N/A'))
    st.caption('sklearn  : ' + bundle.get('sklearn_version', 'N/A'))
    st.divider()
    st.markdown('**Features (' + str(len(FEATURE_COLS)) + ')**')
    for f in FEATURE_COLS:
        st.caption('• ' + f)

tab1, tab2, tab3 = st.tabs([
    '\U0001F50D Score a Transaction',
    '\U0001F4CA Model Performance',
    '\U0001F4E1 Drift Monitor',
])

# ==========================================================================
# TAB 1 — Score a transaction
# ==========================================================================
with tab1:
    st.header('Score a Transaction')
    st.markdown(
        'Enter the raw transaction fields. The engineered features '
        '(balance errors, zero-balance flags, type flag) are computed '
        'automatically, then the model scores the transaction.'
    )
    st.caption(
        'Defaults are a typical PaySim fraud: a TRANSFER that moves the sender\'s '
        'entire balance to an empty account. Try lowering the amount below the '
        'sender\'s balance and watch the score change.'
    )

    with st.form('score_form'):
        c1, c2 = st.columns(2)
        with c1:
            txn_type = st.selectbox('Transaction Type', ['TRANSFER', 'CASH_OUT'],
                                    help='All PaySim fraud is TRANSFER or CASH_OUT.')
            amount = st.number_input('Amount', min_value=0.0, value=181_000.0, step=1_000.0)
            old_bal_orig = st.number_input('Sender balance BEFORE (oldbalanceOrg)',
                                           min_value=0.0, value=181_000.0, step=1_000.0)
            new_bal_orig = st.number_input('Sender balance AFTER (newbalanceOrig)',
                                           min_value=0.0, value=0.0, step=1_000.0)
        with c2:
            old_bal_dest = st.number_input('Receiver balance BEFORE (oldbalanceDest)',
                                           min_value=0.0, value=0.0, step=1_000.0)
            new_bal_dest = st.number_input('Receiver balance AFTER (newbalanceDest)',
                                           min_value=0.0, value=0.0, step=1_000.0)
        submitted = st.form_submit_button('\U0001F50D Score Transaction',
                                          width='stretch')

    if submitted:
        # Must mirror Stage 5 feature engineering exactly
        err_orig = float(new_bal_orig + amount - old_bal_orig)
        err_dest = float(old_bal_dest + amount - new_bal_dest)
        row = {
            'amount'               : float(amount),
            'type_TRANSFER'        : int(txn_type == 'TRANSFER'),
            'oldbalanceOrg'        : float(old_bal_orig),
            'newbalanceOrig'       : float(new_bal_orig),
            'oldbalanceDest'       : float(old_bal_dest),
            'newbalanceDest'       : float(new_bal_dest),
            'errorBalanceOrig'     : err_orig,
            'errorBalanceDest'     : err_dest,
            'flag_orig_zero_after' : int(new_bal_orig == 0.0),
            'flag_dest_zero_before': int(old_bal_dest == 0.0),
            'flag_dest_zero_both'  : int(old_bal_dest == 0.0 and new_bal_dest == 0.0),
        }
        X = pd.DataFrame([row])[FEATURE_COLS]   # enforce training column order
        prob = float(model.predict_proba(X)[0, 1])
        flag = prob >= THRESHOLD

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric('Fraud probability', f'{prob:.1%}')
        m2.metric('Decision threshold', f'{THRESHOLD:.3f}')
        m3.metric('Decision', '\U0001F6A8 FRAUD' if flag else '✅ LEGITIMATE')
        if flag:
            st.error(f'**Flagged as likely fraud** — P(fraud) = {prob:.1%} ≥ {THRESHOLD:.3f}')
        else:
            st.success(f'**Cleared as legitimate** — P(fraud) = {prob:.1%} < {THRESHOLD:.3f}')

        if train_stats:
            st.subheader('How this transaction compares with training data')
            st.caption(
                'Each bar: (value − legitimate average) ÷ (fraud average − legitimate average). '
                '0 = looks like a typical legitimate row, 1 = looks like a typical fraud row. '
                'This is a quick visual, not a SHAP explanation.'
            )
            legit_m = train_stats.get('legit_mean', {})
            fraud_m = train_stats.get('fraud_mean', {})
            devs = []
            for feat in FEATURE_COLS:
                gap = fraud_m.get(feat, 0) - legit_m.get(feat, 0)
                devs.append((row[feat] - legit_m.get(feat, 0)) / gap if abs(gap) > 1e-9 else 0.0)
            devs = np.clip(devs, -3, 3)
            fig_dev = go.Figure(go.Bar(
                x=devs, y=FEATURE_COLS, orientation='h',
                marker_color=['#F44336' if d > 0.5 else '#2196F3' for d in devs],
            ))
            fig_dev.update_layout(xaxis_title='0 = typical legitimate · 1 = typical fraud (clipped at ±3)',
                                  height=400, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_dev, width='stretch')

        st.caption(
            f'Derived — errorBalanceOrig: {err_orig:,.0f} | errorBalanceDest: {err_dest:,.0f} | '
            f"flag_orig_zero_after: {row['flag_orig_zero_after']} | "
            f"flag_dest_zero_before: {row['flag_dest_zero_before']} | "
            f"flag_dest_zero_both: {row['flag_dest_zero_both']}"
        )

    if demo_df is not None:
        st.divider()
        st.subheader('Pre-scored example transactions')
        n_f = int(demo_df['isFraud'].sum())
        caught = int(demo_df.loc[demo_df['isFraud'] == 1, 'flagged'].sum())
        st.caption(
            f'{n_f} frauds + {len(demo_df) - n_f} legitimate rows from the test set '
            f'(enriched with fraud — the real test rate is {metrics["fraud_rate_pct"]:.2f}%). '
            f'Flagged at the tuned threshold: {caught} of {n_f} frauds.'
        )
        show = [c for c in ['isFraud', 'flagged', 'fraud_prob', 'amount', 'oldbalanceOrg',
                            'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'] if c in demo_df.columns]
        st.dataframe(demo_df[show].sort_values('fraud_prob', ascending=False).head(20),
                     width='stretch')

# ==========================================================================
# TAB 2 — Model performance
# ==========================================================================
with tab2:
    st.header('Model Performance — test set')
    st.caption(
        f"{MODEL_NAME}. Test set = final 20% of PaySim time: {metrics['test_rows']:,} "
        f"TRANSFER/CASH_OUT transactions, {metrics['test_frauds']:,} frauds. "
        'Model and threshold were chosen on a separate validation slice; '
        'the test set was used once, for these numbers.'
    )
    c = st.columns(5)
    c[0].metric('PR-AUC', f"{metrics['pr_auc']:.4f}",
                help=f"No-skill baseline: {metrics['fraud_rate_pct'] / 100:.4f}")
    c[1].metric('ROC-AUC', f"{metrics['roc_auc']:.4f}", help='Optimistic under heavy class imbalance')
    c[2].metric('Recall', f"{metrics['recall']:.4f}",
                help=f"{metrics['tp']:,} of {metrics['test_frauds']:,} frauds caught")
    c[3].metric('Precision', f"{metrics['precision']:.4f}",
                help=f"{metrics['fp']:,} false alarms")
    c[4].metric('F1', f"{metrics['f1']:.4f}")

    st.divider()
    left, right = st.columns(2)
    with left:
        if fig_path('pr_curves_comparison.png'):
            st.subheader('Precision-recall curves')
            st.image(fig_path('pr_curves_comparison.png'))
        if fig_path('confusion_matrix_threshold_comparison.png'):
            st.subheader('Confusion matrix — default vs tuned threshold')
            st.image(fig_path('confusion_matrix_threshold_comparison.png'))
    with right:
        if shap_ranking and shap_mean_abs:
            st.subheader('Global feature importance (mean |SHAP|)')
            vals = [shap_mean_abs.get(f, 0) for f in shap_ranking]
            fig_s = go.Figure(go.Bar(x=vals[::-1], y=shap_ranking[::-1],
                                     orientation='h', marker_color='#F44336'))
            fig_s.update_layout(xaxis_title='Mean |SHAP value|', height=420,
                                margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_s, width='stretch')
        if fig_path('threshold_cost_curve.png'):
            st.subheader('Cost vs threshold (validation set)')
            st.image(fig_path('threshold_cost_curve.png'))

    st.divider()
    st.subheader('Signal audit — what is the near-perfect score made of?')
    if ablation:
        rule = ablation['rule']
        st.markdown(
            f"**One rule — `{rule['name']}` — catches {rule['recall']:.1%} of test fraud "
            f"at {rule['precision']:.2%} precision.** PaySim's fraud script empties the "
            "sender's account, so the score mostly reflects the simulator, not model skill. "
            "Features built from the balance *after* the transaction (`newbalance*`, "
            "`errorBalance*`, `flag_orig_zero_after`, `flag_dest_zero_both`) also wouldn't "
            "exist when a real-time system scores a payment."
        )
        tbl = pd.DataFrame(ablation['results'])[['feature_set', 'n_features', 'test_pr_auc']]
        tbl.columns = ['Feature set', 'Features', 'Test PR-AUC']
        tbl['Test PR-AUC'] = tbl['Test PR-AUC'].map('{:.4f}'.format)
        st.dataframe(tbl, width='stretch', hide_index=True)
        st.caption(f"Same XGBoost settings for all three. No-skill baseline: {ablation['baseline']:.4f}.")
        if fig_path('ablation_signal_audit.png'):
            st.image(fig_path('ablation_signal_audit.png'))
    else:
        st.info('Run Stage 15 (signal audit), then Stage 13, to add this section.')

    st.divider()
    st.subheader('Why PR-AUC, not accuracy?')
    fr = metrics['fraud_rate_pct']
    st.markdown(
        f"In the test set **{fr:.2f}%** of transactions are fraud (0.129% across all 6.36M "
        'PaySim rows, before filtering to TRANSFER/CASH_OUT). A model that always says '
        f'"legitimate" scores **{100 - fr:.2f}% accuracy** and catches nothing. '
        'PR-AUC looks only at how well fraud is ranked; its no-skill value equals the '
        f'fraud rate (**{fr / 100:.4f}**).'
    )

# ==========================================================================
# TAB 3 — Drift monitor
# ==========================================================================
with tab3:
    st.header('Concept Drift Monitor')
    if drift:
        st.markdown(
            '**Scenario:** fraudsters notice that emptying an account gets caught, so after '
            f"the injection point they switch to a **{drift['scenario']}**. "
            f"The model is replayed over all {drift['stream_len']:,} test transactions in time "
            'order, and ADWIN watches its error stream (1 = wrong, 0 = right).'
        )
        k = st.columns(4)
        k[0].metric('Fraud recall, no drift', f"{drift['recall_after_original']:.1%}",
                    help='Recall on the same post-injection fraud rows if fraudsters had not adapted')
        k[1].metric('Fraud recall after drift', f"{drift['recall_after_drifted']:.1%}")
        ev = drift['events_drifted']
        k[2].metric('ADWIN alerts', f'{len(ev)}',
                    help=f"Baseline (no-drift) stream: {len(drift['events_baseline'])} alerts")
        k[3].metric('First alert after injection',
                    f"{ev[0] - drift['inject_idx']:,} txns" if ev else 'none')

        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=drift['idx'], y=drift['rolling_original_pct'], mode='lines',
                                   name='No drift', line=dict(color='#90A4AE', width=1.2)))
        fig_d.add_trace(go.Scatter(x=drift['idx'], y=drift['rolling_drifted_pct'], mode='lines',
                                   name='Drifted stream', line=dict(color='#F4511E', width=1.8)))
        fig_d.add_vline(x=drift['inject_idx'], line_dash='dot', line_color='#FF9800',
                        annotation_text='Drift injected', annotation_position='top left')
        if ev:
            fig_d.add_vline(x=ev[0], line_dash='dash', line_color='#C62828',
                            annotation_text='First ADWIN alert', annotation_position='top right')
        fig_d.update_layout(
            xaxis_title='Transaction index (time order)',
            yaxis_title=f"Rolling error rate % ({drift['rolling_window']:,}-txn window)",
            height=430, margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation='h', y=1.08),
        )
        st.plotly_chart(fig_d, width='stretch')
        st.caption(
            f"Error rate before injection {drift['error_rate_before_pct']:.3f}% → "
            f"after {drift['error_rate_after_pct']:.3f}%. ADWIN delta = {drift['adwin_delta']}."
        )
        st.markdown('''**Response playbook:**
1. **Alert** — log the time and notify the model owner.
2. **Diagnose** — real behaviour change, a data-pipeline bug, or a seasonal blip?
3. **Decide** — retrain only when the change is confirmed and fresh labels exist.''')
        st.info(
            'Caveat: this replay assumes fraud labels arrive instantly. In production, '
            'chargebacks and reports arrive days later, so error-based drift alerts lag. '
            'Monitoring input distributions (e.g. the share of transfers that empty an '
            'account) catches this kind of shift sooner.'
        )
    else:
        st.info('Drift replay not in bundle — run Stage 12, then Stage 13.')

    st.divider()
    st.subheader('About ADWIN (Adaptive Windowing)')
    st.markdown('''
**ADWIN** (Bifet & Gavaldà, SDM 2007) keeps a variable-length window over the
model's error stream. At each new transaction it checks whether the window can be
split into an older and a newer part whose error rates differ by more than chance
(a Hoeffding bound). If yes, it raises an alert and drops the older part.
Lower `delta` = more sensitive.
''')
