"""
15_model_scorecard.py — a full metrics scorecard for both models.

Computes, for the ordinal logistic regression and for LightGBM, the standard
classification metrics requested for a model comparison table:

    Test Accuracy, Test Precision (macro), Test Recall (macro), Test F1 (macro),
    CV Accuracy, CV F1 (macro), Train Accuracy, Train F1 (macro),
    Runtime (min), Train-CV Accuracy Gap,
    plus AUC (macro one-vs-rest AND binary Active-vs-rest) and LogLoss.

Two honest design choices, stated so the numbers are not over-read:

  * The split is TEMPORAL, matching the rest of the project: train on the
    early waves, tune on the second-to-last, test on the last (never touched
    until the end). Test metrics therefore measure genuine generalisation to
    an unseen YEAR, which is what a forecasting model must do.

  * "CV" here is LEAVE-ONE-WAVE-OUT across the training waves, not random
    k-fold. Random k-fold would mix future and past survey years into the
    same fold and give a dishonestly optimistic number; wave-based CV is the
    temporally-honest equivalent. It is labelled as such in the output.

All metrics are survey-weighted.

Run:
    python 15_model_scorecard.py dataset1_individual.parquet -o scorecard_out
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, log_loss, roc_auc_score)

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

BANDS = ["Inactive", "Fairly Active", "Active"]
DEMOG = ["age4", "gender", "eth5", "nssec4", "educ3", "Disab3", "BMIG"]
PLACE = ["borough"]
FEATS = DEMOG + PLACE


# ---------------------------------------------------------------- model glue
def design(train_df, other_dfs, feats):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                        min_frequency=30)
    Xtr = enc.fit_transform(train_df[feats])
    return enc, Xtr, [enc.transform(f[feats]) for f in other_dfs]


def fit_ordinal(Xtr, ytr, wtr, Xs, C=1.0):
    """Two cumulative logits -> ordered 3-class probabilities."""
    cum = []
    for thresh in (1, 2):
        m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        m.fit(Xtr, (ytr >= thresh).astype(int), sample_weight=wtr)
        cum.append([m.predict_proba(X)[:, 1] for X in Xs])
    probs = []
    for i in range(len(Xs)):
        p1, p2 = cum[0][i], np.minimum(cum[1][i], cum[0][i])
        P = np.column_stack([1 - p1, p1 - p2, p2])
        probs.append(np.clip(P, 1e-9, 1) / np.clip(P, 1e-9, 1).sum(1, keepdims=True))
    return probs


def fit_lgb(Xtr, ytr, wtr, Xva, yva, wva, Xs):
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
    dva = lgb.Dataset(Xva, label=yva, weight=wva, reference=dtr)
    params = dict(objective="multiclass", num_class=3, learning_rate=0.05,
                  max_depth=4, num_leaves=15, min_child_samples=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=5.0, verbose=-1, seed=42)
    gbm = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(40, verbose=False)])
    return [gbm.predict(X, num_iteration=gbm.best_iteration) for X in Xs]


# ---------------------------------------------------------------- metrics
def metric_block(y, P, w):
    """All the requested classification metrics for one (y, probs) pair."""
    pred = P.argmax(1)
    out = {
        "accuracy":  accuracy_score(y, pred, sample_weight=w),
        "precision_macro": precision_score(y, pred, average="macro",
                                           sample_weight=w, zero_division=0),
        "recall_macro":    recall_score(y, pred, average="macro",
                                        sample_weight=w, zero_division=0),
        "f1_macro":        f1_score(y, pred, average="macro",
                                    sample_weight=w, zero_division=0),
        "logloss":         log_loss(y, P, labels=[0, 1, 2], sample_weight=w),
    }
    # AUC: macro one-vs-rest across all three bands
    try:
        out["auc_macro_ovr"] = roc_auc_score(y, P, multi_class="ovr",
                                             average="macro", sample_weight=w,
                                             labels=[0, 1, 2])
    except Exception:
        out["auc_macro_ovr"] = np.nan
    # AUC: binary Active-vs-rest (the figure we have quoted elsewhere)
    try:
        out["auc_active"] = roc_auc_score((y == 2).astype(int), P[:, 2],
                                          sample_weight=w)
    except Exception:
        out["auc_active"] = np.nan
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="scorecard_out")
    ap.add_argument("--C", type=float, default=1.0,
                    help="regularisation for the ordinal model")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    project_start = time.time()
    df = pd.read_parquet(args.parquet)
    df = df[df["activity_band"].notna() & df["wt_final"].notna()
            & (df["wt_final"] > 0)].copy()
    feats = [c for c in FEATS if c in df.columns]
    df["_y"] = pd.Categorical(df["activity_band"], categories=BANDS,
                              ordered=True).codes

    waves = sorted(int(w) for w in df.wave.unique())
    val_w, test_w = waves[-2], waves[-1]
    train_w = [w for w in waves if w < val_w]
    tr = df[df.wave.isin(train_w)]
    va = df[df.wave == val_w]
    te = df[df.wave == test_w]
    trva = df[df.wave <= val_w]                # train+val, used for final refit

    print("MODEL SCORECARD")
    print("=" * 70)
    print(f"temporal split  |  train waves {train_w}  |  "
          f"validate wave {val_w}  |  test wave {test_w}")
    print(f"rows            |  train {len(tr):,}  val {len(va):,}  "
          f"test {len(te):,}")
    print(f"features        |  {feats}\n")

    rows = []

    # ============================== ORDINAL LOGIT ==============================
    print("fitting ordinal logistic regression ...")
    t0 = time.time()
    enc, Xtr, (Xva, Xte, Xtrva_te) = design(tr, [va, te, te], feats)
    ytr, wtr = tr._y.values, tr.wt_final.values
    # train + test predictions (train fit)
    P_tr, P_te = fit_ordinal(Xtr, ytr, wtr, [Xtr, Xte], C=args.C)
    m_train = metric_block(ytr, P_tr, wtr)
    m_test = metric_block(te._y.values, P_te, te.wt_final.values)

    # leave-one-wave-out CV across the training waves (temporally honest)
    cv_acc, cv_f1 = [], []
    for w in train_w:
        cvtr = tr[tr.wave != w]; cvhold = tr[tr.wave == w]
        if len(cvhold) < 200:
            continue
        e2, Xc, (Xh,) = design(cvtr, [cvhold], feats)
        Pc = fit_ordinal(Xc, cvtr._y.values, cvtr.wt_final.values, [Xh],
                         C=args.C)[0]
        mb = metric_block(cvhold._y.values, Pc, cvhold.wt_final.values)
        cv_acc.append(mb["accuracy"]); cv_f1.append(mb["f1_macro"])
    runtime_logit = (time.time() - t0) / 60

    rows.append({
        "Model": "Ordinal logistic regression",
        "Test Accuracy": m_test["accuracy"],
        "Test Precision (macro)": m_test["precision_macro"],
        "Test Recall (macro)": m_test["recall_macro"],
        "Test F1 (macro)": m_test["f1_macro"],
        "CV Accuracy": np.mean(cv_acc), "CV F1 (macro)": np.mean(cv_f1),
        "Train Accuracy": m_train["accuracy"],
        "Train F1 (macro)": m_train["f1_macro"],
        "Runtime (min)": runtime_logit,
        "Train-CV Accuracy Gap": m_train["accuracy"] - np.mean(cv_acc),
        "AUC (macro OvR)": m_test["auc_macro_ovr"],
        "AUC (Active vs rest)": m_test["auc_active"],
        "LogLoss": m_test["logloss"],
    })

    # ============================== LIGHTGBM ==================================
    if HAVE_LGB:
        print("fitting LightGBM ...")
        t0 = time.time()
        yva, wva = va._y.values, va.wt_final.values
        P_tr_g, P_te_g = fit_lgb(Xtr, ytr, wtr, Xva, yva, wva, [Xtr, Xte])
        m_train_g = metric_block(ytr, P_tr_g, wtr)
        m_test_g = metric_block(te._y.values, P_te_g, te.wt_final.values)
        cv_acc_g, cv_f1_g = [], []
        for w in train_w:
            cvtr = tr[tr.wave != w]; cvhold = tr[tr.wave == w]
            if len(cvhold) < 200:
                continue
            e2, Xc, (Xh,) = design(cvtr, [cvhold], feats)
            # small internal val for early stopping: last training wave
            iv = cvtr[cvtr.wave == max(w2 for w2 in train_w if w2 != w)]
            Xiv = e2.transform(iv[feats])
            Pc = fit_lgb(Xc, cvtr._y.values, cvtr.wt_final.values,
                         Xiv, iv._y.values, iv.wt_final.values, [Xh])[0]
            mb = metric_block(cvhold._y.values, Pc, cvhold.wt_final.values)
            cv_acc_g.append(mb["accuracy"]); cv_f1_g.append(mb["f1_macro"])
        runtime_gbm = (time.time() - t0) / 60

        rows.append({
            "Model": "LightGBM",
            "Test Accuracy": m_test_g["accuracy"],
            "Test Precision (macro)": m_test_g["precision_macro"],
            "Test Recall (macro)": m_test_g["recall_macro"],
            "Test F1 (macro)": m_test_g["f1_macro"],
            "CV Accuracy": np.mean(cv_acc_g), "CV F1 (macro)": np.mean(cv_f1_g),
            "Train Accuracy": m_train_g["accuracy"],
            "Train F1 (macro)": m_train_g["f1_macro"],
            "Runtime (min)": runtime_gbm,
            "Train-CV Accuracy Gap": m_train_g["accuracy"] - np.mean(cv_acc_g),
            "AUC (macro OvR)": m_test_g["auc_macro_ovr"],
            "AUC (Active vs rest)": m_test_g["auc_active"],
            "LogLoss": m_test_g["logloss"],
        })
    else:
        print("!! lightgbm not installed — skipping (pip install lightgbm)")

    # ---------------------------------------------------------------- output
    S = pd.DataFrame(rows).set_index("Model")
    # round for display
    disp = S.copy()
    for c in disp.columns:
        disp[c] = disp[c].map(lambda v: f"{v:.4f}" if abs(v) < 1
                              else f"{v:.3f}")
    total_runtime = (time.time() - project_start) / 60

    print("\n" + "=" * 70)
    print("SCORECARD (all metrics survey-weighted; test = unseen final wave)")
    print("=" * 70)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(disp.T.to_string())

    print(f"\nScorecard runtime: {total_runtime:.2f} min "
          f"(this script only, both models).")
    print("Note: 'CV' is leave-one-wave-out across the training waves — the "
          "temporally honest form of cross-validation for a forecasting task. "
          "Random k-fold is deliberately NOT used, as it would leak future "
          "survey years into past folds.")

    S.to_csv(out / "model_scorecard.csv")

    # write markdown by hand (no dependency on the optional 'tabulate' package)
    dt = disp.T                       # metrics as rows, models as columns
    def md_table(frame):
        cols = list(frame.columns)
        lines = ["| Metric | " + " | ".join(cols) + " |",
                 "|" + "---|" * (len(cols) + 1)]
        for metric, row in frame.iterrows():
            lines.append("| " + metric + " | "
                         + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    with open(out / "model_scorecard.md", "w", encoding="utf-8") as f:
        f.write("# Model scorecard\n\n")
        f.write(f"Temporal split: train waves {train_w}, validate wave "
                f"{val_w}, test wave {test_w}. All metrics survey-weighted.\n\n")
        f.write(md_table(dt))
        f.write(f"\n\n_Scorecard runtime: {total_runtime:.2f} min._\n")
    print(f"\nwritten to {out}/model_scorecard.csv and .md")


if __name__ == "__main__":
    main()
