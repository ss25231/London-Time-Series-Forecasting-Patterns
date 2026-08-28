"""
16_benchmarks.py — the "good compared to what?" floor.

A forecast or a classifier is only impressive relative to a baseline. This
module builds the simplest defensible benchmarks and reports every model as a
SKILL SCORE against them, so "our model is good" always has an explicit
comparison.

Two families, matching the project's two kinds of model:

  CLASSIFIER FLOORS  (for the propensity model — does it beat trivial?)
    * Prevalence classifier: predict the weighted base rate for everyone.
      By construction its AUC = 0.5 and its log-loss = the entropy of the
      base rate. Any real model must beat this.
    * Single-feature model: logistic regression on ONE predictor (age alone,
      then socio-economic group alone). If the full model barely beats "age
      only", most of the signal lives in one variable.

  FORECAST BENCHMARKS  (for the forecast — does it beat "assume no change?")
    Computed PER BOROUGH, because that is the level the engine forecasts at
    and the level the brief cares about. London is only ever a weighted mean
    of boroughs, so validating boroughs validates the thing we actually built.
    * Persistence (naive): each borough forecast to stay at its own last
      observed value.               yhat_{b,t+h} = y_{b,t}
    * Drift: each borough continues its own linear trend.
                                     yhat_{b,t+h} = y_{b,t} + h * slope_b

  SKILL SCORE (used everywhere):
        skill = 1 - error_model / error_benchmark
    > 0  model beats the benchmark ;  = 0  no better than trivial ;
    < 0  worse than doing nothing.

This script computes the classifier floors directly. The forecast benchmarks
are exposed as functions that 17_backtest.py imports, so both scripts share
one definition of "naive" and "drift".

Run:
    python 16_benchmarks.py dataset1_individual.parquet -o benchmark_out
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

BANDS = ["Inactive", "Fairly Active", "Active"]


# ============================================================ classifier floors
def weighted_rate(y, w):
    return float(np.average(y, weights=w))


def prevalence_floor(y, w):
    """Predict the base rate for everyone. The absolute floor."""
    p = weighted_rate(y, w)
    phat = np.full(len(y), p)
    # log-loss of a constant predictor = entropy of the base rate
    ll = log_loss(y, np.column_stack([1 - phat, phat]),
                  labels=[0, 1], sample_weight=w)
    # AUC is exactly 0.5 for a constant score; state it rather than compute
    return {"model": "Prevalence (predict base rate)", "auc": 0.5,
            "logloss": ll, "accuracy": max(p, 1 - p),
            "note": f"base rate = {100*p:.1f}%"}


def single_feature_floor(df, feat, target_col="_yb", wcol="wt_final"):
    """Logistic regression on ONE categorical feature."""
    d = df[df[feat].notna()].copy()
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                        min_frequency=30)
    X = enc.fit_transform(d[[feat]])
    y = d[target_col].values
    w = d[wcol].values
    m = LogisticRegression(max_iter=2000)
    m.fit(X, y, sample_weight=w)
    p = m.predict_proba(X)[:, 1]
    return {"model": f"Single feature: {feat}",
            "auc": roc_auc_score(y, p, sample_weight=w),
            "logloss": log_loss(y, np.column_stack([1 - p, p]),
                                labels=[0, 1], sample_weight=w),
            "accuracy": accuracy_score(y, (p > 0.5).astype(int),
                                       sample_weight=w),
            "note": "one predictor only"}


def skill(error_model, error_benchmark):
    if error_benchmark == 0:
        return np.nan
    return 1.0 - error_model / error_benchmark


# ==================================================== forecast benchmarks (API)
# These are imported by 17_backtest.py so both scripts share one definition.

def borough_series(df, target="is_active"):
    """Weighted borough x wave rate of the target. Returns a wide frame:
    index = borough, columns = wave, values = % meeting the target."""
    y = pd.to_numeric(df[target], errors="coerce")
    d = df.assign(_y=y)
    d = d[d._y.notna() & d.wt_final.notna() & (d.wt_final > 0)]
    g = (d.groupby(["borough", "wave"])
           .apply(lambda x: 100 * np.average(x._y, weights=x.wt_final),
                  include_groups=False)
           .reset_index(name="rate"))
    return g.pivot(index="borough", columns="wave", values="rate")


def naive_forecast(panel, origin_wave, horizon):
    """Persistence: every borough forecast to stay at its origin-wave value."""
    if origin_wave not in panel.columns:
        return None
    last = panel[origin_wave]
    return pd.Series(last.values, index=panel.index,
                     name=f"naive_w{origin_wave + horizon}")


def drift_forecast(panel, origin_wave, horizon, min_points=3):
    """Drift: each borough continues its own linear trend up to the origin."""
    waves = [w for w in panel.columns if w <= origin_wave]
    if len(waves) < min_points:
        return None
    out = {}
    for b, row in panel.iterrows():
        ys = row[waves].values.astype(float)
        xs = np.array(waves, dtype=float)
        ok = ~np.isnan(ys)
        if ok.sum() < min_points:
            out[b] = row[origin_wave]
            continue
        slope, intercept = np.polyfit(xs[ok], ys[ok], 1)
        out[b] = row[origin_wave] + slope * horizon
    return pd.Series(out, name=f"drift_w{origin_wave + horizon}")


# ============================================================ main (classifier)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="benchmark_out")
    ap.add_argument("--target", default="is_active")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    y = pd.to_numeric(df[args.target], errors="coerce")
    ok = y.notna() & df.wt_final.notna() & (df.wt_final > 0)
    df = df[ok].copy()
    df["_yb"] = y[ok].astype(int)
    w = df.wt_final.values

    print("CLASSIFIER FLOORS  (does the propensity model beat trivial?)")
    print("=" * 66)

    rows = [prevalence_floor(df._yb.values, w)]
    for feat in ["age4", "nssec4", "eth5"]:
        if feat in df.columns:
            rows.append(single_feature_floor(df, feat))

    S = pd.DataFrame(rows)[["model", "auc", "logloss", "accuracy", "note"]]
    with pd.option_context("display.width", 200,
                           "display.max_colwidth", 40):
        print(S.round(4).to_string(index=False))

    # reference: what the full model scored (from the scorecard, if present)
    print("\nInterpretation:")
    print("  * Prevalence AUC = 0.500 by construction — the absolute floor.")
    print("  * If a single feature already reaches most of the full model's")
    print("    AUC, the extra features add little; if not, they earn their")
    print("    place. Compare these to the full-model AUC (~0.70) in the")
    print("    model scorecard.")

    S.to_csv(out / "classifier_floors.csv", index=False)

    # a tiny persistence-vs-drift illustration on the borough panel, so this
    # script alone shows the forecast benchmarks exist (the real evaluation is
    # in 17_backtest.py).
    panel = borough_series(df, args.target)
    last_w = max(panel.columns)
    print(f"\nForecast benchmarks are defined here and evaluated in "
          f"17_backtest.py.\nBorough panel: {panel.shape[0]} boroughs x "
          f"{panel.shape[1]} waves (latest = wave {last_w}).")
    panel.to_csv(out / "borough_rate_panel.csv")
    print(f"\nwritten to {out}/classifier_floors.csv and borough_rate_panel.csv")


if __name__ == "__main__":
    main()
