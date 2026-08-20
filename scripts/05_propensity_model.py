"""
05_propensity_model.py — first propensity model on Dataset 1.

Question this answers: how much of Londoners' activity level is predictable
from demographics + borough, and does a gradient-boosted model beat a
regularised linear one on data like this?

Design
------
* Trained on INDIVIDUALS (135k), not cells. Thin cells are irrelevant here.
* Survey-weighted throughout (wt_final). Unweighted results are wrong.
* Temporal holdout: fit on waves 1-6, tune on wave 7, wave 8 touched ONCE.
* Ordinal target: Inactive < Fairly Active < Active. Fitted as two cumulative
  logits (P(>=Fairly), P(>=Active)) which respects the ordering without
  needing a specialist package.
* Evaluated on what actually matters for the brief: not just accuracy, but
  whether predicted BOROUGH-level active shares match observed ones.

Run:
    python 05_propensity_model.py dataset1_individual.parquet -o model_out
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "font.size": 9, "savefig.bbox": "tight"})

BANDS = ["Inactive", "Fairly Active", "Active"]

# --------------------------------------------------------------------------
# Inner / Outer London — statutory definition (London Government Act 1963).
# Not external data; a 33-row lookup. The brief asks for this split directly.
# --------------------------------------------------------------------------
INNER = {
    "E09000001",  # City of London
    "E09000007",  # Camden
    "E09000011",  # Greenwich
    "E09000012",  # Hackney
    "E09000013",  # Hammersmith and Fulham
    "E09000019",  # Islington
    "E09000020",  # Kensington and Chelsea
    "E09000022",  # Lambeth
    "E09000023",  # Lewisham
    "E09000028",  # Southwark
    "E09000030",  # Tower Hamlets
    "E09000032",  # Wandsworth
    "E09000033",  # Westminster
}

# NB these MUST match the column names emitted by harmonise_waves.py.
# Disability is Disab3 and BMI is BMIG. An earlier version used
# "disability"/"bmi", which matched nothing and silently dropped two of the
# strongest predictors of inactivity. disty2/disty5 = long-term condition
# and mental health, available for people with a limiting disability.
# `health` is EXCLUDED: it is absent from waves 1-3 entirely, so once
# missing values are filled with a category the model can read that category
# as "this respondent came from an early wave". A disguised wave indicator in
# a composition model is worse than a missing feature.
DEMOG_FEATURES = ["age4", "gender", "eth5", "nssec4", "educ3",
                  "Disab3", "BMIG"]
PLACE_FEATURES = ["borough", "inner_outer"]
CAT_FEATURES = DEMOG_FEATURES + PLACE_FEATURES

# Ablation sets. The decisive one is "demographics only": if it reproduces
# most of the observed spread BETWEEN boroughs, then borough differences are
# a composition effect and projecting population forward will forecast them.
# If it does not, boroughs differ for reasons beyond who lives in them, and
# the forecast needs borough-specific effects that cannot be projected.
ABLATIONS = {
    "demographics only": DEMOG_FEATURES,
    "place only": PLACE_FEATURES,
    "demographics + place": CAT_FEATURES,
}


# --------------------------------------------------------------------------

def load(path):
    df = pd.read_parquet(path)
    df["inner_outer"] = np.where(
        df["borough"].astype(str).isin(INNER), "Inner", "Outer")

    # Non-disclosure is a behaviour, not a measurement gap: these fields were
    # optional in the survey. Keep as an explicit level, never impute.
    for c in CAT_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("Not disclosed").astype(str)

    keep = df["activity_band"].notna() & df["wt_final"].notna() & (df["wt_final"] > 0)
    print(f"usable rows: {keep.sum():,} of {len(df):,}")
    return df[keep].copy()


def design(train, other_frames, feats):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                        min_frequency=30)
    Xtr = enc.fit_transform(train[feats])
    outs = [enc.transform(f[feats]) for f in other_frames]
    return enc, Xtr, outs


def cumulative_logit(Xtr, ytr, wtr, Xs, C=0.5):
    """Two cumulative logits -> ordered 3-class probabilities.

    P(>=Fairly) and P(>=Active) are each a binary logit. Class probabilities
    follow by differencing, clipped so the ordering cannot invert.
    """
    models, cum = {}, []
    for k, thresh in enumerate([1, 2], start=1):
        m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        m.fit(Xtr, (ytr >= thresh).astype(int), sample_weight=wtr)
        models[thresh] = m
        cum.append([m.predict_proba(X)[:, 1] for X in Xs])

    probs = []
    for i in range(len(Xs)):
        p_ge1, p_ge2 = cum[0][i], cum[1][i]
        p_ge2 = np.minimum(p_ge2, p_ge1)          # enforce monotonicity
        P = np.column_stack([1 - p_ge1, p_ge1 - p_ge2, p_ge2])
        probs.append(np.clip(P, 1e-9, 1) / np.clip(P, 1e-9, 1).sum(1, keepdims=True))
    return models, probs


def wacc(y, pred, w):
    return float((w * (y == pred)).sum() / w.sum())


def wmae_ordinal(y, pred, w):
    return float((w * np.abs(y - pred)).sum() / w.sum())


def evaluate(name, y, P, w, out):
    pred = P.argmax(1)
    row = {
        "model": name,
        "wt_accuracy": round(wacc(y, pred, w), 4),
        "wt_ordinal_MAE": round(wmae_ordinal(y, pred, w), 4),
        "logloss": round(log_loss(y, P, labels=[0, 1, 2], sample_weight=w), 4),
        "AUC_active": round(roc_auc_score((y == 2).astype(int), P[:, 2],
                                          sample_weight=w), 4),
        "pred_pct_active": round(100 * (w * P[:, 2]).sum() / w.sum(), 1),
        "true_pct_active": round(100 * (w * (y == 2)).sum() / w.sum(), 1),
    }
    out.append(row)
    return row


def borough_calibration(df, P, w, label):
    """The test that matters for the brief: do predicted borough active
    shares track observed ones? Accuracy on individuals can look fine while
    borough aggregates are useless."""
    d = pd.DataFrame({
        "borough": df["borough"].values,
        "w": w,
        "p_hat": P[:, 2],
        "y": (df["_y"].values == 2).astype(float),
    })
    g = d.groupby("borough").apply(
        lambda x: pd.Series({
            "n": len(x),
            "obs": 100 * (x.w * x.y).sum() / x.w.sum(),
            "pred": 100 * (x.w * x.p_hat).sum() / x.w.sum(),
        }), include_groups=False)
    g["err"] = g["pred"] - g["obs"]
    mae = g["err"].abs().mean()
    corr = g[["obs", "pred"]].corr().iloc[0, 1]
    print(f"  [{label}] borough MAE {mae:.2f} pp | corr(obs,pred) {corr:.3f} "
          f"| obs spread {g['obs'].std():.2f} pp | pred spread {g['pred'].std():.2f} pp")
    return g




# --------------------------------------------------------------------------
# VISUALISATION
# --------------------------------------------------------------------------

def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.25, linewidth=.6)
    ax.set_axisbelow(True)


def plot_trend(df, outdir):
    """The polarisation finding: what actually moved over 8 years."""
    rows = []
    for w, g in df.groupby("wave"):
        tot = g.wt_final.sum()
        for k, b in enumerate(BANDS):
            rows.append({"wave": w, "band": b,
                         "pct": 100 * g.wt_final[g._y == k].sum() / tot})
    t = pd.DataFrame(rows).pivot(index="wave", columns="band", values="pct")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = {"Inactive": "#c0392b", "Fairly Active": "#e67e22", "Active": "#27ae60"}
    for b in BANDS:
        axes[0].plot(t.index, t[b], marker="o", ms=5, lw=2,
                     color=cols[b], label=b)
    axes[0].axvspan(4.5, 6.5, color="grey", alpha=.12)
    axes[0].text(5.5, axes[0].get_ylim()[1] * .97, "COVID", ha="center",
                 fontsize=8, color="grey")
    axes[0].legend(fontsize=8, frameon=False)
    _style(axes[0], "Activity bands over time", "wave", "% of Londoners")

    # indexed to wave 1 — makes the small but systematic move visible
    idx = t / t.iloc[0] * 100
    for b in BANDS:
        axes[1].plot(idx.index, idx[b], marker="o", ms=5, lw=2,
                     color=cols[b], label=b)
    axes[1].axhline(100, color="black", lw=.8, ls="--")
    _style(axes[1], "Same series, indexed to wave 1 (=100)", "wave",
           "index")
    fig.suptitle("Is London getting more active? The middle is hollowing out.",
                 fontsize=12.5, fontweight="bold", x=.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(outdir / "fig1_trend.png", dpi=150)
    plt.close(fig)


def plot_borough_calibration(gs, outdir, test_wave):
    """Can the model tell boroughs apart, or does it flatten them?"""
    n = len(gs)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.2), squeeze=False)
    for ax, (name, g) in zip(axes[0], gs.items()):
        lo = min(g.obs.min(), g.pred.min()) - 1
        hi = max(g.obs.max(), g.pred.max()) + 1
        ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--",
                label="perfect calibration")
        ax.scatter(g.obs, g.pred, s=g.n / g.n.max() * 180 + 20,
                   alpha=.7, color="#2c7fb8", edgecolor="white", linewidth=.8)
        for b, r in g.iterrows():
            if abs(r.err) > g.err.abs().quantile(.85):
                ax.annotate(str(b)[-3:], (r.obs, r.pred), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
        corr = g[["obs", "pred"]].corr().iloc[0, 1]
        ax.text(.03, .97,
                f"corr = {corr:.2f}\nMAE = {g.err.abs().mean():.2f} pp\n"
                f"observed spread = {g.obs.std():.2f} pp\n"
                f"predicted spread = {g.pred.std():.2f} pp",
                transform=ax.transAxes, va="top", fontsize=8.5,
                bbox=dict(fc="white", ec="#cccccc", alpha=.9))
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        _style(ax, name, "observed % Active", "predicted % Active")
    fig.suptitle(f"Borough calibration, wave {test_wave} holdout — points on the "
                 f"line mean the model reproduces real borough differences",
                 fontsize=12.5, fontweight="bold", x=.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(outdir / "fig2_borough_calibration.png", dpi=150)
    plt.close(fig)


def plot_reliability(preds, y, w, outdir):
    """Are the predicted probabilities honest?"""
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", label="perfect")
    for name, P in preds.items():
        p = P[:, 2]
        bins = np.quantile(p, np.linspace(0, 1, 11))
        bins[-1] += 1e-9
        idx = np.digitize(p, bins[1:-1])
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() < 20:
                continue
            xs.append((w[m] * p[m]).sum() / w[m].sum())
            ys.append((w[m] * (y[m] == 2)).sum() / w[m].sum())
        ax.plot(xs, ys, marker="o", ms=5, lw=1.8, label=name)
    ax.legend(fontsize=8.5, frameon=False)
    _style(ax, "Reliability: predicted vs actual probability of being Active",
           "predicted P(Active)", "observed proportion Active")
    fig.tight_layout()
    fig.savefig(outdir / "fig3_reliability.png", dpi=150)
    plt.close(fig)


def plot_group_rates(df, P, outdir):
    """Who is active? Observed vs modelled, by demographic group.
    This is the chart London Sport will actually look at."""
    groups = [g for g in ["age4", "eth5", "nssec4", "disability",
                          "gender", "inner_outer"] if g in df.columns]
    ncol = 3
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    d = df.copy(); d["_p"] = P[:, 2]
    london = 100 * (d.wt_final * (d._y == 2)).sum() / d.wt_final.sum()

    for ax, gcol in zip(axes, groups):
        s = d.groupby(gcol).apply(lambda x: pd.Series({
            "obs": 100 * (x.wt_final * (x._y == 2)).sum() / x.wt_final.sum(),
            "pred": 100 * (x.wt_final * x._p).sum() / x.wt_final.sum(),
            "n": len(x)}), include_groups=False)
        s = s[s.n >= 100].sort_values("obs")
        ypos = np.arange(len(s))
        ax.barh(ypos, s.obs, height=.62, color="#2c7fb8", label="observed")
        ax.plot(s.pred, ypos, "D", ms=6, color="#e67e22", label="modelled")
        ax.axvline(london, color="black", lw=1, ls="--")
        ax.text(london, len(s) - .3, f" London {london:.0f}%", fontsize=7.5,
                color="black", va="top")
        ax.set_yticks(ypos)
        ax.set_yticklabels([str(i)[:22] for i in s.index], fontsize=8)
        _style(ax, gcol, "% Active (150+ mins/wk)")
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    for ax in axes[len(groups):]:
        ax.axis("off")
    fig.suptitle("Who is active in London? Observed rates vs what the model predicts",
                 fontsize=12.5, fontweight="bold", x=.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(outdir / "fig4_who_is_active.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(res, outdir):
    """Did the extra model complexity buy anything?"""
    r = res[res.model.str.contains("TEST|test")].copy()
    r["name"] = r.model.str.replace(r" \((TEST|test)\)", "", regex=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (col, lab, better) in zip(axes, [
            ("AUC_active", "AUC for P(Active)", "higher is better"),
            ("logloss", "log loss", "lower is better"),
            ("wt_accuracy", "weighted accuracy", "higher is better")]):
        cols = ["#95a5a6" if "prior" in n else "#2c7fb8" for n in r.name]
        ax.bar(r.name, r[col], color=cols, width=.55)
        for i, v in enumerate(r[col]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        _style(ax, f"{lab}  ({better})")
        ax.tick_params(axis="x", rotation=12, labelsize=8)
        ax.margins(y=.18)
    fig.suptitle("Model comparison on the untouched holdout wave — grey = "
                 "no-features reference", fontsize=12.5, fontweight="bold",
                 x=.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, .9])
    fig.savefig(outdir / "fig5_model_comparison.png", dpi=150)
    plt.close(fig)


def plot_importance(gbm, enc, feats, outdir, top=20):
    names = enc.get_feature_names_out(feats)
    imp = pd.Series(gbm.feature_importance("gain"), index=names)
    imp = imp.sort_values(ascending=False).head(top)[::-1]
    fam = [n.split("_")[0] for n in imp.index]
    pal = {f: c for f, c in zip(sorted(set(fam)), plt.cm.tab10.colors)}
    fig, ax = plt.subplots(figsize=(8, .34 * len(imp) + 1.4))
    ax.barh(range(len(imp)), imp.values,
            color=[pal[f] for f in fam], height=.7)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels([n.replace("_", ": ", 1)[:44] for n in imp.index],
                       fontsize=8)
    _style(ax, "What drives the model? (LightGBM gain, top %d)" % top, "gain")
    fig.tight_layout()
    fig.savefig(outdir / "fig6_importance.png", dpi=150)
    plt.close(fig)


def plot_ablation(abl, abl_g, outdir):
    """Composition vs place — the chart that justifies the forecast design."""
    fig = plt.figure(figsize=(13.5, 4.6))
    gs_ = fig.add_gridspec(1, 4, width_ratios=[1.15, 1, 1, 1], wspace=.28)

    ax = fig.add_subplot(gs_[0, 0])
    y = np.arange(len(abl))
    ax.barh(y, abl["spread_reproduced_%"], color="#2c7fb8", height=.6)
    ax.axvline(100, color="black", ls="--", lw=1)
    for i, v in enumerate(abl["spread_reproduced_%"]):
        ax.text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(abl.features, fontsize=9)
    _style(ax, "Share of borough spread reproduced", "% of observed spread")
    ax.set_xlim(0, max(115, abl["spread_reproduced_%"].max() * 1.15))

    for j, (label, g) in enumerate(abl_g.items()):
        ax = fig.add_subplot(gs_[0, j + 1])
        lo = min(g.obs.min(), g.pred.min()) - 1
        hi = max(g.obs.max(), g.pred.max()) + 1
        ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1)
        ax.scatter(g.obs, g.pred, s=45, alpha=.75, color="#2c7fb8",
                   edgecolor="white", linewidth=.7)
        ax.text(.04, .96, f"corr {g[['obs','pred']].corr().iloc[0,1]:.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(fc="white", ec="#ccc", alpha=.9))
        _style(ax, label, "observed %", "predicted %")
    fig.suptitle("Are borough differences about WHO lives there, or WHERE it is?",
                 fontsize=12.5, fontweight="bold", x=.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(outdir / "fig7_ablation.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="model_out")
    ap.add_argument("--test-wave", type=int, default=8)
    ap.add_argument("--val-wave", type=int, default=7)
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    df = load(args.parquet)

    feats = [c for c in CAT_FEATURES if c in df.columns]
    absent = [c for c in CAT_FEATURES if c not in df.columns]
    if absent:
        print(f"  !! REQUESTED BUT ABSENT from Dataset 1: {absent}")
        print("     Check the column names — a silent mismatch drops the")
        print("     feature entirely and the model still runs happily.")
    print(f"features: {feats} + wave")

    df["_y"] = pd.Categorical(df["activity_band"], categories=BANDS,
                              ordered=True).codes

    tr = df[df.wave <= args.val_wave - 1]
    va = df[df.wave == args.val_wave]
    te = df[df.wave == args.test_wave]
    print(f"train waves 1-{args.val_wave-1}: {len(tr):,} | "
          f"val wave {args.val_wave}: {len(va):,} | "
          f"test wave {args.test_wave}: {len(te):,}")

    enc, Xtr, (Xva, Xte) = design(tr, [va, te], feats)
    ytr, yva, yte = tr._y.values, va._y.values, te._y.values
    wtr, wva, wte = tr.wt_final.values, va.wt_final.values, te.wt_final.values
    print(f"design matrix: {Xtr.shape[1]} columns after one-hot")

    results = []

    # ---- reference: predict the training prior for everyone ----
    prior = np.array([(wtr * (ytr == k)).sum() / wtr.sum() for k in range(3)])
    for nm, X, y, w in [("val", Xva, yva, wva), ("test", Xte, yte, wte)]:
        P = np.tile(prior, (X.shape[0], 1))
        r = evaluate(f"prior-only ({nm})", y, P, w, results)
    print("\nreference model (no features) fitted.")

    # ---- ordinal logit ----
    print("\nfitting cumulative-logit ordinal model ...")
    best = None
    for C in [0.05, 0.2, 1.0]:
        _, (Pva,) = cumulative_logit(Xtr, ytr, wtr, [Xva], C=C)
        ll = log_loss(yva, Pva, labels=[0, 1, 2], sample_weight=wva)
        print(f"   C={C}: val logloss {ll:.4f}")
        if best is None or ll < best[0]:
            best = (ll, C)
    C = best[1]
    print(f"   selected C={C}")
    _, (Pva_o, Pte_o) = cumulative_logit(Xtr, ytr, wtr, [Xva, Xte], C=C)
    evaluate("ordinal logit (val)", yva, Pva_o, wva, results)
    evaluate("ordinal logit (TEST)", yte, Pte_o, wte, results)

    # ---- LightGBM ----
    Pte_g = None
    if HAVE_LGB:
        print("\nfitting LightGBM ...")
        dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
        dva = lgb.Dataset(Xva, label=yva, weight=wva, reference=dtr)
        params = dict(objective="multiclass", num_class=3, learning_rate=0.05,
                      max_depth=4, num_leaves=15, min_child_samples=200,
                      feature_fraction=0.8, bagging_fraction=0.8,
                      bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=42)
        gbm = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(40, verbose=False)])
        print(f"   best iteration: {gbm.best_iteration}")
        Pva_g = gbm.predict(Xva, num_iteration=gbm.best_iteration)
        Pte_g = gbm.predict(Xte, num_iteration=gbm.best_iteration)
        evaluate("LightGBM (val)", yva, Pva_g, wva, results)
        evaluate("LightGBM (TEST)", yte, Pte_g, wte, results)
    else:
        print("\n!! lightgbm not installed — pip install lightgbm")

    res = pd.DataFrame(results)
    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(res.to_string(index=False))

    # ------------------------------------------------------------------
    # ABLATION: how much of the borough gap is WHO lives there vs WHERE?
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("ABLATION — decomposing borough differences")
    print("=" * 78)
    abl_rows, abl_g = [], {}
    for label, fs in ABLATIONS.items():
        fs = [c for c in fs if c in df.columns]
        e2, X2tr, (X2va, X2te) = design(tr, [va, te], fs)
        _, (P2va, P2te) = cumulative_logit(X2tr, ytr, wtr, [X2va, X2te], C=C)
        g = borough_calibration(te, P2te, wte, label)
        abl_g[label] = g
        abl_rows.append({
            "features": label,
            "n_cols": X2tr.shape[1],
            "AUC_active": round(roc_auc_score((yte == 2).astype(int),
                                              P2te[:, 2], sample_weight=wte), 4),
            "borough_corr": round(g[["obs", "pred"]].corr().iloc[0, 1], 3),
            "borough_MAE_pp": round(g.err.abs().mean(), 2),
            "pred_spread_pp": round(g.pred.std(), 2),
            "obs_spread_pp": round(g.obs.std(), 2),
        })
    abl = pd.DataFrame(abl_rows)
    abl["spread_reproduced_%"] = (
        100 * abl.pred_spread_pp / abl.obs_spread_pp).round(0)
    print()
    print(abl.to_string(index=False))
    abl.to_csv(outdir / "ablation.csv", index=False)

    share = float(abl.loc[abl.features == "demographics only",
                          "spread_reproduced_%"].iloc[0])
    print(f"\n  READ: demographics alone reproduce {share:.0f}% of the observed")
    print( "  between-borough spread in activity.")
    if share >= 70:
        print("  -> Borough differences are largely COMPOSITIONAL. Projecting")
        print("     borough populations forward will forecast borough activity.")
    elif share >= 40:
        print("  -> Borough differences are PART composition, part place.")
        print("     The forecast needs both a composition layer and stable")
        print("     borough effects held constant into the future.")
    else:
        print("  -> Borough differences are mostly about PLACE, not people.")
        print("     Composition projection alone will not forecast boroughs.")

    print("\n" + "=" * 78)
    print("BOROUGH CALIBRATION (wave %d holdout)" % args.test_wave)
    print("=" * 78)
    print("  Does the model reproduce between-borough differences, or does it")
    print("  flatten them toward the London mean?")
    g_o = borough_calibration(te, Pte_o, wte, "ordinal logit")
    if Pte_g is not None:
        g_g = borough_calibration(te, Pte_g, wte, "LightGBM")

    res.to_csv(outdir / "model_results.csv", index=False)
    g_o.to_csv(outdir / "borough_calibration_ordinal.csv")

    # ---------------- figures ----------------
    print("\nbuilding figures ...")
    plot_trend(df, outdir)
    gs = {"Ordinal logit": g_o}
    if Pte_g is not None:
        gs["LightGBM"] = g_g
    plot_borough_calibration(gs, outdir, args.test_wave)
    plot_ablation(abl, abl_g, outdir)
    preds = {"Ordinal logit": Pte_o}
    if Pte_g is not None:
        preds["LightGBM"] = Pte_g
    plot_reliability(preds, yte, wte, outdir)
    plot_group_rates(te, Pte_o, outdir)
    plot_model_comparison(res, outdir)
    if Pte_g is not None:
        plot_importance(gbm, enc, feats, outdir)
    for f in sorted(outdir.glob("fig*.png")):
        print(f"   {f.name}")

    print(f"\nwritten to {outdir}/")
    print("\nHOW TO READ THIS:")
    print(" * If both models barely beat 'prior-only' on accuracy, that is")
    print("   expected — activity is mostly unexplained by demographics. Look")
    print("   at AUC and logloss instead, which use the probabilities.")
    print(" * If pred spread << obs spread in the borough calibration, the")
    print("   model is shrinking boroughs toward the mean and cannot support")
    print("   borough-level forecasting on its own.")
    print(" * If LightGBM only ties the logit, prefer the logit: it")
    print("   extrapolates more sanely and is far easier to defend.")


if __name__ == "__main__":
    main()
