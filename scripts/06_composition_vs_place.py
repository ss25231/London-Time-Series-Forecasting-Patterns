"""
06_composition_vs_place.py — the decomposition the whole forecast rests on.

The wave-8 result showed the model reproduces 90% of between-borough variation
in activity. But `borough` was a feature, so some of that is the model
memorising each borough's historic level rather than explaining it.

Two very different worlds:

  COMPOSITION  Boroughs differ because their PEOPLE differ (age, ethnicity,
               NS-SEC). Demography changes -> activity changes -> a
               projection-driven forecast has real content.

  PLACE        Boroughs differ for reasons not in the data (facilities,
               green space, culture, transport). Those are sticky, so
               forecasting mostly means carrying the current level forward.

This script separates them by ablation:

  A  demographics only            -> borough spread explained by COMPOSITION
  B  borough only                 -> total systematic borough variation
  C  demographics + borough       -> the full model
  D  A, evaluated leave-one-borough-out  -> can we predict a borough we have
                                            NEVER seen? The real test of
                                            whether composition generalises.

Run:
    python 06_composition_vs_place.py dataset1_individual.parquet -o model_out
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.facecolor": "white", "font.size": 9,
                     "savefig.bbox": "tight"})

BANDS = ["Inactive", "Fairly Active", "Active"]
INNER = {"E09000001", "E09000007", "E09000011", "E09000012", "E09000013",
         "E09000019", "E09000020", "E09000022", "E09000023", "E09000028",
         "E09000030", "E09000032", "E09000033"}

# NB these MUST match the column names emitted by harmonise_waves.py.
# Disability is Disab3 and BMI is BMIG. An earlier version used
# "disability"/"bmi", which matched nothing and silently dropped two of the
# strongest predictors of inactivity. disty2/disty5 = long-term condition
# and mental health, available for people with a limiting disability.
# `health` is EXCLUDED: it is absent from waves 1-3 entirely, so once
# missing values are filled with a category the model can read that category
# as "this respondent came from an early wave". A disguised wave indicator in
# a composition model is worse than a missing feature.
DEMOG = ["age4", "gender", "eth5", "nssec4", "educ3", "Disab3", "BMIG"]


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.25, linewidth=.6); ax.set_axisbelow(True)


def load(path):
    df = pd.read_parquet(path)
    df["inner_outer"] = np.where(df["borough"].astype(str).isin(INNER),
                                 "Inner", "Outer")
    for c in DEMOG + ["borough", "inner_outer"]:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("Not disclosed").astype(str)
    ok = df["activity_band"].notna() & df["wt_final"].notna() & (df["wt_final"] > 0)
    df = df[ok].copy()
    df["_y"] = pd.Categorical(df["activity_band"], categories=BANDS,
                              ordered=True).codes
    return df


def fit_predict(tr, te, feats, C=1.0):
    """Cumulative-logit ordinal model -> P(Active) on `te`."""
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                        min_frequency=30)
    Xtr = enc.fit_transform(tr[feats]); Xte = enc.transform(te[feats])
    m = LogisticRegression(C=C, max_iter=2000)
    m.fit(Xtr, (tr._y.values == 2).astype(int), sample_weight=tr.wt_final.values)
    return m.predict_proba(Xte)[:, 1]


def borough_frame(te, p):
    d = pd.DataFrame({"borough": te.borough.values, "w": te.wt_final.values,
                      "p": p, "y": (te._y.values == 2).astype(float)})
    return d.groupby("borough").apply(lambda x: pd.Series({
        "n": len(x),
        "obs": 100 * (x.w * x.y).sum() / x.w.sum(),
        "pred": 100 * (x.w * x.p).sum() / x.w.sum()}), include_groups=False)


def report(name, g, out):
    corr = g[["obs", "pred"]].corr().iloc[0, 1]
    row = {"model": name, "borough_corr": round(corr, 3),
           "borough_MAE_pp": round((g.pred - g.obs).abs().mean(), 2),
           "obs_spread_pp": round(g.obs.std(), 2),
           "pred_spread_pp": round(g.pred.std(), 2),
           "spread_reproduced_%": round(100 * g.pred.std() / g.obs.std(), 0),
           "var_explained_%": round(100 * max(corr, 0) ** 2, 0)}
    out.append(row)
    print(f"  {name:38s} corr {corr:5.3f} | MAE {row['borough_MAE_pp']:4.2f}pp "
          f"| spread {row['pred_spread_pp']:4.2f}/{row['obs_spread_pp']:4.2f}pp")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="model_out")
    ap.add_argument("--test-wave", type=int, default=8)
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    df = load(args.parquet)
    demog = [c for c in DEMOG if c in df.columns]
    absent = [c for c in DEMOG if c not in df.columns]
    if absent:
        print(f"  !! REQUESTED BUT ABSENT from Dataset 1: {absent}")
    tr = df[df.wave < args.test_wave]
    te = df[df.wave == args.test_wave]
    print(f"train {len(tr):,} | test (wave {args.test_wave}) {len(te):,}")
    print(f"demographic features: {demog}\n")

    res, frames = [], {}

    print("ABLATION (wave %d holdout)" % args.test_wave)
    p_a = fit_predict(tr, te, demog)
    frames["A. Demographics only"] = borough_frame(te, p_a)
    report("A. demographics only", frames["A. Demographics only"], res)

    p_b = fit_predict(tr, te, ["borough"])
    frames["B. Borough only"] = borough_frame(te, p_b)
    report("B. borough identity only", frames["B. Borough only"], res)

    p_c = fit_predict(tr, te, demog + ["borough"])
    frames["C. Demographics + borough"] = borough_frame(te, p_c)
    report("C. demographics + borough", frames["C. Demographics + borough"], res)

    # ---- D: leave-one-borough-out, demographics only ----
    print("\nLEAVE-ONE-BOROUGH-OUT (demographics only)")
    print("  Predicting each borough from a model that has never seen it.")
    rows = []
    for b in sorted(df.borough.unique()):
        tr_b = tr[tr.borough != b]
        te_b = te[te.borough == b]
        if len(te_b) < 50:
            continue
        p = fit_predict(tr_b, te_b, demog)
        w = te_b.wt_final.values
        rows.append({"borough": b, "n": len(te_b),
                     "obs": 100 * (w * (te_b._y.values == 2)).sum() / w.sum(),
                     "pred": 100 * (w * p).sum() / w.sum()})
    g_lobo = pd.DataFrame(rows).set_index("borough")
    frames["D. LOBO (unseen borough)"] = g_lobo
    report("D. LOBO, demographics only", g_lobo, res)

    out = pd.DataFrame(res)
    print("\n" + "=" * 78)
    print(out.to_string(index=False))
    print("=" * 78)

    # ---- interpretation ----
    comp = out.loc[0, "pred_spread_pp"]; full = out.loc[2, "pred_spread_pp"]
    obs = out.loc[0, "obs_spread_pp"]
    print("\nDECOMPOSITION OF BETWEEN-BOROUGH VARIATION")
    print(f"  observed spread across boroughs      {obs:.2f} pp")
    print(f"  reproduced by demographics alone     {comp:.2f} pp "
          f"({100*comp/obs:.0f}%)")
    print(f"  reproduced with borough identity     {full:.2f} pp "
          f"({100*full/obs:.0f}%)")
    share = 100 * comp / obs
    print()
    if share >= 60:
        print("  => COMPOSITION-DOMINATED. Borough differences are largely")
        print("     about who lives there. Pushing projected populations")
        print("     through the propensity model will produce real, moving")
        print("     borough forecasts. Proceed with the projection engine.")
    elif share >= 30:
        print("  => MIXED. Demography explains a substantial minority of")
        print("     borough differences; the rest is place. Forecast with a")
        print("     borough random intercept ON TOP of the composition model,")
        print("     and report the two contributions separately.")
    else:
        print("  => PLACE-DOMINATED. Demography explains little of why")
        print("     boroughs differ. A composition forecast will barely move")
        print("     the numbers; be explicit that borough projections are")
        print("     essentially level-carrying plus a small trend.")

    # ---- figures ----
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.9))
    for ax, (name, g) in zip(axes, frames.items()):
        lo = min(g.obs.min(), g.pred.min()) - 2
        hi = max(g.obs.max(), g.pred.max()) + 2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.scatter(g.obs, g.pred, s=70, alpha=.75, color="#2c7fb8",
                   edgecolor="white")
        corr = g[["obs", "pred"]].corr().iloc[0, 1]
        ax.text(.04, .96, f"corr {corr:.2f}\nspread {g.pred.std():.1f} / "
                          f"{g.obs.std():.1f} pp",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(fc="white", ec="#cccccc"))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        _style(ax, name, "observed % Active", "predicted % Active")
    fig.suptitle("Why do London boroughs differ? Composition vs place",
                 fontsize=13, fontweight="bold", x=.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(outdir / "fig7_composition_vs_place.png", dpi=150)
    plt.close(fig)

    # residual map-ish ranking: which boroughs defy their demography?
    g = frames["A. Demographics only"].copy()
    g["residual"] = g.obs - g.pred
    g = g.sort_values("residual")
    fig, ax = plt.subplots(figsize=(8, .3 * len(g) + 1.6))
    cols = ["#c0392b" if v < 0 else "#27ae60" for v in g.residual]
    ax.barh(range(len(g)), g.residual, color=cols, height=.7)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.index, fontsize=7.5)
    _style(ax, "Which boroughs beat (or miss) what their demography predicts?",
           "observed minus predicted, percentage points")
    ax.text(.99, .01, "green = more active than its population implies",
            transform=ax.transAxes, ha="right", fontsize=8, color="#27ae60")
    fig.tight_layout()
    fig.savefig(outdir / "fig8_borough_residuals.png", dpi=150)
    plt.close(fig)

    out.to_csv(outdir / "composition_vs_place.csv", index=False)
    g.to_csv(outdir / "borough_residuals.csv")
    print(f"\nwritten: fig7_composition_vs_place.png, fig8_borough_residuals.png")
    print("  fig8 is the policy chart: boroughs above zero outperform their")
    print("  demography, and are worth studying for what they do differently.")


if __name__ == "__main__":
    main()
