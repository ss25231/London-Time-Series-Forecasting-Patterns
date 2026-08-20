"""
08_multi_target_ablation.py — does every behaviour split the same way?

Script 06 showed that ~52% of between-borough variation in ACTIVITY is
compositional (who lives there) and ~48% is place. This runs the identical
ablation across every target that passed the feasibility check, because
there is no reason to assume all behaviours behave alike.

The expected contrast: cycling has the largest borough spread in the whole
dataset (8.75pp, signal/noise 4.86) and is plausibly the most
infrastructure-dependent behaviour. If cycling comes back far more
place-driven than activity, that is a stronger, more actionable finding
than either number alone.

Per-target training windows are enforced (see WINDOWS below) because the
club and volunteering items changed base mid-series.

`health` is EXCLUDED as a feature: it is absent in waves 1-3, so once
missing values are filled it acts as a disguised wave indicator.

Run:
    python 08_multi_target_ablation.py dataset1_individual.parquet -o model_out
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

INNER = {"E09000001", "E09000007", "E09000011", "E09000012", "E09000013",
         "E09000019", "E09000020", "E09000022", "E09000023", "E09000028",
         "E09000030", "E09000032", "E09000033"}

# health deliberately omitted — wave-confounded (absent waves 1-3).
DEMOG = ["age4", "gender", "eth5", "nssec4", "educ3", "Disab3", "BMIG"]

# target -> (label, first usable wave, last usable wave)
# Windows follow the wave-stability diagnostic in script 07.
TARGETS = {
    "is_active":        ("Activity 150+ mins", 1, 8),
    "part_cycling":     ("Cycling", 2, 8),
    "part_walking":     ("Walking", 1, 8),
    "part_fitness":     ("Fitness activities", 2, 8),
    "part_team":        ("Team sport", 1, 8),
    "part_racket":      ("Racket sport", 1, 8),
    "part_dance":       ("Dance", 1, 8),
    "active_outdoor":   ("Active outdoors", 2, 8),
    "active_indoor":    ("Active indoors", 2, 8),
    "active_park":      ("Active in parks", 2, 8),
    "active_leisurecentre": ("Active at leisure centre", 2, 8),
    # Clubs: wave 2 excluded — its routing filter was much broader.
    "club_any_all":     ("Club membership", 3, 8),
    "club_fitness_all": ("Club: fitness", 3, 8),
    "club_team_all":    ("Club: team sport", 3, 8),
    # Volunteering: rebased to all adults; still fragile, reported for
    # completeness rather than as a borough forecast.
    "vol_any":          ("Volunteering", 6, 8),
}


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left", pad=6)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.25, linewidth=.6); ax.set_axisbelow(True)


def fit_predict(tr, te, feats, C=0.2):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                        min_frequency=30)
    Xtr = enc.fit_transform(tr[feats]); Xte = enc.transform(te[feats])
    m = LogisticRegression(C=C, max_iter=3000)
    m.fit(Xtr, tr["_t"].values.astype(int), sample_weight=tr.wt_final.values)
    return m.predict_proba(Xte)[:, 1]


def borough_frame(te, p, min_n=30):
    d = pd.DataFrame({"b": te.borough.values, "w": te.wt_final.values,
                      "p": p, "y": te["_t"].values.astype(float)})
    g = d.groupby("b").apply(lambda x: pd.Series({
        "n": len(x),
        "obs": 100 * (x.w * x.y).sum() / x.w.sum(),
        "pred": 100 * (x.w * x.p).sum() / x.w.sum()}), include_groups=False)
    return g[g.n >= min_n]


def run_target(df, col, label, w0, w1):
    s = pd.to_numeric(df[col], errors="coerce")
    d = df[(df.wave >= w0) & (df.wave <= w1) & s.notna()
           & df.wt_final.notna()].copy()
    d["_t"] = s[d.index]
    if d._t.nunique() < 2 or len(d) < 3000:
        return None

    test_w = int(d.wave.max())
    tr, te = d[d.wave < test_w], d[d.wave == test_w]
    if len(tr) < 2000 or len(te) < 1000:
        return None

    g_d = borough_frame(te, fit_predict(tr, te, DEMOG))
    g_p = borough_frame(te, fit_predict(tr, te, ["borough"]))
    if len(g_d) < 15:
        return None

    obs = g_d.obs.std()
    comp = g_d.pred.std()
    place = g_p.pred.std()
    return {
        "target": col, "label": label,
        "waves": f"{w0}-{w1}", "test_wave": test_w,
        "n_train": len(tr), "n_test": len(te),
        "prevalence_%": round(100 * (te.wt_final * te._t).sum()
                              / te.wt_final.sum(), 1),
        "obs_spread_pp": round(obs, 2),
        "composition_pp": round(comp, 2),
        "place_pp": round(place, 2),
        "composition_share_%": round(100 * comp / obs, 0) if obs > 0 else np.nan,
        "corr_demog": round(g_d[["obs", "pred"]].corr().iloc[0, 1], 3),
        "corr_place": round(g_p[["obs", "pred"]].corr().iloc[0, 1], 3),
    }, g_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="model_out")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    df["inner_outer"] = np.where(df.borough.astype(str).isin(INNER),
                                 "Inner", "Outer")
    for c in DEMOG + ["borough"]:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("Not disclosed").astype(str)
    print(f"individuals: {len(df):,}")
    print(f"features (health excluded as wave-confounded): {DEMOG}\n")

    rows, frames = [], {}
    for col, (label, w0, w1) in TARGETS.items():
        if col not in df.columns:
            print(f"  -- {label:26s} not in Dataset 1")
            continue
        out = run_target(df, col, label, w0, w1)
        if out is None:
            print(f"  -- {label:26s} insufficient data")
            continue
        r, g = out
        rows.append(r); frames[label] = g
        print(f"  {label:26s} w{r['waves']:4s} | spread {r['obs_spread_pp']:5.2f}pp "
              f"| composition {r['composition_share_%']:3.0f}%")

    res = pd.DataFrame(rows).sort_values("composition_share_%")
    print("\n" + "=" * 100)
    print("COMPOSITION vs PLACE, BY BEHAVIOUR")
    print("=" * 100)
    print(res[["label", "waves", "prevalence_%", "obs_spread_pp",
               "composition_pp", "place_pp", "composition_share_%",
               "corr_demog"]].to_string(index=False))
    print("=" * 100)
    print("""
  composition_share = how much of the between-borough spread demographics
                      alone reproduce. HIGH means the behaviour follows the
                      people, so population projections will move it. LOW
                      means it follows the place, so it responds to
                      infrastructure and provision, not demographic change.
""")

    fig, ax = plt.subplots(figsize=(9.5, .44 * len(res) + 1.8))
    y = np.arange(len(res))
    ax.barh(y, res["composition_share_%"], height=.66, color="#2c7fb8",
            label="composition (who lives there)")
    ax.barh(y, 100 - res["composition_share_%"],
            left=res["composition_share_%"], height=.66, color="#e67e22",
            label="place (where it is)")
    for i, (_, r) in enumerate(res.iterrows()):
        ax.text(3, i, f"{r['composition_share_%']:.0f}%", va="center",
                fontsize=8.5, color="white", fontweight="bold")
        ax.text(101, i, f"spread {r['obs_spread_pp']:.1f}pp", va="center",
                fontsize=7.5, color="#555555")
    ax.set_yticks(y); ax.set_yticklabels(res.label, fontsize=9)
    ax.set_xlim(0, 118); ax.set_xlabel("share of between-borough variation (%)")
    ax.legend(fontsize=8.5, frameon=False, ncol=2, loc="lower right")
    ax.set_title("Do behaviours follow the people, or the place?",
                 fontsize=12.5, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.25, axis="x"); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outdir / "fig10_composition_by_behaviour.png", dpi=150)
    plt.close(fig)

    res.to_csv(outdir / "multi_target_ablation.csv", index=False)
    for lab, g in frames.items():
        g.to_csv(outdir / f"borough_{lab.replace(' ', '_').replace(':','')}.csv")
    print(f"written: {outdir}/fig10_composition_by_behaviour.png "
          f"and multi_target_ablation.csv")


if __name__ == "__main__":
    main()
