"""
10_visualise.py — exploration and prediction visuals.

Two families of figure:

  EXPLORATORY   what the data says about who is active, before any model
                  A  full band composition by group (not just % active)
                  B  how each group moved over the eight years
                  C  heatmaps of two demographics crossed
                  D  intersectional view (age x class, split by gender)
                  E  inner vs outer London by group
                  F  borough ranked, coloured by inner/outer

  PREDICTION    what the model does with it
                  G  predicted probability distributions by actual band
                  H  where the model over- and under-predicts, by group
                  I  observed vs predicted by borough AND demographic

Everything is survey-weighted. Groups with fewer than a minimum base are
suppressed rather than drawn, because a bar built on 30 people looks
identical to one built on 3,000 and is not comparable.

Run:
    python 10_visualise.py dataset1_individual.parquet -o viz_out
    python 10_visualise.py dataset1_individual.parquet -o viz_out --target part_cycling
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, log_loss, roc_curve

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 9, "savefig.bbox": "tight", "axes.axisbelow": True,
})

# ---- palette ----
INK, TEAL, AMBER, RED, GREY = "#1F3050", "#2E7D8F", "#D98C3F", "#B4472E", "#7A8899"
BAND_COL = {"Inactive": "#C0392B", "Fairly Active": "#E8A33D", "Active": "#2E8B6F"}
DIVERGE = LinearSegmentedColormap.from_list("dv", ["#B4472E", "#F2EFE9", "#2E7D8F"])
SEQ = LinearSegmentedColormap.from_list("sq", ["#F4F1EC", "#7FB3B5", "#1F3050"])

BANDS = ["Inactive", "Fairly Active", "Active"]
INNER = {"E09000001", "E09000007", "E09000011", "E09000012", "E09000013",
         "E09000019", "E09000020", "E09000022", "E09000023", "E09000028",
         "E09000030", "E09000032", "E09000033"}
BOROUGH_NAME = {
    "E09000002": "Barking & Dagenham", "E09000003": "Barnet", "E09000004": "Bexley",
    "E09000005": "Brent", "E09000006": "Bromley", "E09000007": "Camden",
    "E09000008": "Croydon", "E09000009": "Ealing", "E09000010": "Enfield",
    "E09000011": "Greenwich", "E09000012": "Hackney", "E09000013": "Hammersmith & Fulham",
    "E09000014": "Haringey", "E09000015": "Harrow", "E09000016": "Havering",
    "E09000017": "Hillingdon", "E09000018": "Hounslow", "E09000019": "Islington",
    "E09000020": "Kensington & Chelsea", "E09000021": "Kingston", "E09000022": "Lambeth",
    "E09000023": "Lewisham", "E09000024": "Merton", "E09000025": "Newham",
    "E09000026": "Redbridge", "E09000027": "Richmond", "E09000028": "Southwark",
    "E09000029": "Sutton", "E09000030": "Tower Hamlets", "E09000031": "Waltham Forest",
    "E09000032": "Wandsworth", "E09000033": "Westminster",
}

# Display order matters: these are ordered categories, not arbitrary labels.
ORDER = {
    "age4": ["16-24", "25-44", "45-64", "65+", "Not disclosed"],
    "nssec4": ["Higher", "Middle", "Lower", "Student/Other", "Not classified (age)",
               "Not disclosed"],
    "educ3": ["L4+", "L1-3/Other", "None", "Not disclosed"],
    "eth5": ["White British", "White Other", "Asian", "Black", "Mixed/Other",
             "Not disclosed"],
    "gender": ["Male", "Female", "Other", "Not disclosed"],
    "Disab3": None, "BMIG": None,
}
LABEL = {
    "age4": "Age band", "gender": "Gender", "eth5": "Ethnicity",
    "nssec4": "Socio-economic group", "educ3": "Education",
    "Disab3": "Disability", "BMIG": "BMI group", "inner_outer": "Inner / outer London",
}
DEMOG = ["age4", "gender", "eth5", "nssec4", "educ3", "Disab3", "BMIG"]
WAVE_YEAR = {1: "15/16", 2: "16/17", 3: "17/18", 4: "18/19",
             5: "19/20", 6: "20/21", 7: "21/22", 8: "22/23"}


# ---------------------------------------------------------------- helpers
def style(ax, title="", xlabel="", ylabel="", grid="y"):
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left", pad=8,
                     color=INK)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(alpha=.25, linewidth=.6, axis=grid)


def wpct(g, mask_col, w="wt_final"):
    """Weighted percentage where mask_col == 1."""
    m = g[mask_col].notna() & g[w].notna()
    if m.sum() == 0:
        return np.nan
    return 100 * (g.loc[m, mask_col] * g.loc[m, w]).sum() / g.loc[m, w].sum()


def levels(df, col):
    o = ORDER.get(col)
    have = [str(v) for v in df[col].dropna().unique()]
    if o:
        return [v for v in o if v in have] + sorted(set(have) - set(o))
    return sorted(have)


def suppress(n, minimum):
    return n < minimum


# ---------------------------------------------------------------- load
def load(path, target):
    df = pd.read_parquet(path)
    df["inner_outer"] = np.where(df.borough.astype(str).isin(INNER), "Inner", "Outer")
    for c in DEMOG + ["borough", "inner_outer"]:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("Not disclosed").astype(str)
    y = pd.to_numeric(df[target], errors="coerce")
    df = df[y.notna() & df.wt_final.notna() & (df.wt_final > 0)].copy()
    df["_t"] = y[df.index].astype(float)
    df["borough_name"] = df.borough.map(BOROUGH_NAME).fillna(df.borough)
    return df


def fit_ordinal(df, feats):
    """Three-class band probabilities from two cumulative logits.

    P(>= Fairly Active) and P(>= Active) are each a binary model; the three
    band probabilities follow by differencing. Fitting it this way respects
    the natural ordering Inactive < Fairly Active < Active, which a plain
    three-way classifier would ignore.
    """
    y = pd.Categorical(df.activity_band, categories=BANDS, ordered=True).codes
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                        min_frequency=30)
    X = enc.fit_transform(df[feats])
    cum = []
    for thresh in (1, 2):
        m = LogisticRegression(C=1.0, max_iter=3000)
        m.fit(X, (y >= thresh).astype(int), sample_weight=df.wt_final.values)
        cum.append(m.predict_proba(X)[:, 1])
    p_ge1, p_ge2 = cum
    p_ge2 = np.minimum(p_ge2, p_ge1)            # ordering cannot invert
    P = np.column_stack([1 - p_ge1, p_ge1 - p_ge2, p_ge2])
    return np.clip(P, 1e-9, 1) / np.clip(P, 1e-9, 1).sum(1, keepdims=True), y


def fit_model(df, feats):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=30)
    X = enc.fit_transform(df[feats])
    m = LogisticRegression(C=1.0, max_iter=3000)
    m.fit(X, df._t.values.astype(int), sample_weight=df.wt_final.values)
    return m, enc


# ================================================================ A
def figA(df, out, minn, tlabel):
    """Full band composition, not just the headline percentage."""
    if "activity_band" not in df.columns:
        return
    cols = [c for c in DEMOG if c in df.columns]
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.05 * sum(
        len(levels(df, c)) for c in cols) * 0.42 + 1.5),
        gridspec_kw={"height_ratios": [len(levels(df, c)) for c in cols]})
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        lv = levels(df, c)[::-1]
        base, rows = [], []
        for v in lv:
            g = df[df[c] == v]
            if suppress(len(g), minn):
                continue
            tot = g.wt_final.sum()
            rows.append([100 * g.wt_final[g.activity_band == b].sum() / tot
                         for b in BANDS])
            base.append(f"{v}  (n={len(g):,})")
        if not rows:
            ax.axis("off"); continue
        R = np.array(rows); y = np.arange(len(base))
        left = np.zeros(len(base))
        for j, b in enumerate(BANDS):
            ax.barh(y, R[:, j], left=left, height=.66, color=BAND_COL[b],
                    label=b if ax is axes[0] else None)
            for yy, (l, v) in enumerate(zip(left, R[:, j])):
                if v > 6:
                    ax.text(l + v / 2, yy, f"{v:.0f}", ha="center", va="center",
                            fontsize=7.5, color="white", fontweight="bold")
            left += R[:, j]
        ax.set_yticks(y); ax.set_yticklabels(base, fontsize=8)
        ax.set_xlim(0, 100)
        style(ax, LABEL.get(c, c), grid="x")
        ax.set_xticks([0, 25, 50, 75, 100])
    axes[0].legend(fontsize=8, frameon=False, ncol=3,
                   bbox_to_anchor=(1.0, 1.55), loc="upper right")
    axes[-1].set_xlabel("% of group", fontsize=9)
    fig.suptitle("A. Full activity profile by demographic group",
                 fontsize=13, fontweight="bold", x=.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, .985])
    fig.savefig(out / "vizA_band_composition.png", dpi=150); plt.close(fig)


# ================================================================ B
def figB(df, out, minn, tlabel):
    """Did every group move together, or did the gaps widen?"""
    cols = [c for c in ["age4", "eth5", "nssec4", "educ3", "gender", "Disab3"]
            if c in df.columns]
    ncol = 3; nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.3 * nrow),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()
    lon = df.groupby("wave").apply(lambda g: wpct(g, "_t"), include_groups=False)
    for ax, c in zip(axes, cols):
        for i, v in enumerate(levels(df, c)):
            sub = df[df[c] == v]
            if len(sub) < minn * 4:
                continue
            s = sub.groupby("wave").apply(lambda g: wpct(g, "_t") if len(g) >= minn
                                          else np.nan, include_groups=False)
            ax.plot(s.index, s.values, "o-", ms=3.5, lw=1.6, label=str(v)[:20],
                    color=plt.cm.tab10(i % 10), alpha=.9)
        ax.plot(lon.index, lon.values, "k--", lw=1.3, alpha=.55, label="London")
        ax.axvspan(4.5, 6.5, color="grey", alpha=.10)
        ax.set_xticks(range(1, 9))
        ax.set_xticklabels([WAVE_YEAR[w] for w in range(1, 9)], fontsize=7,
                           rotation=45)
        ax.legend(fontsize=6.5, frameon=False, ncol=1, loc="lower left")
        style(ax, LABEL.get(c, c), "", f"% {tlabel}")
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("B. How each group moved over eight years  (shaded = COVID)",
                 fontsize=13, fontweight="bold", x=.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(out / "vizB_group_trends.png", dpi=150); plt.close(fig)


# ================================================================ C
def figC(df, out, minn, tlabel):
    """Two demographics crossed — where do disadvantages stack?"""
    pairs = [("age4", "eth5"), ("age4", "nssec4"), ("eth5", "nssec4"),
             ("age4", "educ3"), ("nssec4", "Disab3"), ("eth5", "gender")]
    pairs = [(a, b) for a, b in pairs if a in df.columns and b in df.columns]
    ncol = 3; nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.1 * ncol, 3.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    lon = wpct(df, "_t")
    for ax, (a, b) in zip(axes, pairs):
        la, lb = levels(df, a), levels(df, b)
        M = np.full((len(la), len(lb)), np.nan)
        N = np.zeros_like(M)
        for i, va in enumerate(la):
            for j, vb in enumerate(lb):
                g = df[(df[a] == va) & (df[b] == vb)]
                N[i, j] = len(g)
                if len(g) >= minn:
                    M[i, j] = wpct(g, "_t")
        im = ax.imshow(M, cmap=DIVERGE, vmin=lon - 22, vmax=lon + 22,
                       aspect="auto")
        for i in range(len(la)):
            for j in range(len(lb)):
                if np.isnan(M[i, j]):
                    ax.text(j, i, "·", ha="center", va="center", color=GREY,
                            fontsize=11)
                else:
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                            fontsize=7.5,
                            color="white" if abs(M[i, j] - lon) > 13 else "#222222",
                            fontweight="bold")
        ax.set_xticks(range(len(lb)))
        ax.set_xticklabels([str(v)[:14] for v in lb], rotation=35, ha="right",
                           fontsize=7.5)
        ax.set_yticks(range(len(la)))
        ax.set_yticklabels([str(v)[:16] for v in la], fontsize=7.5)
        ax.set_title(f"{LABEL.get(a, a)}  ×  {LABEL.get(b, b)}", fontsize=10,
                     fontweight="bold", loc="left", color=INK, pad=6)
        ax.grid(False)
    for ax in axes[len(pairs):]:
        ax.axis("off")
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=.015, pad=.02)
    cb.set_label(f"% {tlabel}  (London = {lon:.0f}%)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle("C. Two demographics crossed — a dot means the base is too small "
                 "to report", fontsize=13, fontweight="bold", x=.02, ha="left",
                 color=INK)
    fig.savefig(out / "vizC_crosstabs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ================================================================ D
def figD(df, out, minn, tlabel):
    """The intersectional view: class gradient within each age band, by gender."""
    if not {"age4", "nssec4", "gender"} <= set(df.columns):
        return
    genders = [g for g in ["Male", "Female"] if g in df.gender.unique()]
    ages = [a for a in levels(df, "age4") if a != "Not disclosed"]
    cls = [c for c in levels(df, "nssec4") if c in ("Higher", "Middle", "Lower")]
    fig, axes = plt.subplots(1, len(genders), figsize=(5.6 * len(genders), 4.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(ages)); wid = .26
    for ax, gd in zip(axes, genders):
        for k, c in enumerate(cls):
            vals, ns = [], []
            for a in ages:
                g = df[(df.gender == gd) & (df.age4 == a) & (df.nssec4 == c)]
                ns.append(len(g))
                vals.append(wpct(g, "_t") if len(g) >= minn else np.nan)
            bars = ax.bar(x + (k - 1) * wid, vals, wid * .92,
                          color=[INK, TEAL, AMBER][k], label=c)
            for b, v, nn in zip(bars, vals, ns):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + .8, f"{v:.0f}",
                            ha="center", fontsize=7.5, color="#333333")
        ax.set_xticks(x); ax.set_xticklabels(ages, fontsize=9)
        ax.axhline(wpct(df, "_t"), color="black", ls="--", lw=1, alpha=.5)
        style(ax, gd, "", f"% {tlabel}")
        ax.legend(fontsize=8, frameon=False, title="Socio-economic group",
                  title_fontsize=8)
    fig.suptitle("D. The class gradient holds within every age band and both genders",
                 fontsize=13, fontweight="bold", x=.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(out / "vizD_intersectional.png", dpi=150); plt.close(fig)


# ================================================================ E
def figE(df, out, minn, tlabel):
    """Does the inner/outer gap apply to everyone, or only some groups?"""
    cols = [c for c in ["age4", "eth5", "nssec4", "Disab3"] if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.9 * len(cols), 4.2))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        lv = [v for v in levels(df, c) if v != "Not disclosed"]
        inn, outr, keep = [], [], []
        for v in lv:
            gi = df[(df[c] == v) & (df.inner_outer == "Inner")]
            go = df[(df[c] == v) & (df.inner_outer == "Outer")]
            if len(gi) < minn or len(go) < minn:
                continue
            inn.append(wpct(gi, "_t")); outr.append(wpct(go, "_t")); keep.append(v)
        y = np.arange(len(keep))
        for i, (a, b) in enumerate(zip(outr, inn)):
            ax.plot([a, b], [i, i], color="#CBD5DD", lw=2.5, zorder=1)
        ax.scatter(outr, y, s=55, color=AMBER, label="Outer", zorder=2)
        ax.scatter(inn, y, s=55, color=TEAL, label="Inner", zorder=2)
        ax.set_yticks(y); ax.set_yticklabels([str(k)[:18] for k in keep], fontsize=8)
        style(ax, LABEL.get(c, c), f"% {tlabel}", grid="x")
        if ax is axes[0]:
            ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("E. Inner versus outer London, within each group",
                 fontsize=13, fontweight="bold", x=.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(out / "vizE_inner_outer.png", dpi=150); plt.close(fig)


# ================================================================ F
def figF(df, out, minn, tlabel):
    """Boroughs ranked, with sample size shown so thin bases are visible."""
    g = df.groupby("borough_name").apply(lambda x: pd.Series({
        "pct": wpct(x, "_t"), "n": len(x)}), include_groups=False)
    g = g[g.n >= minn].sort_values("pct")
    inner_names = {BOROUGH_NAME[k] for k in INNER if k in BOROUGH_NAME}
    cols = [TEAL if b in inner_names else AMBER for b in g.index]
    fig, ax = plt.subplots(figsize=(8.6, .3 * len(g) + 1.6))
    y = np.arange(len(g))
    ax.barh(y, g.pct, height=.68, color=cols)
    lon = wpct(df, "_t")
    ax.axvline(lon, color=INK, ls="--", lw=1.2)
    ax.text(lon, len(g) - .2, f" London {lon:.1f}%", fontsize=8, color=INK, va="top")
    for i, (v, nn) in enumerate(zip(g.pct, g.n)):
        ax.text(v + .4, i, f"{v:.1f}", va="center", fontsize=7.5, color="#333333")
        ax.text(1.0, i, f"n={int(nn):,}", va="center", fontsize=6.5, color="white")
    ax.set_yticks(y); ax.set_yticklabels(g.index, fontsize=8)
    style(ax, "", f"% {tlabel}", grid="x")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TEAL, label="Inner London"),
                       Patch(color=AMBER, label="Outer London")],
              fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("F. Boroughs ranked  (all eight years pooled)", fontsize=13,
                 fontweight="bold", x=.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(out / "vizF_borough_ranking.png", dpi=150); plt.close(fig)


# ================================================================ G
def figG(df, p, out, tlabel):
    """Can the model separate the two groups at all?"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    for lab, val, col in [(f"Not {tlabel}", 0, RED), (tlabel.capitalize(), 1, TEAL)]:
        d = p[df._t.values == val]
        ax.hist(d, bins=40, alpha=.6, color=col, label=f"{lab}  (n={len(d):,})",
                density=True)
    ax.legend(fontsize=8.5, frameon=False)
    style(ax, "Predicted probability, split by what actually happened",
          "predicted probability", "density", grid="y")

    ax = axes[1]
    dec = pd.qcut(p, 10, labels=False, duplicates="drop")
    tab = pd.DataFrame({"dec": dec, "y": df._t.values, "w": df.wt_final.values})
    s = tab.groupby("dec").apply(lambda x: 100 * (x.y * x.w).sum() / x.w.sum(),
                                 include_groups=False)
    ax.bar(s.index + 1, s.values, color=INK, width=.7)
    ax.axhline(100 * np.average(df._t, weights=df.wt_final), color=AMBER,
               ls="--", lw=1.4, label="overall rate")
    for i, v in zip(s.index + 1, s.values):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=7.5, color="#333333")
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8.5, frameon=False)
    style(ax, "Actual rate within each predicted-risk tenth",
          "tenth of the population, ranked by predicted probability",
          f"% {tlabel}")
    fig.suptitle("G. Does the model separate people who are active from those who "
                 "are not?", fontsize=13, fontweight="bold", x=.02, ha="left",
                 color=INK)
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(out / "vizG_prediction_separation.png", dpi=150); plt.close(fig)


# ================================================================ H
def figH(df, p, out, minn, tlabel):
    """Where is the model wrong, and for whom?"""
    d = df.copy(); d["_p"] = p
    cols = [c for c in DEMOG + ["inner_outer"] if c in d.columns]
    rows = []
    for c in cols:
        for v in levels(d, c):
            g = d[d[c] == v]
            if len(g) < minn:
                continue
            obs = 100 * np.average(g._t, weights=g.wt_final)
            pred = 100 * np.average(g._p, weights=g.wt_final)
            rows.append({"grp": f"{LABEL.get(c, c)}: {v}", "err": pred - obs,
                         "n": len(g)})
    r = pd.DataFrame(rows).sort_values("err")
    fig, ax = plt.subplots(figsize=(8.8, .28 * len(r) + 1.6))
    y = np.arange(len(r))
    ax.barh(y, r.err, height=.68,
            color=[RED if e < 0 else TEAL for e in r.err])
    ax.axvline(0, color="black", lw=1)
    for i, (e, nn) in enumerate(zip(r.err, r.n)):
        ax.text(e + (.08 if e >= 0 else -.08), i, f"{e:+.1f}",
                va="center", ha="left" if e >= 0 else "right", fontsize=7,
                color="#333333")
    ax.set_yticks(y); ax.set_yticklabels(r.grp, fontsize=7.5)
    style(ax, "", "model prediction minus reality, percentage points", grid="x")
    ax.text(.99, .01, "left of zero = the model under-estimates this group",
            transform=ax.transAxes, ha="right", fontsize=8, color=GREY)
    fig.suptitle("H. Where the model gets it wrong", fontsize=13,
                 fontweight="bold", x=.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(out / "vizH_model_error_by_group.png", dpi=150); plt.close(fig)


# ================================================================ I
def figI(df, p, out, minn, tlabel):
    """Observed against predicted, at borough level and by group."""
    d = df.copy(); d["_p"] = p
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    b = d.groupby("borough_name").apply(lambda x: pd.Series({
        "obs": 100 * np.average(x._t, weights=x.wt_final),
        "pred": 100 * np.average(x._p, weights=x.wt_final),
        "n": len(x)}), include_groups=False)
    b = b[b.n >= minn]
    ax = axes[0]
    lo, hi = min(b.obs.min(), b.pred.min()) - 2, max(b.obs.max(), b.pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.scatter(b.obs, b.pred, s=60, color=TEAL, alpha=.8, edgecolor="white")
    resid = (b.pred - b.obs).abs()
    for nm in resid.sort_values(ascending=False).head(6).index:
        ax.annotate(nm, (b.obs[nm], b.pred[nm]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points", color="#333333")
    ax.text(.03, .96, f"correlation {b.obs.corr(b.pred):.2f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(fc="white", ec="#CCCCCC"))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    style(ax, "By borough", f"observed % {tlabel}", f"predicted % {tlabel}",
          grid="both")

    rows = []
    for c in [x for x in DEMOG if x in d.columns]:
        for v in levels(d, c):
            g = d[d[c] == v]
            if len(g) < minn:
                continue
            rows.append({"obs": 100 * np.average(g._t, weights=g.wt_final),
                         "pred": 100 * np.average(g._p, weights=g.wt_final),
                         "lab": f"{v}", "fam": c})
    r = pd.DataFrame(rows)
    ax = axes[1]
    lo, hi = min(r.obs.min(), r.pred.min()) - 3, max(r.obs.max(), r.pred.max()) + 3
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    for i, fam in enumerate(r.fam.unique()):
        sub = r[r.fam == fam]
        ax.scatter(sub.obs, sub.pred, s=55, alpha=.85, edgecolor="white",
                   color=plt.cm.tab10(i % 10), label=LABEL.get(fam, fam))
    worst = (r.pred - r.obs).abs().sort_values(ascending=False).head(5).index
    for i in worst:
        ax.annotate(r.lab[i], (r.obs[i], r.pred[i]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points", color="#333333")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    style(ax, "By demographic group", f"observed % {tlabel}",
          f"predicted % {tlabel}", grid="both")
    fig.suptitle("I. Observed versus predicted — points on the line mean the model "
                 "reproduces reality", fontsize=13, fontweight="bold", x=.02,
                 ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(out / "vizI_observed_vs_predicted.png", dpi=150); plt.close(fig)


# ================================================================ J
def figJ(df, P, out, minn):
    """Observed band profile against the modelled one, side by side.

    Reading it: where the two bars match, the demographic variables account
    for that group's activity profile. Where they diverge, something the
    model cannot see is doing the work.
    """
    cols = [c for c in DEMOG if c in df.columns]
    heights = [len(levels(df, c)) for c in cols]
    fig, axes = plt.subplots(len(cols), 1,
                             figsize=(10.5, sum(heights) * 0.78 + 2.0),
                             gridspec_kw={"height_ratios": heights})
    axes = np.atleast_1d(axes)
    w = df.wt_final.values

    for ax, c in zip(axes, cols):
        lv = levels(df, c)[::-1]
        ylabels, obs_rows, pred_rows, ypos = [], [], [], []
        pos = 0
        for v in lv:
            m = (df[c] == v).values
            if m.sum() < minn:
                continue
            ww = w[m]; tot = ww.sum()
            obs_rows.append([
                100 * ww[(df.activity_band.values[m] == b)].sum() / tot
                for b in BANDS])
            pred_rows.append([100 * np.average(P[m, j], weights=ww)
                              for j in range(3)])
            ylabels.append(f"{v}  (n={int(m.sum()):,})")
            ypos.append(pos); pos += 1
        if not obs_rows:
            ax.axis("off"); continue

        O, Pr = np.array(obs_rows), np.array(pred_rows)
        yp = np.array(ypos, dtype=float)
        off, bh = 0.19, 0.34
        for rows, shift, alpha, hatch in ((O, -off, 1.0, None),
                                          (Pr, off, .55, "///")):
            left = np.zeros(len(rows))
            for j, b in enumerate(BANDS):
                ax.barh(yp + shift, rows[:, j], left=left, height=bh,
                        color=BAND_COL[b], alpha=alpha, hatch=hatch,
                        edgecolor="white", linewidth=.4)
                left += rows[:, j]
        # flag groups the model misses on the Active band
        for i, yy in enumerate(yp):
            d = Pr[i, 2] - O[i, 2]
            if abs(d) >= 1.5:
                ax.text(101, yy, f"{d:+.1f}", va="center", fontsize=7,
                        color=RED if d < 0 else TEAL, fontweight="bold")
        ax.set_yticks(yp); ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlim(0, 106)
        ax.set_xticks([0, 25, 50, 75, 100])
        style(ax, LABEL.get(c, c), grid="x")

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=BAND_COL[b], label=b) for b in BANDS]
    handles += [Patch(facecolor="#BBBBBB", label="observed (upper bar)"),
                Patch(facecolor="#BBBBBB", alpha=.55, hatch="///",
                      label="modelled (lower bar)")]
    axes[0].legend(handles=handles, fontsize=8, frameon=False, ncol=5,
                   bbox_to_anchor=(1.0, 1.9), loc="upper right")
    axes[-1].set_xlabel("% of group", fontsize=9)
    fig.suptitle("J. Observed band profile versus the modelled one",
                 fontsize=13, fontweight="bold", x=.02, ha="left", color=INK)
    fig.text(.02, .002, "Numbers on the right show the gap on the Active band "
             "(modelled minus observed), where it exceeds 1.5 points.",
             fontsize=8, color=GREY)
    fig.tight_layout(rect=[0, .012, 1, .985])
    fig.savefig(out / "vizJ_band_observed_vs_modelled.png", dpi=150)
    plt.close(fig)


# ================================================================ K
def figK(df, P, out, minn):
    """The same comparison as a single summary: how far off is each group,
    band by band? Easier to scan than J when you want the misses only."""
    cols = [c for c in DEMOG + ["inner_outer"] if c in df.columns]
    w = df.wt_final.values
    rows = []
    for c in cols:
        for v in levels(df, c):
            m = (df[c] == v).values
            if m.sum() < minn:
                continue
            ww = w[m]; tot = ww.sum()
            for j, b in enumerate(BANDS):
                o = 100 * ww[(df.activity_band.values[m] == b)].sum() / tot
                pr = 100 * np.average(P[m, j], weights=ww)
                rows.append({"grp": f"{LABEL.get(c, c)}: {v}", "band": b,
                             "diff": pr - o})
    r = pd.DataFrame(rows)
    piv = r.pivot(index="grp", columns="band", values="diff")[BANDS]
    piv = piv.reindex(piv["Active"].abs().sort_values().index)

    fig, ax = plt.subplots(figsize=(7.6, .30 * len(piv) + 1.8))
    vmax = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap=DIVERGE, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > vmax * .55 else "#222222",
                    fontweight="bold" if abs(v) > 1.5 else "normal")
    ax.set_xticks(range(3)); ax.set_xticklabels(BANDS, fontsize=9)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=7.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    cb.set_label("modelled minus observed (percentage points)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle("K. How far the model is off, by group and band",
                 fontsize=13, fontweight="bold", x=.02, ha="left", color=INK)
    fig.text(.02, .003, "Blue = the model over-states this band for this group; "
             "red = it under-states it. Rows sorted by the size of the miss on "
             "Active.", fontsize=8, color=GREY)
    fig.tight_layout(rect=[0, .015, 1, .97])
    fig.savefig(out / "vizK_band_error_heatmap.png", dpi=150); plt.close(fig)


# ================================================================ L
def _split_fit(df, feats, val_wave, test_wave):
    """Honest evaluation split: train on the early waves, tune on val_wave,
    test on test_wave. Returns predictions on val and test so the accuracy
    figures show performance on data the model never trained on."""
    tr = df[df.wave < val_wave]
    va = df[df.wave == val_wave]
    te = df[df.wave == test_wave]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                        min_frequency=30)
    Xtr = enc.fit_transform(tr[feats])
    m = LogisticRegression(C=1.0, max_iter=3000)
    m.fit(Xtr, tr._t.values.astype(int), sample_weight=tr.wt_final.values)
    out = {}
    for nm, part in (("val", va), ("test", te)):
        if len(part):
            out[nm] = (part, m.predict_proba(enc.transform(part[feats]))[:, 1])
    # prior-only reference: predict the training rate for everyone
    prior = np.average(tr._t, weights=tr.wt_final)
    return out, prior, tr, va, te


def figL(df, feats, out, tlabel, val_wave, test_wave):
    """The headline accuracy scorecard: model vs a no-features baseline,
    on a year the model never saw."""
    parts, prior, tr, va, te = _split_fit(df, feats, val_wave, test_wave)
    if "test" not in parts:
        return
    te_df, p_te = parts["test"]
    y = te_df._t.values.astype(int); w = te_df.wt_final.values

    auc = roc_auc_score(y, p_te, sample_weight=w)
    ll = log_loss(y, p_te, sample_weight=w, labels=[0, 1])
    ll_prior = log_loss(y, np.full_like(p_te, prior), sample_weight=w,
                        labels=[0, 1])
    acc = np.average((( p_te >= .5).astype(int) == y), weights=w)
    acc_base = max(np.average(y, weights=w), 1 - np.average(y, weights=w))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # --- panel 1: ROC curve ---
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y, p_te, sample_weight=w)
    ax.plot(fpr, tpr, color=INK, lw=2.4, label=f"model  (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1.2,
            label="chance  (AUC 0.500)")
    ax.fill_between(fpr, tpr, fpr, color=TEAL, alpha=.12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    style(ax, "How well can it rank people?",
          "false positive rate", "true positive rate", grid="both")

    # --- panel 2: metric bars, model vs baseline ---
    ax = axes[1]
    labels = ["AUC", "Accuracy", "Log loss\n(lower better)"]
    model_v = [auc, acc, ll]
    base_v = [0.5, acc_base, ll_prior]
    x = np.arange(3); bw = .36
    b1 = ax.bar(x - bw / 2, base_v, bw, color=GREY, label="no-features baseline")
    b2 = ax.bar(x + bw / 2, model_v, bw, color=INK, label="model")
    for bars, vals in ((b1, base_v), (b2, model_v)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + .012, f"{v:.3f}",
                    ha="center", fontsize=8, color="#333333")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=8.5, frameon=False)
    style(ax, "Model versus knowing nothing", grid="y")
    ax.set_ylim(0, max(model_v + base_v) * 1.18)

    # --- panel 3: does it generalise? val vs test ---
    ax = axes[2]
    rows = []
    for nm in ("val", "test"):
        if nm in parts:
            pd_, pp = parts[nm]
            yy = pd_._t.values.astype(int); ww = pd_.wt_final.values
            rows.append((nm, roc_auc_score(yy, pp, sample_weight=ww)))
    names = [f"tuning yr\n(wave {val_wave})", f"unseen yr\n(wave {test_wave})"]
    vals = [r[1] for r in rows]
    bars = ax.bar(names[:len(vals)], vals, .5, color=[TEAL, AMBER][:len(vals)])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + .006, f"{v:.3f}",
                ha="center", fontsize=9, color="#333333", fontweight="bold")
    ax.set_ylim(0.5, max(vals) * 1.06)
    style(ax, "No overfitting: unseen \u2248 tuning", ylabel="AUC", grid="y")
    ax.text(.5, .04, "test \u2265 tuning is the opposite of overfitting",
            transform=ax.transAxes, ha="center", fontsize=8, color=GREY)

    fig.suptitle(f"L. Model accuracy on a held-out year  ({tlabel})",
                 fontsize=13, fontweight="bold", x=.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(out / "vizL_accuracy_scorecard.png", dpi=150); plt.close(fig)
    return dict(auc=auc, acc=acc, logloss=ll)


# ================================================================ M
def figM(df, feats, out, tlabel, minn):
    """Forecast-relevant accuracy: predicted vs observed BOROUGH rates on
    the held-out year, which is what the forecast is actually judged on.
    Also runs the leave-one-borough-out check as a second panel."""
    val_wave = df.wave.max() - 1; test_wave = df.wave.max()
    tr = df[df.wave < test_wave]; te = df[df.wave == test_wave]

    def _fit(train, score):
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                            min_frequency=30)
        X = enc.fit_transform(train[feats])
        m = LogisticRegression(C=1.0, max_iter=3000)
        m.fit(X, train._t.values.astype(int),
              sample_weight=train.wt_final.values)
        return m.predict_proba(enc.transform(score[feats]))[:, 1]

    # (a) standard holdout: train on early waves, predict test-wave boroughs
    p_te = _fit(tr, te)
    d = pd.DataFrame({"b": te.borough_name.values, "w": te.wt_final.values,
                      "p": p_te, "y": te._t.values})
    g = d.groupby("b").apply(lambda x: pd.Series({
        "n": len(x),
        "obs": 100 * (x.w * x.y).sum() / x.w.sum(),
        "pred": 100 * (x.w * x.p).sum() / x.w.sum()}), include_groups=False)
    g = g[g.n >= minn]

    # (b) leave-one-borough-out on the test wave, trained without that borough
    rows = []
    for b in sorted(te.borough.unique()):
        trb = tr[tr.borough != b]; teb = te[te.borough == b]
        if len(teb) < minn:
            continue
        pp = _fit(trb, teb); ww = teb.wt_final.values
        rows.append({"b": teb.borough_name.iloc[0],
                     "obs": 100 * np.average(teb._t, weights=ww),
                     "pred": 100 * np.average(pp, weights=ww)})
    gl = pd.DataFrame(rows).set_index("b")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, gg, ttl, seen in ((axes[0], g, "Boroughs the model trained on",
                               True),
                              (axes[1], gl,
                               "Boroughs HIDDEN during training", False)):
        lo = min(gg.obs.min(), gg.pred.min()) - 2
        hi = max(gg.obs.max(), gg.pred.max()) + 2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
        ax.scatter(gg.obs, gg.pred, s=62, color=INK if seen else AMBER,
                   alpha=.8, edgecolor="white")
        corr = gg.obs.corr(gg.pred); mae = (gg.pred - gg.obs).abs().mean()
        for nm in (gg.pred - gg.obs).abs().sort_values(ascending=False).head(5).index:
            ax.annotate(str(nm)[:14], (gg.obs[nm], gg.pred[nm]), fontsize=6.5,
                        xytext=(4, 3), textcoords="offset points", color="#333")
        ax.text(.03, .96, f"correlation {corr:.2f}\naverage error {mae:.2f} pp",
                transform=ax.transAxes, va="top", fontsize=9.5,
                bbox=dict(fc="white", ec="#CCCCCC"))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        style(ax, ttl, f"observed % {tlabel}", f"predicted % {tlabel}",
              grid="both")
    fig.suptitle("M. Forecast accuracy: predicted vs real borough rates on a "
                 "held-out year", fontsize=13, fontweight="bold", x=.02,
                 ha="left", color=INK)
    fig.text(.02, .005, "The right panel is the key test for forecasting: each "
             "borough is predicted by a model that never saw it. Near-identical "
             "accuracy means the model generalises rather than memorises.",
             fontsize=8, color=GREY)
    fig.tight_layout(rect=[0, .02, 1, .93])
    fig.savefig(out / "vizM_forecast_accuracy.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="viz_out")
    ap.add_argument("--target", default="is_active")
    ap.add_argument("--label", default=None,
                    help="wording used on axes, e.g. 'meeting the guideline'")
    ap.add_argument("--min-n", type=int, default=100,
                    help="suppress any group with fewer respondents than this")
    ap.add_argument("--wave", type=int, default=None,
                    help="restrict to a single wave (default: all pooled)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tlabel = args.label or ("meeting the guideline" if args.target == "is_active"
                            else args.target.replace("_", " "))

    df = load(args.parquet, args.target)
    if args.wave:
        df = df[df.wave == args.wave]
    print(f"target: {args.target} | rows: {len(df):,} | "
          f"weighted rate: {wpct(df, '_t'):.1f}%")
    print(f"groups smaller than {args.min_n} respondents are suppressed\n")

    print("exploratory ...")
    figA(df, out, args.min_n, tlabel); print("   A  band composition")
    figB(df, out, args.min_n, tlabel); print("   B  group trends over time")
    figC(df, out, args.min_n, tlabel); print("   C  crosstab heatmaps")
    figD(df, out, args.min_n, tlabel); print("   D  intersectional view")
    figE(df, out, args.min_n, tlabel); print("   E  inner vs outer")
    figF(df, out, args.min_n, tlabel); print("   F  borough ranking")

    print("\nfitting model for prediction visuals ...")
    feats = [c for c in DEMOG + ["borough"] if c in df.columns]
    m, enc = fit_model(df, feats)
    p = m.predict_proba(enc.transform(df[feats]))[:, 1]
    print(f"   features: {feats}")

    figG(df, p, out, tlabel); print("   G  prediction separation")
    figH(df, p, out, args.min_n, tlabel); print("   H  error by group")
    figI(df, p, out, args.min_n, tlabel); print("   I  observed vs predicted")

    print("\naccuracy (held-out evaluation) ...")
    nw = df.wave.nunique()
    if nw >= 3:
        vw, tw = df.wave.max() - 1, df.wave.max()
        sc = figL(df, feats, out, tlabel, vw, tw); print("   L  accuracy scorecard")
        figM(df, feats, out, tlabel, args.min_n); print("   M  forecast accuracy")
        if sc:
            print(f"      AUC {sc['auc']:.3f} | accuracy {sc['acc']:.3f} "
                  f"| log loss {sc['logloss']:.3f}  (on unseen wave {tw})")
    else:
        print("   (need >=3 waves for a held-out accuracy split; skipped)")

    # three-band comparison — only meaningful for the activity target
    if "activity_band" in df.columns and df.activity_band.notna().any():
        print("\nfitting three-band ordinal model ...")
        P, _ = fit_ordinal(df, feats)
        figJ(df, P, out, args.min_n); print("   J  observed vs modelled bands")
        figK(df, P, out, args.min_n); print("   K  band error heatmap")
    else:
        print("\n(skipping J and K: activity_band not available for this target)")

    print(f"\nwritten to {out}/")
    for f in sorted(out.glob("viz*.png")):
        print(f"   {f.name}")


if __name__ == "__main__":
    main()
