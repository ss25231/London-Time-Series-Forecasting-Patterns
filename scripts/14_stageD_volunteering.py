"""
14_stageD_volunteering.py — Section 4 of the brief.

Volunteering is the most constrained target in the whole project, established
by the earlier diagnostics:
  * demographics explain only ~15% of the between-borough variation and the
    borough correlation is ~0.11  -> NO borough forecast
  * the question base changed twice (a rotation change at wave 5, and a
    routing change) -> the time trend is unreliable
So Section 4 is delivered as HONEST DESCRIPTION at London and demographic
level, with the base-change limitation stated on every figure. Nothing is
forecast, because the data cannot support it.

Covers:
  D1  volunteering likelihood by background (age, gender, ethnicity,
      disability, socio-economic group)  -> description
  D2  volunteer ROLE types (coach, officiate, organise, support) by
      demographic  -> description, among volunteers

Windows:
  vol_any        : waves 2-8 (rebased earlier); we report the stable window
  vol_*_all      : rebased to all adults; roles restricted to waves 6-8 where
                   the base is consistent

Run:
    python 14_stageD_volunteering.py dataset1_individual.parquet -o stageD_out
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

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.facecolor": "white", "font.size": 11,
                     "savefig.bbox": "tight"})

INK, TEAL, AMBER, RED, GREY = "#1F3050", "#2E7D8F", "#D98C3F", "#B4472E", "#7A8899"
SEQ = LinearSegmentedColormap.from_list("s", ["#F4F1EC", "#7FB3B5", INK])
WAVE_YEAR = {w: 2014 + w for w in range(1, 26)}  # wave 1 = 2015; auto-extends

DISAB_MAP = {"1": "Disabled (limiting)", "1.0": "Disabled (limiting)",
             "2": "Condition, not limiting", "2.0": "Condition, not limiting",
             "3": "No disability", "3.0": "No disability"}
LABEL = {"age4": "Age", "gender": "Gender", "eth5": "Ethnicity",
         "nssec4": "Socio-economic group", "Disab3": "Disability"}
ORDER = {
    "age4": ["16-24", "25-44", "45-64", "65+"],
    "gender": ["Male", "Female"],
    "eth5": ["White British", "White Other", "Asian", "Black", "Mixed/Other"],
    "nssec4": ["Higher", "Middle", "Lower", "Student/Other"],
    "Disab3": ["Disabled (limiting)", "Condition, not limiting", "No disability"],
}
GROUP_COLS = ["age4", "gender", "eth5", "nssec4", "Disab3"]

ROLES = [("vol_coach_all", "Coaching /\ninstructing"),
         ("vol_officiate_all", "Refereeing /\nofficiating"),
         ("vol_organise_all", "Organising /\nadmin"),
         ("vol_support_all", "Stewarding /\nother support")]


def style(ax, title="", xlabel="", ylabel="", grid="y"):
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left",
                     pad=6, color=INK)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(alpha=.25, linewidth=.6, axis=grid); ax.set_axisbelow(True)


def wrate(g, col):
    s = pd.to_numeric(g[col], errors="coerce")
    ok = s.notna() & g.wt_final.notna()
    if ok.sum() == 0:
        return np.nan
    return 100 * (s[ok] * g.wt_final[ok]).sum() / g.wt_final[ok].sum()


def levels(df, col):
    have = [str(v) for v in df[col].dropna().unique()
            if str(v) not in ("Not disclosed", "nan")]
    o = ORDER.get(col)
    if o:
        ordered = [v for v in o if v in have]
        return ordered if ordered else sorted(have)
    return sorted(have)


def load(path):
    df = pd.read_parquet(path)
    if "Disab3" in df.columns:
        s = df["Disab3"].astype("object")
        df["Disab3"] = s.map(lambda x: str(x) if pd.notna(x) else x)\
                        .map(DISAB_MAP).fillna(df["Disab3"].astype("object"))
    for c in GROUP_COLS:
        if c in df.columns:
            df[c] = df[c].astype("object")
    return df


# ------------------------------------------------------------------ D1
def d1_volunteering(df, out):
    """Volunteering rate over time (context) + by demographic group."""
    # time context — flagged as unreliable
    trows = []
    for w, g in df.groupby("wave"):
        r = wrate(g, "vol_any")
        cov = 100 * (pd.to_numeric(g["vol_any"], errors="coerce").notna()
                     & g.wt_final.notna()).mean()
        if not np.isnan(r):
            trows.append({"year": WAVE_YEAR[w], "vol_%": r, "coverage_%": cov})
    tt = pd.DataFrame(trows)
    tt.to_csv(out / "D1_volunteering_over_time.csv", index=False)

    # by demographic (pool all waves with vol_any present)
    base = df[pd.to_numeric(df["vol_any"], errors="coerce").notna()]
    london = wrate(base, "vol_any")
    grows = []
    fig, axes = plt.subplots(1, len(GROUP_COLS),
                             figsize=(2.7 * len(GROUP_COLS), 4.4), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, gcol in zip(axes, GROUP_COLS):
        lv = levels(base, gcol)
        vals, ns = [], []
        for v in lv:
            g = base[base[gcol] == v]
            vals.append(wrate(g, "vol_any")); ns.append(len(g))
            grows.append({"dimension": LABEL[gcol], "group": v,
                          "vol_%": round(vals[-1], 1), "n": len(g)})
        y = np.arange(len(lv))[::-1]
        ax.barh(y, vals, color=TEAL, height=.66)
        ax.axvline(london, color=INK, ls="--", lw=1)
        for i, (vv, nn) in enumerate(zip(vals, ns)):
            ax.text(vv + .3, y[i], f"{vv:.0f}", va="center", fontsize=8)
        ax.set_yticks(y); ax.set_yticklabels([str(x)[:16] for x in lv],
                                             fontsize=8)
        style(ax, LABEL[gcol], xlabel="% volunteering", grid="x")
    pd.DataFrame(grows).to_csv(out / "D1_volunteering_by_group.csv", index=False)
    fig.suptitle(f"D1. Volunteering in sport by background  "
                 f"(London average {london:.0f}%; description, not forecast)",
                 fontsize=12.5, fontweight="bold", x=.02, ha="left", color=INK)
    fig.text(.02, .01, "Pooled across waves. The volunteering question base "
             "changed during the series, so it is reported as a demographic "
             "profile, not a trend or a borough forecast.", fontsize=8,
             color=GREY)
    fig.tight_layout(rect=[0, .03, 1, .94])
    fig.savefig(out / "figD1_by_group.png", dpi=150); plt.close(fig)
    return tt, pd.DataFrame(grows), london


# ------------------------------------------------------------------ D2
def d2_roles(df, out):
    """Compute the role-by-demographic tables and write their CSVs. The cramped
    5-panel grid figure is deprecated (it overcrowded); the clean per-dimension
    heatmaps figX_roles_<dim>.png replace it, so no grid figure is drawn here."""
    have = [(c, l) for c, l in ROLES if c in df.columns]
    if not have:
        return None
    base = df[df.wave >= 6].copy()
    overall = {l: wrate(base, c) for c, l in have}
    stored = {}
    for gcol in GROUP_COLS:
        lv = levels(base, gcol)
        M = np.full((len(lv), len(have)), np.nan)
        for i, v in enumerate(lv):
            g = base[base[gcol] == v]
            for j, (c, _) in enumerate(have):
                M[i, j] = wrate(g, c)
        stored[gcol] = pd.DataFrame(M, index=lv, columns=[l for _, l in have])
        stored[gcol].to_csv(out / f"D2_roles_{gcol}.csv")
    return overall, stored


# ------------------------------------------------------------------ main

def d1_separate_charts(df, out):
    """Volunteering rate by each demographic dimension, as SEPARATE single
    horizontal-bar charts. Shared theme, full readable labels."""
    try:
        import viz_theme as VT
    except Exception:
        return
    import numpy as np, pandas as pd
    base = df[pd.to_numeric(df["vol_any"], errors="coerce").notna()].copy()
    london = wrate(base, "vol_any")   # same estimator as the grouped D1 chart

    def wr(g):
        s = pd.to_numeric(g["vol_any"], errors="coerce")
        ok = s.notna() & g.wt_final.notna()
        return 100 * (s[ok] * g.wt_final[ok]).sum() / g.wt_final[ok].sum() \
            if ok.sum() else np.nan

    for gcol in GROUP_COLS:
        lv = levels(base, gcol)
        if not lv:
            continue
        vals = [wr(base[base[gcol] == v]) for v in lv]
        # readable disability labels
        disp = lv
        fig, ax = VT.new_ax(9.5, 0.6 * len(lv) + 2.4)
        ax.grid(axis="y", alpha=0); ax.grid(axis="x", alpha=.22)
        y = np.arange(len(lv))[::-1]
        ax.barh(y, vals, color=VT.COL["teal"], height=.62, edgecolor="white")
        ax.axvline(london, color=VT.COL["navy"], ls="--", lw=1.3)
        ax.text(london, len(lv) - 0.4, f"  London average {london:.0f}%",
                fontsize=10.5, color=VT.COL["navy"], va="top")
        for v, yy in zip(vals, y):
            ax.text(v + 0.3, yy, f"{v:.0f}%", va="center", fontsize=11.5)
        ax.set_yticks(y); ax.set_yticklabels([str(x)[:32] for x in disp])
        ax.set_xlabel("Share who volunteer in sport (%)")
        ax.set_xlim(0, max(vals) * 1.18)
        VT.title_block(ax, f"Volunteering in sport by {VT.LABELS.get(gcol, gcol).lower()}",
                       "London adults, pooled across years")
        VT.finish(fig, out / f"figX_vol_{gcol}.png",
                  "The volunteering question base changed during the series, so "
                  "this is a demographic profile, not a trend or forecast.")
    print("   separate volunteering charts written per demographic")



def d2_roles_separate(df, out):
    """Volunteer roles by demographic, as SEPARATE heatmaps (one per
    dimension) with a side colourbar — matching the within-borough style."""
    try:
        import viz_theme as VT
    except Exception:
        return
    import numpy as np, pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    SEQ = LinearSegmentedColormap.from_list(
        "seq", ["#F4F1EC", "#9DC3C8", VT.COL["teal"], VT.COL["navy"]])
    have = [(c, l.replace(chr(10), " ")) for c, l in ROLES if c in df.columns]
    if not have:
        return
    base = df[df.wave >= 6].copy()

    def wr(g, col):
        s = pd.to_numeric(g[col], errors="coerce")
        ok = s.notna() & g.wt_final.notna()
        return 100 * (s[ok] * g.wt_final[ok]).sum() / g.wt_final[ok].sum() \
            if ok.sum() else np.nan

    for gcol in GROUP_COLS:
        lv = levels(base, gcol)
        if not lv:
            continue
        M = np.full((len(have), len(lv)), np.nan)      # roles x groups
        for ri, (c, _) in enumerate(have):
            for gi, v in enumerate(lv):
                M[ri, gi] = wr(base[base[gcol] == v], c)
        fig, ax = VT.new_ax(max(8, 1.1 * len(lv) + 3.5), 0.8 * len(have) + 3)
        im = ax.imshow(M, cmap=SEQ, aspect="auto",
                       vmin=0, vmax=np.nanmax(M) if np.isfinite(M).any() else 1)
        for ri in range(len(have)):
            for gi in range(len(lv)):
                v = M[ri, gi]
                if not np.isnan(v):
                    ax.text(gi, ri, f"{v:.1f}", ha="center", va="center",
                            fontsize=10.5,
                            color="white" if v > np.nanmax(M) * .6 else "#222")
        ax.set_xticks(range(len(lv)))
        ax.set_xticklabels([str(v)[:18] for v in lv], rotation=25, ha="right")
        ax.set_yticks(range(len(have)))
        ax.set_yticklabels([l for _, l in have])
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label("Share of all adults (%)", fontsize=11)
        VT.title_block(ax,
                       f"Volunteer roles by {VT.LABELS.get(gcol, gcol).lower()}",
                       "London adults, 2020\u201322; darker = higher share")
        VT.finish(fig, out / f"figX_roles_{gcol}.png",
                  "Share of all adults taking each role. Restricted to waves "
                  "with a consistent question base; description, not a forecast.")
    print("   separate volunteer-role heatmaps written per demographic")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="stageD_out")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = load(args.parquet)
    print("STAGE D — Section 4 volunteering")
    print("=" * 66)
    if "vol_any" not in df.columns:
        print("  vol_any not found — cannot proceed")
        return
    cov = 100 * pd.to_numeric(df["vol_any"], errors="coerce").notna().mean()
    print(f"vol_any coverage: {cov:.0f}% of rows\n")

    print("D1  volunteering by background ...")
    tt, gdf, london = d1_volunteering(df, out)
    print(f"London volunteering rate (pooled): {london:.1f}%")
    print("\ncoverage by year (note the base change):")
    print(tt.round(1).to_string(index=False))
    print("\nby group:")
    print(gdf.to_string(index=False))

    print("\nD1b separate volunteering charts by demographic ...")
    d1_separate_charts(df, out)

    print("\nD2b separate volunteer-role charts ...")
    d2_roles_separate(df, out)

    print("\nD2  volunteer roles by demographic ...")
    r = d2_roles(df, out)
    if r:
        overall, stored = r
        print("overall role mix (% of all adults, waves 2020-22):")
        for k, v in overall.items():
            print(f"   {k.replace(chr(10), ' '):28s} {v:.1f}%")

    print(f"\nwritten to {out}/")
    for f in sorted(out.glob("*")):
        print(f"   {f.name}")
    print("\nStage D status:")
    print("  D1 volunteering by background .. described (no borough forecast)")
    print("  D2 volunteer roles ............. described (waves 2020-22)")
    print("\n  Volunteering is reported as description at London and demographic")
    print("  level only. A borough forecast is not supportable: demographics")
    print("  explain ~15% of borough variation and the question base changed")
    print("  during the series.")


if __name__ == "__main__":
    main()
