"""
13_stageC_indoor_outdoor.py — Section 3 of the brief.

Indoor vs outdoor is asked of a ROTATING SUBSAMPLE — only respondents with
asked_inout == True answer it (~30-38% of the sample). So the honest posture
here is mostly DESCRIPTION, with one caveated London-level forecast.

Covers:
  C1  indoor/outdoor balance over time + a caveated 3-year forecast
  C2  indoor vs outdoor preference by activity level  (DESCRIPTION — the
      strongest part of section 3; a cross-tab, no forecast needed)
  C3  inner vs outer London  x  activity level  x  socio-economic group
      (DESCRIPTION)

Everything is computed on the asked_inout base with its own reweighting, and
every figure states the subsample limitation on its face.

Run:
    python 13_stageC_indoor_outdoor.py dataset1_individual.parquet -o stageC_out
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
INDOOR_C, OUTDOOR_C = "#3C6E8F", "#D98C3F"
SEQ = LinearSegmentedColormap.from_list("s", ["#F4F1EC", "#7FB3B5", INK])
DV = LinearSegmentedColormap.from_list("d", ["#B4472E", "#F2EFE9", TEAL])

INNER = {"E09000001", "E09000007", "E09000011", "E09000012", "E09000013",
         "E09000019", "E09000020", "E09000022", "E09000023", "E09000028",
         "E09000030", "E09000032", "E09000033"}
WAVE_YEAR = {w: 2014 + w for w in range(1, 26)}  # wave 1 = 2015; auto-extends
BANDS = ["Inactive", "Fairly Active", "Active"]
NSSEC_ORDER = ["Higher", "Middle", "Lower", "Student/Other"]

SETTINGS = [("active_home", "At home"), ("active_leisurecentre", "Leisure centre"),
            ("active_park", "Park"), ("active_road", "Street / road"),
            ("active_countryside", "Countryside / coast")]


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


def load(path):
    df = pd.read_parquet(path)
    df["inner_outer"] = np.where(df.borough.astype(str).isin(INNER),
                                 "Inner", "Outer")
    for c in ["nssec4", "activity_band"]:
        if c in df.columns:
            df[c] = df[c].astype("object")
    # the in/out subsample
    if "asked_inout" in df.columns:
        base = df[df.asked_inout == True].copy()
    else:
        base = df[df["active_indoor"].notna()].copy()
    return df, base


# ------------------------------------------------------------------ C1
def c1_balance(base, out):
    """Indoor and outdoor participation over time, on the subsample, with a
    caveated 3-year linear projection."""
    rows = []
    for w, g in base.groupby("wave"):
        rows.append({"wave": w, "year": WAVE_YEAR[w],
                     "indoor": wrate(g, "active_indoor"),
                     "outdoor": wrate(g, "active_outdoor"),
                     "n": int(g.asked_inout.sum()) if "asked_inout" in g else len(g)})
    t = pd.DataFrame(rows).dropna(subset=["indoor", "outdoor"])
    t.to_csv(out / "C1_balance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(t.year, t.indoor, "o-", color=INDOOR_C, lw=2.2, ms=5, label="Indoor")
    ax.plot(t.year, t.outdoor, "o-", color=OUTDOOR_C, lw=2.2, ms=5,
            label="Outdoor")
    ax.axvspan(2018.5, 2020.5, color="grey", alpha=.10)
    ax.text(2019.5, ax.get_ylim()[1], " COVID", fontsize=8, color=GREY,
            va="top")

    # caveated 3-year projection: linear on the last non-COVID slope
    ly = t.year.max()
    for col, c in (("indoor", INDOOR_C), ("outdoor", OUTDOOR_C)):
        today_v = t[col].iloc[-1]
        # value label on today's (last observed) point
        ax.annotate(f"{today_v:.0f}%", (ly, today_v), xytext=(0, 10),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, color=c, fontweight="bold")
        nc = t[~t.wave.isin([5, 6])]
        if len(nc) >= 3:
            sl, ic = np.polyfit(nc.year, nc[col], 1)
            fy = [ly, ly + 3]
            fv = [today_v, today_v + sl * 3]
            ax.plot(fy, fv, "--", color=c, lw=1.8, alpha=.85)
            ax.scatter([ly + 3], [fv[-1]], facecolor="white", edgecolor=c,
                       linewidth=1.6, s=55, zorder=3)
            # value label on the final predicted point
            ax.annotate(f"{fv[-1]:.0f}%", (ly + 3, fv[-1]), xytext=(8, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=11, color=c, fontweight="bold")
    ax.set_xlim(t.year.min() - 0.4, ly + 4.5)
    ax.axvline(t.year.max(), color="#AAAAAA", ls=":", lw=1)
    ax.legend(fontsize=9, frameon=False)
    style(ax, "", "survey year (start)", "% of subsample participating")
    fig.suptitle("C1. Indoor vs outdoor activity over time  (asked of a ~30% "
                 "subsample)", fontsize=12.5, fontweight="bold", x=.02,
                 ha="left", color=INK)
    fig.text(.02, .01, "Solid = observed survey years (no interpolation). "
             "Dashed with open marker = projection on the pre-COVID slope; "
             "indicative only, as this question reaches ~30% of respondents.",
             fontsize=9, color=GREY)
    fig.tight_layout(rect=[0, .03, 1, .95])
    fig.savefig(out / "figC1_balance.png", dpi=150); plt.close(fig)
    return t


# ------------------------------------------------------------------ C2
def c2_by_activity_level(base, out):
    """Indoor/outdoor and specific settings, split by activity band.
    Pure description — the cleanest part of section 3.

    The Inactive band is excluded: 'active in setting X' is 0 by definition
    for someone doing no activity, so that row carries no information and
    would misleadingly read as 'measured and found to do nothing'."""
    bands = [b for b in BANDS if b in base.activity_band.unique()
             and b != "Inactive"]
    # panel 1: indoor vs outdoor by band
    rows = []
    for b in bands:
        g = base[base.activity_band == b]
        rows.append({"band": b, "Indoor": wrate(g, "active_indoor"),
                     "Outdoor": wrate(g, "active_outdoor")})
    io = pd.DataFrame(rows).set_index("band").reindex(bands)

    # panel 2: settings by band
    srows = []
    for b in bands:
        g = base[base.activity_band == b]
        rec = {"band": b}
        for col, lab in SETTINGS:
            if col in g.columns:
                rec[lab] = wrate(g, col)
        srows.append(rec)
    S = pd.DataFrame(srows).set_index("band").reindex(bands)
    io.to_csv(out / "C2_indoor_outdoor_by_band.csv")
    S.to_csv(out / "C2_settings_by_band.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4),
                             gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    x = np.arange(len(bands)); w = .38
    ax.bar(x - w / 2, io.Indoor, w, color=INDOOR_C, label="Indoor")
    ax.bar(x + w / 2, io.Outdoor, w, color=OUTDOOR_C, label="Outdoor")
    for i, b in enumerate(bands):
        ax.text(i - w / 2, io.Indoor[b] + .6, f"{io.Indoor[b]:.0f}",
                ha="center", fontsize=8)
        ax.text(i + w / 2, io.Outdoor[b] + .6, f"{io.Outdoor[b]:.0f}",
                ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(bands, fontsize=9)
    ax.legend(fontsize=9, frameon=False)
    style(ax, "Indoor vs outdoor, by activity level",
          ylabel="% participating")

    ax = axes[1]
    im = ax.imshow(S.values, cmap=SEQ, aspect="auto")
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            v = S.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > np.nanmax(S.values) * .6 else "#222")
    ax.set_xticks(range(S.shape[1]))
    ax.set_xticklabels(S.columns, rotation=25, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(bands))); ax.set_yticklabels(bands, fontsize=9)
    ax.grid(False)
    style(ax, "Where each activity level is active", grid=None)
    fig.suptitle("C2. Indoor/outdoor preference by activity level  "
                 "(description)", fontsize=13, fontweight="bold", x=.02,
                 ha="left", color=INK)
    fig.text(.02, .005, "Inactive people are omitted: being 'active in a "
             "setting' is zero by definition for them. Numbers are % of each "
             "active band participating in that setting.", fontsize=8,
             color=GREY)
    fig.tight_layout(rect=[0, .03, 1, .94])
    fig.savefig(out / "figC2_by_activity_level.png", dpi=150); plt.close(fig)
    return io, S


# ------------------------------------------------------------------ C3
def c3_inner_outer_ses(base, out):
    """Inner vs outer London x socio-economic group, for indoor and outdoor.
    Description of the section-3 relationship the brief names explicitly."""
    ses = [s for s in NSSEC_ORDER if s in base.nssec4.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    stored = {}
    for ax, (col, lab, c) in zip(axes,
                                 [("active_indoor", "Indoor", INDOOR_C),
                                  ("active_outdoor", "Outdoor", OUTDOOR_C)]):
        rows = []
        for s in ses:
            gi = base[(base.nssec4 == s) & (base.inner_outer == "Inner")]
            go = base[(base.nssec4 == s) & (base.inner_outer == "Outer")]
            rows.append({"ses": s, "Inner": wrate(gi, col),
                         "Outer": wrate(go, col)})
        d = pd.DataFrame(rows).set_index("ses").reindex(ses)
        stored[lab] = d
        y = np.arange(len(ses))
        for i, s in enumerate(ses):
            ax.plot([d.Outer[s], d.Inner[s]], [i, i], color="#CBD5DD", lw=3,
                    zorder=1)
        ax.scatter(d.Outer, y, s=70, color=AMBER, zorder=2, label="Outer")
        ax.scatter(d.Inner, y, s=70, color=TEAL, zorder=2, label="Inner")
        ax.set_yticks(y); ax.set_yticklabels(ses, fontsize=9)
        style(ax, f"{lab} activity", xlabel="% participating", grid="x")
        if ax is axes[0]:
            ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    for lab, d in stored.items():
        d.to_csv(out / f"C3_{lab.lower()}_inner_outer_ses.csv")
    fig.suptitle("C3. Inner vs outer London x socio-economic group  "
                 "(description)", fontsize=13, fontweight="bold", x=.02,
                 ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(out / "figC3_inner_outer_ses.png", dpi=150); plt.close(fig)
    return stored


# ------------------------------------------------------------------ main

def c2_separate_charts(base, out):
    """Indoor/outdoor by activity level, as SEPARATE single charts (one per
    activity band) plus a settings chart per band. Shared theme."""
    try:
        import viz_theme as VT
    except Exception:
        return
    import numpy as np, pandas as pd
    bands = [b for b in ["Fairly Active", "Active"]
             if b in base.activity_band.unique()]

    def wr(g, col):
        s = pd.to_numeric(g[col], errors="coerce")
        ok = s.notna() & g.wt_final.notna()
        return 100 * (s[ok] * g.wt_final[ok]).sum() / g.wt_final[ok].sum() \
            if ok.sum() else np.nan

    # one indoor-vs-outdoor chart PER activity level
    for b in bands:
        g = base[base.activity_band == b]
        vals = [wr(g, "active_indoor"), wr(g, "active_outdoor")]
        fig, ax = VT.new_ax(7.5, 5.6)
        bars = ax.bar(["Indoor", "Outdoor"], vals,
                      color=[VT.COL["teal"], VT.COL["amber"]], width=.55)
        for x, v in zip([0, 1], vals):
            ax.text(x, v + 1, f"{v:.0f}%", ha="center", fontsize=13,
                    fontweight="bold")
        ax.set_ylabel("Share participating (%)")
        ax.set_ylim(0, max(vals) * 1.2)
        VT.title_block(ax, f"Indoor vs outdoor activity \u2014 {b} adults",
                       "London, setting subsample")
        VT.finish(fig, out / f"figX_io_{b.replace(' ', '_').lower()}.png",
                  "Asked of about a third of respondents. Description, not a "
                  "forecast.")
    print("   separate indoor/outdoor charts written per activity level")



def c3_separate_charts(base, out):
    """Inner vs outer London by socio-economic group, as two SEPARATE themed
    dumbbell charts (one indoor, one outdoor)."""
    try:
        import viz_theme as VT
    except Exception:
        return
    import numpy as np, pandas as pd
    ses = [s for s in ["Higher", "Middle", "Lower", "Student/Other"]
           if s in base.nssec4.unique()]

    def wr(g, col):
        s = pd.to_numeric(g[col], errors="coerce")
        ok = s.notna() & g.wt_final.notna()
        return 100 * (s[ok] * g.wt_final[ok]).sum() / g.wt_final[ok].sum() \
            if ok.sum() else np.nan

    for col, lab in [("active_indoor", "Indoor"), ("active_outdoor", "Outdoor")]:
        fig, ax = VT.new_ax(9, 0.7 * len(ses) + 2.6)
        y = np.arange(len(ses))
        inner, outer = [], []
        for s in ses:
            gi = base[(base.nssec4 == s) & (base.inner_outer == "Inner")]
            go = base[(base.nssec4 == s) & (base.inner_outer == "Outer")]
            inner.append(wr(gi, col)); outer.append(wr(go, col))
        for i, (a, b) in enumerate(zip(outer, inner)):
            ax.plot([a, b], [i, i], color=VT.COL["lightgrey"], lw=3, zorder=1)
        ax.scatter(outer, y, s=90, color=VT.COL["amber"], zorder=2,
                   label="Outer London")
        ax.scatter(inner, y, s=90, color=VT.COL["teal"], zorder=2,
                   label="Inner London")
        ax.set_yticks(y); ax.set_yticklabels(ses)
        ax.set_xlabel(f"{lab} activity participation (%)")
        ax.grid(axis="y", alpha=0); ax.grid(axis="x", alpha=.22)
        # legend below the axes, out of the data entirely
        ax.legend(frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=2)
        VT.title_block(ax,
                       f"{lab} activity: inner vs outer London by group",
                       "By socio-economic group")
        VT.finish(fig, out / f"figX_C3_{lab.lower()}.png",
                  "Setting subsample (about a third of respondents). "
                  "Description, not a forecast.")
    print("   separate inner/outer x SES charts written")



def c2_by_setting_separate(base, out):
    """Two separate charts: (1) indoor activity by activity level, (2) outdoor
    activity by activity level. Same idea as the combined chart but one per
    setting, so each is clean and single-message."""
    try:
        import viz_theme as VT
    except Exception:
        return
    import numpy as np, pandas as pd
    bands = [b for b in ["Fairly Active", "Active"]
             if b in base.activity_band.unique()]

    def wr(g, col):
        s = pd.to_numeric(g[col], errors="coerce")
        ok = s.notna() & g.wt_final.notna()
        return 100 * (s[ok] * g.wt_final[ok]).sum() / g.wt_final[ok].sum() \
            if ok.sum() else np.nan

    for col, lab, colour in [("active_indoor", "Indoor", VT.COL["teal"]),
                             ("active_outdoor", "Outdoor", VT.COL["amber"])]:
        vals = [wr(base[base.activity_band == b], col) for b in bands]
        fig, ax = VT.new_ax(8, 5.4)
        x = np.arange(len(bands))
        ax.bar(x, vals, color=colour, width=.55)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=13,
                    fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(bands)
        ax.set_ylabel("Share participating (%)")
        ax.set_xlabel("Activity level")
        ax.set_ylim(0, max([v for v in vals if not np.isnan(v)] + [1]) * 1.2)
        VT.title_block(ax, f"{lab} activity, by activity level",
                       "London, setting subsample")
        VT.finish(fig, out / f"figX_setting_{lab.lower()}_by_level.png",
                  "Asked of about a third of respondents. The more active a "
                  "person is, the more they do "
                  f"{lab.lower()} activity. Description, not a forecast.")
    print("   separate indoor-by-level and outdoor-by-level charts written")



def c2_setting_heatmaps_separate(base, out):
    """Two heatmaps like figC2's settings panel, but SEPARATE: one for indoor
    settings x activity level, one for outdoor settings x activity level."""
    try:
        import viz_theme as VT
    except Exception:
        return
    import numpy as np, pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    SEQ = LinearSegmentedColormap.from_list(
        "seq", ["#F4F1EC", "#9DC3C8", VT.COL["teal"], VT.COL["navy"]])

    bands = [b for b in ["Fairly Active", "Active"]
             if b in base.activity_band.unique()]
    indoor = [("active_home", "At home"),
              ("active_leisurecentre", "Leisure centre")]
    outdoor = [("active_park", "Park"),
               ("active_road", "Street or road"),
               ("active_countryside", "Countryside or coast")]

    def wr(g, col):
        s = pd.to_numeric(g[col], errors="coerce")
        ok = s.notna() & g.wt_final.notna()
        return 100 * (s[ok] * g.wt_final[ok]).sum() / g.wt_final[ok].sum() \
            if ok.sum() else np.nan

    for settings, name in [(indoor, "Indoor"), (outdoor, "Outdoor")]:
        settings = [(c, l) for c, l in settings if c in base.columns]
        if not settings:
            continue
        M = np.full((len(settings), len(bands)), np.nan)
        for si, (c, _) in enumerate(settings):
            for bi, b in enumerate(bands):
                M[si, bi] = wr(base[base.activity_band == b], c)
        fig, ax = VT.new_ax(max(7.5, 1.6 * len(bands) + 3),
                            0.9 * len(settings) + 3)
        im = ax.imshow(M, cmap=SEQ, aspect="auto",
                       vmin=0, vmax=np.nanmax(M) if np.isfinite(M).any() else 1)
        for si in range(len(settings)):
            for bi in range(len(bands)):
                v = M[si, bi]
                if not np.isnan(v):
                    ax.text(bi, si, f"{v:.0f}%", ha="center", va="center",
                            fontsize=12,
                            color="white" if v > np.nanmax(M) * .6 else "#222",
                            fontweight="bold")
        ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands)
        ax.set_yticks(range(len(settings)))
        ax.set_yticklabels([l for _, l in settings])
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label("Share participating (%)", fontsize=11)
        VT.title_block(ax, f"{name} settings, by activity level",
                       "Where each activity level is active, indoors"
                       if name == "Indoor" else
                       "Where each activity level is active, outdoors")
        VT.finish(fig, out / f"figX_settings_heat_{name.lower()}.png",
                  "Setting subsample (about a third of respondents). Inactive "
                  "people are omitted (zero by definition). Description.")
    print("   separate indoor/outdoor SETTINGS heatmaps written")



def c3_by_activity_level(base, out):
    """Inner vs outer London by socio-economic group, as THREE separate charts
    — one per activity level. For each level, the value is the share of that
    inner/outer x socio-economic group who fall in that activity band, so you
    can see how the inner/outer gap differs for Active, Fairly Active and
    Inactive people."""
    try:
        import viz_theme as VT
    except Exception:
        return
    import numpy as np, pandas as pd
    ses = [s for s in ["Higher", "Middle", "Lower", "Student/Other"]
           if s in base.nssec4.unique()]
    bands = [b for b in ["Active", "Fairly Active", "Inactive"]
             if b in base.activity_band.unique()]

    def band_share(g, band):
        ok = g.wt_final.notna()
        if ok.sum() == 0:
            return np.nan
        return 100 * g.wt_final[ok & (g.activity_band == band)].sum() \
            / g.wt_final[ok].sum()

    for band in bands:
        fig, ax = VT.new_ax(9, 0.7 * len(ses) + 2.8)
        y = np.arange(len(ses))
        inner, outer = [], []
        for s in ses:
            gi = base[(base.nssec4 == s) & (base.inner_outer == "Inner")]
            go = base[(base.nssec4 == s) & (base.inner_outer == "Outer")]
            inner.append(band_share(gi, band))
            outer.append(band_share(go, band))
        for i, (a, b) in enumerate(zip(outer, inner)):
            ax.plot([a, b], [i, i], color=VT.COL["lightgrey"], lw=3, zorder=1)
        ax.scatter(outer, y, s=95, color=VT.COL["amber"], zorder=2,
                   label="Outer London")
        ax.scatter(inner, y, s=95, color=VT.COL["teal"], zorder=2,
                   label="Inner London")
        # value labels next to each dot
        for i in range(len(ses)):
            for val, col in [(outer[i], VT.COL["amber"]),
                             (inner[i], VT.COL["teal"])]:
                if not np.isnan(val):
                    ax.annotate(f"{val:.0f}%", (val, i), xytext=(0, 9),
                                textcoords="offset points", ha="center",
                                fontsize=9.5, color=col, fontweight="bold")
        ax.set_yticks(y); ax.set_yticklabels(ses)
        ax.set_xlabel(f"Share of group who are {band} (%)")
        ax.grid(axis="y", alpha=0); ax.grid(axis="x", alpha=.22)
        ax.legend(frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=2)
        VT.title_block(ax,
                       f"{band} people: inner vs outer London by group",
                       "Share of each socio-economic group at this activity "
                       "level")
        safe = band.replace(" ", "_").lower()
        VT.finish(fig, out / f"figX_C3level_{safe}.png",
                  "Share of each inner/outer x socio-economic group who fall "
                  "in this activity band. All respondents (not the setting "
                  "subsample). Description, not a forecast.")
    print("   inner/outer x SES charts by activity level written (3 charts)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="stageC_out")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df, base = load(args.parquet)
    cover = 100 * len(base) / len(df)
    print("STAGE C — Section 3 indoor vs outdoor")
    print("=" * 66)
    print(f"in/out subsample: {len(base):,} of {len(df):,} rows ({cover:.0f}%)")
    print(f"waves with in/out data: {sorted(base.wave.unique())}\n")

    print("C1  indoor/outdoor balance over time ...")
    t = c1_balance(base, out)
    print(t[["year", "indoor", "outdoor", "n"]].round(1).to_string(index=False))

    print("\nC2  preference by activity level ...")
    io, S = c2_by_activity_level(base, out)
    print("indoor vs outdoor by band:")
    print(io.round(1).to_string())

    print("\nC2b separate indoor/outdoor charts by activity level ...")
    c2_separate_charts(base, out)
    c2_by_setting_separate(base, out)
    c2_setting_heatmaps_separate(base, out)

    print("\nC3  inner vs outer x socio-economic group ...")
    stored = c3_inner_outer_ses(base, out)
    print("C3b separate inner/outer charts ...")
    c3_separate_charts(base, out)
    print("C3c inner/outer charts by activity level ...")
    # use the FULL sample here (activity band is known for everyone), not the
    # setting subsample, so all three levels are well populated
    c3_by_activity_level(df, out)
    print("indoor, inner vs outer:")
    print(stored["Indoor"].round(1).to_string())

    print(f"\nwritten to {out}/")
    for f in sorted(out.glob("*")):
        print(f"   {f.name}")
    print("\nStage C status:")
    print("  C1 indoor/outdoor balance ...... forecast (wide, ~30% subsample)")
    print("  C2 preference by activity level  described")
    print("  C3 inner/outer x SES ........... described")


if __name__ == "__main__":
    main()
