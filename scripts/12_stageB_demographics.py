"""
12_stageB_demographics.py — Section 2 of the brief.

Reuses the engine in 09_forecast.py. Where Stage A forecast each activity for
London as a whole, Stage B forecasts activity FOR EACH DEMOGRAPHIC GROUP and
identifies which activities each group most engages in.

Covers:
  B1  activity forecast for every age / gender / ethnicity / disability / SES
      group  (3-year point + 10-year composition scenario)
  B2  each group's most-likely activities  (ranked participation)
  B3  activity by demographic WITHIN borough  -> POOLED, wide intervals
      (borough x demographic cells are ~6 people per year, so single-year
      estimates are unusable; we pool all waves and suppress thin cells)
  B4  participation among people with a long-term health condition
      -> DESCRIPTION + CAVEAT (health is waves 4-8 only; disty2 is defined
      only for people with a limiting disability)

How the group forecast works
----------------------------
The propensity model gives each person a probability of being active. To
forecast a GROUP, we hold that group's within-type behaviour fixed and let
the rest of its composition drift with the projected population — then read
off the group's rate. Because the behaviour trend is estimated London-wide,
group differences in the forecast come from composition, which is exactly
what the brief asks to see.

Run:
    python 12_stageB_demographics.py dataset1_individual.parquet -o stageB_out
"""

import argparse
import importlib.util
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.facecolor": "white", "font.size": 11,
                     "savefig.bbox": "tight"})

INK, TEAL, AMBER, RED, GREY = "#1F3050", "#2E7D8F", "#D98C3F", "#B4472E", "#7A8899"
BAND_COL = {"Inactive": "#C0392B", "Fairly Active": "#E8A33D", "Active": "#2E8B6F"}

LABEL = {"age4": "Age", "gender": "Gender", "eth5": "Ethnicity",
         "nssec4": "Socio-economic group", "educ3": "Education",
         "Disab3": "Disability", "BMIG": "BMI group"}

# Disab3 is stored as a numeric code (1/2/3). Active Lives coding:
#   1 = Disabled (day-to-day activities limited)
#   2 = Long-term condition, not limiting
#   3 = No disability or long-term condition
DISAB_MAP = {"1": "Disabled (limiting)", "1.0": "Disabled (limiting)",
             "2": "Condition, not limiting", "2.0": "Condition, not limiting",
             "3": "No disability", "3.0": "No disability"}
ORDER = {
    "age4": ["16-24", "25-44", "45-64", "65+"],
    "gender": ["Male", "Female"],
    "eth5": ["White British", "White Other", "Asian", "Black", "Mixed/Other"],
    "nssec4": ["Higher", "Middle", "Lower", "Student/Other"],
    "Disab3": ["Disabled (limiting)", "Condition, not limiting", "No disability"],
}
GROUP_COLS = ["age4", "gender", "eth5", "nssec4", "Disab3"]

ACTIVITIES = {
    "part_walking": "Walking", "part_cycling": "Cycling",
    "part_fitness": "Fitness", "part_team": "Team sports",
    "part_racket": "Racket sports", "part_dance": "Dance",
    "part_active_travel": "Active travel",
}


def normalise_disab(df):
    """Map the numeric Disab3 code to readable labels, tolerating floats."""
    if "Disab3" in df.columns:
        s = df["Disab3"].astype("object")
        key = s.map(lambda x: str(x) if pd.notna(x) else x)
        df["Disab3"] = key.map(DISAB_MAP).fillna(df["Disab3"].astype("object"))
    return df


def load_engine():
    here = Path(__file__).resolve().parent
    cand = here / "09_forecast.py"
    if not cand.exists():
        cand = Path("09_forecast.py")
    spec = importlib.util.spec_from_file_location("fc09", cand)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def levels(df, col):
    have = [str(v) for v in df[col].dropna().unique()
            if str(v) not in ("Not disclosed", "nan")]
    o = ORDER.get(col)
    if o:
        ordered = [v for v in o if v in have]
        # if the fixed order matched nothing (different label scheme), fall
        # back to whatever the column actually contains
        return ordered if ordered else sorted(have)
    return sorted(have)


# ------------------------------------------------------------------ B1
def forecast_groups(eng, path, horizon, shrink):
    """For every group, an observed rate, a 3-year forecast, and a 10-year
    composition scenario. Behaviour trend is London-wide; the group-level
    movement is compositional."""
    df = normalise_disab(eng.load(path, "is_active"))
    last_w = int(df.wave.max())
    hz = [last_w + i for i in range(1, horizon + 1)]
    h10 = last_w + 10

    m, enc, feats = eng.fit_propensity(df)
    # per-person predicted probability, attached back to the frame
    df = df.copy()
    df["_p"] = m.predict_proba(enc.transform(df[feats]))[:, 1]

    trend_df, slope, intercept, covid = eng.behaviour_trend(df, m, enc, feats)

    def resid(w):
        return slope * w + intercept
    anchor = resid(last_w)

    # observed composition and its projection
    comp = eng.observed_composition(df)
    proj3 = eng.project_composition(comp, hz, shrink=shrink)
    proj10 = eng.project_composition(comp, [h10], shrink=shrink)

    # cell-level propensity (borough x age x gender x eth)
    prop = eng.propensity_by_cell(df, m, enc, feats)

    # Model-predicted Active probability for every cell (borough x age x
    # gender x eth). This is the propensity surface the projection re-weights.
    prop = prop.copy()
    prop["p_hat"] = prop["p_hat"].clip(0, 1)

    def group_rate(shares, gcol, gval, trend_pp):
        """Weighted Active rate for one demographic group under a given
        population projection. For group columns that ARE cell dimensions
        (age, gender, eth) we filter the cells; for the others (nssec, disab)
        the cell propensities are averaged within the group as observed,
        because those are not projected dimensions."""
        j = shares.merge(prop, on=["borough"] + eng.CELL, how="left")
        bmean = prop.groupby("borough").apply(
            lambda x: np.average(x.p_hat, weights=x.n), include_groups=False)
        j["p_hat"] = j.p_hat.fillna(j.borough.map(bmean))
        if gcol in eng.CELL:
            j = j[j[gcol] == gval]
        else:
            # group not a projected dimension: scale every cell's propensity by
            # this group's observed activity ratio to the cell, then reweight.
            ref = df[df[gcol] == gval]
            ratio = (np.average((ref.activity_band == "Active"),
                                weights=ref.wt_final)
                     / max(np.average((df.activity_band == "Active"),
                                      weights=df.wt_final), 1e-6))
            j["p_hat"] = (j["p_hat"] * ratio).clip(0, 1)
        rate = 100 * (j.share * j.p_hat).sum() / j.share.sum()
        return rate + trend_pp

    rows = []
    for gcol in GROUP_COLS:
        for v in levels(df, gcol):
            sub = df[df[gcol] == v]
            obs = 100 * np.average((sub.activity_band == "Active"),
                                   weights=sub.wt_final)
            base = group_rate(comp[comp.wave == last_w], gcol, v, anchor)
            # calibrate the whole trajectory so the group's baseline equals its
            # observed rate; the forecast then reads on the same scale as the
            # numbers people recognise, and only the CHANGE comes from the model.
            offset = obs - base
            f3 = group_rate(proj3[proj3.wave == hz[-1]], gcol, v, resid(hz[-1]))
            s_comp = group_rate(proj10[proj10.wave == h10], gcol, v, anchor)
            s_trend = group_rate(proj10[proj10.wave == h10], gcol, v, resid(h10))
            s_rev = group_rate(proj10[proj10.wave == h10], gcol, v,
                               anchor + covid)
            rows.append({
                "dimension": LABEL[gcol], "group": v, "n": len(sub),
                "observed": round(obs, 1),
                "forecast_3yr": round(f3 + offset, 1),
                "scen_composition": round(s_comp + offset, 1),
                "scen_trend": round(s_trend + offset, 1),
                "scen_reverts": round(s_rev + offset, 1),
            })
    return pd.DataFrame(rows), eng.WAVE_YEAR[last_w] + horizon, slope, covid


# ------------------------------------------------------------------ B2
def preferred_activities(df):
    """Which activities each group does most, as a rate table for heatmaps."""
    have = [c for c in ACTIVITIES if c in df.columns]
    out = {}
    for gcol in GROUP_COLS:
        rows = []
        for v in levels(df, gcol):
            sub = df[df[gcol] == v]
            rec = {"group": v}
            for c in have:
                s = pd.to_numeric(sub[c], errors="coerce")
                ok = s.notna() & sub.wt_final.notna()
                rec[ACTIVITIES[c]] = (100 * (s[ok] * sub.wt_final[ok]).sum()
                                      / sub.wt_final[ok].sum()) if ok.sum() else np.nan
            rows.append(rec)
        if rows:
            out[gcol] = pd.DataFrame(rows).set_index("group")
    return out


# ------------------------------------------------------------------ B4
def health_description(df):
    """B4 — activity among people reporting a long-term condition.
    health is waves 4-8; disty2_POP flags a long-term condition among people
    with a limiting disability. Both described, not forecast."""
    rows = []
    w_all = df.wt_final
    # general self-rated health (waves 4-8)
    if "health" in df.columns:
        h = pd.to_numeric(df["health"], errors="coerce")
        act = (df.activity_band == "Active").astype(float)
        for code, lab in [(1, "Very good health"), (2, "Good"),
                          (3, "Fair"), (4, "Bad"), (5, "Very bad")]:
            m = (h == code) & df.wt_final.notna()
            if m.sum() > 200:
                rows.append({"measure": "Self-rated health", "group": lab,
                             "n": int(m.sum()),
                             "active_%": 100 * np.average(act[m], weights=w_all[m])})
    # long-term condition flag
    for col, lab in [("disty2_POP", "Long-term condition\n(among disabled)"),
                     ("disty5_POP", "Mental-health condition\n(among disabled)")]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            act = (df.activity_band == "Active").astype(float)
            for val, tag in [(1, "reports it"), (0, "does not")]:
                m = (s == val) & df.wt_final.notna()
                if m.sum() > 200:
                    rows.append({"measure": lab, "group": tag, "n": int(m.sum()),
                                 "active_%": 100 * np.average(act[m],
                                                              weights=w_all[m])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def fig_group_forecast(fc, fc_year, out):
    dims = fc.dimension.unique()
    fig, axes = plt.subplots(1, len(dims), figsize=(2.9 * len(dims), 4.6),
                             sharex=False)
    axes = np.atleast_1d(axes)
    for ax, dim in zip(axes, dims):
        d = fc[fc.dimension == dim]
        y = np.arange(len(d))[::-1]
        ax.hlines(y, d.scen_reverts, d.scen_trend, color="#DCE3EA", lw=5,
                  zorder=1)
        ax.scatter(d.observed, y, color=GREY, s=42, zorder=2, label="today")
        ax.scatter(d.forecast_3yr, y, color=INK, s=52, zorder=3,
                   label=f"{fc_year}")
        ax.set_yticks(y); ax.set_yticklabels(d.group, fontsize=8.5)
        ax.set_title(dim, fontsize=10.5, fontweight="bold", loc="left",
                     color=INK)
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.25, axis="x"); ax.set_axisbelow(True)
    axes[0].set_xlabel("% Active", fontsize=8)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("B1. Activity forecast by demographic group  "
                 "(bar = 10-year scenario range)", fontsize=13,
                 fontweight="bold", x=.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(out / "figB1_group_forecast.png", dpi=150); plt.close(fig)


def fig_preferred(tables, out):
    from matplotlib.colors import LinearSegmentedColormap
    SEQ = LinearSegmentedColormap.from_list("s", ["#F4F1EC", "#7FB3B5", INK])
    dims = list(tables.keys())
    fig, axes = plt.subplots(1, len(dims), figsize=(3.2 * len(dims), 4.4))
    axes = np.atleast_1d(axes)
    for ax, gcol in zip(axes, dims):
        T = tables[gcol]
        im = ax.imshow(T.values, cmap=SEQ, aspect="auto")
        for i in range(T.shape[0]):
            # bold the top activity in each row
            top = np.nanargmax(T.values[i])
            for j in range(T.shape[1]):
                v = T.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if v > np.nanmax(T.values) * .6 else "#222",
                            fontweight="bold" if j == top else "normal")
        ax.set_xticks(range(T.shape[1]))
        ax.set_xticklabels(T.columns, rotation=40, ha="right", fontsize=7.5)
        ax.set_yticks(range(T.shape[0]))
        ax.set_yticklabels(T.index, fontsize=8)
        ax.set_title(LABEL[gcol], fontsize=10.5, fontweight="bold", loc="left",
                     color=INK, pad=6)
        ax.grid(False)
        last_im = im
    fig.suptitle("B2. Which activities each group does most  "
                 "(bold = top activity for that group)", fontsize=13,
                 fontweight="bold", x=.02, ha="left", color=INK)
    # shared colourbar on the right, as the key/legend
    cb = fig.colorbar(last_im, ax=list(axes), fraction=0.025, pad=0.02)
    cb.set_label("Share participating (%)", fontsize=10)
    fig.savefig(out / "figB2_preferred_activities.png", dpi=150,
                bbox_inches="tight"); plt.close(fig)


def fig_within_borough(eng, df, out, minn):
    """B3 — activity by demographic within borough, pooled across years.
    Shows the age gradient inside each borough; suppresses thin cells."""
    bn = {b: n for b, n in zip(
        ["E09000002", "E09000032"], ["Barking & Dagenham", "Wandsworth"])}
    piv = df.groupby(["borough", "age4"]).apply(
        lambda g: 100 * np.average((g.activity_band == "Active"),
                                   weights=g.wt_final) if len(g) >= minn else np.nan,
        include_groups=False).unstack()
    piv = piv[[c for c in ORDER["age4"] if c in piv.columns]]
    # order boroughs by overall activity
    order = df.groupby("borough").apply(
        lambda g: 100 * np.average((g.activity_band == "Active"),
                                   weights=g.wt_final), include_groups=False)
    piv = piv.reindex(order.sort_values().index)
    from matplotlib.colors import LinearSegmentedColormap
    DV = LinearSegmentedColormap.from_list("d", ["#B4472E", "#F2EFE9", TEAL])
    fig, ax = plt.subplots(figsize=(6.2, .28 * len(piv) + 1.6))
    im = ax.imshow(piv.values, cmap=DV, aspect="auto",
                   vmin=np.nanpercentile(piv.values, 5),
                   vmax=np.nanpercentile(piv.values, 95))
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, "·" if np.isnan(v) else f"{v:.0f}", ha="center",
                    va="center", fontsize=7,
                    color="#555" if np.isnan(v) else "#222")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, fontsize=9)
    ax.set_yticks(range(piv.shape[0]))
    try:
        import viz_theme as _VT
        names = [_VT.BOROUGH_NAME.get(str(b), str(b)) for b in piv.index]
    except Exception:
        names = [str(b) for b in piv.index]
    ax.set_yticklabels(names, fontsize=7.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    cb.set_label("% Active", fontsize=8.5)
    fig.suptitle("B3. Age gradient within each borough  (all years pooled; "
                 "dot = base too small)", fontsize=11.5, fontweight="bold",
                 x=.02, ha="left", color=INK)
    fig.text(.02, .01, "Single-year borough x demographic cells are too small "
             "to report, so all eight years are pooled.", fontsize=8, color=GREY)
    fig.tight_layout(rect=[0, .02, 1, .97])
    fig.savefig(out / "figB3_within_borough.png", dpi=150); plt.close(fig)


def fig_health(hd, out):
    if hd.empty:
        return
    measures = hd.measure.unique()
    fig, axes = plt.subplots(1, len(measures), figsize=(3.6 * len(measures), 3.8))
    axes = np.atleast_1d(axes)
    for ax, mz in zip(axes, measures):
        d = hd[hd.measure == mz]
        y = np.arange(len(d))
        ax.barh(y, d["active_%"], color=TEAL, height=.6)
        for i, (v, n) in enumerate(zip(d["active_%"], d.n)):
            ax.text(v + .5, i, f"{v:.0f}%", va="center", fontsize=8)
        ax.set_yticks(y); ax.set_yticklabels(d.group, fontsize=8.5)
        ax.set_title(mz, fontsize=10, fontweight="bold", loc="left", color=INK)
        ax.set_xlabel("% Active", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.25, axis="x"); ax.set_axisbelow(True)
    fig.suptitle("B4. Activity and health conditions  (described, not forecast)",
                 fontsize=13, fontweight="bold", x=.02, ha="left", color=INK)
    fig.text(.02, .01, "Self-rated health is collected from 2018-19 onward; "
             "long-term-condition flags apply to people with a limiting "
             "disability. Snapshot, not a projection.", fontsize=8, color=GREY)
    fig.tight_layout(rect=[0, .04, 1, .94])
    fig.savefig(out / "figB4_health.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------ main

def stageB_separate_charts(fc, tables, hd, out):
    """Separate, clearer Stage B charts:
       B1 - a clean forecast dot chart per demographic (no overlap)
       B2 - a heatmap per demographic (activities down, groups across) so the
            message reads instantly with no legend clutter
       B4 - a simple ranked bar per health measure
    """
    try:
        import viz_theme as VT
    except Exception:
        return
    import numpy as np, pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    SEQ = LinearSegmentedColormap.from_list(
        "seq", ["#F4F1EC", "#9DC3C8", VT.COL["teal"], VT.COL["navy"]])

    # ---- B1: ranked bar per dimension; bar = forecast, tick = today ----
    for dim in fc.dimension.unique():
        d = fc[fc.dimension == dim].copy().sort_values("forecast_3yr")
        fig, ax = VT.new_ax(9.5, 0.7 * len(d) + 2.8)
        y = np.arange(len(d))
        # forecast as the bar
        ax.barh(y, d.forecast_3yr, height=.62, color=VT.COL["teal"],
                edgecolor="white", zorder=2, label="Forecast (3-year)")
        # "today" as a dark vertical tick on each bar
        ax.scatter(d.observed, y, marker="|", s=380, color=VT.COL["navy"],
                   linewidths=2.4, zorder=4, label="Today")
        # forecast value at the end of each bar
        for f, yy in zip(d.forecast_3yr, y):
            ax.text(f + 0.6, yy, f"{f:.0f}%", va="center", fontsize=11.5,
                    color="#333333")
        ax.set_yticks(y); ax.set_yticklabels(d.group)
        ax.set_xlabel("Share who are Active (%)")
        ax.set_xlim(0, d.forecast_3yr.max() * 1.16)
        ax.grid(axis="y", alpha=0); ax.grid(axis="x", alpha=.22)
        # legend below the axes so it never covers a bar
        ax.legend(frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.14), ncol=2)
        VT.title_block(ax, f"Activity forecast by {dim.lower()}",
                       "Bar = three-year forecast; vertical tick = today's rate")
        VT.finish(fig, out / f"figX_B1_{dim.split()[0].lower()}.png",
                  "Groups ordered by forecast. The forecast is calibrated to "
                  "each group's observed rate, so the tick and bar are close "
                  "when little change is projected.")

    # ---- B2: heatmap per dimension (no legend, reads instantly) ----
    for gcol, T in tables.items():
        acts = list(T.columns); grps = list(T.index)
        M = T.values.astype(float)
        fig, ax = VT.new_ax(max(8.5, 1.05 * len(grps) + 3.5),
                            0.62 * len(acts) + 3.2)
        im = ax.imshow(M.T, cmap=SEQ, aspect="auto",
                       vmin=np.nanmin(M), vmax=np.nanmax(M))
        # annotate every cell, bold the top activity per group (per column)
        for gi in range(len(grps)):
            col = M[gi, :]
            top = int(np.nanargmax(col)) if np.isfinite(col).any() else -1
            for ai in range(len(acts)):
                v = M[gi, ai]
                if not np.isnan(v):
                    ax.text(gi, ai, f"{v:.0f}", ha="center", va="center",
                            fontsize=10.5,
                            color="white" if v > np.nanmax(M) * .62 else "#222",
                            fontweight="bold" if ai == top else "normal")
        ax.set_xticks(range(len(grps)))
        ax.set_xticklabels([str(g)[:16] for g in grps], rotation=25, ha="right")
        ax.set_yticks(range(len(acts))); ax.set_yticklabels(acts)
        ax.grid(False)
        # colourbar as the only "legend", clearly labelled
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label("Share participating (%)", fontsize=11)
        VT.title_block(ax,
                       f"Preferred activities by {VT.LABELS.get(gcol, gcol).lower()}",
                       "Darker = higher participation; bold = top activity for "
                       "that group")
        VT.finish(fig, out / f"figX_B2_{gcol}.png",
                  "Each cell is the share of that group who did the activity.")

    # ---- B4: health, simple ranked bars ----
    if hd is not None and not hd.empty:
        for mz in hd.measure.unique():
            dd = hd[hd.measure == mz]
            fig, ax = VT.new_ax(9, 0.6 * len(dd) + 2.6)
            y = np.arange(len(dd))[::-1]
            ax.barh(y, dd["active_%"], color=VT.COL["teal"], height=.62,
                    edgecolor="white")
            for v, yy in zip(dd["active_%"], y):
                ax.text(v + 0.6, yy, f"{v:.0f}%", va="center", fontsize=11.5)
            ax.set_yticks(y); ax.set_yticklabels([str(x)[:30] for x in dd.group])
            ax.set_xlabel("Share who are Active (%)")
            ax.set_xlim(0, max(dd["active_%"]) * 1.16)
            ax.grid(axis="y", alpha=0); ax.grid(axis="x", alpha=.22)
            VT.title_block(ax, f"Activity by {mz.lower()}",
                           "London adults; described, not forecast")
            safe = mz.split("(")[0].strip().replace(" ", "_").replace("/", "_")[:24]
            VT.finish(fig, out / f"figX_B4_{safe}.png",
                      "Self-rated health is collected from 2018\u201319 onward. "
                      "Association, not one-way cause.")
    print("   separate Stage B charts written (clean forecasts, heatmaps, health)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="stageB_out")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--min-n", type=int, default=40)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    eng = load_engine()
    print("STAGE B — Section 2 demographic forecasts")
    print("=" * 66)

    print("B1  forecasting each demographic group ...")
    fc, fc_year, slope, covid = forecast_groups(eng, args.parquet,
                                                args.horizon, args.shrink)
    print(fc.to_string(index=False))
    fc.to_csv(out / "B1_group_forecasts.csv", index=False)
    fig_group_forecast(fc, fc_year, out)

    df = normalise_disab(eng.load(args.parquet, "is_active"))

    print("\nB2  preferred activities by group ...")
    tables = preferred_activities(df)
    for gcol, T in tables.items():
        T.to_csv(out / f"B2_activities_{gcol}.csv")
    fig_preferred(tables, out)
    print("   written per-dimension activity tables")

    print("\nB3  activity by demographic within borough (pooled) ...")
    fig_within_borough(eng, df, out, args.min_n)
    print("   written pooled age-gradient heatmap")

    print("\nB4  health conditions (description) ...")
    hd = health_description(df)
    if not hd.empty:
        print(hd.round(1).to_string(index=False))
        hd.to_csv(out / "B4_health.csv", index=False)
        fig_health(hd, out)

    print("\nStage B separate themed charts ...")
    stageB_separate_charts(fc, tables, hd if not hd.empty else None, out)

    print(f"\nwritten to {out}/")
    for f in sorted(out.glob("*")):
        print(f"   {f.name}")
    print("\nStage B status:")
    print("  B1 group forecasts ............. produced")
    print("  B2 preferred activities ........ produced")
    print("  B3 within-borough demographic .. pooled (single-year too thin)")
    print("  B4 health conditions ........... described (health waves 4-8 only)")


if __name__ == "__main__":
    main()
