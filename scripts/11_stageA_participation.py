"""
11_stageA_participation.py — Section 1 of the brief, completed.

Reuses the forecast engine in 09_forecast.py (imported, not reimplemented) and
applies it across the participation targets the brief lists, then produces the
free-time description that cannot be forecast.

Covers:
  A2  club membership forecast, London-wide + boroughs
  A3  activity-specific forecasts: walking, cycling, fitness, team, racket,
      dance, active travel  (London-wide, 3-year + 10-year scenarios)
  A4  free-time factors — DESCRIPTION ONLY (wave 8 is the only wave that
      carries limfreti*, so it can be described but honestly not forecast)

Each target is labelled with the honest wave window and whether it is a
forecast or a description. Nothing is forced past what the data supports.

Run:
    python 11_stageA_participation.py dataset1_individual.parquet -o stageA_out
"""

import argparse
import importlib.util
import sys
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


def load_engine():
    """Import 09_forecast.py as a module so its functions can be reused."""
    here = Path(__file__).resolve().parent
    cand = here / "09_forecast.py"
    if not cand.exists():
        cand = Path("09_forecast.py")
    spec = importlib.util.spec_from_file_location("fc09", cand)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# target -> (nice label, first usable wave, last usable wave, kind)
# kind: "forecast" runs the engine; "describe" is cross-tab only.
TARGETS = [
    ("club_any_all",   "Club membership (any sport)", 3, 8, "forecast"),
    ("club_team_all",  "Team-sport club",             3, 8, "forecast"),
    ("club_fitness_all", "Fitness club / gym",        3, 8, "forecast"),
    ("part_walking",   "Walking",                     1, 8, "forecast"),
    ("part_cycling",   "Cycling",                     2, 8, "forecast"),
    ("part_fitness",   "Fitness activities",          2, 8, "forecast"),
    ("part_team",      "Team sports",                 1, 8, "forecast"),
    ("part_racket",    "Racket sports",               1, 8, "forecast"),
    ("part_dance",     "Dance",                       1, 8, "forecast"),
    ("part_active_travel", "Active travel",           1, 8, "forecast"),
]


def london_rate(df, col, w0, w1, eng):
    """Weighted rate per wave, restricted to the usable window."""
    d = df[(df.wave >= w0) & (df.wave <= w1)]
    s = pd.to_numeric(d[col], errors="coerce")
    ok = s.notna() & d.wt_final.notna()
    out = {}
    for wv, g in d[ok].groupby("wave"):
        gs = pd.to_numeric(g[col], errors="coerce")
        out[wv] = 100 * (gs * g.wt_final).sum() / g.wt_final.sum()
    return pd.Series(out)


def forecast_one(eng, path, col, w0, w1, horizon, shrink, boot):
    """Run the engine's pieces for a single target and return a small summary.
    Mirrors 09's main() but trimmed to the numbers Stage A needs."""
    df = eng.load(path, col)
    df = df[(df.wave >= w0) & (df.wave <= w1)].copy()
    if df.wave.nunique() < 4 or len(df) < 5000:
        return None

    last_w = int(df.wave.max())
    hz = [last_w + i for i in range(1, horizon + 1)]

    m, enc, feats = eng.fit_propensity(df)
    prop = eng.propensity_by_cell(df, m, enc, feats)
    trend_df, slope, intercept, covid = eng.behaviour_trend(df, m, enc, feats)
    comp = eng.observed_composition(df)

    def resid(w):
        return slope * w + intercept

    anchor = resid(last_w)
    base_shares = comp[comp.wave == last_w][["borough"] + eng.CELL + ["share"]]
    base = eng.score(base_shares.assign(wave=last_w), prop, trend_pp=anchor)

    wpop = df[df.wave == last_w].groupby("borough").wt_final.sum()
    lon_base = np.average(base.set_index("borough").rate.reindex(wpop.index),
                          weights=wpop)

    # 3-year forecast
    proj = eng.project_composition(comp, hz, shrink=shrink)
    s_last = proj[proj.wave == hz[-1]]
    fc = eng.score(s_last, prop, trend_pp=resid(hz[-1]))
    lon_fc = np.average(fc.set_index("borough").rate.reindex(wpop.index),
                        weights=wpop)
    comp_only = eng.score(s_last, prop, trend_pp=anchor)
    lon_comp = np.average(comp_only.set_index("borough").rate.reindex(wpop.index),
                          weights=wpop) - lon_base

    # 10-year scenarios
    h10 = last_w + 10
    s10 = eng.project_composition(comp, [h10], shrink=shrink)
    sc_comp = np.average(eng.score(s10, prop, trend_pp=anchor)
                         .set_index("borough").rate.reindex(wpop.index),
                         weights=wpop)
    sc_trend = np.average(eng.score(s10, prop, trend_pp=resid(h10))
                          .set_index("borough").rate.reindex(wpop.index),
                          weights=wpop)
    sc_rev = np.average(eng.score(s10, prop, trend_pp=anchor + covid)
                        .set_index("borough").rate.reindex(wpop.index),
                        weights=wpop)

    # light bootstrap for a London interval on the 3-year point
    boots = []
    rng = np.random.default_rng(11)
    for _ in range(boot):
        idx = rng.integers(0, len(df), len(df))
        db = df.iloc[idx]
        try:
            mb, eb, fb = eng.fit_propensity(db)
            pb = eng.propensity_by_cell(db, mb, eb, fb)
            cb = eng.observed_composition(db)
            jb = eng.project_composition(cb, [hz[-1]], shrink=shrink)
            _, sb, ib, _ = eng.behaviour_trend(db, mb, eb, fb)
            rb = eng.score(jb, pb, trend_pp=sb * hz[-1] + ib)
            wp = db[db.wave == last_w].groupby("borough").wt_final.sum()
            boots.append(np.average(
                rb.set_index("borough").rate.reindex(wp.index).fillna(lon_fc),
                weights=wp))
        except Exception:
            continue
    lo, hi = (np.percentile(boots, [5, 95]) if len(boots) > 5
              else (np.nan, np.nan))

    # per-borough final for the boroughs figure
    bor = fc.set_index("borough").rate.reindex(wpop.index)
    bor_base = base.set_index("borough").rate.reindex(wpop.index)

    return {
        "col": col, "waves": f"{w0}-{w1}", "last_year": eng.WAVE_YEAR[last_w],
        "baseline": lon_base, "fc3": lon_fc, "lo": lo, "hi": hi,
        "comp3": lon_comp, "beh3": lon_fc - lon_base - lon_comp,
        "sc_comp": sc_comp, "sc_trend": sc_trend, "sc_rev": sc_rev,
        "series": london_rate(df, col, w0, w1, eng),
        "fc_year": eng.WAVE_YEAR[last_w] + horizon,
        "borough_base": bor_base, "borough_fc": bor,
        "slope": slope,
    }


def describe_freetime(df, out):
    """A4 — free time. Wave 8 only, so DESCRIPTION not forecast."""
    w8 = df[df.wave == df.wave.max()]
    items = [("limfreti_care", "Caring / family responsibilities"),
             ("limfreti_time", "Long hours / second job")]
    items = [(c, l) for c, l in items if c in df.columns]
    if not items:
        return None

    rows = []
    for c, lab in items:
        s = pd.to_numeric(w8[c], errors="coerce")
        ok = s.notna() & w8.wt_final.notna()
        share = 100 * (s[ok] * w8.wt_final[ok]).sum() / w8.wt_final[ok].sum()
        # activity rate among those who report the constraint vs those who don't
        act = pd.to_numeric(w8["is_active"], errors="coerce")
        both = ok & act.notna()
        with_c = both & (s == 1)
        without = both & (s == 0)
        a_with = (100 * (act[with_c] * w8.wt_final[with_c]).sum()
                  / w8.wt_final[with_c].sum()) if with_c.sum() else np.nan
        a_without = (100 * (act[without] * w8.wt_final[without]).sum()
                     / w8.wt_final[without].sum()) if without.sum() else np.nan
        rows.append({"factor": lab, "reports_pct": share,
                     "active_if_reports": a_with,
                     "active_if_not": a_without,
                     "gap": a_without - a_with})
    r = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 0.8 * len(r) + 2.0))
    y = np.arange(len(r))
    ax.barh(y - .18, r.active_if_not, .34, color=TEAL, label="does NOT report it")
    ax.barh(y + .18, r.active_if_reports, .34, color=AMBER, label="reports it")
    for i, row in r.iterrows():
        ax.text(row.active_if_not + .4, i - .18, f"{row.active_if_not:.0f}%",
                va="center", fontsize=8)
        ax.text(row.active_if_reports + .4, i + .18,
                f"{row.active_if_reports:.0f}%", va="center", fontsize=8)
        xr = ax.get_xlim()[1]
        ax.text(xr * 0.99, i, f"{row.reports_pct:.0f}% report this",
                fontsize=7.5, color=GREY, ha="right", va="center")
    ax.set_yticks(y); ax.set_yticklabels(r.factor, fontsize=10)
    ax.set_xlabel("% meeting the activity guideline")
    ax.set_xlim(0, max(r.active_if_not.max(), r.active_if_reports.max()) * 1.18)
    ax.legend(fontsize=8.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.12), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.25, axis="x"); ax.set_axisbelow(True)
    ax.set_title("A4. Free-time constraints and activity  (2022–23 only — "
                 "cannot be forecast)", fontsize=12, fontweight="bold",
                 loc="left", color=INK)
    fig.text(.02, .01, "limfreti* is collected in the final wave only, so this "
             "is a snapshot, not a trend. The gap is descriptive.",
             fontsize=8, color=GREY)
    fig.tight_layout(rect=[0, .03, 1, 1])
    fig.savefig(out / "figA4_freetime.png", dpi=150); plt.close(fig)
    r.to_csv(out / "A4_freetime.csv", index=False)
    return r


# ---------------------------------------------------------------- figures

def stageA_separate_forecasts(results, out):
    """One themed forecast chart PER activity, replacing the grid panel.
    Shows observed years (solid, filled), the 3-year projection (dashed, open)
    with a 90% band, and the last-observed divider."""
    try:
        import viz_theme as VT
    except Exception:
        return
    import numpy as np
    res = [r for r in results if r]
    for r in res:
        s = r["series"]
        yrs = [2014 + w for w in s.index]
        ly = 2014 + s.index.max()
        fig, ax = VT.new_ax(9.5, 5.6)
        # observed
        ax.plot(yrs, s.values, "o-", color=VT.COL["navy"], lw=2.3, ms=6,
                label="Observed")
        # 90% band
        if not np.isnan(r["lo"]):
            ax.fill_between([ly, r["fc_year"]], [s.values[-1], r["lo"]],
                            [s.values[-1], r["hi"]], color=VT.COL["amber"],
                            alpha=.18, label="90% range")
        # forecast line + open marker
        ax.plot([ly, r["fc_year"]], [s.values[-1], r["fc3"]], "--",
                color=VT.COL["amber"], lw=2.2)
        ax.plot([r["fc_year"]], [r["fc3"]], "o", ms=8, color="white",
                markeredgecolor=VT.COL["amber"], markeredgewidth=1.8,
                label="Forecast")
        ax.axvline(ly, color=VT.COL["grey"], ls=":", lw=1.3)
        ax.text(ly, ax.get_ylim()[0], " last observed year", rotation=90,
                va="bottom", ha="right", fontsize=9.5, color=VT.COL["grey"])
        ax.text(r["fc_year"] + 0.1, r["fc3"], f"  {r['fc3']:.1f}%", va="center",
                fontsize=12, color=VT.COL["amber"], fontweight="bold")
        # value label on today's (last observed) point
        ax.annotate(f"{s.values[-1]:.1f}%", (ly, s.values[-1]),
                    xytext=(0, 11), textcoords="offset points", ha="center",
                    va="bottom", fontsize=11.5, color=VT.COL["navy"],
                    fontweight="bold")
        ax.set_xlabel("Survey year")
        ax.set_ylabel("Share of adults (%)")
        ax.set_xlim(min(yrs) - 0.3, r["fc_year"] + 1.4)
        ax.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD",
                  framealpha=.95, loc="best")
        VT.title_block(ax, f"{r['label']}: observed and forecast",
                       "London participation, three-year projection with 90% range")
        safe = r["label"].lower().replace(" ", "_").replace("/", "_")
        VT.finish(fig, out / f"figX_A_{safe}.png",
                  "Solid line and filled dots are observed survey years; "
                  "dashed line and open dot are the model forecast.")
    print("   separate per-activity forecast charts written")


def stageA_separate_scenarios(results, out):
    """One clean ten-year scenario chart per activity. Uses a labelled vertical
    layout so scenario names never overlap, and plain-English scenario names."""
    try:
        import viz_theme as VT
    except Exception:
        return
    import numpy as np
    res = [r for r in results if r]
    for r in res:
        # "Today" must match the forecast plot, which anchors to the last
        # OBSERVED survey value. The scenario values were built off the model
        # baseline, so shift them by the same offset to keep the chart
        # consistent and on the observed scale.
        today = r["series"].values[-1]
        offset = today - r["baseline"]
        sc_rev = r["sc_rev"] + offset
        sc_comp = r["sc_comp"] + offset
        sc_trend = r["sc_trend"] + offset
        # plain-English names, ordered worst -> best for a readable ladder
        rows = [
            ("If recent gains reverse\n(a COVID-scale setback)", sc_rev,
             VT.COL["red"]),
            ("If behaviour stops improving\n(population change only)",
             sc_comp, VT.COL["amber"]),
            ("If recent gains continue", sc_trend, VT.COL["teal"]),
        ]
        fig, ax = VT.new_ax(9.5, 4.6)
        y = np.arange(len(rows))
        vals = [v for _, v, _ in rows]
        # baseline reference line = observed "today", matching the forecast plot
        ax.axvline(today, color=VT.COL["grey"], ls="--", lw=1.3, zorder=1)
        ax.text(today, len(rows) - 0.35,
                f"Today: {today:.1f}%", fontsize=10.5,
                color=VT.COL["grey"], ha="center", va="bottom")
        for i, (name, v, col) in enumerate(rows):
            ax.plot([today, v], [i, i], color=col, lw=2.5, alpha=.5,
                    zorder=2)
            ax.scatter(v, i, s=150, color=col, zorder=3)
            # label just ABOVE the dot so the connector line never crosses it
            ax.annotate(f"{v:.1f}%", (v, i), xytext=(0, 12),
                        textcoords="offset points", va="bottom", ha="center",
                        fontsize=12, color=col, fontweight="bold")
        ax.set_yticks(y); ax.set_yticklabels([n for n, _, _ in rows],
                                             fontsize=11.5)
        span = max(vals + [today]) - min(vals + [today])
        span = max(span, 1.0)
        ax.set_xlim(min(vals + [today]) - span * 0.35,
                    max(vals + [today]) + span * 0.35)
        ax.set_xlabel("Projected share of adults (%) in ten years")
        ax.set_ylim(-0.6, len(rows) - 0.1)
        ax.grid(axis="y", alpha=0); ax.grid(axis="x", alpha=.22)
        VT.title_block(ax, f"{r['label']}: ten-year scenarios",
                       "Three what-if paths; the spread is the honest "
                       "long-run uncertainty")
        safe = r["label"].lower().replace(" ", "_").replace("/", "_")
        VT.finish(fig, out / f"figX_Ascen_{safe}.png",
                  "These are bounded what-ifs, not a single prediction. "
                  "\u201cPopulation change only\u201d holds behaviour fixed and "
                  "lets only London's changing age and demographic mix act.")
    print("   separate per-activity scenario charts written")


def fig_london_panel(results, out):
    """A2+A3: every forecast target, observed series + 3-year point."""
    res = [r for r in results if r]
    n = len(res); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, res):
        s = r["series"]
        yrs = [2014 + w for w in s.index]
        ax.plot(yrs, s.values, "o-", color=INK, lw=1.8, ms=4, label="observed")
        ly = 2014 + s.index.max()
        ax.plot([ly, r["fc_year"]], [s.values[-1], r["fc3"]], "o--",
                color=AMBER, lw=1.8, ms=6, markerfacecolor="white",
                markeredgecolor=AMBER, markeredgewidth=1.4, label="forecast")
        ax.axvline(ly, color="#AAAAAA", ls=":", lw=1)
        if not np.isnan(r["lo"]):
            ax.fill_between([ly, r["fc_year"]], [s.values[-1], r["lo"]],
                            [s.values[-1], r["hi"]], color=AMBER, alpha=.15)
        ax.set_title(r["label"], fontsize=10, fontweight="bold", loc="left",
                     color=INK)
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.25); ax.set_axisbelow(True)
        ax.set_ylabel("% of adults", fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("A2–A3. London participation forecasts by activity  "
                 "(3-year point with 90% band)", fontsize=13, fontweight="bold",
                 x=.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(out / "figA_london_forecasts.png", dpi=150,
                bbox_inches="tight"); plt.close(fig)


def fig_scenarios_table(results, out):
    """10-year scenario spread per target — the honest long view."""
    res = [r for r in results if r]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(res) + 1.6))
    y = np.arange(len(res))[::-1]
    for i, r in zip(y, res):
        off = r["series"].values[-1] - r["baseline"]   # observed-scale shift
        sc_rev, sc_comp, sc_trend = (r["sc_rev"] + off, r["sc_comp"] + off,
                                     r["sc_trend"] + off)
        ax.plot([sc_rev, sc_trend], [i, i], color="#D6DEE6", lw=6,
                solid_capstyle="round", zorder=1)
        ax.scatter(sc_comp, i, color=AMBER, s=55, zorder=3,
                   label="composition only" if i == y[0] else None)
        ax.scatter(sc_trend, i, color=TEAL, s=55, zorder=3,
                   label="trend continues" if i == y[0] else None)
        ax.scatter(sc_rev, i, color=RED, s=55, zorder=3,
                   label="trend reverts" if i == y[0] else None)
        ax.scatter(r["series"].values[-1], i, color=INK, marker="|", s=200,
                   zorder=4, label="today" if i == y[0] else None)
    ax.set_yticks(y); ax.set_yticklabels([r["label"] for r in res], fontsize=9)
    ax.set_xlabel("% of adults")
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(.5, 1.06))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.25, axis="x"); ax.set_axisbelow(True)
    fig.suptitle("A2–A3. Ten-year scenario range by activity  (spread = "
                 "honest long-run uncertainty)", fontsize=12.5,
                 fontweight="bold", x=.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(out / "figA_scenarios.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- main

# ---- free-time REASONS, by age and by gender (wave 8 only) ----------------
FREETIME_LABELS = {
    "freetime1": "Work commitments",
    "freetime2": "Study commitments",
    "freetime3": "Looking after children",
    "freetime4": "Caring for an adult",
    "freetime5": "Health or disability",
    "freetime6": "No one to do it with",
}


def freetime_reasons(df, out):
    """Main reasons free time is limited, broken down by age and by gender.
    Wave 8 only (the question is asked once), so DESCRIPTION not forecast.
    Produces four separate figures:
       reasons by age  |  reasons by gender
       activity impact by age  |  activity impact by gender
    """
    try:
        import viz_theme as VT
    except Exception:
        return
    import numpy as np, pandas as pd

    w8 = df[df.wave == df.wave.max()].copy()
    reasons = [(c, l) for c, l in FREETIME_LABELS.items() if c in w8.columns]
    # keep only reasons that actually have responses
    reasons = [(c, l) for c, l in reasons
               if pd.to_numeric(w8[c], errors="coerce").notna().sum() > 200
               and pd.to_numeric(w8[c], errors="coerce").fillna(0).sum() > 0]
    if not reasons:
        print("   (no usable free-time reason columns; skipped)")
        return

    def wpct(g, col):
        s = pd.to_numeric(g[col], errors="coerce")
        ok = s.notna() & g.wt_final.notna()
        return 100 * (s[ok] * g.wt_final[ok]).sum() / g.wt_final[ok].sum() \
            if ok.sum() else np.nan

    # ============ (A) reasons by AGE — grouped horizontal bars ============
    ages = [a for a in ["16-24", "25-44", "45-64", "65+"]
            if a in w8["age4"].unique()]
    fig, ax = VT.new_ax(11, 6.6)
    ax.grid(axis="y", alpha=0); ax.grid(axis="x", alpha=.22)
    n_r = len(reasons); n_a = len(ages)
    yb = np.arange(n_r); h = 0.8 / n_a
    age_cols = [VT.COL["navy"], VT.COL["teal"], VT.COL["amber"], VT.COL["red"]]
    for j, a in enumerate(ages):
        g = w8[w8["age4"] == a]
        vals = [wpct(g, c) for c, _ in reasons]
        ax.barh(yb + (j - (n_a - 1) / 2) * h, vals, h * .92,
                color=age_cols[j % 4], label=a)
    ax.set_yticks(yb); ax.set_yticklabels([l for _, l in reasons])
    ax.invert_yaxis()
    ax.set_xlabel("Share of adults reporting this reason (%)")
    ax.legend(frameon=False, title="Age group", loc="lower right")
    VT.title_block(ax, "Why free time for activity is limited, by age",
                   "London adults, 2022\u201323")
    VT.finish(fig, out / "figX4_freetime_reasons_by_age.png",
              "Asked once (2022\u201323), of about a third of respondents. "
              "Reason labels follow the standard Active Lives coding.")

    # ============ (B) reasons by GENDER ============
    genders = [g for g in ["Male", "Female"] if g in w8["gender"].unique()]
    fig, ax = VT.new_ax(10.5, 6.4)
    ax.grid(axis="y", alpha=0); ax.grid(axis="x", alpha=.22)
    n_g = len(genders); h = 0.8 / n_g
    gcols = [VT.COL["navy"], VT.COL["amber"]]
    for j, gd in enumerate(genders):
        g = w8[w8["gender"] == gd]
        vals = [wpct(g, c) for c, _ in reasons]
        ax.barh(yb + (j - (n_g - 1) / 2) * h, vals, h * .92,
                color=gcols[j % 2], label=gd)
    ax.set_yticks(yb); ax.set_yticklabels([l for _, l in reasons])
    ax.invert_yaxis()
    ax.set_xlabel("Share of adults reporting this reason (%)")
    ax.legend(frameon=False, title="Gender", loc="lower right")
    VT.title_block(ax, "Why free time for activity is limited, by gender",
                   "London adults, 2022\u201323")
    VT.finish(fig, out / "figX5_freetime_reasons_by_gender.png",
              "Asked once (2022\u201323), of about a third of respondents. "
              "Reason labels follow the standard Active Lives coding.")

    # ============ (C) activity impact by AGE ============
    # For each age, activity rate among those reporting ANY constraint vs none
    def impact_fig(dim, levels, cols, fname, title):
        fig, ax = VT.new_ax(10, 6)
        any_c = w8[[c for c, _ in reasons]].apply(
            lambda r: pd.to_numeric(r, errors="coerce").fillna(0).max(), axis=1)
        w8["_anyc"] = any_c
        act = (w8["activity_band"] == "Active").astype(float) \
            if "activity_band" in w8 else pd.to_numeric(w8["is_active"],
                                                        errors="coerce")
        x = np.arange(len(levels)); wbar = 0.38
        rep, norep = [], []
        for lv in levels:
            g = w8[w8[dim] == lv]
            m_rep = g["_anyc"] == 1; m_no = g["_anyc"] == 0
            aw = act.reindex(g.index)
            rep.append(100 * np.average(aw[m_rep], weights=g.wt_final[m_rep])
                       if m_rep.sum() else np.nan)
            norep.append(100 * np.average(aw[m_no], weights=g.wt_final[m_no])
                         if m_no.sum() else np.nan)
        ax.bar(x - wbar / 2, norep, wbar, color=VT.COL["teal"],
               label="No free-time constraint")
        ax.bar(x + wbar / 2, rep, wbar, color=VT.COL["amber"],
               label="Reports a constraint")
        for i in range(len(levels)):
            if not np.isnan(norep[i]):
                ax.text(i - wbar / 2, norep[i] + 0.6, f"{norep[i]:.0f}%",
                        ha="center", fontsize=11)
            if not np.isnan(rep[i]):
                ax.text(i + wbar / 2, rep[i] + 0.6, f"{rep[i]:.0f}%",
                        ha="center", fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(levels)
        ax.set_ylabel("Share who are Active (%)")
        ax.set_xlabel(VT.LABELS.get(dim, dim))
        # leave headroom above the tallest bar so the legend sits inside the
        # plot, below the title, without covering any bar or value label
        ax.set_ylim(0, max([v for v in (rep + norep) if not np.isnan(v)]) * 1.32)
        ax.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD",
                  framealpha=0.95, loc="upper center", ncol=2,
                  borderpad=0.6)
        VT.title_block(ax, title, "London adults, 2022\u201323")
        VT.finish(fig, out / fname,
                  "\u201cReports a constraint\u201d = reports at least one "
                  "reason their free time is limited. Snapshot, not a forecast.")

    impact_fig("age4", ages, age_cols,
               "figX6_freetime_impact_by_age.png",
               "Activity when free time is limited, by age")
    impact_fig("gender", genders, gcols,
               "figX7_freetime_impact_by_gender.png",
               "Activity when free time is limited, by gender")
    print("   free-time reason figures written (by age and by gender)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="stageA_out")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--boot", type=int, default=30)
    ap.add_argument("--shrink", type=float, default=0.5)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    eng = load_engine()
    df_probe = pd.read_parquet(args.parquet)
    have = set(df_probe.columns)

    print("STAGE A — Section 1 participation forecasts")
    print("=" * 66)
    results, summary = [], []
    for col, label, w0, w1, kind in TARGETS:
        if col not in have:
            print(f"  {label:28s} — column absent, skipped")
            continue
        print(f"  {label:28s} waves {w0}-{w1} ... ", end="", flush=True)
        r = forecast_one(eng, args.parquet, col, w0, w1,
                         args.horizon, args.shrink, args.boot)
        if r is None:
            print("insufficient data")
            continue
        r["label"] = label
        results.append(r)
        # Put the 10-year scenarios on the SAME observed scale the plots use.
        # The scenario values are built off the model baseline; the plots shift
        # them by (observed today - baseline) so they read on the observed
        # scale. The CSV must apply the identical shift, or the numbers here
        # won't match the figX_Ascen_*.png charts.
        offset = r["series"].values[-1] - r["baseline"]
        summary.append({
            "activity": label, "waves": r["waves"],
            f"today_%": round(r["series"].values[-1], 1),
            f"{r['fc_year']}_%": round(r["fc3"], 1),
            "90%_low": round(r["lo"], 1), "90%_high": round(r["hi"], 1),
            "10yr_composition": round(r["sc_comp"] + offset, 1),
            "10yr_trend": round(r["sc_trend"] + offset, 1),
            "10yr_reverts": round(r["sc_rev"] + offset, 1),
        })
        print(f"{r['series'].values[-1]:.1f}% -> {r['fc3']:.1f}% "
              f"[{r['lo']:.1f}, {r['hi']:.1f}]")

    S = pd.DataFrame(summary)
    print("\n" + "=" * 66)
    print("SECTION 1 FORECAST SUMMARY")
    print("=" * 66)
    print(S.to_string(index=False))
    S.to_csv(out / "A_participation_summary.csv", index=False)

    if results:
        fig_london_panel(results, out)
        fig_scenarios_table(results, out)
        print("Stage A separate themed forecast charts ...")
        stageA_separate_forecasts(results, out)
        stageA_separate_scenarios(results, out)

    print("\nA4 — free-time factors (description only) ...")
    ft = describe_freetime(df_probe, out)
    print("A4b free-time reasons by age and gender ...")
    freetime_reasons(df_probe, out)
    if ft is not None:
        print(ft.round(1).to_string(index=False))

    print(f"\nwritten to {out}/")
    for f in sorted(out.glob("*")):
        print(f"   {f.name}")
    print("\nStage A status:")
    print("  A1 overall activity ............ DONE (script 09)")
    print("  A2 club membership ............. forecast produced")
    print("  A3 activity-specific ........... forecast produced")
    print("  A4 free-time factors ........... described (cannot forecast: 1 wave)")


if __name__ == "__main__":
    main()
