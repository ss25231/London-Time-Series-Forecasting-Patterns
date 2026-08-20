"""
09_forecast.py — the deliverable.

Architecture (settled by the diagnostics in scripts 05-08):

  forecast = propensity(demographics, borough)  x  projected composition
                                                +  behaviour trend

  * PROPENSITY is time-free and fitted on all 135k individuals. It carries
    the demographic gradient and a borough intercept.
  * COMPOSITION is projected forward from the 8 observed waves using the
    survey's own calibrated weights, which are raked to ONS borough
    population estimates. Borough-specific trends are shrunk toward the
    London trend so no borough extrapolates off a cliff.
  * BEHAVIOUR TREND is the London-level residual over time. The observed
    series is flat (63.5% -> 64.3% across 8 years), so this term is small
    by construction — which is a finding, not a shortcut.

Every projected change decomposes into:
    composition effect  = the population changed
    behaviour effect    = people of a given type changed

Those have different policy responses, and separating them is the point.

Horizon: 3 years as a forecast with intervals. The brief's 10-year ask is
answered as labelled SCENARIOS, because an 8-point flat series cannot
support a decade-long point forecast and pretending otherwise would be
false precision.

Run:
    python 09_forecast.py dataset1_individual.parquet -o forecast_out
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
plt.rcParams.update({"figure.facecolor": "white", "font.size": 11,
                     "savefig.bbox": "tight"})

INNER = {"E09000001", "E09000007", "E09000011", "E09000012", "E09000013",
         "E09000019", "E09000020", "E09000022", "E09000023", "E09000028",
         "E09000030", "E09000032", "E09000033"}

# health excluded: absent in waves 1-3, so it acts as a wave indicator.
DEMOG = ["age4", "gender", "eth5", "nssec4", "educ3", "Disab3", "BMIG"]
CELL = ["age4", "gender", "eth5"]          # composition is projected on these
WAVE_YEAR = {w: 2014 + w for w in range(1, 26)}  # wave 1 = 2015; auto-extends
COVID_WAVES = {5, 6}


def _style(ax, title="", xlabel="", ylabel=""):
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.25, linewidth=.6); ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# 1. DATA
# --------------------------------------------------------------------------

def load(path, target):
    df = pd.read_parquet(path)
    df["inner_outer"] = np.where(df.borough.astype(str).isin(INNER),
                                 "Inner", "Outer")
    for c in DEMOG + ["borough", "inner_outer"]:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("Not disclosed").astype(str)
    y = pd.to_numeric(df[target], errors="coerce")
    ok = y.notna() & df.wt_final.notna() & (df.wt_final > 0) & df.borough.notna()
    df = df[ok].copy(); df["_y"] = y[ok].astype(int)
    df["year"] = df.wave.map(WAVE_YEAR)
    return df


# --------------------------------------------------------------------------
# 2. PROPENSITY (time-free)
# --------------------------------------------------------------------------

def fit_propensity(df, C=1.0):
    feats = [c for c in DEMOG + ["borough"] if c in df.columns]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                        min_frequency=30)
    X = enc.fit_transform(df[feats])
    m = LogisticRegression(C=C, max_iter=3000)
    m.fit(X, df._y.values, sample_weight=df.wt_final.values)
    return m, enc, feats


def propensity_by_cell(df, m, enc, feats):
    """Predicted P(target) for every borough x demographic cell.

    Uses the wave-8 population as the reference for within-cell composition
    of the non-cell features (NS-SEC, education, disability, BMI)."""
    ref = df[df.wave == df.wave.max()].copy()
    ref["p"] = m.predict_proba(enc.transform(ref[feats]))[:, 1]
    g = ref.groupby(["borough"] + CELL, observed=True).apply(
        lambda x: pd.Series({
            "p_hat": np.average(x.p, weights=x.wt_final),
            "n": len(x)}), include_groups=False).reset_index()
    return g


# --------------------------------------------------------------------------
# 3. COMPOSITION: observed, then projected
# --------------------------------------------------------------------------

def observed_composition(df):
    """Share of each borough's weighted population in each demographic cell,
    per wave. wt_final is calibrated to ONS borough population estimates, so
    this IS London's observed demography, not a survey artefact."""
    g = (df.groupby(["borough", "wave"] + CELL, observed=True)["wt_final"]
           .sum().reset_index(name="w"))
    tot = g.groupby(["borough", "wave"])["w"].transform("sum")
    g["share"] = g["w"] / tot
    return g


def project_composition(comp, horizon_waves, shrink=0.5, min_waves=6):
    """Linear trend per borough-cell on the logit of its share, shrunk toward
    the London-wide trend for that cell.

    Shrinkage matters: a borough-cell trend fitted on 8 noisy points will
    happily extrapolate to absurd shares. Pulling each borough halfway to the
    London trend keeps projections sane without flattening real divergence.
    """
    def _logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    comp = comp.copy(); comp["lg"] = _logit(comp["share"])

    # London-wide trend per cell
    lon = (comp.groupby(CELL + ["wave"])["w"].sum().reset_index())
    tot = lon.groupby("wave")["w"].transform("sum")
    lon["share"] = lon["w"] / tot; lon["lg"] = _logit(lon["share"])
    lon_slope = {}
    for key, g in lon.groupby(CELL):
        if len(g) >= min_waves:
            lon_slope[key] = np.polyfit(g.wave, g.lg, 1)[0]

    rows = []
    for (b, *cellkey), g in comp.groupby(["borough"] + CELL):
        key = tuple(cellkey)
        g = g.sort_values("wave")
        if len(g) >= min_waves:
            slope, intercept = np.polyfit(g.wave, g.lg, 1)
        else:
            slope, intercept = 0.0, g.lg.mean()
        # shrink toward the London trend for this cell
        ls = lon_slope.get(key, 0.0)
        slope = shrink * slope + (1 - shrink) * ls
        last_w = g.wave.max(); last_lg = g.lg.iloc[-1]
        for h in horizon_waves:
            lg = last_lg + slope * (h - last_w)
            rows.append({"borough": b, **dict(zip(CELL, cellkey)),
                         "wave": h, "share_raw": 1 / (1 + np.exp(-lg))})
    out = pd.DataFrame(rows)
    out["share"] = out["share_raw"] / out.groupby(
        ["borough", "wave"])["share_raw"].transform("sum")
    return out.drop(columns="share_raw")


# --------------------------------------------------------------------------
# 4. BEHAVIOUR TREND
# --------------------------------------------------------------------------

def behaviour_trend(df, m, enc, feats):
    """London-level residual over time: observed rate minus what the
    time-free propensity model predicts for that wave's population.

    A flat residual means people of a given type are behaving the same way
    over time, and all movement in the headline number is compositional."""
    rows = []
    for w, g in df.groupby("wave"):
        p = m.predict_proba(enc.transform(g[feats]))[:, 1]
        obs = np.average(g._y, weights=g.wt_final)
        pred = np.average(p, weights=g.wt_final)
        rows.append({"wave": w, "observed": 100 * obs, "predicted": 100 * pred,
                     "residual": 100 * (obs - pred),
                     "covid": int(w in COVID_WAVES)})
    t = pd.DataFrame(rows)
    non_covid = t[t.covid == 0]
    slope, intercept = np.polyfit(non_covid.wave, non_covid.residual, 1)
    covid_effect = (t[t.covid == 1].residual.mean()
                    - (slope * t[t.covid == 1].wave.mean() + intercept))
    return t, slope, intercept, covid_effect


# --------------------------------------------------------------------------
# 5. SCORING
# --------------------------------------------------------------------------

def score(shares, prop, trend_pp=0.0):
    """Borough rate = sum over cells of (share x cell propensity), + trend."""
    j = shares.merge(prop, on=["borough"] + CELL, how="left")
    # cells with no observed respondents fall back to the borough mean
    bmean = prop.groupby("borough").apply(
        lambda x: np.average(x.p_hat, weights=x.n), include_groups=False)
    j["p_hat"] = j.p_hat.fillna(j.borough.map(bmean))
    j["contrib"] = j.share * j.p_hat
    out = j.groupby(["borough", "wave"])["contrib"].sum().reset_index(
        name="rate")
    out["rate"] = 100 * out["rate"] + trend_pp
    return out


# --------------------------------------------------------------------------
# 6. MAIN
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="forecast_out")
    ap.add_argument("--target", default="is_active")
    ap.add_argument("--horizon", type=int, default=3,
                    help="years ahead to forecast with intervals")
    ap.add_argument("--boot", type=int, default=60,
                    help="bootstrap replicates for intervals")
    ap.add_argument("--shrink", type=float, default=0.5,
                    help="0 = all boroughs follow the London trend, "
                         "1 = each borough follows its own")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    df = load(args.parquet, args.target)
    last_w = int(df.wave.max()); last_y = WAVE_YEAR[last_w]
    horizon = [last_w + i for i in range(1, args.horizon + 1)]
    print(f"target: {args.target} | rows: {len(df):,} | "
          f"waves {df.wave.min()}-{last_w} ({WAVE_YEAR[df.wave.min()]}"
          f"-{last_y})")
    print(f"forecasting waves {horizon} = "
          f"{[last_y + i for i in range(1, args.horizon + 1)]}\n")

    # ---- propensity ----
    m, enc, feats = fit_propensity(df)
    prop = propensity_by_cell(df, m, enc, feats)
    print(f"propensity fitted on {len(feats)} feature blocks; "
          f"{len(prop):,} borough-cells scored")

    # ---- trend ----
    t, slope, intercept, covid = behaviour_trend(df, m, enc, feats)
    print("\nBEHAVIOUR TREND (London residual: observed - propensity)")
    print(t.round(2).to_string(index=False))
    print(f"\n  non-COVID slope: {slope:+.3f} pp per wave")
    print(f"  COVID effect (waves 5-6): {covid:+.2f} pp")
    if abs(slope) < 0.15:
        print("  => Essentially FLAT. Behaviour of a given demographic type is")
        print("     stable over 8 years; movement in the headline number is")
        print("     compositional. The forecast is a composition forecast.")

    # ---- composition ----
    comp = observed_composition(df)
    proj = project_composition(comp, horizon, shrink=args.shrink)
    print(f"\ncomposition projected for {proj.borough.nunique()} boroughs "
          f"x {len(horizon)} years")

    # ---- baseline and forecast ----
    # The residual has a LEVEL as well as a slope. The propensity model sits
    # ~1.3pp below the observed rate at wave 8, so the forecast must be
    # anchored to the FITTED RESIDUAL at each wave, not to the slope
    # increment alone — otherwise every projection inherits that offset and
    # the series appears to fall off a cliff at the forecast boundary.
    def resid_hat(w):
        return slope * w + intercept

    anchor = resid_hat(last_w)
    print(f"\n  fitted residual at wave {last_w}: {anchor:+.2f} pp "
          f"(anchors the forecast to the observed level)")

    base_shares = comp[comp.wave == last_w][["borough"] + CELL + ["share"]]
    base_shares = base_shares.assign(wave=last_w)
    base = score(base_shares, prop, trend_pp=anchor)

    fc_rows = []
    for h in horizon:
        s = proj[proj.wave == h]
        # (a) composition only: population moves, behaviour frozen at wave-8
        comp_only = score(s, prop, trend_pp=anchor).rename(
            columns={"rate": "comp_only"})
        # (b) full: composition moves AND the behaviour trend continues
        full = score(s, prop, trend_pp=resid_hat(h)).rename(
            columns={"rate": "forecast"})
        r = comp_only.merge(full, on=["borough", "wave"])
        fc_rows.append(r)
    fc = pd.concat(fc_rows)

    fc = fc.merge(base[["borough", "rate"]].rename(columns={"rate": "baseline"}),
                  on="borough")
    fc["composition_effect"] = fc.comp_only - fc.baseline
    fc["behaviour_effect"] = fc.forecast - fc.comp_only
    fc["total_change"] = fc.forecast - fc.baseline
    fc["year"] = fc.wave.map(lambda w: last_y + (w - last_w))

    # ---- bootstrap intervals ----
    print(f"\nbootstrapping ({args.boot} replicates) ...", flush=True)
    boots = []
    rng = np.random.default_rng(42)
    for b in range(args.boot):
        idx = rng.integers(0, len(df), len(df))
        db = df.iloc[idx]
        try:
            mb, eb, fb = fit_propensity(db)
            pb = propensity_by_cell(db, mb, eb, fb)
            cb = observed_composition(db)
            jb = project_composition(cb, [horizon[-1]], shrink=args.shrink)
            tb, sb, ib, _ = behaviour_trend(db, mb, eb, fb)
            rb = score(jb, pb, trend_pp=sb * horizon[-1] + ib)
            boots.append(rb.set_index("borough")["rate"].rename(b))
        except Exception:
            continue
        if (b + 1) % 20 == 0:
            print(f"   {b + 1}/{args.boot}", flush=True)
    if boots:
        B = pd.concat(boots, axis=1)
        ci = pd.DataFrame({"lo": B.quantile(.05, axis=1),
                           "hi": B.quantile(.95, axis=1)})
        fc = fc.merge(ci, left_on="borough", right_index=True, how="left")
        fc.loc[fc.wave != horizon[-1], ["lo", "hi"]] = np.nan

    # ---- London aggregate ----
    wpop = df[df.wave == last_w].groupby("borough").wt_final.sum()
    lon_base = np.average(base.set_index("borough").rate.reindex(wpop.index),
                          weights=wpop)
    lon_fc = (fc[fc.wave == horizon[-1]].set_index("borough")
              .reindex(wpop.index))
    lon_final = np.average(lon_fc.forecast, weights=wpop)
    lon_comp = np.average(lon_fc.composition_effect, weights=wpop)
    lon_beh = np.average(lon_fc.behaviour_effect, weights=wpop)

    obs_last = np.average(df[df.wave == last_w]._y,
                          weights=df[df.wave == last_w].wt_final) * 100
    gap = lon_base - obs_last
    print(f"\n  ANCHOR CHECK: observed {obs_last:.2f}% vs modelled baseline "
          f"{lon_base:.2f}% (gap {gap:+.2f} pp)")
    if abs(gap) > 0.5:
        print("  !! The baseline does not reproduce the last observed wave.")
        print("     Any forecast built on it inherits that offset.")
    else:
        print("     OK — the forecast starts where the data ends.")

    print("\n" + "=" * 78)
    print(f"LONDON FORECAST — {args.target}")
    print("=" * 78)
    print(f"  {last_y}-{str(last_y+1)[2:]} baseline      {lon_base:5.1f}%")
    print(f"  {last_y+args.horizon}-{str(last_y+args.horizon+1)[2:]} forecast      "
          f"{lon_final:5.1f}%")
    print(f"     of which composition {lon_comp:+5.2f} pp")
    print(f"                behaviour {lon_beh:+5.2f} pp")

    # ---- 10-year scenarios ----
    h10 = last_w + 10
    s10 = project_composition(comp, [h10], shrink=args.shrink)
    sc = {}
    sc["Composition only"] = score(s10, prop, trend_pp=anchor)
    sc["Trend continues"] = score(s10, prop, trend_pp=resid_hat(h10))
    sc["Trend reverts (COVID-like)"] = score(s10, prop,
                                             trend_pp=anchor + covid)
    scen = []
    for name, r in sc.items():
        v = np.average(r.set_index("borough").rate.reindex(wpop.index),
                       weights=wpop)
        scen.append({"scenario": name, "london_%": round(v, 1),
                     "change_pp": round(v - lon_base, 2)})
    scen = pd.DataFrame(scen)
    print(f"\n  TEN-YEAR SCENARIOS ({last_y + 10}-{str(last_y+11)[2:]}) — "
          f"scenarios, NOT a point forecast:")
    print(scen.to_string(index=False))
    print("""
  An 8-point series with no trend cannot support a decade-long point
  forecast. These are bounded what-ifs, and the spread between them is
  the honest statement of uncertainty.""")

    # ---- outputs ----
    final = fc[fc.wave == horizon[-1]].sort_values("forecast")
    final.to_csv(outdir / "borough_forecast.csv", index=False)
    fc.to_csv(outdir / "borough_forecast_all_years.csv", index=False)
    scen.to_csv(outdir / "scenarios_10yr.csv", index=False)
    t.to_csv(outdir / "behaviour_trend.csv", index=False)

    # ================= NEW STANDALONE FIGURES (shared theme) =================
    try:
        import viz_theme as VT
    except Exception:
        VT = None

    if VT is not None:
        # ---- NEW 1: overall activity trend, all three bands, observed+proj ----
        # observed band shares per wave, from the individual file
        band_rows = []
        for w, g in df.groupby("wave"):
            tot = g.wt_final.sum()
            row = {"wave": w, "year": VT.WAVE_YEAR[w]}
            for b in ["Inactive", "Fairly Active", "Active"]:
                row[b] = 100 * g.wt_final[g.activity_band == b].sum() / tot
            band_rows.append(row)
        bt = pd.DataFrame(band_rows).sort_values("wave")
        bt.to_csv(outdir / "overall_band_trend.csv", index=False)

        # ---- project each band forward the same number of years as the
        #      main forecast, using each band's own non-COVID linear trend ----
        HZ = args.horizon
        last_w = int(bt.wave.max()); last_yr = VT.WAVE_YEAR[last_w]
        proj_years = [last_yr + i for i in range(1, HZ + 1)]
        non_covid = bt[~bt.wave.isin([5, 6])]
        band_proj = {}
        for b in ["Active", "Fairly Active", "Inactive"]:
            sl, ic = np.polyfit(non_covid.year, non_covid[b], 1)
            last_val = bt[b].iloc[-1]
            band_proj[b] = [last_val + sl * i for i in range(1, HZ + 1)]

        fig, ax = VT.new_ax(11, 6.4)
        for b in ["Active", "Fairly Active", "Inactive"]:
            col = VT.BAND_COLOUR[b]
            # observed: solid line, filled markers
            ax.plot(bt.year, bt[b], "o-", lw=2.4, ms=6, color=col, label=b)
            # forecast: dashed line, open markers, joined to the last observed pt
            fx = [last_yr] + proj_years
            fy = [bt[b].iloc[-1]] + band_proj[b]
            ax.plot(fx, fy, "--", lw=2.2, color=col)
            ax.plot(proj_years, band_proj[b], "o", ms=7, color="white",
                    markeredgecolor=col, markeredgewidth=1.8)
            # value label at the far right (forecast end)
            ax.text(proj_years[-1] + 0.15, band_proj[b][-1],
                    f"{band_proj[b][-1]:.1f}%", va="center", ha="left",
                    fontsize=12, color=col, fontweight="bold")
            # value label on TODAY's (last observed) point
            ax.annotate(f"{bt[b].iloc[-1]:.1f}%", (last_yr, bt[b].iloc[-1]),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10.5, color=col,
                        fontweight="bold")
        # COVID shading and last-observed divider
        ax.axvspan(2018.5, 2020.5, color="#000000", alpha=.05)
        ax.text(2019.5, ax.get_ylim()[1], "COVID", fontsize=10.5,
                color=VT.COL["grey"], ha="center", va="top")
        ax.axvline(last_yr, color=VT.COL["grey"], ls=":", lw=1.3)
        ax.text(last_yr, ax.get_ylim()[0], " last observed year", rotation=90,
                va="bottom", ha="right", fontsize=9.5, color=VT.COL["grey"])
        ax.set_xlabel("Survey year")
        ax.set_ylabel("Share of London adults (%)")
        all_w = list(bt.wave)
        ax.set_xticks(list(bt.year) + proj_years)
        ax.set_xticklabels([VT.YEAR_LABEL[w] for w in all_w] +
                           [f"{y}\u2013{str(y+1)[2:]}" for y in proj_years],
                           rotation=45, ha="right")
        ax.set_xlim(bt.year.min() - 0.3, proj_years[-1] + 1.4)
        ax.legend(frameon=False, loc="center left", title="Activity band")
        VT.title_block(ax, "London activity levels: observed and projected",
                       "Share of adults in each activity band, with a "
                       f"{HZ}-year projection")
        VT.finish(fig, outdir / "figX1_overall_band_trend.png",
                  "Solid lines and filled dots are observed survey years; "
                  "dashed lines and open dots are projected. Bands: Active = "
                  "150+ minutes a week, Fairly Active = 30\u2013149, Inactive = "
                  "under 30.")

        # ---- NEW 2 & 3: borough forecast rankings ----
        fr = final.copy()
        fr["name"] = fr["borough"].map(VT.BOROUGH_NAME).fillna(fr["borough"])
        fr = fr.sort_values("forecast", ascending=False)

        def borough_bar(sub, title, subtitle, colour, fname, note):
            sub = sub.sort_values("forecast")   # ascending for barh readability
            fig, ax = VT.new_ax(10, 0.55 * len(sub) + 2.2)
            ax.grid(axis="y", alpha=0)          # horizontal bars: grid on x
            ax.grid(axis="x", alpha=.22, linewidth=.7)
            y = np.arange(len(sub))
            bars = ax.barh(y, sub["forecast"], color=colour, height=.66,
                           edgecolor="white")
            # confidence interval whiskers if present
            has_ci = "lo" in sub and sub["lo"].notna().any()
            if has_ci:
                ax.hlines(y, sub["lo"], sub["hi"], color="#33333355", lw=1.6,
                          zorder=3)
            # place the value label beyond the whisker end so they never clash
            label_x = sub["hi"] if has_ci else sub["forecast"]
            for v, lx, yy in zip(sub["forecast"], label_x, y):
                ax.text(lx + 0.6, yy, f"{v:.1f}%", va="center", ha="left",
                        fontsize=11.5, color="#333333")
            ax.set_yticks(y); ax.set_yticklabels(sub["name"])
            ax.set_xlabel("Forecast share of adults who are Active (%)")
            right = (sub["hi"].max() if has_ci else sub["forecast"].max())
            ax.set_xlim(0, right * 1.16)
            VT.title_block(ax, title, subtitle)
            VT.finish(fig, outdir / fname, note)

        yr = final["year"].iloc[0] if "year" in final else ""
        borough_bar(fr.head(10), "Ten most active boroughs (forecast)",
                    f"Projected share of adults meeting the guideline, {yr}",
                    VT.COL["teal"], "figX2_top10_boroughs.png",
                    "Bars show the central forecast; thin lines show the 90% "
                    "confidence range where available.")
        borough_bar(fr.tail(5), "Five least active boroughs (forecast)",
                    f"Projected share of adults meeting the guideline, {yr}",
                    VT.COL["amber"], "figX3_bottom5_boroughs.png",
                    "Bars show the central forecast; thin lines show the 90% "
                    "confidence range where available.")

        # ================= FIGURES =================
    # fig11: London series, observed then projected
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(t.wave.map(WAVE_YEAR), t.observed, "o-", color="#2c7fb8", lw=2,
            ms=6, label="observed (survey data)")
    fy = [last_y + (h - last_w) for h in horizon]
    fv = [np.average(fc[fc.wave == h].set_index("borough").forecast
                     .reindex(wpop.index), weights=wpop) for h in horizon]
    ax.plot([last_y] + fy, [t.observed.iloc[-1]] + fv, "o--", color="#e67e22",
            lw=2, ms=7, markerfacecolor="white", markeredgecolor="#e67e22",
            markeredgewidth=1.6, label="forecast (model projection)")
    ax.axvline(last_y, color="#888888", ls=":", lw=1.2)
    ax.text(last_y, ax.get_ylim()[0], " last observed year", rotation=90,
            va="center", ha="right", fontsize=9, color="#888888")
    if boots:
        lo = np.average(lon_fc.lo, weights=wpop)
        hi = np.average(lon_fc.hi, weights=wpop)
        ax.fill_between([last_y, fy[-1]], [t.observed.iloc[-1], lo],
                        [t.observed.iloc[-1], hi], color="#e67e22", alpha=.18,
                        label="90% interval")
    ax.axvspan(2018.5, 2020.5, color="grey", alpha=.12)
    ax.text(2019.5, ax.get_ylim()[1], " COVID", fontsize=8, color="grey",
            va="top")
    ax.legend(fontsize=8.5, frameon=False)
    _style(ax, f"London: {args.target}, observed and projected",
           "survey year (start)", "% of adults")
    fig.tight_layout(); fig.savefig(outdir / "fig11_london_forecast.png", dpi=150)
    plt.close(fig)

    # fig12: borough forecast with intervals
    f = final.copy()
    fig, ax = plt.subplots(figsize=(9, .32 * len(f) + 1.6))
    y = np.arange(len(f))
    if "lo" in f and f.lo.notna().any():
        ax.hlines(y, f.lo, f.hi, color="#bdc3c7", lw=3.5)
    ax.plot(f.baseline, y, "o", ms=5, color="#95a5a6", label=f"{last_y} actual")
    ax.plot(f.forecast, y, "D", ms=6, color="#2c7fb8",
            label=f"{last_y + args.horizon} forecast")
    ax.set_yticks(y); ax.set_yticklabels(f.borough, fontsize=7.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax, f"Borough forecast, {args.target} (bars = 90% interval)",
           "% of adults")
    fig.tight_layout(); fig.savefig(outdir / "fig12_borough_forecast.png", dpi=150)
    plt.close(fig)

    # fig13: composition vs behaviour decomposition
    d = final.sort_values("total_change")
    fig, ax = plt.subplots(figsize=(9, .32 * len(d) + 1.8))
    y = np.arange(len(d))
    ax.barh(y, d.composition_effect, height=.62, color="#2c7fb8",
            label="composition (population change)")
    ax.barh(y, d.behaviour_effect, left=d.composition_effect, height=.62,
            color="#e67e22", label="behaviour (trend)")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(d.borough, fontsize=7.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax, f"Why does each borough change by {last_y + args.horizon}?",
           "projected change, percentage points")
    ax.text(.99, .02, "behaviour trend is estimated London-wide, so it is the "
                      "same for every borough;\nall between-borough variation "
                      "here is compositional",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="#555555")
    fig.tight_layout()
    fig.savefig(outdir / "fig13_decomposition.png", dpi=150)
    plt.close(fig)

    # fig14: 10-year scenarios
    fig, ax = plt.subplots(figsize=(9, 4.6))
    obs_years = list(t.wave.map(WAVE_YEAR))
    ax.plot(obs_years, t.observed, "o-", color="#2c7fb8", lw=2, ms=5,
            label="observed")
    today_val = t.observed.iloc[-1]
    # value label on today's point
    ax.annotate(f"{today_val:.1f}%", (last_y, today_val), xytext=(0, 10),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=10.5, color="#2c7fb8", fontweight="bold")
    cols = ["#27ae60", "#e67e22", "#c0392b"]
    for (nm, c) in zip(scen.scenario, cols):
        v = scen.loc[scen.scenario == nm, "london_%"].iloc[0]
        ax.plot([last_y, last_y + 10], [today_val, v], "--", color=c, lw=2,
                label=nm)
        ax.plot([last_y + 10], [v], "o", color=c, ms=7)
        # value label on each final predicted point
        ax.annotate(f"{v:.1f}%", (last_y + 10, v), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=10.5, color=c, fontweight="bold")
    
    # Whole-number year ticks and labels for every year
    xticks = list(range(min(obs_years), last_y + 11))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) for x in xticks], rotation=45, ha="right")
    ax.set_xlim(min(obs_years) - 0.4, last_y + 10 + 2.0)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    _style(ax, "Ten-year scenarios (not a point forecast)",
           "survey year (start)", "% of adults")
    fig.tight_layout(); fig.savefig(outdir / "fig14_scenarios.png", dpi=150)
    plt.close(fig)

    print(f"\nwritten to {outdir}/")
    for p in sorted(outdir.glob("*")):
        print(f"   {p.name}")


if __name__ == "__main__":
    main()
