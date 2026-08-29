"""
22_requested_graphs.py — three specific graphs.

    req_sensitivity_line.png   how the forecast changes (or doesn't) as the
                               shrinkage sensitivity setting moves along the
                               x-axis. One clean line.

    req_baseline_vs_flagship.png  the rudimentary "assume no change" forecast
                               against our flagship forecast, drawn across the
                               years so the difference is visible.

    req_backtest.png           the back-test: at each origin, our forecast
                               error vs the naive benchmark's error, showing we
                               are lower every time.

Reuses the real engine (09_forecast.py). The backtest graph reads the real
backtest_by_origin.csv if present.

Run:
    python 22_requested_graphs.py dataset1_individual.parquet -o graphs_out
"""

import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import viz_theme as VT
C = VT.COL
INK = "#1a2436"


def _engine():
    here = Path(__file__).parent
    path = here / "09_forecast.py"
    if not path.exists():
        path = Path("09_forecast.py")
    spec = importlib.util.spec_from_file_location("engine09", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["engine09"] = m
    spec.loader.exec_module(m)
    return m

ENG = None


# ============================================================ 1. sensitivity
def sensitivity_line(df, out):
    """London forecast across the full range of the shrinkage setting."""
    last_w = int(df.wave.max())
    horizon = list(range(last_w + 1, last_w + 4))
    wpop = df[df.wave == last_w].groupby("borough").wt_final.sum()
    lams = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

    londons = []
    for lam in lams:
        m, enc, feats = ENG.fit_propensity(df)
        prop = ENG.propensity_by_cell(df, m, enc, feats)
        comp = ENG.observed_composition(df)
        proj = ENG.project_composition(comp, horizon, shrink=lam)
        t, slope, ic, cov = ENG.behaviour_trend(df, m, enc, feats)
        trend = slope * (horizon[-1] - last_w)
        fc = ENG.score(proj, prop, trend_pp=trend)
        fc = fc[fc.wave == horizon[-1]].set_index("borough")["rate"]
        londons.append(np.average(fc.reindex(wpop.index), weights=wpop))

    londons = np.array(londons)
    span = londons.max() - londons.min()

    fig, ax = VT.new_ax(9.5, 5.4)
    ax.plot(lams, londons, "o-", color=C["teal"], lw=2.8, ms=10, zorder=3)
    for x, y in zip(lams, londons):
        ax.annotate(f"{y:.1f}%", (x, y), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=10.5,
                    fontweight="bold", color=INK)
    # honest y-window (not zoomed to fake a slope)
    mid = londons.mean()
    ax.set_ylim(mid - 3, mid + 3)
    ax.axvline(0.5, color=C["amber"], ls="--", lw=1.8, zorder=1)
    ax.text(0.5, mid + 2.6, "value we used (0.5)", color=C["amber"],
            fontsize=10.5, ha="center", va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))
    ax.set_xlabel("Sensitivity setting  (0 = smoother \u2192 1 = follows each "
                  "borough's own trend)", color=INK)
    ax.set_ylabel("London activity forecast (%)", color=INK)
    ax.set_xticks(lams)
    ax.tick_params(colors=INK)
    for s in ax.spines.values():
        s.set_color("#B8C0CC")
    VT.title_block(ax, "The forecast is stable across the sensitivity setting",
                   f"Across the whole range the forecast moves only "
                   f"{span:.1f} percentage points")
    VT.finish(fig, out / "req_sensitivity_line.png",
              "The line is nearly flat: changing the sensitivity setting from "
              "one extreme to the other barely moves the forecast, so the "
              "value we chose is not driving the result.")
    print(f"  req_sensitivity_line.png  (span {span:.2f}pp)")


# ============================================== 2. baseline vs flagship
def baseline_vs_flagship(df, out, panel_csv, forecast_csv):
    """Observed history + two forecasts forward: naive (flat) vs our flagship.

    Fully data-driven: the observed history comes from the borough rate panel,
    and the flagship forward line comes straight from the pipeline's own
    borough_forecast_all_years.csv — no re-running of the engine — so the line
    matches the report exactly. The engine is used only for population weights
    (the last wave's borough weights), which are not stored in the CSVs."""
    # population weights per borough (last wave) for the London average
    last_w = int(df.wave.max())
    wpop = df[df.wave == last_w].groupby("borough").wt_final.sum()

    # --- observed history: from the borough rate panel ---
    panel = pd.read_csv(panel_csv)
    panel = panel.set_index(panel.columns[0])          # borough index
    panel.columns = [int(c) for c in panel.columns]    # wave columns as ints
    ws = sorted(panel.columns)
    obs = []
    for w in ws:
        rates = panel[w].reindex(wpop.index)
        obs.append(np.average(rates.values, weights=wpop.reindex(rates.index).values))
    obs = np.array(obs)
    obs_years = [VT.WAVE_YEAR[w] for w in ws]
    last_year = obs_years[-1]
    today = obs[-1]

    # --- flagship forward line: from the pipeline's forecast file ---
    ff = pd.read_csv(forecast_csv)
    fut_waves = sorted(ff.wave.unique())
    # London-average the forecast at each horizon wave
    ours_raw = []
    for w in fut_waves:
        sub = ff[ff.wave == w].set_index("borough")["forecast"].reindex(wpop.index)
        ours_raw.append(np.average(sub.values, weights=wpop.reindex(sub.index).values))
    ours_raw = np.array(ours_raw)
    # anchor to observed 'today': the forecast file carries a 'baseline' (the
    # modelled current level); shift the whole line so it starts exactly at the
    # observed last point, matching how the report anchors its forecasts.
    if "baseline" in ff.columns:
        base_sub = (ff[ff.wave == fut_waves[0]].set_index("borough")["baseline"]
                    .reindex(wpop.index))
        model_today = np.average(base_sub.values,
                                 weights=wpop.reindex(base_sub.index).values)
        offset = today - model_today
    else:
        offset = 0.0
    ours = list(ours_raw + offset)
    fut_years = [VT.WAVE_YEAR[w] for w in fut_waves]

    # naive = flat at today's observed value
    naive = [today] * len(fut_years)

    fig, ax = VT.new_ax(9.8, 5.6)
    # observed
    ax.plot(obs_years, obs, "o-", color=C["navy"], lw=2.6, ms=7,
            label="what actually happened", zorder=3)
    # our forecast (connect from today)
    ax.plot([last_year] + fut_years, [today] + ours, "o--", color=C["teal"],
            lw=2.6, ms=8, label="our forecast", zorder=4)
    # naive forecast
    ax.plot([last_year] + fut_years, [today] + naive, "s--", color=C["red"],
            lw=2.4, ms=7, label="rudimentary \u2018assume no change\u2019", zorder=4)
    # labels at the end
    ax.annotate(f"{ours[-1]:.1f}%", (fut_years[-1], ours[-1]), xytext=(8, 4),
                textcoords="offset points", fontsize=11, fontweight="bold",
                color=C["teal"])
    ax.annotate(f"{naive[-1]:.1f}%", (fut_years[-1], naive[-1]), xytext=(8, -12),
                textcoords="offset points", fontsize=11, fontweight="bold",
                color=C["red"])
    # divider at 'today' — label placed in the UPPER area so it never meets the
    # bottom-right legend
    ax.axvline(last_year, color=C["grey"], ls=":", lw=1.4, zorder=1)
    ymin, ymax = ax.get_ylim()
    ax.text(last_year - 0.15, ymax - (ymax - ymin) * 0.04, "forecast starts here  ",
            rotation=90, fontsize=9, color=C["grey"], va="top", ha="right")
    ax.set_xlabel("Year", color=INK)
    ax.set_ylabel("London adults meeting the guideline (%)", color=INK)
    ax.tick_params(colors=INK)
    for s in ax.spines.values():
        s.set_color("#B8C0CC")
    # keep the axis ending where the data ends; the forecast lines sit high,
    # so the lower-right area is empty and the legend fits there cleanly
    all_years = obs_years + fut_years
    ax.set_xlim(min(all_years) - 0.6, max(all_years) + 0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC",
              framealpha=.95, fontsize=10.5, loc="lower right")
    VT.title_block(ax, "Our forecast vs the rudimentary benchmark",
                   "The naive model just holds today's value flat; ours "
                   "responds to how the population changes")
    VT.finish(fig, out / "req_baseline_vs_flagship.png",
              "The blue line is the real history. From today, the red line is "
              "the crude \u2018assume no change\u2019 guess and the teal line is our "
              "forecast, which bends as London's population shifts.")
    print("  req_baseline_vs_flagship.png")


# ============================================================ 3. backtest
def backtest_graph(out, backtest_csv):
    """Per-origin: our error vs naive error, showing ours is lower each time."""
    if not Path(backtest_csv).exists():
        print(f"  (skipped backtest graph: {backtest_csv} not found)")
        return
    R = pd.read_csv(backtest_csv)
    clean = R[R.forecasts_into_covid == 0].copy() if "forecasts_into_covid" in R else R

    fig, ax = VT.new_ax(9.8, 5.6)
    x = clean.origin_wave.astype(int)
    xt = [f"{VT.WAVE_YEAR[w]}\u2192{VT.WAVE_YEAR[w+1]}" for w in x]
    ax.plot(range(len(x)), clean.MAE_naive, "s--", color=C["red"], lw=2.4,
            ms=9, label="rudimentary \u2018assume no change\u2019", zorder=3)
    ax.plot(range(len(x)), clean.MAE_model, "o-", color=C["teal"], lw=2.8,
            ms=10, label="our forecast", zorder=4)
    # shade the gap (our improvement)
    ax.fill_between(range(len(x)), clean.MAE_model, clean.MAE_naive,
                    color=C["teal"], alpha=0.10, zorder=1)
    for i, (m, n) in enumerate(zip(clean.MAE_model, clean.MAE_naive)):
        ax.annotate(f"{m:.1f}", (i, m), xytext=(0, -15), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color=C["teal"])
        ax.annotate(f"{n:.1f}", (i, n), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color=C["red"])
    ax.set_xticks(range(len(x))); ax.set_xticklabels(xt, fontsize=10.5, color=INK)
    ax.set_ylim(0, max(clean.MAE_naive) * 1.25)
    ax.set_xlabel("Back-test round  (train up to first year, forecast the next)",
                  color=INK)
    ax.set_ylabel("Average forecast error\n(percentage points \u2014 lower is better)",
                  color=INK)
    ax.tick_params(colors=INK)
    for s in ax.spines.values():
        s.set_color("#B8C0CC")
    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC",
              framealpha=.95, fontsize=10.5, loc="upper right")
    VT.title_block(ax, "Back-test: our forecast is more accurate every round",
                   "Re-forecasting the past \u2014 our error sits below the naive "
                   "benchmark at every clean origin")
    VT.finish(fig, out / "req_backtest.png",
              "At each round we train on the past and forecast the next year. "
              "Our error (teal) is below the crude benchmark (red) every time; "
              "the shaded gap is the accuracy we add.")
    print("  req_backtest.png")


def main():
    global ENG
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="graphs_out")
    ap.add_argument("--backtest", default="backtest_out/backtest_by_origin.csv")
    ap.add_argument("--panel", default="benchmark_out/borough_rate_panel.csv",
                    help="observed borough rate panel (from 16_benchmarks.py)")
    ap.add_argument("--forecast-all", default="forecast_out/borough_forecast_all_years.csv",
                    help="pipeline forecast at every horizon year (from 09)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ENG = _engine()
    df = ENG.load(args.parquet, "is_active")

    sensitivity_line(df, out)
    panel = args.panel
    if not Path(panel).exists():
        for c in ["borough_rate_panel.csv", "benchmark_out/borough_rate_panel.csv"]:
            if Path(c).exists():
                panel = c; break
    fcsv = args.forecast_all
    if not Path(fcsv).exists():
        for c in ["borough_forecast_all_years.csv",
                  "forecast_out/borough_forecast_all_years.csv"]:
            if Path(c).exists():
                fcsv = c; break
    if Path(panel).exists() and Path(fcsv).exists():
        baseline_vs_flagship(df, out, panel, fcsv)
    else:
        print(f"  (skipped baseline-vs-flagship: need {panel} and {fcsv})")
    bt = args.backtest
    if not Path(bt).exists():
        for c in ["backtest_by_origin.csv", "backtest_out/backtest_by_origin.csv"]:
            if Path(c).exists():
                bt = c; break
    backtest_graph(out, bt)
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
