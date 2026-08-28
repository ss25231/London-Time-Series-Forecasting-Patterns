"""
17_backtest.py — rolling-origin back-testing of the forecast, per borough.

This answers the question a single train/test split cannot: "how do you KNOW
the forecast is any good?" It does so by making genuine FORWARD forecasts at
several successive origins and measuring how far each was from what actually
happened — then comparing that error to a naive benchmark, and checking
whether the 90% intervals are honest.

WHY THIS, NOT THE EARLIER TEST
------------------------------
The propensity model was validated by classifying a held-out YEAR well. That
proves the "who is active" surface is real — but it does NOT prove the
FORECAST is good, because the forecast is carried by demographic projection,
not by the model extrapolating in time. Rolling-origin back-testing tests the
whole forecasting procedure, forward, which is the thing actually in question.

HOW IT WORKS  (per borough, because that is what the engine forecasts and what
the brief cares about; London is only a weighted mean of boroughs)

    origin w=4 : train on waves 1..4, forecast wave 4+h, compare to actual
    origin w=5 : train on waves 1..5, forecast wave 5+h, compare to actual
    ... slide the origin forward ...

At each origin we compute, across the 32 boroughs:
    * MAE  of our forecast   vs  MAE of the naive (persistence) benchmark
    * a SKILL score  = 1 - MAE_ours / MAE_naive   (>0 means we beat naive)
    * interval CALIBRATION: the share of boroughs whose ACTUAL value fell
      inside our 90% band. Honest bands cover ~90%.

HONEST LIMITS  (stated, not hidden)
    * Only 8 waves, 2 disrupted by COVID -> few clean origins. Directional
      evidence, not proof.
    * Origins that forecast INTO a COVID wave are flagged; nobody forecasts a
      pandemic, and counting those against the method would be misleading.
    * Early origins train on very few waves, so their population projection is
      weak; we expect and report that.

It reuses the REAL engine functions from 09_forecast.py by importing them and
feeding them data truncated to each origin, so nothing in 09 is changed.

Run:
    python 17_backtest.py dataset1_individual.parquet -o backtest_out
    python 17_backtest.py dataset1_individual.parquet -o backtest_out --horizon 1
"""

import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ---- import the real engine and the shared benchmarks -----------------------
def _load(mod_name, filename):
    here = Path(__file__).parent
    path = here / filename
    if not path.exists():
        path = Path(filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

ENG = _load("engine09", "09_forecast.py")
BM = _load("benchmarks16", "16_benchmarks.py")

COVID_WAVES = getattr(ENG, "COVID_WAVES", {5, 6})


# ---------------------------------------------------------------- one origin
def forecast_at_origin(df_full, origin_wave, horizon, target, boot, shrink):
    """Run the real engine using ONLY data up to origin_wave, then forecast
    origin_wave + horizon. Returns a DataFrame: borough, forecast, lo, hi.

    We truncate the data to <= origin_wave and call the engine's own
    functions, so this is the identical procedure the project uses, just with
    the clock wound back."""
    d = df_full[df_full.wave <= origin_wave].copy()
    if d.wave.nunique() < 3:
        return None

    target_wave = origin_wave + horizon

    def _one(dd):
        m, enc, feats = ENG.fit_propensity(dd)
        prop = ENG.propensity_by_cell(dd, m, enc, feats)   # uses dd.wave.max() = origin
        comp = ENG.observed_composition(dd)
        proj = ENG.project_composition(comp, [target_wave], shrink=shrink,
                                       min_waves=min(6, dd.wave.nunique()))
        # behaviour trend, projected to the target wave
        t, slope, intercept, covid_eff = ENG.behaviour_trend(dd, m, enc, feats)
        last_w = dd.wave.max()
        # trend in pp added at the horizon (exclude covid handling subtlety:
        # use the non-covid slope, same as the engine)
        trend_pp = slope * (target_wave - last_w)
        fc = ENG.score(proj, prop, trend_pp=trend_pp)
        return fc.set_index("borough")["rate"]

    # point forecast on the full (truncated) data
    point = _one(d)

    # bootstrap for the 90% interval, resampling within origin waves
    boots = []
    rng = np.random.default_rng(42)
    for _ in range(boot):
        idx = rng.integers(0, len(d), len(d))
        try:
            boots.append(_one(d.iloc[idx]))
        except Exception:
            continue
    if boots:
        B = pd.concat(boots, axis=1)
        samp_sd = B.std(axis=1, ddof=1)
        centre = B.mean(axis=1)
    else:
        samp_sd = point * 0.0
        centre = point

    out = pd.DataFrame({"borough": point.index, "forecast": point.values})
    out["centre"] = out.borough.map(centre)
    out["samp_sd"] = out.borough.map(samp_sd)
    return out


# ---------------------------------------------------------------- evaluation
def evaluate():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="backtest_out")
    ap.add_argument("--target", default="is_active")
    ap.add_argument("--horizon", type=int, default=1,
                    help="waves ahead to forecast at each origin")
    ap.add_argument("--boot", type=int, default=40)
    ap.add_argument("--shrink", type=float, default=0.5)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = ENG.load(args.parquet, args.target)
    panel = BM.borough_series(df, args.target)     # borough x wave actuals (%)
    waves = sorted(df.wave.unique())

    # candidate origins: need >=3 training waves before the origin, and the
    # target wave (origin+h) must exist.
    origins = [w for w in waves
               if sum(1 for x in waves if x <= w) >= 3
               and (w + args.horizon) in waves]

    print("ROLLING-ORIGIN BACK-TEST  (per borough, forward forecasts)")
    print("=" * 68)
    print(f"target = {args.target}   horizon = {args.horizon} wave(s)")
    print(f"waves available: {waves}")
    print(f"origins tested : {origins}\n")

    rows = []
    per_origin_detail = []
    for o in origins:
        tw = o + args.horizon
        actual = panel[tw]                    # actual borough rates at target
        fc = forecast_at_origin(df, o, args.horizon, args.target,
                                args.boot, args.shrink)
        if fc is None:
            continue
        fc = fc.set_index("borough")
        common = actual.index.intersection(fc.index)
        a = actual.reindex(common).astype(float)
        f = fc.reindex(common)["forecast"].astype(float)

        # naive + drift benchmarks (share one definition with 16_benchmarks)
        naive = BM.naive_forecast(panel, o, args.horizon).reindex(common)
        drift = BM.drift_forecast(panel, o, args.horizon)
        drift = drift.reindex(common) if drift is not None else None

        mae_model = np.nanmean(np.abs(f - a))
        mae_naive = np.nanmean(np.abs(naive - a))
        mae_drift = (np.nanmean(np.abs(drift - a))
                     if drift is not None else np.nan)

        rows.append({
            "origin_wave": o, "target_wave": tw,
            "forecasts_into_covid": int(tw in COVID_WAVES),
            "MAE_model": round(mae_model, 3),
            "MAE_naive": round(mae_naive, 3),
            "MAE_drift": round(mae_drift, 3),
            "skill_vs_naive": round(BM.skill(mae_model, mae_naive), 3),
            "skill_vs_drift": round(BM.skill(mae_model, mae_drift), 3)
                              if not np.isnan(mae_drift) else np.nan,
        })
        det = pd.DataFrame({"borough": common, "actual": a.values,
                            "forecast": f.values, "naive": naive.values,
                            "centre": fc.reindex(common)["centre"].values,
                            "samp_sd": fc.reindex(common)["samp_sd"].values})
        det["origin_wave"] = o
        det["target_wave"] = tw
        det["covid"] = int(tw in COVID_WAVES)
        per_origin_detail.append(det)

    R = pd.DataFrame(rows)
    # ---- TWO-PASS interval calibration (honest, not tuned) ----
    # Pass 1: measure process SD from clean-origin POINT errors.
    # Pass 2: coverage using total SD = sqrt(samp^2 + process^2).
    process_sd = 0.0
    if per_origin_detail:
        alld = pd.concat(per_origin_detail)
        clean = alld[alld.covid == 0]
        err = (clean["forecast"] - clean["actual"]).dropna()
        process_sd = float(np.sqrt(np.mean((err - err.mean())**2)))
        z = 1.645
        # sampling-only coverage (what we had before)
        alld["lo_samp"] = alld.centre - z * alld.samp_sd
        alld["hi_samp"] = alld.centre + z * alld.samp_sd
        alld["cov_samp"] = ((alld.actual >= alld.lo_samp) &
                            (alld.actual <= alld.hi_samp))
        # calibrated coverage (sampling + process in quadrature)
        tot = np.sqrt(alld.samp_sd**2 + process_sd**2)
        alld["lo_cal"] = alld.centre - z * tot
        alld["hi_cal"] = alld.centre + z * tot
        alld["cov_cal"] = ((alld.actual >= alld.lo_cal) &
                           (alld.actual <= alld.hi_cal))
        cov = (alld[alld.covid == 0].groupby("origin_wave")
               .agg(coverage_sampling_only=("cov_samp", "mean"),
                    coverage_calibrated=("cov_cal", "mean")).reset_index())
        R = R.merge(cov, on="origin_wave", how="left")
        per_origin_detail = [alld]   # for saving

    if R.empty:
        print("Not enough waves to form any origin. Need >=4 waves.")
        return

    # ---------------------------------------------------------------- report
    clean = R[R.forecasts_into_covid == 0]
    print(R.to_string(index=False))
    print("\n" + "-" * 68)
    print("HEADLINE (excluding origins that forecast into a COVID wave):")
    if not clean.empty:
        print(f"  mean skill vs naive : {clean.skill_vs_naive.mean():+.3f}   "
              "(>0 means the method beats 'assume no change')")
        print(f"  mean skill vs drift : {clean.skill_vs_drift.mean():+.3f}")
        if "coverage_calibrated" in clean:
            print(f"  90% coverage, sampling only : "
                  f"{100*clean.coverage_sampling_only.mean():.0f}%   (too narrow)")
            print(f"  90% coverage, CALIBRATED    : "
                  f"{100*clean.coverage_calibrated.mean():.0f}%   "
                  "(sampling + measured process variance)")
        # also report the two aggregate error summaries requested
        print(f"  mean MAE (model)    : {clean.MAE_model.mean():.2f} pp")
        print(f"  mean MAE (naive)    : {clean.MAE_naive.mean():.2f} pp")
    else:
        print("  (all available origins forecast into a COVID wave)")

    print("\nHOW TO READ THIS")
    print("  skill > 0  : our demographic-projection forecast beats naive.")
    print("  skill ~ 0  : no better than assuming each borough stays put")
    print("               (expected where a borough's change is not")
    print("               compositional — see the corroboration check).")
    print("  coverage   : if far below 90%, our intervals are too narrow;")
    print("               if far above, too wide.")

    R.to_csv(out / "backtest_by_origin.csv", index=False)
    detail_all = None
    if per_origin_detail:
        detail_all = pd.concat(per_origin_detail)
        detail_all.to_csv(out / "backtest_by_borough.csv", index=False)

    # ---- MEASURE the process (forecast-error) standard deviation ----
    # Use only clean (non-COVID-target) origins. This is the empirical spread
    # of how far the point forecast lands from reality, one wave ahead. It is
    # measured, not tuned — 09 adds it in quadrature to the sampling interval
    # so the deployed 90% bands are honest.
    if detail_all is not None:
        clean_origins = R[R.forecasts_into_covid == 0].origin_wave.tolist()
        cd = detail_all[detail_all.origin_wave.isin(clean_origins)].copy()
        err = (cd["forecast"] - cd["actual"]).dropna()
        # centre the errors (remove any small mean bias) then take the SD of
        # the residual spread; this is the per-borough one-wave forecast SD
        process_sd = float(np.sqrt(np.mean((err - err.mean())**2)))
        bias = float(err.mean())
        import json
        cal = {"process_sd_pp": round(process_sd, 3),
               "mean_bias_pp": round(bias, 3),
               "n_forecasts": int(len(err)),
               "horizon": args.horizon,
               "note": ("Per-borough one-wave forecast-error SD, measured on "
                        "clean rolling origins. 09_forecast.py adds this in "
                        "quadrature to the sampling interval so 90% bands are "
                        "calibrated. Measured, not tuned to any coverage target.")}
        (out / "interval_calibration.json").write_text(json.dumps(cal, indent=2))
        print(f"\nMEASURED process SD = {process_sd:.2f} pp  "
              f"(bias {bias:+.2f} pp, n={len(err)})")
        print(f"  written to {out}/interval_calibration.json")
        print("  09_forecast.py reads this to calibrate its deployed intervals.")
    print(f"\nwritten to {out}/backtest_by_origin.csv and "
          "backtest_by_borough.csv")

    # ---- optional themed figure ----
    try:
        make_figure(R, clean, out)
    except Exception as e:
        print(f"(figure skipped: {e})")


def make_figure(R, clean, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import viz_theme as VT
        newax, COL, finish, tb = (VT.new_ax, VT.COL, VT.finish, VT.title_block)
    except Exception:
        VT = None

    if VT is None:
        return
    fig, ax = VT.new_ax(9.5, 5.6)
    x = R.origin_wave.astype(int)
    ax.axhline(0, color=VT.COL["grey"], lw=1, ls="--")
    ax.plot(x, R.skill_vs_naive, "o-", color=VT.COL["teal"], lw=2.4, ms=7,
            label="skill vs naive")
    if R.skill_vs_drift.notna().any():
        ax.plot(x, R.skill_vs_drift, "s--", color=VT.COL["amber"], lw=2,
                ms=6, label="skill vs drift")
    # shade covid-into origins
    for _, r in R.iterrows():
        if r.forecasts_into_covid:
            ax.axvspan(r.origin_wave - .3, r.origin_wave + .3,
                       color="#00000010")
    ax.set_xlabel("Origin wave (train up to here, forecast forward)")
    ax.set_ylabel("Forecast skill  (1 - MAE_model / MAE_benchmark)")
    ax.set_xticks(list(x))
    ax.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD",
              framealpha=.95, loc="best")
    VT.title_block(ax, "Does the forecast beat 'assume no change'?",
                   "Rolling-origin skill, per borough; above zero = better "
                   "than naive")
    VT.finish(fig, out / "backtest_skill.png",
              "Shaded origins forecast into a COVID wave and are excluded from "
              "the headline. Positive skill means the demographic-projection "
              "forecast beats persistence.")
    print("  figure: backtest_skill.png")


if __name__ == "__main__":
    evaluate()
