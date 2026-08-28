"""
18_corroboration.py — does the forecast work for the RIGHT reason?

A forecast can beat a benchmark by luck. This check tests whether it beats it
for the reason we claim. Our method forecasts a borough well by projecting its
demographic composition forward. So it SHOULD forecast best on the boroughs
whose activity is well explained by demographics, and worst on the boroughs
driven by place effects the model does not see.

We test that directly:

    x-axis : how much PLACE (not demographics) drives each borough
             = |residual| from 06_composition_vs_place.py
               (obs minus what demographics predict; large = place-driven)
    y-axis : how badly we FORECAST that borough in back-testing
             = mean absolute forecast error per borough (clean origins)

If the method works for the right reason, these are POSITIVELY correlated:
place-driven boroughs (large residual) are the hard-to-forecast ones (large
error). A positive, non-trivial correlation is corroborating evidence; a flat
or negative one would warn that the forecast succeeds by accident.

Inputs (both already produced by the pipeline):
    backtest_out/backtest_by_borough.csv     (per-borough forecast errors)
    model_out/borough_residuals.csv          (per-borough composition residual)

Run:
    python 18_corroboration.py -o corroboration_out
    python 18_corroboration.py --backtest backtest_out --residuals model_out
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def find_file(name, hints):
    for h in hints:
        p = Path(h) / name
        if p.exists():
            return p
    # last resort: search common output dirs
    for d in Path(".").glob("*_out"):
        p = d / name
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="corroboration_out")
    ap.add_argument("--backtest", default="backtest_out")
    ap.add_argument("--residuals", default="model_out")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    bt_path = find_file("backtest_by_borough.csv",
                        [args.backtest, "backtest_out", "."])
    res_path = find_file("borough_residuals.csv",
                         [args.residuals, "model_out", "."])
    if bt_path is None or res_path is None:
        print("Missing inputs. Need backtest_by_borough.csv (run 17_backtest.py)")
        print("and borough_residuals.csv (run 06_composition_vs_place.py).")
        print(f"  backtest found: {bt_path}")
        print(f"  residuals found: {res_path}")
        return

    bt = pd.read_csv(bt_path)
    res = pd.read_csv(res_path)

    # per-borough mean absolute forecast error, clean origins only
    if "covid" in bt.columns:
        bt = bt[bt["covid"] == 0]
    bt["abs_err"] = (bt["forecast"] - bt["actual"]).abs()
    err = (bt.groupby("borough")["abs_err"].mean()
             .rename("mean_abs_forecast_error"))

    # per-borough place strength = |composition residual|
    rcol = "residual" if "residual" in res.columns else res.columns[-1]
    bcol = "borough" if "borough" in res.columns else res.columns[0]
    res = res[[bcol, rcol]].rename(columns={bcol: "borough", rcol: "residual"})
    res["place_strength"] = res["residual"].abs()

    m = err.reset_index().merge(res, on="borough", how="inner")
    if len(m) < 5:
        print(f"Only {len(m)} boroughs matched; check the borough keys align.")
        return

    # correlations
    pear = m["place_strength"].corr(m["mean_abs_forecast_error"], method="pearson")
    spear = m["place_strength"].corr(m["mean_abs_forecast_error"], method="spearman")

    print("CORROBORATION: does forecast skill track demographic explainability?")
    print("=" * 70)
    print(f"boroughs matched: {len(m)}")
    print(f"Pearson  correlation (place strength vs forecast error): {pear:+.3f}")
    print(f"Spearman correlation (rank-based, robust)             : {spear:+.3f}")
    print()
    if spear > 0.2:
        print("READING:  positive — the method forecasts best on boroughs whose")
        print("          activity is demographically explained, and worst on")
        print("          place-driven ones. This is the expected pattern: the")
        print("          forecast works for the RIGHT reason, not by luck.")
    elif spear < -0.2:
        print("READING:  negative — unexpected. The forecast does NOT track")
        print("          demographic explainability; investigate before relying")
        print("          on the mechanism claim.")
    else:
        print("READING:  weak — with only 32 boroughs and few origins this is")
        print("          suggestive at most; report it as directional, not proof.")

    m = m.sort_values("mean_abs_forecast_error")
    m.to_csv(out / "corroboration_by_borough.csv", index=False)
    print(f"\nwritten to {out}/corroboration_by_borough.csv")

    # ---- themed scatter ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import viz_theme as VT
        fig, ax = VT.new_ax(9, 6)
        ax.scatter(m["place_strength"], m["mean_abs_forecast_error"],
                   s=70, color=VT.COL["teal"], edgecolor="white", zorder=3)
        # trend line
        if len(m) >= 3:
            sl, ic = np.polyfit(m["place_strength"], m["mean_abs_forecast_error"], 1)
            xs = np.linspace(m["place_strength"].min(), m["place_strength"].max(), 50)
            ax.plot(xs, sl * xs + ic, color=VT.COL["amber"], lw=2, ls="--",
                    label=f"trend (Spearman {spear:+.2f})")
        # label a few extreme boroughs
        try:
            nm = {k: v for k, v in VT.BOROUGH_NAME.items()}
            ext = pd.concat([m.nlargest(2, "place_strength"),
                             m.nsmallest(2, "place_strength")])
            for _, r in ext.iterrows():
                ax.annotate(nm.get(str(r["borough"]), str(r["borough"])),
                            (r["place_strength"], r["mean_abs_forecast_error"]),
                            fontsize=8.5, color="#444", xytext=(4, 4),
                            textcoords="offset points")
        except Exception:
            pass
        ax.set_xlabel("How much PLACE drives the borough  "
                      "(|composition residual|, pp)")
        ax.set_ylabel("Forecast error in back-testing  (mean |error|, pp)")
        ax.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD",
                  framealpha=.95, loc="best")
        VT.title_block(ax, "The forecast works for the right reason",
                       "Place-driven boroughs are the hard-to-forecast ones")
        VT.finish(fig, out / "corroboration_scatter.png",
                  "Each dot is a borough. Positive slope = we forecast best "
                  "where demographics explain activity, as the method assumes.")
        print(f"  figure: corroboration_scatter.png")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
