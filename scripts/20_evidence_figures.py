"""
20_evidence_figures.py — figures that DEMONSTRATE the forecast predicts well.

Built directly on the real per-borough back-test output (not illustrations):
every point is an actual forward forecast compared with what really happened.

    evid_pred_vs_actual.png     predicted vs actual, ours vs naive, on the
                                diagonal = perfect
    evid_error_distribution.png how far off each method was (ours is tighter)
    evid_beats_naive.png        borough-by-borough: where we beat "no change"
    evid_band_catches.png       example boroughs: actual value sits inside our
                                calibrated 90% band

Input: backtest_by_borough.csv (from 17_backtest.py)

Run:
    python 20_evidence_figures.py backtest_by_borough.csv -o evidence_out
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import viz_theme as VT
C = VT.COL
INK = "#1a2436"


def load(path):
    df = pd.read_csv(path)
    # clean origins only (exclude forecasts into a COVID wave)
    if "covid" in df.columns:
        df = df[df["covid"] == 0].copy()
    df["abs_err_model"] = (df["forecast"] - df["actual"]).abs()
    df["abs_err_naive"] = (df["naive"] - df["actual"]).abs()
    return df


# ---------------------------------------------------------------- 1
def pred_vs_actual(df, out):
    """Predicted vs actual, our model and naive side by side. On the diagonal
    means a perfect forecast."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.9), sharex=True, sharey=True)
    lo = min(df.actual.min(), df.forecast.min(), df.naive.min()) - 2
    hi = max(df.actual.max(), df.forecast.max(), df.naive.max()) + 2

    for ax, col, name, colr in [(a1, "forecast", "Our forecast", C["teal"]),
                                (a2, "naive", "\u201cAssume no change\u201d", C["red"])]:
        ax.plot([lo, hi], [lo, hi], color=C["grey"], ls="--", lw=1.6, zorder=1)
        ax.scatter(df.actual, df[col], s=46, color=colr, alpha=0.75,
                   edgecolor="white", linewidth=0.6, zorder=3)
        # mean absolute error annotation
        mae = (df[col] - df.actual).abs().mean()
        ax.text(0.04, 0.95, f"average miss: {mae:.1f} points",
                transform=ax.transAxes, fontsize=12, fontweight="bold",
                color=INK, va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=colr, lw=1.5))
        ax.set_title(name, fontsize=14, fontweight="bold", color=colr, pad=8)
        ax.set_xlabel("What actually happened (%)", color=INK, fontsize=12)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.tick_params(colors=INK)
        for s in ax.spines.values():
            s.set_color("#B8C0CC")
        ax.grid(alpha=0.25)
    a1.set_ylabel("What the method predicted (%)", color=INK, fontsize=12)
    fig.suptitle("Do the predictions land on reality?  Closer to the dashed line is better",
                 fontsize=15, fontweight="bold", color=C["navy"], x=0.5, y=0.99)
    fig.text(0.5, 0.015,
             "Each dot is one borough, one year ahead. Our forecasts hug the "
             "line more tightly than assuming no change \u2014 they miss by less.",
             ha="center", fontsize=10.5, color=INK)
    fig.subplots_adjust(top=0.86, bottom=0.16, wspace=0.12)
    fig.savefig(out / "evid_pred_vs_actual.png", dpi=150,
                bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print("  evid_pred_vs_actual.png")


# ---------------------------------------------------------------- 2
def error_distribution(df, out):
    """Two stacked panels (ours vs no-change) sharing an x-axis, so each is
    clean and the leftward shift of our errors is obvious. No overlapping bars."""
    m_model = df.abs_err_model.mean(); m_naive = df.abs_err_naive.mean()
    top = max(df.abs_err_naive.max(), df.abs_err_model.max()) + 1
    bins = np.linspace(0, top, 15)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.6), sharex=True)
    # panel 1: our forecast
    ax1.hist(df.abs_err_model, bins=bins, color=C["teal"], edgecolor="white",
             zorder=3)
    ax1.axvline(m_model, color=INK, lw=2.2, ls="--", zorder=4)
    ax1.text(m_model + 0.2, ax1.get_ylim()[1]*0.82,
             f"average miss {m_model:.1f} points", color=INK, fontsize=11.5,
             fontweight="bold", va="top")
    ax1.set_title("Our forecast", fontsize=13.5, fontweight="bold",
                  color=C["teal"], loc="left", pad=6)
    # panel 2: no change
    ax2.hist(df.abs_err_naive, bins=bins, color=C["red"], edgecolor="white",
             zorder=3)
    ax2.axvline(m_naive, color=INK, lw=2.2, ls="--", zorder=4)
    ax2.text(m_naive + 0.2, ax2.get_ylim()[1]*0.82,
             f"average miss {m_naive:.1f} points", color=INK, fontsize=11.5,
             fontweight="bold", va="top")
    ax2.set_title("\u201cAssume no change\u201d", fontsize=13.5,
                  fontweight="bold", color=C["red"], loc="left", pad=6)

    for ax in (ax1, ax2):
        ax.tick_params(colors=INK)
        ax.set_ylabel("Number of\nborough-forecasts", color=INK, fontsize=10.5)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color("#B8C0CC")
    ax2.set_xlabel("How far the prediction missed (percentage points)",
                   color=INK, fontsize=12)

    fig.suptitle("Our misses cluster near zero; \u201cno change\u201d spreads further right",
                 fontsize=15, fontweight="bold", color=C["navy"], x=0.5, y=0.99)
    fig.text(0.5, 0.015,
             "Same rounds, same boroughs, shown separately so each is clear. "
             "The teal bars sit further left \u2014 smaller misses, more often.",
             ha="center", fontsize=10.5, color=INK)
    fig.subplots_adjust(top=0.9, bottom=0.14, hspace=0.32)
    fig.savefig(out / "evid_error_distribution.png", dpi=150,
                bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print("  evid_error_distribution.png")


# ---------------------------------------------------------------- 3
def beats_naive(df, out):
    """Per-borough: average error ours vs naive, sorted, showing where we win."""
    g = (df.groupby("borough")[["abs_err_model", "abs_err_naive"]].mean()
           .reset_index())
    g["improvement"] = g.abs_err_naive - g.abs_err_model   # >0 means we beat it
    g = g.sort_values("improvement")
    try:
        g["name"] = g.borough.map(VT.BOROUGH_NAME).fillna(g.borough)
    except Exception:
        g["name"] = g.borough

    fig, ax = VT.new_ax(10, 0.32 * len(g) + 2.4)
    y = np.arange(len(g))
    cols = [C["teal"] if v >= 0 else C["red"] for v in g.improvement]
    ax.barh(y, g.improvement, color=cols, height=0.7, edgecolor="white", zorder=3)
    ax.axvline(0, color=INK, lw=1.2, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(g.name, fontsize=8.8, color=INK)
    ax.set_xlabel("How much better our forecast was than assuming no change\n"
                  "(percentage points of error removed)", color=INK)
    ax.tick_params(colors=INK)
    for s in ax.spines.values():
        s.set_color("#B8C0CC")
    ax.grid(axis="x", alpha=0.25)
    # count wins
    wins = int((g.improvement >= 0).sum())
    ax.text(0.98, 0.02, f"we beat \u201cno change\u201d in {wins} of {len(g)} boroughs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11.5,
            fontweight="bold", color=C["teal"],
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C["teal"], lw=1.4))
    ax.set_title("")  # manual titles, robust on this tall figure
    # reserve a top band and place both lines inside it, well separated
    fig.subplots_adjust(top=0.93)
    fig.text(0.04, 0.975, "Where the forecast beats \u201cassume no change\u201d",
             fontsize=16, fontweight="bold", color=VT.COL["navy"], ha="left",
             va="top")
    fig.text(0.04, 0.955, "Borough by borough, averaged over the clean test rounds",
             fontsize=11.5, color=VT.COL["grey"], ha="left", va="top")
    fig.text(0.04, 0.02,
             "Teal bars to the right are boroughs where our forecast was more "
             "accurate; the few red bars are where assuming no change did as "
             "well or slightly better.",
             fontsize=10, color=VT.COL["grey"], ha="left")
    fig.subplots_adjust(bottom=0.11)
    fig.savefig(out / "evid_beats_naive.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("  evid_beats_naive.png")


# ---------------------------------------------------------------- 4
def band_catches(df, out):
    """Example boroughs at the latest origin: actual vs our forecast + band."""
    latest = df[df.origin_wave == df.origin_wave.max()].copy()
    try:
        latest["name"] = latest.borough.map(VT.BOROUGH_NAME).fillna(latest.borough)
    except Exception:
        latest["name"] = latest.borough
    # pick a readable spread of 12 boroughs across the activity range
    latest = latest.sort_values("actual")
    pick = latest.iloc[np.linspace(0, len(latest)-1, 12).astype(int)]

    fig, ax = VT.new_ax(10.5, 6.6)
    y = np.arange(len(pick))
    # calibrated band as a horizontal range
    ax.hlines(y, pick.lo_cal, pick.hi_cal, color=C["lightgrey"], lw=9,
              zorder=1, label="our 90% range")
    ax.scatter(pick.forecast, y, s=90, color=C["teal"], zorder=3,
               label="our forecast", edgecolor="white", linewidth=1)
    ax.scatter(pick.actual, y, s=120, color=C["navy"], marker="D", zorder=4,
               label="what actually happened", edgecolor="white", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(pick.name, fontsize=9.5, color=INK)
    ax.set_xlabel("Share meeting the activity guideline (%)", color=INK)
    ax.tick_params(colors=INK)
    for s in ax.spines.values():
        s.set_color("#B8C0CC")
    ax.grid(axis="x", alpha=0.25)
    inside = int(((pick.actual >= pick.lo_cal) & (pick.actual <= pick.hi_cal)).sum())
    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC",
              framealpha=0.95, fontsize=10.5, loc="lower right")
    VT.title_block(ax, "The actual value lands inside our range",
                   f"Latest test round \u2014 the diamond falls in the band for "
                   f"{inside} of these {len(pick)} boroughs")
    VT.finish(fig, out / "evid_band_catches.png",
              "The grey bar is our calibrated 90% range, the teal dot our "
              "forecast, the navy diamond what actually happened. When the "
              "diamond sits in the bar, the range did its job.")
    print("  evid_band_catches.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backtest_csv", nargs="?",
                    default="backtest_out/backtest_by_borough.csv")
    ap.add_argument("-o", "--out", default="evidence_out")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    path = args.backtest_csv
    if not Path(path).exists():
        for c in ["backtest_by_borough.csv",
                  "backtest_out/backtest_by_borough.csv"]:
            if Path(c).exists():
                path = c; break
    print(f"reading {path}")
    df = load(path)
    print(f"{df.borough.nunique()} boroughs, {len(df)} clean borough-forecasts")
    pred_vs_actual(df, out)
    error_distribution(df, out)
    beats_naive(df, out)
    band_catches(df, out)
    print(f"written to {out}/")


if __name__ == "__main__":
    main()
