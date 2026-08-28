"""
19_explainer_figures.py —  explanatory figures.

A set of figures that explain the methodology and validation in simple visual
terms, for a non-technical reader. All use the shared theme, dark readable
text, generous spacing, and no overlapping elements. Real project numbers.

    expl_benchmark_ladder.png   baseline -> single feature -> full -> LightGBM
    expl_model_schematic.png    how the propensity model turns a person into a
                                probability
    expl_rolling_origin.png     how rolling-origin back-testing slides forward
    expl_calibration.png        the 53% -> 94% interval fix
    expl_validation_summary.png the three-number validation scorecard

Run:
    python 19_explainer_figures.py -o explainer_out
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import viz_theme as VT
C = VT.COL
INK = "#1a2436"          # dark text everywhere (never light)


# ---------------------------------------------------------------- 1. benchmark
def benchmark_ladder(out):
    """AUC ladder: floor -> single features -> full model -> LightGBM."""
    labels = ["Guess the\nbase rate\n(no model)",
              "Age only", "Ethnicity\nonly", "Socio-economic\ngroup only",
              "Our full\nmodel", "LightGBM\n(complex)"]
    auc = [0.500, 0.572, 0.568, 0.619, 0.704, 0.708]
    cols = [C["grey"], C["lightgrey"], C["lightgrey"], C["amber"],
            C["teal"], C["navy"]]

    fig, ax = VT.new_ax(11, 6.4)
    x = np.arange(len(auc))
    bars = ax.bar(x, auc, color=cols, width=0.66, edgecolor="white", zorder=3)
    # floor reference line
    ax.axhline(0.5, color=C["grey"], ls="--", lw=1.4, zorder=1)
    ax.text(len(auc) - 0.4, 0.505, "chance level (0.50)", fontsize=10.5,
            color=INK, va="bottom", ha="right")
    # value labels above bars
    for xi, v in zip(x, auc):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom",
                fontsize=12.5, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, color=INK)
    ax.set_ylabel("Ability to tell active from inactive people\n(AUC — higher is better)",
                  color=INK)
    ax.set_ylim(0.48, 0.73)
    ax.tick_params(colors=INK)
    for s in ax.spines.values():
        s.set_color("#B8C0CC")
    VT.title_block(ax, "How good is our model? Compared with what?",
                   "Each bar earns its height above the one before it")
    VT.finish(fig, out / "expl_benchmark_ladder.png",
              "Reading left to right: guessing gets you 0.50. One feature helps "
              "a little. Our full model reaches 0.70, and the far more complex "
              "LightGBM barely differs \u2014 so we keep the simpler model.")
    print("  expl_benchmark_ladder.png")


# ---------------------------------------------------------------- 2. schematic
def model_schematic(out):
    """A person -> features -> model -> three band probabilities."""
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(cx, cy, w, h, title, lines, fill, edge, tcol=INK):
        ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                     boxstyle="round,pad=0.4,rounding_size=1.4",
                     facecolor=fill, edgecolor=edge, linewidth=2, zorder=2))
        ax.text(cx, cy + h/2 - 5, title, ha="center", va="top",
                fontsize=12.5, fontweight="bold", color=tcol, zorder=3)
        ax.text(cx, cy + h/2 - 13, lines, ha="center", va="top",
                fontsize=10.5, color="#333", zorder=3, linespacing=1.5)

    def arrow(x1, x2, y, label):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                     mutation_scale=22, color=C["grey"], linewidth=2.4, zorder=1))
        ax.text((x1 + x2)/2, y + 5, label, ha="center", fontsize=9.5,
                color=INK, style="italic", zorder=4)

    ax.text(50, 96, "How the model turns a person into a probability",
            ha="center", fontsize=16, fontweight="bold", color=C["navy"])

    box(13, 58, 22, 52, "One person",
        "age band\ngender\nethnicity\nclass\neducation\ndisability\nBMI group\nborough",
        "#F6EAD8", C["amber"])
    arrow(25, 41, 60, "")
    ax.text(33, 66, "feed the\ncharacteristics in", ha="center", va="center",
            fontsize=9.5, color=INK, style="italic", zorder=5)
    box(53, 58, 26, 40, "The model",
        "weighs each\ncharacteristic by\nwhat it learned\nfrom 135,497\npeople",
        "#E7EAF1", C["navy"])
    arrow(67, 79, 60, "")
    ax.text(73, 66, "out come three\nprobabilities", ha="center", va="center",
            fontsize=9.5, color=INK, style="italic", zorder=5)
    box(90, 58, 22, 52, "Chance of each",
        "", "#E3EEF0", C["teal"])
    # three band bars inside the last box
    bands = [("Active", 0.62, C["teal"]), ("Fairly", 0.13, C["amber"]),
             ("Inactive", 0.25, C["red"])]
    by = 60
    for name, val, col in bands:
        ax.text(81, by + 4.6, f"{name}  {int(val*100)}%", fontsize=9.5,
                color=INK, va="bottom", ha="left", zorder=5)
        ax.barh(by, val * 14, left=81, height=4.6, color=col, zorder=4)
        by -= 11
    ax.text(50, 18,
            "The model never predicts the future. It only answers: given who "
            "this person is, how likely are they to be active?\nThe forecasting "
            "comes later, by projecting how many people of each type London "
            "will have.",
            ha="center", fontsize=10.5, color=INK, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.6", fc="#F4F1EC", ec="#C8CDD5"))
    fig.savefig(out / "expl_model_schematic.png", dpi=150,
                bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print("  expl_model_schematic.png")


# ---------------------------------------------------------------- 3. rolling
def rolling_origin(out):
    """Visualise the sliding train/forecast/compare windows."""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0.3, 8.9); ax.set_ylim(-0.2, 4.6)

    waves = list(range(1, 9))
    rows = [  # (train_up_to, target, label, is_covid)
        (3, 4, "Round 1", False),
        (4, 5, "Round 2", True),
        (5, 6, "Round 3", True),
        (6, 7, "Round 4", False),
        (7, 8, "Round 5", False),
    ]
    for i, (tr, tg, lab, covid) in enumerate(rows):
        y = len(rows) - 1 - i
        # training block
        ax.add_patch(plt.Rectangle((1 - 0.35, y - 0.32), tr - 1 + 0.7, 0.64,
                     facecolor=C["teal"], alpha=0.85, edgecolor="white", zorder=2))
        ax.text((1 + tr) / 2, y, "train", ha="center", va="center",
                fontsize=10.5, color="white", fontweight="bold", zorder=3)
        # forecast arrow to target
        ax.add_patch(FancyArrowPatch((tr + 0.4, y), (tg - 0.4, y),
                     arrowstyle="-|>", mutation_scale=16, color=C["amber"],
                     linewidth=2.4, zorder=3))
        # target marker
        mcol = C["grey"] if covid else C["navy"]
        ax.scatter(tg, y, s=210, color=mcol, zorder=4, edgecolor="white",
                   linewidth=1.5)
        ax.text(tg, y, str(tg), ha="center", va="center", fontsize=10,
                color="white", fontweight="bold", zorder=5)
        # round label + covid note
        ax.text(0.55, y, lab, ha="right", va="center", fontsize=11,
                color=INK, fontweight="bold")
        if covid:
            ax.text(tg + 0.35, y, "(COVID year — set aside)", ha="left",
                    va="center", fontsize=9.5, color=C["red"], style="italic")
        else:
            ax.text(tg + 0.35, y, "compare forecast with what really happened",
                    ha="left", va="center", fontsize=9.5, color=INK)

    ax.set_xticks(waves)
    ax.set_xticklabels([f"'{14+w}/{15+w}"[:3] + f"/{15+w}" for w in waves]
                       if False else [f"Wave {w}" for w in waves],
                       fontsize=10, color=INK)
    ax.set_yticks([])
    ax.set_xlabel("Survey year", color=INK, fontsize=12)
    ax.tick_params(colors=INK)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#B8C0CC")
    ax.set_title("Rolling-origin back-testing: pretend we're in the past, then check",
                 fontsize=15, fontweight="bold", color=C["navy"], pad=16,
                 loc="left")
    fig.text(0.5, 0.02,
             "Each round trains only on the years available at the time, "
             "forecasts the next year, then checks against what actually "
             "happened. Rounds landing on a COVID year are set aside.",
             ha="center", fontsize=10, color=INK)
    fig.subplots_adjust(bottom=0.13, top=0.9)
    fig.savefig(out / "expl_rolling_origin.png", dpi=150,
                bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print("  expl_rolling_origin.png")


# ---------------------------------------------------------------- 4. calibration
def calibration(out):
    """Before/after coverage: 53% -> 94%, with what changed."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.6),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # left: coverage bars before/after
    covs = [53, 94]
    names = ["Before", "After"]
    cols = [C["red"], C["teal"]]
    xb = [0, 1]
    ax1.bar(xb, covs, color=cols, width=0.6, zorder=3, edgecolor="white")
    ax1.axhline(90, color=C["grey"], ls="--", lw=1.6, zorder=2)
    ax1.text(1.46, 90, "target 90%", fontsize=10.5, color=INK, va="center",
             ha="left")
    for x, v in zip(xb, covs):
        ax1.text(x, v + 1.5, f"{v}%", ha="center", va="bottom", fontsize=15,
                 fontweight="bold", color=INK)
    ax1.set_xticks(xb); ax1.set_xticklabels(names, fontsize=12, color=INK)
    ax1.text(0, -9, "sampling only", ha="center", va="top", fontsize=9.5,
             color="#555", clip_on=False)
    ax1.text(1, -9, "sampling + process", ha="center", va="top", fontsize=9.5,
             color="#555", clip_on=False)
    ax1.set_ylabel("How often the true value landed\ninside our 90% range",
                   color=INK)
    ax1.set_ylim(0, 108); ax1.set_xlim(-0.6, 1.9)
    ax1.tick_params(colors=INK)
    for s in ax1.spines.values():
        s.set_color("#B8C0CC")
    ax1.set_title("The fix in one picture", fontsize=13, fontweight="bold",
                  color=C["navy"], loc="left", pad=10)

    # right: explanation of the two variance sources as stacked widths
    ax2.axis("off")
    ax2.set_xlim(0, 100); ax2.set_ylim(0, 100)
    ax2.text(2, 92, "Why the range was too narrow", fontsize=13,
             fontweight="bold", color=C["navy"])
    # bar 1: sampling only
    ax2.text(2, 76, "Before — we counted only one kind of uncertainty:",
             fontsize=10.5, color=INK)
    ax2.add_patch(plt.Rectangle((2, 64), 32, 8, facecolor=C["teal"], zorder=3))
    ax2.text(18, 68, "sampling", ha="center", va="center", color="white",
             fontsize=10, fontweight="bold", zorder=4)
    ax2.text(37, 68, "\u2190 too short", fontsize=10, color=C["red"],
             va="center", style="italic")
    # bar 2: sampling + process
    ax2.text(2, 50, "After — we added the year-to-year wobble we had left out:",
             fontsize=10.5, color=INK)
    ax2.add_patch(plt.Rectangle((2, 38), 32, 8, facecolor=C["teal"], zorder=3))
    ax2.add_patch(plt.Rectangle((34, 38), 30, 8, facecolor=C["amber"], zorder=3))
    ax2.text(18, 42, "sampling", ha="center", va="center", color="white",
             fontsize=10, fontweight="bold", zorder=4)
    ax2.text(49, 42, "process", ha="center", va="center", color="white",
             fontsize=10, fontweight="bold", zorder=4)
    ax2.text(67, 42, "\u2190 honest width", fontsize=10, color=C["teal"],
             va="center", style="italic")
    ax2.text(2, 20,
             "The extra width (3.4 points) was measured from the back-test,\n"
             "not chosen to hit 90%. Coverage then landed at 94% on its own \u2014\n"
             "which is why we trust it.",
             fontsize=10, color=INK, linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.6", fc="#F4F1EC", ec="#C8CDD5"))

    fig.suptitle("Making the uncertainty honest",
                 fontsize=16, fontweight="bold", color=C["navy"], x=0.5, y=0.99)
    fig.subplots_adjust(top=0.86, wspace=0.32, bottom=0.12)
    fig.savefig(out / "expl_calibration.png", dpi=150,
                bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print("  expl_calibration.png")


# ---------------------------------------------------------------- 5. summary
def validation_summary(out):
    """Three-number scorecard: skill, coverage, corroboration."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    cards = [
        ("+20%", "more accurate", "than assuming no change\n(forecast skill)",
         C["teal"]),
        ("94%", "honest coverage", "of our 90% ranges\n(after calibration)",
         C["navy"]),
        ("0.74", "right reason", "we forecast best where\ndemographics explain\n(correlation)",
         C["amber"]),
    ]
    cx = [18, 50, 82]
    for (big, mid, small, col), x in zip(cards, cx):
        ax.add_patch(FancyBboxPatch((x - 14, 26), 28, 52,
                     boxstyle="round,pad=0.5,rounding_size=2",
                     facecolor="white", edgecolor=col, linewidth=2.6, zorder=2))
        ax.text(x, 66, big, ha="center", va="center", fontsize=30,
                fontweight="bold", color=col, zorder=3)
        ax.text(x, 52, mid, ha="center", va="center", fontsize=13,
                fontweight="bold", color=INK, zorder=3)
        ax.text(x, 40, small, ha="center", va="center", fontsize=10.5,
                color="#333", zorder=3, linespacing=1.5)

    ax.text(50, 90, "Does the forecast work? Three checks, three answers.",
            ha="center", fontsize=16, fontweight="bold", color=C["navy"])
    ax.text(50, 12,
            "Accurate  \u00b7  honest about its uncertainty  \u00b7  and right for the "
            "right reason.",
            ha="center", fontsize=11.5, color=INK, style="italic")
    fig.savefig(out / "expl_validation_summary.png", dpi=150,
                bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print("  expl_validation_summary.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="explainer_out")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print("building explanatory figures ...")
    benchmark_ladder(out)
    model_schematic(out)
    rolling_origin(out)
    calibration(out)
    validation_summary(out)
    print(f"written to {out}/")


if __name__ == "__main__":
    main()
