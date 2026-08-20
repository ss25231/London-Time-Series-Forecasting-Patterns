"""
viz_theme.py — one shared style for every figure in the project.

Import this in any stage script and use:
    from viz_theme import COL, new_ax, finish, barh_labels, LABELS
so that every graph has the same palette, fonts, spacing and export
settings. This is the single place to change the look project-wide.

Design goals:
  * one figure per file, never a grid
  * generous margins, no text overlap
  * clear labels, no cryptic abbreviations
  * navy / teal / amber palette, consistent across stages
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- palette -------------------------------------------------------------
COL = {
    "navy":   "#1F3050",   # primary / headings / "Active"
    "teal":   "#2E7D8F",   # secondary / positive
    "amber":  "#D98C3F",   # accent / forecast / "Fairly Active"
    "red":    "#B4472E",   # "Inactive" / negative
    "green":  "#2E8B6F",   # occasional positive
    "grey":   "#7A8899",   # muted / reference lines
    "lightgrey": "#D8DEE4",
    "ink":    "#1F3050",
}

# consistent meaning for the three activity bands, everywhere
BAND_COLOUR = {
    "Inactive":      COL["red"],
    "Fairly Active": COL["amber"],
    "Active":        COL["teal"],
}

# full, human-readable borough names (no codes on charts)
BOROUGH_NAME = {
    "E09000002": "Barking and Dagenham", "E09000003": "Barnet",
    "E09000004": "Bexley", "E09000005": "Brent", "E09000006": "Bromley",
    "E09000007": "Camden", "E09000008": "Croydon", "E09000009": "Ealing",
    "E09000010": "Enfield", "E09000011": "Greenwich", "E09000012": "Hackney",
    "E09000013": "Hammersmith and Fulham", "E09000014": "Haringey",
    "E09000015": "Harrow", "E09000016": "Havering", "E09000017": "Hillingdon",
    "E09000018": "Hounslow", "E09000019": "Islington",
    "E09000020": "Kensington and Chelsea", "E09000021": "Kingston upon Thames",
    "E09000022": "Lambeth", "E09000023": "Lewisham", "E09000024": "Merton",
    "E09000025": "Newham", "E09000026": "Redbridge",
    "E09000027": "Richmond upon Thames", "E09000028": "Southwark",
    "E09000029": "Sutton", "E09000030": "Tower Hamlets",
    "E09000031": "Waltham Forest", "E09000032": "Wandsworth",
    "E09000033": "Westminster",
}

# readable labels for coded groups
LABELS = {
    "age4": "Age group", "gender": "Gender", "eth5": "Ethnicity",
    "nssec4": "Socio-economic group", "educ3": "Education",
    "Disab3": "Disability status", "BMIG": "Body-mass-index group",
    "inner_outer": "Inner or outer London",
}
DISAB_LABELS = {
    "1": "Disabled (day-to-day activities limited)", "1.0": "Disabled (day-to-day activities limited)",
    "2": "Long-term condition (not limiting)", "2.0": "Long-term condition (not limiting)",
    "3": "No disability or condition", "3.0": "No disability or condition",
}

WAVE_YEAR = {w: 2014 + w for w in range(1, 26)}          # wave 1 = 2015
YEAR_LABEL = {w: f"{2014 + w}\u2013{str(2015 + w)[2:]}" for w in range(1, 26)}


# ---- base rcParams -------------------------------------------------------
def apply_rc():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
        "figure.autolayout": False,
    })


# ---- figure scaffolding --------------------------------------------------
def new_ax(w=9.5, h=6.0):
    """A single clean axes. One figure, one chart — never a grid."""
    apply_rc()
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.22, linewidth=.7)
    ax.set_axisbelow(True)
    return fig, ax


def title_block(ax, title, subtitle=None):
    """Bold title, optional grey subtitle above the axes, left-aligned.
    The subtitle sits clearly above the plotting area and the title above
    that, so they never overlap regardless of figure size."""
    if subtitle:
        # subtitle just above the axes, title higher still
        ax.text(0, 1.04, subtitle, transform=ax.transAxes, fontsize=11.5,
                color=COL["grey"], va="bottom", ha="left")
        ax.set_title(title, loc="left", pad=32, color=COL["navy"])
    else:
        ax.set_title(title, loc="left", pad=12, color=COL["navy"])


def finish(fig, path, note=None):
    """Add an optional source/caveat note beneath the axes, then save.
    The note is drawn just under the axes using an offset measured from the
    axes bottom, so it never overlaps the x-axis label and never leaves a
    large empty band. bbox_inches='tight' crops to content."""
    if note:
        fig.canvas.draw()                      # finalise layout first
        ax = fig.axes[0]
        # find the lowest point of the axes area in figure coordinates,
        # then drop the note a fixed distance below it
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted())
        y = bbox.y0 - 0.06
        fig.text(0.01, y, note, fontsize=10.5, color=COL["grey"],
                 ha="left", va="top", wrap=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)


def barh_value_labels(ax, values, y, fmt="{:.0f}", pad=0.4, color="#333333",
                      size=11.5):
    """Put a readable numeric label at the end of each horizontal bar."""
    for v, yy in zip(values, y):
        ax.text(v + pad, yy, fmt.format(v), va="center", ha="left",
                fontsize=size, color=color)


def readable_disab(series):
    return series.astype("object").map(
        lambda x: DISAB_LABELS.get(str(x), x))
