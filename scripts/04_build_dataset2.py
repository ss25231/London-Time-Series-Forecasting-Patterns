"""
03_build_dataset2.py — Dataset 1 -> the cell panel used for modelling and
delivery (Dataset 2).

Grain: one row per (borough, wave, age4, gender, eth5)
       = 32 boroughs x 8 waves x 4 ages x 2 genders x 5 ethnicities

Activeness is NOT part of the key — it is the target, so it lives in
columns as p_active / p_fairly / p_inactive.

Every row carries reliability columns (n_cell, w_cell, se_active). These
are what make the disaggregation safe rather than reckless: a cell built
on 3 respondents and one built on 50 look identical in p_active alone.

Run:
    python 03_build_dataset2.py dataset1_individual.parquet -o dataset2_cells.parquet
"""

import argparse

import numpy as np
import pandas as pd

CELL_KEYS = ["borough", "wave", "age4", "gender", "eth_cell"]

# Coarsening for the cell grain. Dataset 1 keeps the full detail; only the
# CELL panel is coarsened, because cells are where thin bases bite.
ETH3 = {
    "White British": "White", "White Other": "White",
    "Asian": "Asian",
    "Black": "Black/Mixed/Other", "Mixed/Other": "Black/Mixed/Other",
    # Non-disclosure is a respondent choice and a real behavioural signal
    # (non-disclosers are less active), so it survives coarsening intact.
    "Not disclosed": "Not disclosed",
}
# Gend3 carries an "Other" level with a very small base. It cannot support
# a borough-level cell, so it is folded into Unknown for the panel only.
GENDER2 = {"Male": "Male", "Female": "Female",
           "Not disclosed": "Not disclosed"}
BANDS = ["Inactive", "Fairly Active", "Active"]

# Secondary outcomes aggregated as weighted means of 0/1 indicators.
# Club and volunteer-role items are ROUTED, so the as-collected versions are
# not comparable across waves. The *_all columns rebase them to a share of
# ALL adults and are the ones to aggregate. Club figures are only comparable
# from wave 3 (wave 2 used a much broader routing filter).
BINARY_OUTCOMES = [
    "club_any_all", "club_fitness_all", "club_team_all",
    "club_racket_combat_all", "club_outdoor_all",
    "vol_any", "vol_coach_all", "vol_officiate_all",
    "vol_organise_all", "vol_support_all",
    "limfreti_care", "limfreti_time",
    "part_walking", "part_cycling", "part_active_travel", "part_dance",
    "part_team", "part_racket", "part_outdoor_water", "part_leisure_games",
    "part_combat_target", "part_fitness",
    "active_indoor", "active_outdoor", "active_home",
    "active_leisurecentre", "active_park", "active_road",
    "active_countryside",
]

MIN_CELL_PUBLISH = 5     # suppress below this in any published output


def wmean(values: pd.Series, weights: pd.Series) -> float:
    ok = values.notna() & weights.notna() & (weights > 0)
    if not ok.any():
        return np.nan
    return float((values[ok] * weights[ok]).sum() / weights[ok].sum())


def build_cells(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(CELL_KEYS, observed=True, dropna=False)
    for keys, g in grouped:
        w = g["wt_final"]
        rec = dict(zip(CELL_KEYS, keys))

        # --- reliability ---
        band_ok = g["activity_band"].notna() & w.notna()
        n = int(band_ok.sum())
        rec["n_cell"] = n
        rec["w_cell"] = float(w[band_ok].sum()) if n else 0.0

        # --- compositional target: three shares summing to 1 ---
        if n:
            ws = (g.loc[band_ok]
                    .groupby("activity_band", observed=False)["wt_final"].sum())
            tot = ws.sum()
            for b, col in zip(BANDS, ["p_inactive", "p_fairly", "p_active"]):
                rec[col] = float(ws.get(b, 0.0) / tot) if tot > 0 else np.nan
        else:
            rec.update(p_inactive=np.nan, p_fairly=np.nan, p_active=np.nan)

        # design-naive SE — adequate for reliability weighting, not for
        # publishing confidence intervals (use a bootstrap for those)
        p = rec.get("p_active")
        rec["se_active"] = (
            float(np.sqrt(max(p * (1 - p), 0) / n)) if n and p is not None
            and not pd.isna(p) else np.nan)

        # --- continuous activity ---
        rec["mean_mins"] = wmean(g["mins_winsor"], w) if "mins_winsor" in g else np.nan
        rec["median_mins"] = float(g["mins_per_week"].median()) \
            if "mins_per_week" in g and g["mins_per_week"].notna().any() else np.nan

        # --- secondary outcomes, each with its own base ---
        for c in BINARY_OUTCOMES:
            if c in g.columns:
                rec[f"p_{c}"] = wmean(g[c], w)
                rec[f"n_{c}"] = int((g[c].notna() & w.notna()).sum())

        # --- indoor / outdoor base size (subsample only, waves 2-8) ---
        if "asked_inout" in g.columns:
            rec["n_inout"] = int(g["asked_inout"].sum())

        rows.append(rec)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="dataset2_cells.parquet")
    ap.add_argument("--min-cell", type=int, default=MIN_CELL_PUBLISH)
    ap.add_argument("--eth5", dest="eth3", action="store_false",
                    help="keep 5 ethnicity levels in the panel (default: 3)")
    ap.add_argument("--gender3", dest="gender2", action="store_false",
                    help="keep the 'Other' gender level (default: fold to Unknown)")
    ap.set_defaults(eth3=True, gender2=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"individuals: {len(df):,}")

    # --- build the coarsened cell keys ---------------------------------
    if args.eth3:
        df["eth_cell"] = df["eth5"].astype("object").map(ETH3)
    else:
        df["eth_cell"] = df["eth5"].astype("object")
    if args.gender2:
        df["gender"] = df["gender"].astype("object").map(GENDER2)

    missing = [k for k in CELL_KEYS if k not in df.columns]
    if missing:
        raise KeyError(f"Dataset 1 is missing cell keys: {missing}")

    # Missing cell keys become an explicit "Unknown" level rather than being
    # dropped. Non-response on ethnicity is correlated with ethnicity AND with
    # activity, so dropping those rows biases every estimate upward. Keeping
    # them as Unknown preserves the London totals; you simply exclude the
    # Unknown cells from demographic breakdowns.
    n_unknown = {}
    for k in ["age4", "gender", "eth_cell"]:
        # Cast off categorical dtype first: pandas refuses to fillna a
        # Categorical with a level that is not already in its categories.
        df[k] = df[k].astype("object")
        n_unknown[k] = int(df[k].isna().sum())
        df[k] = df[k].fillna("Not disclosed")
    print("rows with a blank cell key, recoded to 'Not disclosed':")
    for k, v in n_unknown.items():
        print(f"    {k:10s} {v:6,} ({100*v/len(df):.1f}%)")

    before = len(df)
    df = df.dropna(subset=["borough", "wave"])
    if before - len(df):
        print(f"dropped {before-len(df):,} rows with no borough/wave "
              f"({100*(before-len(df))/before:.2f}%)")

    cells = build_cells(df)

    # --- publication suppression flag (keep the row, flag it) ---
    cells["publishable"] = cells["n_cell"] >= args.min_cell
    # inverse-variance-ish training weight: reliability, not population size
    cells["train_weight"] = cells["n_cell"].astype(float)

    # --- borough-wave totals, useful as a stable aggregate feature ---
    bw = (cells.groupby(["borough", "wave"], observed=True)
                .apply(lambda g: pd.Series({
                    "borough_n": g["n_cell"].sum(),
                    "borough_p_active": np.average(
                        g["p_active"].fillna(0), weights=g["w_cell"].fillna(0) + 1e-9),
                }), include_groups=False)
                .reset_index())
    cells = cells.merge(bw, on=["borough", "wave"], how="left")

    # --- cell share of borough population (this is what GLA projections
    #     will replace when you forecast composition forward) ---
    cells["cell_share_of_borough"] = cells["w_cell"] / cells.groupby(
        ["borough", "wave"], observed=True)["w_cell"].transform("sum")

    cells = cells.sort_values(CELL_KEYS).reset_index(drop=True)
    cells.to_parquet(args.out, index=False)

    # ---- report ----
    print(f"\ncells: {len(cells):,}")
    print(f"expected if fully crossed: "
          f"{cells.borough.nunique()} x {cells.wave.nunique()} x "
          f"{cells.age4.nunique()} x {cells.gender.nunique()} x "
          f"{cells.eth_cell.nunique()} = "
          f"{cells.borough.nunique()*cells.wave.nunique()*cells.age4.nunique()*cells.gender.nunique()*cells.eth_cell.nunique():,}")
    print("\nn_cell distribution:")
    print(cells["n_cell"].describe(
        percentiles=[.05, .1, .25, .5, .75, .95]).round(1).to_string())
    print(f"\ncells with n < {args.min_cell}: "
          f"{(~cells['publishable']).sum():,} "
          f"({100*(~cells['publishable']).mean():.1f}%)  -> suppress on output")
    print(f"cells with n == 0: {(cells['n_cell'] == 0).sum():,}")
    print("\nmedian n_cell by wave:")
    print(cells.groupby("wave")["n_cell"].median().to_string())
    print("\nLondon weighted p_active by wave (from cells):")
    lon = (cells.groupby("wave")
           .apply(lambda g: np.average(g["p_active"].fillna(0),
                                       weights=g["w_cell"].fillna(0) + 1e-9),
                  include_groups=False))
    print((100 * lon).round(1).to_string())
    print("\nRECONCILIATION — cell panel vs individual file:")
    ind = (df[df["activity_band"].notna() & df["wt_final"].notna()]
           .groupby("wave")
           .apply(lambda g: 100 * (g["is_active"] * g["wt_final"]).sum()
                  / g["wt_final"].sum(), include_groups=False))
    rec = pd.DataFrame({"individual": ind, "cells": 100 * lon})
    rec["diff"] = (rec["cells"] - rec["individual"]).round(2)
    print(rec.round(2).to_string())
    if rec["diff"].abs().max() > 0.05:
        print("  !! the panel does not reconcile — rows are being lost")
    else:
        print("  OK: the panel reproduces the individual file exactly.")

    print(f"\nwritten to {args.out}")
    print("\nREMINDER: weight the model loss by train_weight (or 1/se_active^2). "
          "Never publish a cell where publishable == False.")


if __name__ == "__main__":
    main()
