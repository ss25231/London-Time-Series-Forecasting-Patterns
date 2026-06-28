"""
active_lives_utils.py — shared functions for all Active Lives London wave notebooks.

Import into each wave's notebook:
    from active_lives_utils import (
        LONDON_LA2021, decode, london_filter, weighted_share,
        weighted_distribution, check_routing, validate_wave,
    )

The per-wave codebook and CONFIG live in each wave's own notebook (they differ by year);
the logic here is wave-agnostic and never changes.
"""
import pandas as pd
import numpy as np

# Verified London borough map (LA_2021 code -> (GSS, name)). Stable across waves checked so far
# (2016-17 and 2022-23 both use this numbering). City of London (E09000001) excluded downstream.
LONDON_LA2021 = {
    9:("E09000002","Barking and Dagenham"), 8:("E09000003","Barnet"),
    17:("E09000004","Bexley"), 30:("E09000005","Brent"), 35:("E09000006","Bromley"),
    44:("E09000007","Camden"), 59:("E09000001","City of London"),
    69:("E09000008","Croydon"), 80:("E09000009","Ealing"), 94:("E09000010","Enfield"),
    110:("E09000011","Greenwich"), 112:("E09000012","Hackney"),
    115:("E09000013","Hammersmith and Fulham"), 117:("E09000014","Haringey"),
    120:("E09000015","Harrow"), 125:("E09000016","Havering"),
    129:("E09000017","Hillingdon"), 132:("E09000018","Hounslow"),
    139:("E09000019","Islington"), 138:("E09000020","Kensington and Chelsea"),
    143:("E09000021","Kingston upon Thames"), 146:("E09000022","Lambeth"),
    151:("E09000023","Lewisham"), 164:("E09000024","Merton"), 175:("E09000025","Newham"),
    201:("E09000026","Redbridge"), 206:("E09000027","Richmond upon Thames"),
    247:("E09000028","Southwark"), 261:("E09000029","Sutton"),
    278:("E09000030","Tower Hamlets"), 285:("E09000031","Waltham Forest"),
    286:("E09000032","Wandsworth"), 300:("E09000033","Westminster"),
}


def decode(df, config, codebook):
    """Add readable-label columns (concept name) decoded from each raw coded variable.
    Skips any concept whose variable is absent or unlabelled, so it tolerates waves that
    lack some variables."""
    out = df.copy()
    for concept, var in config.items():
        if var in codebook and var in out.columns:
            out[concept] = out[var].map(codebook[var])
        elif var in out.columns:
            out[concept] = out[var]
    if config.get("weight") in out.columns:
        out["weight"] = pd.to_numeric(out[config["weight"]], errors="coerce")
    return out


def london_filter(df, la_col="LA_2021", la_map=LONDON_LA2021):
    """Map LA codes to GSS + borough name and keep the 32 London boroughs (exclude City of London)."""
    out = df.copy()
    out["gss"] = out[la_col].map({k: v[0] for k, v in la_map.items()})
    out["LA_name"] = out[la_col].map({k: v[1] for k, v in la_map.items()})
    out = out[out["gss"].notna() & (out["gss"] != "E09000001")].copy()
    return out


def weighted_share(df, group_cols, target_col, target_value):
    """Weighted proportion target_col==target_value, with unweighted base n."""
    d = df.dropna(subset=["weight", target_col]).copy()
    d["_num"] = (d[target_col] == target_value).astype(float) * d["weight"]
    if group_cols:
        g = d.groupby(group_cols, dropna=False)
        out = (g["_num"].sum() / g["weight"].sum()).rename("weighted_share").reset_index()
        out["n"] = g.size().values
        return out
    return d["_num"].sum() / d["weight"].sum()


def weighted_distribution(df, group_col, target_col="activity_band"):
    """Weighted distribution across all categories of target_col, by group."""
    d = df.dropna(subset=["weight", target_col, group_col]).copy()
    tab = d.groupby([group_col, target_col])["weight"].sum().unstack(fill_value=0)
    return tab.div(tab.sum(axis=1), axis=0)


def check_routing(df, gate_col, gate_value, dependent_cols, label=""):
    """Routing integrity: among rows where gate_col==gate_value, how complete are dependents?
    ~0% missing = clean universal routing; high+uniform = rotation subsample."""
    cols = [c for c in dependent_cols if c in df.columns]
    sub = df[df[gate_col] == gate_value]
    if not cols or len(sub) == 0:
        print(f"{label or gate_col}=={gate_value!r}: no data"); return
    miss = sub[cols].isna().mean()
    all_blank = sub[cols].isna().all(axis=1).sum()
    print(f"{label or gate_col}=={gate_value!r} (n={len(sub)}): "
          f"max dep-missing={miss.max():.3f}, all-blank={all_blank} ({all_blank/len(sub):.1%})")


def validate_wave(df, wave_label, la_map=LONDON_LA2021, expected_active=None, tol=0.05,
                  need=("wt_final","LA_2021","MEMS7GR_SPORTCOUNT_A01","Eth7","Disab3","Age9")):
    """Value-level checks that renaming columns does NOT guarantee. Prints PASS/CHECK and
    returns the list of issues (empty = clean). Run FIRST on any new wave."""
    issues = []
    miss = [c for c in need if c not in df.columns]
    if miss:
        issues.append(f"MISSING COLUMNS: {miss}")

    coded = [c for c in ["MEMS7GR_SPORTCOUNT_A01","Eth7","Disab3"] if c in df.columns]
    if coded and (df[coded] < 0).any().any():
        issues.append("Negative survey codes present — strip -90..-99 before analysis")

    if "Disab3" in df.columns:
        lim = (df["Disab3"] == 1).mean()
        if not (0.05 < lim < 0.35):
            issues.append(f"Limiting-disability rate {lim:.0%} implausible — check Disab3 coding")

    if "LA_2021" in df.columns:
        cov = df["LA_2021"].map({k: v[0] for k, v in la_map.items()}).notna().mean()
        if cov < 0.95:
            issues.append(f"Only {cov:.0%} of rows map to a London borough — LA_2021 numbering differs; rebuild map")

    if expected_active is not None and {"MEMS7GR_SPORTCOUNT_A01","wt_final"} <= set(df.columns):
        d = df.dropna(subset=["MEMS7GR_SPORTCOUNT_A01","wt_final"])
        rate = ((d["MEMS7GR_SPORTCOUNT_A01"] == 2) * d["wt_final"]).sum() / d["wt_final"].sum()
        if abs(rate - expected_active) > tol:
            issues.append(f"Weighted active rate {rate:.1%} differs from expected {expected_active:.0%} by >{tol:.0%}")

    print(f"=== validate_wave: {wave_label} -> {'PASS' if not issues else 'CHECK NEEDED'} ===")
    for i in issues:
        print("  -", i)
    return issues
