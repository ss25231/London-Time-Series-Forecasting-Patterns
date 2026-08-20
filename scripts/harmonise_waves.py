"""
harmonise_waves.py — Step 2: merge the 8 Active Lives London waves into one
harmonised individual-level file (Dataset 1).

Design principles
-----------------
* Never load a full 600-column wave. `usecols` is applied at read time.
* Missing codes are resolved BEFORE any recoding, using per-variable rules
  (-98 is a real zero for routed questions, NaN for rotated ones).
* Columns absent from a wave are created as NaN and flagged, never silently
  filled. A `wave_has_<block>` flag records availability so models can
  restrict their training window honestly.
* Every derived column is built by an explicit named function, so the
  provenance of each feature is auditable.

Run:
    python harmonise_waves.py /path/to/folder -o dataset1_individual.parquet
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. COLUMN CONTRACT — what we want out of every wave
# --------------------------------------------------------------------------

# Geography: whichever of these exists; first match wins (most recent first).
# Verified against the 8 dictionaries + manifest:
#   LA_2023 is present in waves 1-7, LA_2021 in wave 8. Both carry GSS labels.
#   LA_2023's integer codes are NOT identical between w1/w7 and w2-6, which is
#   exactly why every wave is mapped through its own label table.
LA_PRIORITY = ["LA_2023", "LA_2021", "LA_2020", "LA_2019", "LA_2015", "LA_2009"]

# wt_time / wt_online_time are NOT in any of the 8 London extracts - only
# wt_final. See the note in the handover about requesting time-series weights.
DESIGN = ["serial", "wt_final", "xStrata", "group", "mode", "Quarter"]

DEMOG = [
    "Age9", "Gend3", "Eth7", "Eth2", "Disab3", "Disab2_POP",
    "NSSEC5", "Educ6", "health", "BMIG", "disty2_POP", "disty5_POP",
]

# PRIMARY = MEMS7_SPORTCOUNT_A01, the "sport (count definition)" measure,
# which EXCLUDES gardening. MEMS7_ALL is the PHE variant that adds gardening
# back in; it is retained only as a sensitivity check, never as the target.
# Both are present in all 8 waves with stable value schemes.
ACTIVITY_TARGETS = [
    "MEMS7_SPORTCOUNT_A01",       # <- primary
    "MEMS7GR_SPORTCOUNT_A01",     # Sport England's own band, for cross-check
    "MEMS7GR7_SPORTCOUNT_A01",
    "MEMS7_ALL", "MEMS7GR_ALL",   # incl. gardening — sensitivity only
    "Number_Activities_150_Gr5", "Number_Activities_Gr5",
]

# C-level blocks only — they do not overlap. B06 kept because there is no
# C-level equivalent for gym/fitness.
BLOCKS = {
    "walking": "C01", "cycling": "C02", "active_travel": "C03",
    "dance": "C04", "team": "C05", "racket": "C06",
    "outdoor_water": "C07", "leisure_games": "C08", "combat_target": "C09",
    "fitness": "B06",
}
BLOCK_SUFFIX = {
    "C01": "WALKALL_C01", "C02": "CYCALL_C02", "C03": "ACTTRAV_C03",
    "C04": "DANCEALL_C04", "C05": "TEAMSPORT_C05", "C06": "RACKETSPORT_C06",
    "C07": "ADVWATERSPORT_C07", "C08": "LEISURE_C08", "C09": "COMBATTARGET_C09",
    "B06": "FITNESS_B06",
}

CLUB_RAW = [
    "club_any", "vol_any",
     "CLUB_SPORTCOUNT_A01", "Number_Club_Gr2", "Club_ExcFitness",
    "CLUB_FITNESS_B06", "CLUB_TEAMSPORT_C05", "CLUB_RUNATHMULTI_C11",
    "CLUB_RACKETSPORT_C06", "CLUB_COMBATTARGET_C09",
    "CLUB_CYCALL_C02", "CLUB_ADVWATERSPORT_C07", "CLUB_WALKALL_C01",
]

VOL_RAW = [
    "VolAny", "VolFrqB_Pop", "VolCnt_GR2", "VolDur_GR2", "VolLong_GR3",
] + [f"volint{i}_vol" for i in range(1, 8)]

# The CONTINUOUS overall in/out minutes (MEMS7_IN_SPORTCOUNT_A01) do not
# exist in any wave - only the GROUPED bands. So indoor/outdoor becomes a
# pair of "active in this setting" binaries, not a share of minutes.
# Available waves 2-8 (absent from wave 1).
INOUT_RAW = [
    "MEMS7GR_IN_SPORTCOUNT_A01", "MEMS7GR_OUT_SPORTCOUNT_A01",
    "MEMS7GR_IN_HOME_SPORTCOUNT_A01", "MEMS7GR_IN_LEISURE_SPORTCOUNT_A01",
    "MEMS7GR_OUT_LOCAL_PARK_SPORTCOUNT_A01",
    "MEMS7GR_OUT_LOCAL_ROAD_SPORTCOUNT_A01",
    "MEMS7GR_OUT_COUNTRYCOAST_SPORTCOUNT_A01",
]

DRIVERS = [
    "Motiva_POP", "motivb_POP", "READYAB1_POP", "READYOP1_POP",
    "inclus_a", "inclus_b", "inclus_c",
] + [f"limfreti{i}" for i in range(1, 6)]


def wanted_columns() -> list:
    cols = list(DESIGN) + LA_PRIORITY + DEMOG + ACTIVITY_TARGETS
    cols += CLUB_RAW + VOL_RAW + INOUT_RAW + DRIVERS
    for suf in BLOCK_SUFFIX.values():
        cols += [f"FREQUENCY_{suf}", f"FREQUENCYGR_{suf}",
                 f"DURATIONGR_{suf}", f"MONTHS_12_{suf}"]
    return sorted(set(cols))


# All eight CSVs are named "y1" regardless of wave, so the wave number must
# come from the YEAR RANGE, which is the only unambiguous token in the name.
# e.g. active_lives_london_y1_2019-20.csv -> wave 5
# Wave 1 = 2015-16; each subsequent survey year is the next wave. Generated
# rather than listed so that adding a future file (e.g. 2023-24) needs no
# code change — the mapping extends automatically.
_FIRST_START = 2015          # 2015-16 is wave 1
YEAR_TO_WAVE = {f"{y}-{str(y + 1)[2:]}": (y - _FIRST_START + 1)
                for y in range(_FIRST_START, _FIRST_START + 25)}
YEAR_RE = re.compile(r"(20\d{2}-\d{2})")


def infer_wave(path):
    """Wave from the year range in the filename. Never trust the 'yN' token."""
    m = YEAR_RE.search(path.name)
    if m and m.group(1) in YEAR_TO_WAVE:
        return YEAR_TO_WAVE[m.group(1)]
    raise ValueError(
        f"Cannot infer wave from {path.name!r}. Expected a year range like "
        f"2015-16 in the filename. The 'y1' token in these files is unreliable "
        f"- every wave is named y1."
    )


# --------------------------------------------------------------------------
# 2. MISSING CODES
# --------------------------------------------------------------------------

HARD_MISSING = [-99, -97, -96, -95, -94]  # always NaN

# -98 = "not applicable: survey routing". For these families it is a genuine
# zero (respondent was filtered out because they don't do the thing).
NEG98_IS_ZERO_PREFIXES = (
    "CLUB_", "volint", "MONTHS_12_", "FREQUENCY", "DURATION",
    "Number_Club", "Club_Exc", "VolCnt", "Number_Activities",
)

# For these, -98 means the question was never asked (rotation) -> keep NaN.
NEG98_IS_MISSING = set(DRIVERS) | set(INOUT_RAW) | {
    "VolDur_GR2", "VolLong_GR3", "VolFrqB_Pop",
}


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            continue
        s = df[c]
        s = s.mask(s.isin(HARD_MISSING))
        if c == "BMIG":
            s = s.mask(s == -1)
        if c in NEG98_IS_MISSING:
            s = s.mask(s == -98)
        elif c.startswith(NEG98_IS_ZERO_PREFIXES):
            s = s.where(s != -98, 0)
        else:
            s = s.mask(s == -98)
        df[c] = s
    return df


# --------------------------------------------------------------------------
# 3. HARMONISED RECODES
# --------------------------------------------------------------------------

# Eth7 -> 5 stable levels. Chinese(5) folds into Asian; Other(7) into Mixed/Other.
ETH5 = {1: "White British", 2: "White Other", 3: "Asian", 5: "Asian",
        4: "Black", 6: "Mixed/Other", 7: "Mixed/Other"}

# Age9 -> 4 cell-safe bands. Age9: 2=16-24,3=25-34,4=35-44,5=45-54,
# 6=55-64,7=65-74,8=75-84,9=85+
AGE4 = {2: "16-24", 3: "25-44", 4: "25-44", 5: "45-64",
        6: "45-64", 7: "65+", 8: "65+", 9: "65+"}

# CRITICAL: waves 1-4 carry a fifth NS-SEC code, "Aged <16 or 75+", which
# waves 5-8 do not (those respondents get a real NS-SEC instead). Mapping it
# to any substantive category would fabricate a socio-economic position and
# would bias the 65+ band differently before and after wave 5. It becomes its
# own explicit level so the discontinuity stays visible.
NSSEC4 = {1: "Higher", 2: "Middle", 3: "Lower", 4: "Student/Other",
          5: "Not classified (age)"}
EDUC3 = {1: "L4+", 2: "L1-3/Other", 3: "L1-3/Other", 4: "L1-3/Other",
         5: "L1-3/Other", 6: "None"}
GENDER = {1: "Male", 2: "Female", 3: "Other"}
ACT3 = {0: "Inactive", 1: "Fairly Active", 2: "Active"}



# --------------------------------------------------------------------------
# 3a. ACTIVITY BAND — derived from MEMS7_ALL, not inherited
# --------------------------------------------------------------------------
# Sport England's own MEMS7GR_ALL grouping is kept only as a cross-check.
# Deriving the band ourselves guarantees an identical definition in all
# eight waves regardless of any change they made to their grouping.
#     0-29 mins = Inactive | 30-149 = Fairly Active | 150+ = Active

BANDS = ["Inactive", "Fairly Active", "Active"]


def activity_band(mins: pd.Series) -> pd.Series:
    """Ordered categorical so ordinal models work and levels never sort
    alphabetically. Rounded before cutting so float dust at 149.9999
    cannot fall into the wrong band."""
    m = pd.to_numeric(mins, errors="coerce").round(0)
    b = pd.cut(m, bins=[-np.inf, 29, 149, np.inf], labels=BANDS)
    return b.cat.set_categories(BANDS, ordered=True)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    # --- keys ---
    # Age, gender and ethnicity were OPTIONAL disclosures in the survey.
    # A blank is a decision not to disclose, not a data-collection failure,
    # so it becomes an explicit level and is never imputed. The evidence in
    # this dataset is that non-disclosers are less active than average, so
    # imputing them would erase a real signal and bias every estimate.
    NOT_DISCLOSED = "Not disclosed"
    df["age4"] = df["Age9"].map(AGE4).fillna(NOT_DISCLOSED)
    df["gender"] = df["Gend3"].map(GENDER).fillna(NOT_DISCLOSED)
    df["eth5"] = df["Eth7"].map(ETH5).fillna(NOT_DISCLOSED)

    # Non-disclosure as a feature in its own right: how many of the three
    # optional demographics did this respondent decline to give?
    df["n_undisclosed"] = sum(
        (df[c] == NOT_DISCLOSED).astype(int)
        for c in ["age4", "gender", "eth5"])
    df["any_undisclosed"] = (df["n_undisclosed"] > 0).astype("float")
    df["nssec4"] = df["NSSEC5"].map(NSSEC4)
    df["educ3"] = df["Educ6"].map(EDUC3)
    # --- PRIMARY TARGET: sport-definition minutes (gardening EXCLUDED),
    #     and our own band derived from them ---
    mins = pd.to_numeric(df["MEMS7_SPORTCOUNT_A01"], errors="coerce")
    df["mins_per_week"] = mins.clip(lower=0)
    df["activity_band"] = activity_band(mins)

    df["is_active"] = (df["activity_band"] == "Active").astype("float").where(df["activity_band"].notna())
    df["is_fairly_active"] = (df["activity_band"] == "Fairly Active").astype("float").where(df["activity_band"].notna())
    df["is_inactive"] = (df["activity_band"] == "Inactive").astype("float").where(df["activity_band"].notna())
    # Sport England's own grouping of the SAME measure — cross-check only.
    df["band_sportengland"] = df["MEMS7GR_SPORTCOUNT_A01"].map(ACT3)

    # Gardening-inclusive variant, kept purely as a sensitivity check so you
    # can quantify how much of London's "active" share is gardening.
    mins_gard = pd.to_numeric(df["MEMS7_ALL"], errors="coerce")
    df["mins_incl_gardening"] = mins_gard.clip(lower=0)
    df["is_active_incl_gardening"] = (
        (mins_gard >= 150).astype("float").where(mins_gard.notna()))

    # --- rotation / base flags (so you can subset honestly) ---
    # NB asked_inout is set AFTER the in/out block below, since it depends
    # on a column derived there.
    def _asked(col):
        return df[col].notna() if col in df.columns else pd.Series(
            False, index=df.index)

    df["asked_freetime"] = _asked("limfreti2")
    df["asked_inclusion"] = _asked("inclus_a")

    # --- 10 blocks -> 2 binaries each ---
    for name, code in BLOCKS.items():
        suf = BLOCK_SUFFIX[code]
        freq, dur = f"FREQUENCY_{suf}", f"DURATIONGR_{suf}"
        df[f"part_{name}"] = (
            (df[freq] >= 2).astype("float").where(df[freq].notna())
            if freq in df else np.nan
        )
        df[f"meets150_{name}"] = (
            (df[dur] == 4).astype("float").where(df[dur].notna())
            if dur in df else np.nan
        )

    # --- club groupings ---
    def anyof(cols):
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return pd.Series(np.nan, index=df.index)
        sub = df[cols]
        return sub.max(axis=1).where(sub.notna().any(axis=1))

    # Headline binaries under the names Dataset 2 expects.
    df["club_any"] = anyof(["CLUB_SPORTCOUNT_A01"])
    df["vol_any"] = anyof(["VolAny"])

    df["club_fitness"] = anyof(["CLUB_FITNESS_B06"])
    df["club_team"] = anyof(["CLUB_TEAMSPORT_C05", "CLUB_RUNATHMULTI_C11"])
    df["club_racket_combat"] = anyof(["CLUB_RACKETSPORT_C06", "CLUB_COMBATTARGET_C09"])
    df["club_outdoor"] = anyof(["CLUB_CYCALL_C02", "CLUB_ADVWATERSPORT_C07", "CLUB_WALKALL_C01"])

    # --- volunteer role types (base = volunteers) ---
    df["vol_coach"] = anyof(["volint3_vol"])
    df["vol_officiate"] = anyof(["volint4_vol"])
    df["vol_organise"] = anyof(["volint1_vol", "volint5_vol"])
    df["vol_support"] = anyof(["volint2_vol", "volint6_vol", "volint7_vol"])

    # --- indoor / outdoor: grouped bands only, so "fairly active or better
    #     in this setting" binaries rather than shares of minutes ---
    for name, src in [
        ("active_indoor", "MEMS7GR_IN_SPORTCOUNT_A01"),
        ("active_outdoor", "MEMS7GR_OUT_SPORTCOUNT_A01"),
        ("active_home", "MEMS7GR_IN_HOME_SPORTCOUNT_A01"),
        ("active_leisurecentre", "MEMS7GR_IN_LEISURE_SPORTCOUNT_A01"),
        ("active_park", "MEMS7GR_OUT_LOCAL_PARK_SPORTCOUNT_A01"),
        ("active_road", "MEMS7GR_OUT_LOCAL_ROAD_SPORTCOUNT_A01"),
        ("active_countryside", "MEMS7GR_OUT_COUNTRYCOAST_SPORTCOUNT_A01"),
    ]:
        if src in df.columns:
            v = pd.to_numeric(df[src], errors="coerce")
            df[name] = (v >= 1).astype("float").where(v.notna())
        else:
            df[name] = np.nan
    df["asked_inout"] = df["active_indoor"].notna()

    # --- rebased targets: everything as a share of ALL ADULTS -------------
    # These are the columns to model. The un-rebased versions are retained
    # for reference but must never be pooled across waves.
    # Rebase to all adults — but ONLY in waves that actually asked. Filling
    # a wave that never carried the question with 0 would assert "nobody in
    # London belonged to a club in 2015-16", which is not a measurement.
    # Was the club question asked in this wave? Determined by whether the
    # source column carries any real response, not by a flag set later.
    asked_clubs = ("CLUB_SPORTCOUNT_A01" in df.columns
                   and pd.to_numeric(df["CLUB_SPORTCOUNT_A01"],
                                     errors="coerce").notna().any())
    for src in ["club_any", "club_fitness", "club_team",
                "club_racket_combat", "club_outdoor"]:
        if src in df.columns:
            v = pd.to_numeric(df[src], errors="coerce").fillna(0.0)
            df[f"{src}_all"] = v if asked_clubs else np.nan
    # Wave 2 used a far broader routing filter (59% coverage vs ~35%), so its
    # rebased rate is ~24.7% against 13-16% elsewhere. Not comparable.
    df["club_base_comparable"] = asked_clubs

    vol_known = df["vol_any"].notna() if "vol_any" in df.columns else None
    for src in ["vol_coach", "vol_officiate", "vol_organise", "vol_support"]:
        if src in df.columns:
            v = pd.to_numeric(df[src], errors="coerce").fillna(0.0)
            df[f"{src}_all"] = v.where(vol_known) if vol_known is not None else v

    # --- free-time constraints ---
    # Combined groupings (kept for backwards compatibility) ...
    df["limfreti_care"] = anyof(["limfreti1", "limfreti2"])
    df["limfreti_time"] = anyof(["limfreti3", "limfreti4"])
    # ... plus the individual reasons, so a "main reasons" breakdown can be
    # drawn. These follow the standard Active Lives free-time battery; adjust
    # FREETIME_LABELS in the plotting code if your dictionary differs.
    for i in range(1, 9):
        src = f"limfreti{i}"
        if src in df.columns:
            df[f"freetime{i}"] = pd.to_numeric(df[src], errors="coerce")

    # --- optional demographics: make non-disclosure explicit -------------
    NOT_DISC = "Not disclosed"
    disclosure_cols = ["age4", "gender", "eth5", "nssec4", "educ3"]
    flags = {}
    for c in disclosure_cols:
        if c in df.columns:
            flags[f"disclosed_{c}"] = df[c].notna().astype("float")
            df[c] = df[c].astype("object").fillna(NOT_DISC)
    if flags:
        df = pd.concat([df, pd.DataFrame(flags, index=df.index)], axis=1)
        # Non-disclosure is itself a behaviour and predicts activity, so
        # expose the count as a feature rather than burying it.
        df["n_undisclosed"] = (len(flags)
                               - pd.DataFrame(flags, index=df.index).sum(axis=1))

    return df


# --------------------------------------------------------------------------
# 4. GEOGRAPHY
# --------------------------------------------------------------------------

def resolve_borough(df: pd.DataFrame, la_labels: dict) -> pd.DataFrame:
    """Pick the most recent LA column present and map it to a GSS code.

    `la_labels` maps {la_column_name: {numeric_code: "E09000007 Camden"}} and
    comes from the UKDA data dictionary for that wave. Codes are NOT stable
    across LA vintages, which is exactly why the label lookup is per-column.
    """
    src = next((c for c in LA_PRIORITY if c in df.columns and df[c].notna().any()), None)
    if src is None:
        raise ValueError("No LA column found in this wave")
    lut = la_labels.get(src, {})
    lab = df[src].map(lut)
    gss = lab.str.extract(r"(E\d{8})", expand=False)
    df = pd.concat([df, pd.DataFrame({
        "borough_gss": gss,
        "borough": gss,
        "borough_name": lab.str.replace(r"^E\d{8}\s*", "", regex=True),
        "la_source_col": src,
    }, index=df.index)], axis=1)
    ldn = df["borough_gss"].str.startswith("E09", na=False)
    return df[ldn].copy()


# --------------------------------------------------------------------------
# 5. PER-WAVE LOAD + STACK
# --------------------------------------------------------------------------

def load_wave(path: Path, wave: int, la_labels: dict) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    want = [c for c in wanted_columns() if c in header]
    missing = [c for c in wanted_columns() if c not in header]

    df = pd.read_csv(path, usecols=want, low_memory=False)

    # Drop wholly blank rows (the Y8 file has ~6.5k of them).
    df = df.dropna(how="all")

    df = clean_missing(df)
    # Create absent columns in one shot — assigning them one at a time
    # fragments the frame badly (600 columns x 8 waves).
    if missing:
        df = pd.concat(
            [df, pd.DataFrame(np.nan, index=df.index, columns=missing)],
            axis=1)
    df = df.copy()                          # de-fragment before deriving
    df = resolve_borough(df, la_labels)
    df = add_derived(df).copy()

    df["wave"] = wave
    df["wave_label"] = f"Y{wave}"
    # Clubs: wave 2's routing filter was much broader than waves 3-8, so its
    # rebased figure is not comparable. Mask rather than silently pool.
    if wave <= 2:
        for c in [c for c in df.columns if c.startswith("club_")
                  and c.endswith("_all")]:
            df[c] = np.nan
    df["club_base_comparable"] = wave >= 3
    # Availability flags — which optional blocks this wave actually carried.
    # Verified availability (from wave_manifest.json):
    #   wave 1 has no clubs, no volunteering, no fitness/cycling blocks,
    #   no indoor/outdoor and no attitudes - it supports the activity target
    #   and demographics only.
    df["has_volunteering"] = "VolAny" in header
    df["has_clubs"] = "CLUB_SPORTCOUNT_A01" in header
    df["has_inout"] = "MEMS7GR_IN_SPORTCOUNT_A01" in header
    df["has_inclusion"] = "inclus_a" in header
    df["has_freetime"] = "limfreti1" in header
    df["has_health"] = "health" in header
    df["has_fitness_block"] = "FREQUENCY_FITNESS_B06" in header
    return df


KEEP_FINAL = (
    ["serial", "wave", "wave_label", "Quarter", "wt_final", "wt_time", "xStrata",
     "group", "mode", "borough", "borough_gss", "borough_name",
     "has_clubs", "has_fitness_block", "has_freetime", "has_health",
     "age4", "gender", "eth5", "nssec4", "educ3", "health", "BMIG",
     "n_undisclosed", "disclosed_age4", "disclosed_gender", "disclosed_eth5",
     "disclosed_nssec4", "disclosed_educ3",
     "n_undisclosed", "any_undisclosed",
     "Disab3", "disty2_POP", "disty5_POP",
     "activity_band", "is_active", "is_fairly_active", "is_inactive",
     "mins_per_week", "mins_winsor", "implausible_mins", "band_sportengland",
     "mins_incl_gardening", "is_active_incl_gardening",
     "MEMS7GR7_SPORTCOUNT_A01", "Number_Activities_150_Gr5",
     "club_any", "vol_any", "club_base_comparable",
     "club_any_all", "club_fitness_all", "club_team_all",
     "club_racket_combat_all", "club_outdoor_all",
     "vol_coach_all", "vol_officiate_all", "vol_organise_all",
     "vol_support_all",
     "CLUB_SPORTCOUNT_A01", "Number_Club_Gr2", "Club_ExcFitness",
     "club_fitness", "club_team", "club_racket_combat", "club_outdoor",
     "VolAny", "VolFrqB_Pop", "VolCnt_GR2",
     "vol_coach", "vol_officiate", "vol_organise", "vol_support",
     "active_indoor", "active_outdoor", "active_home",
     "active_leisurecentre", "active_park", "active_road",
     "active_countryside",
     "limfreti_care", "limfreti_time",
     "freetime1", "freetime2", "freetime3", "freetime4",
     "freetime5", "freetime6", "freetime7", "freetime8",
     "Motiva_POP", "READYAB1_POP", "READYOP1_POP",
     "inclus_a", "inclus_b", "inclus_c",
     "asked_inout", "asked_freetime", "asked_inclusion",
     "has_volunteering", "has_inout", "has_inclusion"]
    + [f"part_{n}" for n in BLOCKS]
    + [f"meets150_{n}" for n in BLOCKS]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--la-labels", required=True,
                    help="la_labels_by_wave.json from parse_dictionaries.py")
    ap.add_argument("-o", "--out", default="dataset1_individual.parquet")
    ap.add_argument("--mins-cap", type=float, default=2520.0,
                    help="ceiling for winsorising weekly minutes. 2520 = 6h/day "
                         "of moderate-equivalent activity. Sport England cap "
                         "individual COMPONENTS at 1680; applying that to the "
                         "TOTAL clips ~10%% of Londoners, which is too heavy.")
    args = ap.parse_args()

    la_labels = json.load(open(args.la_labels))   # la_labels_by_wave.json

    frames = []
    for p in sorted(Path(args.folder).glob("*.csv"), key=infer_wave):
        wave = infer_wave(p)
        wla = la_labels.get(str(wave)) or la_labels.get(wave)
        if wla is None:
            raise KeyError(f"No LA labels for wave {wave}. Run "
                           f"parse_dictionaries.py first.")
        wla = {k: {float(a): b for a, b in v.items()} for k, v in wla.items()}
        print(f"loading {p.name} (wave {wave}) ...", flush=True)
        frames.append(load_wave(p, wave, wla))

    df = pd.concat(frames, ignore_index=True, sort=False)

    # Winsorise minutes so one extreme respondent cannot dominate a small
    # cell mean in Dataset 2. The 99th percentile alone is far too permissive
    # here — MEMS7_ALL is uncapped and its top tail runs to implausible
    # values (vigorous activity counts double, and gardening is included).
    # We take the tighter of the 99th percentile and a substantive ceiling.
    m = df["mins_per_week"]

    # --- data-quality guard -------------------------------------------------
    # A week contains 10,080 minutes. MEMS counts vigorous activity double, so
    # values up to 20,160 are arithmetically possible but not credible.
    # Flag rather than delete, so the decision stays visible.
    WEEK_MINUTES = 10_080
    df["implausible_mins"] = (m > WEEK_MINUTES)
    n_imp = int(df["implausible_mins"].sum())
    if n_imp:
        print(f"\n!! {n_imp:,} respondents ({100*n_imp/len(df):.2f}%) report more "
              f"than {WEEK_MINUTES:,} minutes/week — physically impossible.")
        print("   Flagged as implausible_mins. They are still winsorised below,")
        print("   but consider excluding them from any continuous-minutes model.")

    print("\nminutes per week, distribution:")
    print(m.describe(percentiles=[.5, .75, .9, .95, .99, .999]).round(0).to_string())
    df = df.copy()
    q99 = m.quantile(0.99)
    cap = min(q99, args.mins_cap)
    df["mins_winsor"] = m.clip(upper=cap)
    print(f"  99th pct = {q99:.0f} | ceiling = {args.mins_cap} "
          f"| cap applied = {cap:.0f}")
    print(f"  respondents winsorised: {(m > cap).sum():,} "
          f"({100*(m > cap).mean():.2f}%)")
    df = df.copy()   # de-fragment after the last column addition

    df = df[[c for c in KEEP_FINAL if c in df.columns]]

    # Downcast to keep the stacked file small.
    df = df.loc[:, ~df.columns.duplicated()]   # guard against dupes in KEEP_FINAL
    for c in df.select_dtypes("float64").columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in ["age4", "gender", "eth5", "nssec4", "educ3",
              "activity_band", "borough_gss", "borough_name", "wave_label"]:
        if c in df:
            df[c] = df[c].astype("category")

    df.to_parquet(args.out, index=False)

    print(f"\nrows: {len(df):,}   columns: {df.shape[1]}")
    print("\nrespondents per wave:")
    print(df.groupby("wave").size().to_string())
    print("\nweighted % by band, per wave  << CHECK AGAINST SPORT ENGLAND >>")
    ok = df["activity_band"].notna() & df["wt_final"].notna()
    tab = (df[ok].groupby(["wave", "activity_band"], observed=True)["wt_final"]
           .sum().unstack())
    print((100 * tab.div(tab.sum(axis=1), axis=0)).round(1).to_string())

    if "is_active_incl_gardening" in df:
        print("\nHOW MUCH DIFFERENCE DOES GARDENING MAKE? (weighted % Active)")
        ok = df["wt_final"].notna()
        cmp_ = df[ok].groupby("wave").apply(lambda g: pd.Series({
            "excl_gardening": 100 * (g.is_active * g.wt_final).sum()
                              / g.wt_final[g.is_active.notna()].sum(),
            "incl_gardening": 100 * (g.is_active_incl_gardening * g.wt_final).sum()
                              / g.wt_final[g.is_active_incl_gardening.notna()].sum(),
        }), include_groups=False)
        cmp_["gardening_adds"] = cmp_.incl_gardening - cmp_.excl_gardening
        print(cmp_.round(1).to_string())

    print("\nour band vs Sport England's, agreement by wave:")
    if "band_sportengland" in df:
        for w, g in df.groupby("wave"):
            ct = pd.crosstab(g["activity_band"], g["band_sportengland"])
            if ct.size:
                agree = 100 * np.trace(ct.reindex(index=BANDS, columns=BANDS,
                                                  fill_value=0).values) / ct.values.sum()
                print(f"    wave {w}: {agree:.1f}%")
        print("    (expect ~100%; lower means light-activity is handled")
        print("     differently in their grouping — worth knowing, not a bug)")
    print("\nNON-DISCLOSURE OF OPTIONAL DEMOGRAPHICS (age / gender / ethnicity):")
    ok = df["activity_band"].notna() & df["wt_final"].notna()
    g = df[ok].groupby("n_undisclosed")
    dis = g.apply(lambda d: pd.Series({
        "n": len(d),
        "pct_of_sample": 100 * len(d) / ok.sum(),
        "pct_active": 100 * (d["is_active"] * d["wt_final"]).sum()
                      / d["wt_final"].sum(),
    }), include_groups=False)
    print(dis.round(1).to_string())
    print("  Blanks are respondent choices, not missing data. They are kept")
    print("  as an explicit 'Not disclosed' level and never imputed.")

    print("\nkey-variable coverage by wave (% non-missing):")
    for c in ["VolAny", "share_indoor", "limfreti_care", "nssec4", "eth5"]:
        if c in df:
            cov = df.groupby("wave", observed=True)[c].apply(lambda s: round(100 * s.notna().mean(), 1))
            print(f"  {c:16s} {cov.tolist()}")
    if "n_undisclosed" in df:
        print("\nNON-DISCLOSURE (optional items: age, gender, ethnicity, NS-SEC, education)")
        print("  % declining each item, by wave:")
        dc = [c for c in df.columns if c.startswith("disclosed_")]
        print((100 * (1 - df.groupby("wave")[dc].mean())).round(1).to_string())
        ok = df["is_active"].notna() & df["wt_final"].notna()
        g = df[ok].assign(any_undisc=df["n_undisclosed"] > 0)
        cmp_ = g.groupby("any_undisc").apply(
            lambda d: 100 * (d.is_active * d.wt_final).sum() / d.wt_final.sum(),
            include_groups=False)
        print("\n  weighted % Active — disclosed everything vs withheld something:")
        print(cmp_.round(1).rename({False: "disclosed all",
                                    True: "withheld >=1"}).to_string())
        print("  (a gap here means non-disclosure is informative, so these")
        print("   respondents must stay in the models, not be dropped)")

    print("\nREBASED TARGETS (% of ALL adults) — these should be stable:")
    reb = [c for c in ["club_any_all", "club_team_all", "vol_coach_all",
                       "vol_organise_all"] if c in df.columns]
    if reb:
        ok = df["wt_final"].notna()
        def _wm(g, c):
            m = g[c].notna()
            if m.sum() < 30:
                return np.nan
            return 100 * (g.loc[m, c] * g.loc[m, "wt_final"]).sum() \
                / g.loc[m, "wt_final"].sum()
        tab = df[ok].groupby("wave").apply(
            lambda g: pd.Series({c: _wm(g, c) for c in reb}),
            include_groups=False)
        print(tab.round(1).to_string())
        print("  NaN = not collected, or base not comparable. Club columns are")
        print("  masked in waves 1-2 (w1 not asked; w2 used a much broader")
        print("  routing filter, giving ~24.7% against 13-16% in waves 3-8).")

    print("\nWAVE-LEVEL AVAILABILITY (what each wave can actually support):")
    flags = [c for c in ["has_clubs", "has_volunteering", "has_inout",
                         "has_freetime", "has_inclusion", "has_health",
                         "has_fitness_block"] if c in df.columns]
    if flags:
        print(df.groupby("wave")[flags].max().astype(int).to_string())
        print("  0 = the wave does not carry that family at all; any model")
        print("      using it must restrict its training window accordingly.")

    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
