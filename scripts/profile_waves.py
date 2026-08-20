"""
profile_waves.py — Step 1 of the Active Lives London merge.

Reads ONLY the headers (plus a tiny sample) of every wave file and writes a
compact manifest describing which columns exist in which wave. Never loads a
full file, so it runs in seconds on 8 x 30MB.

Usage:
    python profile_waves.py /path/to/folder_with_csvs  -o wave_manifest.json

Then upload wave_manifest.json (small, ~50-200KB) rather than the CSVs.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# All eight CSVs are named "y1" regardless of wave, so the wave number must
# come from the YEAR RANGE, which is the only unambiguous token in the name.
# e.g. active_lives_london_y1_2019-20.csv -> wave 5
YEAR_TO_WAVE = {
    "2015-16": 1, "2016-17": 2, "2017-18": 3, "2018-19": 4,
    "2019-20": 5, "2020-21": 6, "2021-22": 7, "2022-23": 8,
}
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


def profile_file(path: Path, sample_rows: int = 2000) -> dict:
    """Header + light sample. Nothing else is read."""
    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()

    # Small sample, all columns as object first so nothing coerces oddly.
    sample = pd.read_csv(path, nrows=sample_rows, low_memory=False)

    col_info = {}
    for c in cols:
        s = sample[c] if c in sample.columns else pd.Series(dtype=float)
        num = pd.to_numeric(s, errors="coerce")
        non_null = num.dropna()
        # Distinct non-missing-code values, capped — tells us the coding scheme.
        valid = non_null[non_null > -90]
        uniq = sorted(valid.unique().tolist())[:25] if len(valid) else []
        col_info[c] = {
            "dtype": str(s.dtype),
            "pct_null_in_sample": round(100 * float(s.isna().mean()), 1) if len(s) else None,
            "pct_negcode_in_sample": (
                round(100 * float((non_null <= -90).mean()), 1) if len(non_null) else None
            ),
            "min": float(valid.min()) if len(valid) else None,
            "max": float(valid.max()) if len(valid) else None,
            "n_distinct_valid": int(valid.nunique()) if len(valid) else 0,
            "sample_values": [round(float(x), 4) for x in uniq],
        }

    # Cheap row count without loading the frame.
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        n_lines = sum(1 for _ in fh)

    return {
        "file": path.name,
        "wave": infer_wave(path),
        "approx_rows": n_lines - 1,
        "n_columns": len(cols),
        "columns": cols,
        "column_info": col_info,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder containing the 8 wave CSVs")
    ap.add_argument("-o", "--out", default="wave_manifest.json")
    ap.add_argument("--pattern", default="*.csv")
    args = ap.parse_args()

    files = sorted(Path(args.folder).glob(args.pattern))
    if not files:
        sys.exit(f"No files matching {args.pattern} in {args.folder}")

    waves = []
    for f in files:
        print(f"profiling {f.name} ...", flush=True)
        waves.append(profile_file(f))

    # Presence matrix: which columns appear in which waves.
    all_cols = sorted({c for w in waves for c in w["columns"]})
    presence = {
        c: [w.get("wave") or w["file"] for w in waves if c in w["columns"]]
        for c in all_cols
    }
    n_waves = len(waves)
    in_all = [c for c, ws in presence.items() if len(ws) == n_waves]
    partial = {c: ws for c, ws in presence.items() if len(ws) < n_waves}

    manifest = {
        "n_files": n_waves,
        "waves": waves,
        "columns_in_every_wave": in_all,
        "columns_partial": partial,
        "summary": {
            "total_distinct_columns": len(all_cols),
            "present_in_all": len(in_all),
            "present_in_some": len(partial),
        },
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    print("\n--- summary ---")
    print(f"files:                  {n_waves}")
    print(f"distinct columns:       {len(all_cols)}")
    print(f"in every wave:          {len(in_all)}")
    print(f"in only some waves:     {len(partial)}")
    print(f"\nwritten to {args.out}")
    print("Upload that file — it is all I need to build the crosswalk.")


if __name__ == "__main__":
    main()
