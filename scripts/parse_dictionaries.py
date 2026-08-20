"""
parse_dictionaries.py — Step 0: turn the 8 UKDA data dictionaries into
machine-readable JSON.

Produces three outputs:

  dict_vars_by_wave.json   {wave: {varname: {label, level, values}}}
  la_labels_by_wave.json   {wave: {LA_column: {code: "E09000007 Camden"}}}
  dict_comparison.csv      one row per variable, showing which waves carry it
                           and whether its value labels changed between waves

The LA lookup is the critical one: numeric LA codes are NOT stable across
LA vintages, so borough identity must be resolved through the per-wave label
table, never by assuming code 114 means the same place in Y1 and Y8.

Handles both .docx and .rtf dictionaries.

Run:
    python parse_dictionaries.py /path/to/dictionaries -o ./parsed
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


# --------------------------------------------------------------------------
# text extraction
# --------------------------------------------------------------------------

def extract_text(path: Path) -> str:
    """docx or rtf -> plain/markdown text. Tries extract-text, then pandoc."""
    if shutil.which("extract-text"):
        try:
            out = subprocess.run(["extract-text", str(path)],
                                 capture_output=True, text=True, timeout=600)
            if out.returncode == 0 and len(out.stdout) > 1000:
                return out.stdout
        except Exception:
            pass

    if shutil.which("pandoc"):
        fmt = "rtf" if path.suffix.lower() == ".rtf" else "docx"
        out = subprocess.run(["pandoc", str(path), "-f", fmt, "-t", "plain",
                              "--wrap=none"],
                             capture_output=True, text=True, timeout=900)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout

    # last resort for docx: read word/document.xml directly
    if path.suffix.lower() == ".docx":
        import zipfile
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "replace")
        xml = re.sub(r"</w:p>", "\n", xml)
        xml = re.sub(r"<w:tab/>", "\t", xml)
        return re.sub(r"<[^>]+>", "", xml)

    raise RuntimeError(f"Could not extract text from {path.name}. "
                       f"Install pandoc: apt-get install pandoc")


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

# The UKDA layout is consistent, but bold markers differ between the docx
# (markdown **) and rtf (plain) extraction paths, so both are optional.
VAR_RE = re.compile(
    r"(?:\*\*)?Pos\.\s*=\s*(?:\*\*)?\s*([\d,]+)\s*\t?\s*"
    r"(?:\*\*)?Variable\s*=\s*(?:\*\*)?\s*(\S+)\s*\t?\s*"
    r"(?:\*\*)?Variable label\s*=\s*(?:\*\*)?\s*(.*)")
LEVEL_RE = re.compile(r"measurement level is \*?(\w+)\*?")
VALUE_RE = re.compile(r"Value\s*=\s*(-?[\d.]+)\s*\t?\s*Label\s*=\s*(.*)")

WAVE_RE = re.compile(r"data_year_(\d+)", re.I)

# A real UKDA dictionary has thousands of variables. Anything below this
# means extraction or parsing broke, not that the file is small.
MIN_VARS = 500


def parse_dictionary(text: str) -> dict:
    """Split on Pos. markers and pull label / level / value labels."""
    parts = re.split(r"(?:\*\*)?Pos\.\s*=\s*", text)[1:]
    out = {}
    for p in parts:
        m = VAR_RE.match("Pos. = " + p) or VAR_RE.match(p)
        if not m:
            # try matching without the reconstructed prefix
            m2 = re.match(r"(?:\*\*)?([\d,]+)\s*\t?\s*(?:\*\*)?Variable\s*=\s*"
                          r"(?:\*\*)?(\S+)\s*\t?\s*(?:\*\*)?Variable label\s*=\s*"
                          r"(?:\*\*)?(.*)", p)
            if not m2:
                continue
            pos, name, label = m2.groups()
        else:
            pos, name, label = m.groups()

        lvl = LEVEL_RE.search(p)
        vals = [(float(v), lab.strip()) for v, lab in VALUE_RE.findall(p)]
        out[name] = {
            "pos": int(pos.replace(",", "")),
            "label": label.strip().rstrip("*"),
            "level": lvl.group(1) if lvl else "",
            "values": vals,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder containing the 8 dictionaries")
    ap.add_argument("-o", "--out", default="./parsed")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue past a dictionary that fails to parse")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Permissive discovery: converted files may have picked up a different
    # name or a doubled extension (foo.rtf.docx) during conversion.
    files = [f for f in sorted(Path(args.folder).iterdir())
             if f.suffix.lower() in (".docx", ".rtf", ".doc")
             and not f.name.startswith("~$")]
    if not files:
        sys.exit(f"No .docx/.rtf dictionaries found in {args.folder}")
    print(f"found {len(files)} dictionary file(s):")
    for f in files:
        print(f"   {f.name}")
    if len(files) != 8:
        print(f"  !! expected 8, found {len(files)} — check the folder")
    print()

    vars_by_wave, la_by_wave = {}, {}

    for f in files:
        m = WAVE_RE.search(f.name)
        if not m:
            print(f"  ! cannot infer wave from {f.name}, skipping")
            continue
        wave = int(m.group(1))
        print(f"parsing wave {wave}: {f.name} ...", flush=True)

        text = extract_text(f)
        v = parse_dictionary(text)

        if len(v) < MIN_VARS:
            dump = Path(args.out) / f"FAILED_wave{wave}_extract_sample.txt"
            dump.write_text(text[:20000], encoding="utf-8")
            print(f"\n  *** PARSE FAILED for {f.name} ***")
            print(f"      only {len(v)} variables found (expected >{MIN_VARS}).")
            print(f"      The conversion probably changed the layout.")
            print(f"      A 20KB sample of the extracted text was written to:")
            print(f"        {dump}")
            print(f"      Send me that file and I will fix the parser.\n")
            if not args.keep_going:
                sys.exit("Stopping. Re-run with --keep-going to skip bad files.")
            continue
        vars_by_wave[wave] = v

        # LA lookups: every column whose value labels look like GSS codes
        la = {}
        for name, d in v.items():
            if not name.startswith("LA") and name != "xStrata":
                continue
            gss = {str(code): lab for code, lab in d["values"]
                   if re.match(r"^E\d{8}", lab)}
            if gss:
                la[name] = gss
        la_by_wave[wave] = la
        print(f"    {len(v):,} variables; LA columns with GSS labels: "
              f"{list(la.keys()) or '*** NONE — borough mapping will fail ***'}")
        if la:
            n_ldn = len({g for lut in la.values() for g in lut.values()
                         if g.startswith("E09")})
            print(f"    London (E09) authorities labelled: {n_ldn}")

    if not vars_by_wave:
        sys.exit("No dictionaries parsed successfully.")

    counts = {w: len(v) for w, v in vars_by_wave.items()}
    med = sorted(counts.values())[len(counts) // 2]
    for w, c in sorted(counts.items()):
        flag = "  !! far below the others" if c < 0.5 * med else ""
        print(f"  wave {w}: {c:,} variables{flag}")

    (outdir / "dict_vars_by_wave.json").write_text(json.dumps(vars_by_wave))
    (outdir / "la_labels_by_wave.json").write_text(json.dumps(la_by_wave, indent=1))

    # ---------------- cross-wave comparison ----------------
    waves = sorted(vars_by_wave)
    allvars = sorted({n for w in waves for n in vars_by_wave[w]})

    with open(outdir / "dict_comparison.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["variable", "in_waves", "n_waves", "first_wave", "last_wave",
                     "label_y_latest", "value_scheme_stable", "schemes_seen"])
        for n in allvars:
            present = [w for w in waves if n in vars_by_wave[w]]
            schemes = {}
            for w in present:
                sig = "|".join(
                    f"{c:g}={l}" for c, l in vars_by_wave[w][n]["values"] if c > -90)
                schemes.setdefault(sig, []).append(w)
            wr.writerow([
                n, ";".join(map(str, present)), len(present),
                present[0], present[-1],
                vars_by_wave[present[-1]][n]["label"],
                "YES" if len(schemes) == 1 else "NO",
                " || ".join(f"[w{','.join(map(str, ws))}] {s[:180]}"
                            for s, ws in schemes.items()),
            ])

    stable = sum(1 for n in allvars
                 if len({"|".join(f"{c:g}={l}" for c, l in vars_by_wave[w][n]["values"] if c > -90)
                         for w in waves if n in vars_by_wave[w]}) == 1)
    print(f"\n--- summary ---")
    print(f"waves parsed:            {waves}")
    print(f"distinct variables:      {len(allvars):,}")
    print(f"in every wave:           "
          f"{sum(1 for n in allvars if all(n in vars_by_wave[w] for w in waves)):,}")
    print(f"stable value scheme:     {stable:,}")
    print(f"\nwritten to {outdir}/")
    print("Upload dict_comparison.csv — that is the file that tells me exactly "
          "what changed between waves.")


if __name__ == "__main__":
    main()
