"""
Codebook generator for Active Lives waves.

Reads a wave's .sav value labels and emits a clean {variable: {code: label}} codebook,
so codebook entries are correct BY CONSTRUCTION rather than hand-typed. Hand-typing was
wrong five times in this project (Disab3, NSSEC5, Educ6, Eth7, volunteer roles); reading
the labels from the file eliminates that entire class of error.

Usage in a notebook:
    from codebook_gen import build_codebook, compare_codebooks, missing_codes_from_labels
    cb, vol_roles = build_codebook(SAV_PATH, CONCEPT_VARS, ROLE_VARS)
    compare_codebooks(BASELINE_CB, cb)        # flags any drift vs the 2022-23 baseline
"""
import pyreadstat


def build_codebook(sav_path, concept_vars, role_vars=None, drop_negative=True):
    """
    Read value labels (metadata only — fast even on large files) and emit:
      - codebook: {var: {code: label}} for each var in concept_vars that has labels
      - role_labels: {var: question_text} for role_vars (uses the column LABEL, not value labels)
      - missing_codes: the set of negative codes used as missing across the file

    drop_negative=True excludes negative survey codes (-90..-99) from the value maps,
    since those are missing markers, not real categories.
    """
    _, meta = pyreadstat.read_sav(sav_path, metadataonly=True)
    vlabels = meta.variable_value_labels
    clabels = meta.column_names_to_labels

    codebook = {}
    for var in concept_vars:
        if var in vlabels:
            mapping = {int(k): v for k, v in vlabels[var].items()
                       if not (drop_negative and k < 0)}
            codebook[var] = mapping

    role_labels = {}
    if role_vars:
        for var in role_vars:
            role_labels[var] = clabels.get(var, var)

    # Collect the negative (missing) codes actually present anywhere
    missing = set()
    for var, m in vlabels.items():
        for k in m:
            if k < 0:
                missing.add(int(k))

    return codebook, role_labels, sorted(missing)


def compare_codebooks(baseline, candidate, label="candidate"):
    """
    Compare a freshly-generated codebook against the baseline (2022-23) and report drift.
    Returns a list of differences; empty list = identical coding.
    This is the guard: if a new wave codes a variable differently, you see it here BEFORE analysing.
    """
    diffs = []
    for var, base_map in baseline.items():
        if var not in candidate:
            diffs.append(f"{var}: MISSING from {label}")
            continue
        cand_map = candidate[var]
        # codes present in one but not the other
        only_base = set(base_map) - set(cand_map)
        only_cand = set(cand_map) - set(base_map)
        if only_base:
            diffs.append(f"{var}: codes in baseline but not {label}: {sorted(only_base)}")
        if only_cand:
            diffs.append(f"{var}: codes in {label} but not baseline: {sorted(only_cand)}")
        # same code, different label
        for code in set(base_map) & set(cand_map):
            if str(base_map[code]).strip() != str(cand_map[code]).strip():
                diffs.append(f"{var}[{code}]: baseline={base_map[code]!r} vs {label}={cand_map[code]!r}")

    print(f"=== codebook comparison: baseline vs {label} -> "
          f"{'IDENTICAL' if not diffs else str(len(diffs)) + ' DIFFERENCE(S)'} ===")
    for d in diffs:
        print("  -", d)
    return diffs


if __name__ == "__main__":
    # Self-test against the synthetic test.sav
    CONCEPTS = ["MEMS7GR_SPORTCOUNT_A01", "Eth7", "Disab3", "Age9"]
    cb, roles, missing = build_codebook("test.sav", CONCEPTS)
    print("Generated codebook:")
    for v, m in cb.items():
        print(" ", v, "->", m)
    print("Missing codes found:", missing)

    # Simulate a baseline that DIFFERS (wrong Disab3 + renamed Eth code) to prove the comparison works
    fake_baseline = {
        "Disab3": {1: "No disability", 2: "Non-limiting disability", 3: "Limiting disability"},  # reversed!
        "Eth7":   {1: "White British", 2: "White Other", 3: "South Asian", 4: "Black",
                   5: "Chinese", 6: "Mixed", 7: "Other ethnic group"},  # code 3 mislabelled
        "Age9":   {1: "14-15", 2: "16-24", 3: "25-34", 4: "35-44", 5: "45-54",
                   6: "55-64", 7: "65-74", 8: "75-84", 9: "85+"},  # identical
    }
    print()
    compare_codebooks(fake_baseline, cb, "generated-from-sav")
