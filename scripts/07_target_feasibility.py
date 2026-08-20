"""
07_target_feasibility.py — can we actually model these targets?

Before spending a day running propensity + ablation across five targets,
check whether each one has enough signal to say anything.

An ablation asks "how much of the BETWEEN-BOROUGH spread do demographics
reproduce?". That question is only meaningful if the observed spread is
bigger than the noise. With 195 respondents per borough and a 17% base
rate, sampling error alone produces ~2.7pp of apparent borough variation
— and a model can happily "explain" 50% of pure noise.

So for each target this reports:
  * effective base (non-missing), overall and in the holdout wave
  * prevalence
  * respondents per borough
  * observed borough spread
  * EXPECTED spread from sampling noise alone
  * excess spread = sqrt(observed^2 - noise^2)   <- the real signal
  * signal-to-noise ratio, and a verdict

Run:
    python 07_target_feasibility.py dataset1_individual.parquet
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.facecolor": "white", "font.size": 9,
                     "savefig.bbox": "tight"})

# target -> (label, brief section)
TARGETS = {
    "is_active":            ("Activity: 150+ mins/wk", "1. Activity levels"),
    # REBASED versions (share of all adults) — these are the modellable ones.
    "club_any_all":         ("Club membership (rebased)", "1c. Club membership"),
    "club_fitness_all":     ("Club: fitness (rebased)", "1c"),
    "club_team_all":        ("Club: team sports (rebased)", "1c"),
    "vol_any":              ("Volunteering in sport", "4. Volunteering"),
    "vol_coach_all":        ("Volunteer: coaching (rebased)", "4"),
    "vol_organise_all":     ("Volunteer: organising (rebased)", "4"),
    "active_indoor":        ("Active indoors", "3. Indoor/outdoor"),
    "active_outdoor":       ("Active outdoors", "3"),
    "active_park":          ("Active in parks", "3"),
    "active_leisurecentre": ("Active at leisure centre", "3"),
    "part_walking":         ("Participates: walking", "2b. Activity types"),
    "part_fitness":         ("Participates: fitness", "2b"),
    "part_team":            ("Participates: team sport", "2b"),
    "part_cycling":         ("Participates: cycling", "2b"),
    "part_racket":          ("Participates: racket", "2b"),
    "part_dance":           ("Participates: dance", "2b"),
}


def assess(df, col, test_wave):
    s = pd.to_numeric(df[col], errors="coerce")
    ok = s.notna() & df.wt_final.notna()
    if ok.sum() < 500:
        return None

    d = df[ok].copy(); d["_t"] = s[ok]
    te = d[d.wave == test_wave]
    if len(te) < 200:
        te = d[d.wave == d.wave.max()]

    prev = (te.wt_final * te._t).sum() / te.wt_final.sum()

    g = te.groupby("borough", observed=True).apply(lambda x: pd.Series({
        "n": len(x),
        "p": (x.wt_final * x._t).sum() / x.wt_final.sum()}),
        include_groups=False)
    g = g[g.n >= 30]
    if len(g) < 15:
        return None

    obs_sd = g.p.std() * 100
    # expected SD from binomial sampling alone, given each borough's n
    noise_sd = float(np.sqrt(np.mean(prev * (1 - prev) / g.n))) * 100
    excess = float(np.sqrt(max(obs_sd ** 2 - noise_sd ** 2, 0)))
    snr = excess / noise_sd if noise_sd > 0 else np.nan

    waves = sorted(d.wave.unique())
    return {
        "target": col,
        "label": TARGETS[col][0],
        "brief": TARGETS[col][1],
        "waves": f"{min(waves)}-{max(waves)}",
        "n_total": int(ok.sum()),
        "coverage_%": round(100 * ok.mean(), 1),
        "n_holdout": len(te),
        "n_per_borough": int(g.n.median()),
        "prevalence_%": round(100 * prev, 1),
        "obs_spread_pp": round(obs_sd, 2),
        "noise_spread_pp": round(noise_sd, 2),
        "excess_spread_pp": round(excess, 2),
        "signal_noise": round(snr, 2),
    }


def wave_profile(df, col):
    """Per-wave coverage and prevalence. Detects two distinct problems:
      1. the column is ABSENT in a wave (structural)
      2. the ANSWERING BASE changes between waves (rotation/routing change)
    Both break naive pooling across waves, but in different ways."""
    s = pd.to_numeric(df[col], errors="coerce")
    rows = []
    for w, g in df.assign(_t=s).groupby("wave"):
        ok = g._t.notna() & g.wt_final.notna()
        cov = 100 * ok.mean()
        prev = (100 * (g.wt_final[ok] * g._t[ok]).sum() / g.wt_final[ok].sum()
                if ok.sum() > 30 else np.nan)
        rows.append({"wave": w, "coverage": cov, "prevalence": prev,
                     "n": int(ok.sum())})
    return pd.DataFrame(rows).set_index("wave")


def detect_breaks(wp, cov_jump=15.0, prev_jump=5.0):
    """Flag waves where coverage or prevalence shifts abruptly."""
    notes, usable = [], []
    for w, r in wp.iterrows():
        if r.n < 200:
            notes.append(f"w{w}: absent/too sparse (n={r.n})")
        else:
            usable.append(w)
    cov = wp.loc[usable, "coverage"] if usable else pd.Series(dtype=float)
    prev = wp.loc[usable, "prevalence"] if usable else pd.Series(dtype=float)
    for i in range(1, len(usable)):
        a, b = usable[i - 1], usable[i]
        dc = cov[b] - cov[a]
        dp = prev[b] - prev[a]
        if abs(dc) > cov_jump:
            notes.append(f"w{a}->w{b}: BASE SHIFT {dc:+.0f}pp coverage")
        if abs(dp) > prev_jump:
            notes.append(f"w{a}->w{b}: LEVEL JUMP {dp:+.1f}pp prevalence")
    return usable, notes


def verdict(r):
    if r["signal_noise"] >= 2.0:
        return "STRONG  — ablation will be meaningful"
    if r["signal_noise"] >= 1.0:
        return "USABLE  — pool waves to tighten"
    if r["signal_noise"] >= 0.5:
        return "WEAK    — London-level only, not borough"
    return "NO      — borough spread is mostly noise"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("-o", "--out", default="model_out")
    ap.add_argument("--test-wave", type=int, default=8)
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    print(f"individuals: {len(df):,}\n")

    # ---- 1. feature availability: is any FEATURE a wave proxy? ----
    print("=" * 108)
    print("FEATURE AVAILABILITY BY WAVE — a feature absent in early waves")
    print("becomes a disguised wave indicator once missing values are filled.")
    print("=" * 108)
    FEATS = ["age4", "gender", "eth5", "nssec4", "educ3", "health",
             "Disab3", "BMIG", "disty2_POP", "disty5_POP"]
    fa = {}
    for c in FEATS:
        if c not in df.columns:
            continue
        cov = df.groupby("wave")[c].apply(lambda s: round(100 * s.notna().mean()))
        fa[c] = cov
    fa = pd.DataFrame(fa).T
    print(fa.to_string())
    confounded = [c for c in fa.index if (fa.loc[c] < 5).any()]
    if confounded:
        print(f"\n  !! WAVE-CONFOUNDED FEATURES: {confounded}")
        print("     These are ~0% present in at least one wave. If missing")
        print("     values are filled with a category, the model can read")
        print("     that category as 'this respondent is from an early wave'.")
        print("     FIX: either restrict training to waves where the feature")
        print("     exists, or drop the feature. Do not silently fill.")
    else:
        print("\n  OK: no feature is wholly absent from any wave.")

    # ---- 2. per-target wave profile ----
    print("\n" + "=" * 108)
    print("TARGET STABILITY ACROSS WAVES")
    print("=" * 108)
    windows = {}
    for col in TARGETS:
        if col not in df.columns:
            continue
        wp = wave_profile(df, col)
        usable, notes = detect_breaks(wp)
        windows[col] = usable
        flag = "  <-- CHECK" if notes else ""
        print(f"\n{TARGETS[col][0]}{flag}")
        print("   coverage% " + " ".join(f"w{w}:{wp.coverage[w]:5.1f}"
                                         for w in wp.index))
        print("   prevalnc% " + " ".join(
            f"w{w}:{wp.prevalence[w]:5.1f}" if pd.notna(wp.prevalence[w])
            else f"w{w}:   --" for w in wp.index))
        for n in notes:
            print(f"     ! {n}")
        if usable:
            print(f"   -> usable waves: {usable[0]}-{usable[-1]} "
                  f"({len(usable)} waves)")

    rows = []
    for col in TARGETS:
        if col not in df.columns:
            print(f"  !! {col:22s} NOT IN DATASET 1")
            continue
        r = assess(df, col, args.test_wave)
        if r is None:
            print(f"  !! {col:22s} too sparse to assess")
            continue
        r["usable_waves"] = (f"{windows[col][0]}-{windows[col][-1]}"
                             if windows.get(col) else "none")
        r["n_usable_waves"] = len(windows.get(col, []))
        rows.append(r)

    out = pd.DataFrame(rows).sort_values("signal_noise", ascending=False)
    out["verdict"] = out.apply(verdict, axis=1)

    print("=" * 108)
    print("TARGET FEASIBILITY — is there enough borough signal to model?")
    print("=" * 108)
    show = ["label", "usable_waves", "coverage_%", "n_per_borough", "prevalence_%",
            "obs_spread_pp", "noise_spread_pp", "excess_spread_pp",
            "signal_noise", "verdict"]
    print(out[show].to_string(index=False))
    print("=" * 108)
    print("""
READ THIS CAREFULLY:
  obs_spread    = how much boroughs appear to differ
  noise_spread  = how much they would differ from sampling error ALONE
  excess_spread = sqrt(obs^2 - noise^2) — the real between-borough signal
  signal_noise  = excess / noise

  A target with signal_noise < 0.5 has borough differences that are mostly
  sampling error. Running an ablation on it produces a number, and that
  number is meaningless. Model those at London or demographic level only.
""")

    fig, ax = plt.subplots(figsize=(9, .42 * len(out) + 1.8))
    y = np.arange(len(out))[::-1]
    ax.barh(y, out.obs_spread_pp, height=.68, color="#bdc3c7",
            label="observed spread")
    ax.barh(y, out.excess_spread_pp, height=.68, color="#2c7fb8",
            label="real signal (excess over noise)")
    ax.plot(out.noise_spread_pp, y, "|", ms=16, color="#c0392b", mew=2.2,
            label="sampling noise floor")
    ax.set_yticks(y); ax.set_yticklabels(out.label, fontsize=8.5)
    ax.set_xlabel("percentage points of between-borough spread")
    ax.set_title("How much borough variation is real, and how much is noise?",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.25, axis="x"); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outdir / "fig9_target_feasibility.png", dpi=150)
    out.to_csv(outdir / "target_feasibility.csv", index=False)
    print(f"written: {outdir}/fig9_target_feasibility.png, target_feasibility.csv")


if __name__ == "__main__":
    main()
