"""
run_all.py — run the entire project end to end, in order, with timing.

For anyone inheriting this codebase: put your input folders in place (see
below), then run

    python run_all.py

and wait. Every stage runs in dependency order, each is timed, and a summary
table with the total runtime is printed at the end. Nothing else is required.

------------------------------------------------------------------------------
EXPECTED INPUTS (edit the CONFIG block below if your folders are named
differently):

    dictionaries/     the eight data-dictionary files
    csvs/             the eight annual Active Lives CSV extracts
    parsed/           created by the pipeline (dictionary output lands here)

OUTPUTS: each stage writes to its own folder (model_out/, forecast_out/,
stageA_out/, ... ). Figures, CSVs and the model scorecard all land there.
------------------------------------------------------------------------------

Options:
    python run_all.py --quick        # fast pass: small bootstrap (rough
                                       intervals, same point estimates)
    python run_all.py --skip 05 09   # skip stages by number/name
    python run_all.py --from 09      # start partway through (inputs must exist)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------- CONFIG
# Folder names for the raw inputs. Change here if yours differ.
DICT_DIR = "dictionaries"
CSV_DIR = "csvs"
PARSED_DIR = "parsed"
INDIV = "dataset1_individual.parquet"
CELLS = "dataset2_cells.parquet"
LA_LABELS = f"{PARSED_DIR}/la_labels_by_wave.json"
SCRIPTS = "scripts"          # where the .py files live; "." if alongside


def script(name):
    """Resolve a script path whether scripts sit in ./scripts or alongside."""
    p = Path(SCRIPTS) / name
    return str(p if p.exists() else name)


# Each stage: (id, description, command-as-list, produces-a-file-to-check)
def build_plan(quick):
    boot = ["--boot", "5"] if quick else []
    return [
        ("parse", "Parse the data dictionaries",
         [script("parse_dictionaries.py"), DICT_DIR, "-o", PARSED_DIR], None),

        ("profile", "Profile the waves (variable availability, breaks)",
         [script("profile_waves.py"), CSV_DIR, "-o",
          f"{PARSED_DIR}/wave_manifest.json"], None),

        ("harmonise", "Harmonise eight waves -> individual dataset",
         [script("harmonise_waves.py"), CSV_DIR, "--la-labels", LA_LABELS,
          "-o", INDIV], INDIV),

        ("dataset2", "Build the aggregated cell panel",
         [script("04_build_dataset2.py"), INDIV, "-o", CELLS], CELLS),

        ("05", "Propensity model + diagnostics",
         [script("05_propensity_model.py"), INDIV, "-o", "model_out"], None),

        ("06", "Composition vs place decomposition",
         [script("06_composition_vs_place.py"), INDIV, "-o", "model_out"], None),

        ("07", "Target feasibility screening",
         [script("07_target_feasibility.py"), INDIV, "-o", "model_out"], None),

        ("08", "Multi-target ablation (provision gradient)",
         [script("08_multi_target_ablation.py"), INDIV, "-o", "model_out"], None),

        ("09", "Forecasting engine (London + boroughs + scenarios)",
         [script("09_forecast.py"), INDIV, "-o", "forecast_out"] + boot, None),

        ("10", "Exploratory + prediction visualisations",
         [script("10_visualise.py"), INDIV, "-o", "viz_out"], None),

        ("11", "Stage A — participation forecasts",
         [script("11_stageA_participation.py"), INDIV, "-o", "stageA_out"]
         + boot, None),

        ("12", "Stage B — demographic forecasts",
         [script("12_stageB_demographics.py"), INDIV, "-o", "stageB_out"], None),

        ("13", "Stage C — indoor vs outdoor",
         [script("13_stageC_indoor_outdoor.py"), INDIV, "-o", "stageC_out"],
         None),

        ("14", "Stage D — volunteering",
         [script("14_stageD_volunteering.py"), INDIV, "-o", "stageD_out"], None),

        ("15", "Model scorecard (metrics for both models)",
         [script("15_model_scorecard.py"), INDIV, "-o", "scorecard_out"], None),

        # ---- validation (professor-review additions) ----
        # Order matters: benchmarks and the back-test must run before the FINAL
        # forecast, because the back-test measures the interval calibration that
        # 09 reads. So 09 runs once above (uncalibrated), the back-test writes
        # the calibration, then 09 runs again below to deploy calibrated bands.
        ("16", "Benchmarks (classifier floors + naive/drift forecasts)",
         [script("16_benchmarks.py"), INDIV, "-o", "benchmark_out"], None),

        ("17", "Rolling-origin back-test (+ interval calibration)",
         [script("17_backtest.py"), INDIV, "-o", "backtest_out",
          "--horizon", "1"] + boot, None),

        ("09b", "Re-run forecast with CALIBRATED intervals",
         [script("09_forecast.py"), INDIV, "-o", "forecast_out"] + boot, None),

        ("18", "Corroboration (skill vs demographic explainability)",
         [script("18_corroboration.py"), "-o", "corroboration_out",
          "--backtest", "backtest_out", "--residuals", "model_out"], None),

        ("19", "Explanatory figures (how the methods work)",
         [script("19_explainer_figures.py"), "-o", "explainer_out"], None),

        ("20", "Evidence figures (that the forecast predicts well)",
         [script("20_evidence_figures.py"),
          "backtest_out/backtest_by_borough.csv", "-o", "evidence_out"], None),

        ("21", "Sensitivity checks (shrinkage \u03bb + interval sources)",
         [script("21_sensitivity.py"), INDIV, "-o", "sensitivity_out"] + boot,
         None),

        ("22", "Requested explanatory graphs (sensitivity, baseline vs "
         "flagship, back-test)",
         [script("22_requested_graphs.py"), INDIV, "-o", "graphs_out",
          "--panel", "benchmark_out/borough_rate_panel.csv",
          "--forecast-all", "forecast_out/borough_forecast_all_years.csv",
          "--backtest", "backtest_out/backtest_by_origin.csv"], None),
    ]


def human(sec):
    m, s = divmod(int(sec), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fast pass with a small bootstrap")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="stage ids to skip, e.g. --skip 05 10")
    ap.add_argument("--from", dest="start", default=None,
                    help="start from this stage id (earlier outputs must exist)")
    ap.add_argument("--python", default=sys.executable,
                    help="python interpreter to use")
    args = ap.parse_args()

    plan = build_plan(args.quick)
    ids = [p[0] for p in plan]
    if args.start and args.start in ids:
        plan = plan[ids.index(args.start):]

    print("=" * 74)
    print("  ACTIVE LIVES LONDON — FULL PIPELINE")
    print("=" * 74)
    print(f"  interpreter : {args.python}")
    print(f"  scripts in  : {SCRIPTS}/")
    print(f"  mode        : {'QUICK (small bootstrap)' if args.quick else 'FULL'}")
    print(f"  stages      : {len(plan)}")
    print("=" * 74 + "\n")

    results = []
    t_total = time.time()

    for sid, desc, cmd, produces in plan:
        if sid in args.skip:
            print(f"[skip]  {sid:9s} {desc}")
            results.append((sid, desc, "skipped", 0))
            continue

        print(f"[run ]  {sid:9s} {desc}")
        print(f"        $ {args.python} {' '.join(cmd)}")
        t0 = time.time()
        try:
            proc = subprocess.run([args.python] + cmd, check=False)
            dur = time.time() - t0
            if proc.returncode == 0:
                # optional output-file existence check
                if produces and not Path(produces).exists():
                    print(f"        !! finished but expected output "
                          f"'{produces}' not found")
                    results.append((sid, desc, "no-output", dur))
                else:
                    print(f"        done in {human(dur)}\n")
                    results.append((sid, desc, "ok", dur))
            else:
                print(f"        !! FAILED (exit {proc.returncode}) after "
                      f"{human(dur)}\n")
                results.append((sid, desc, "FAILED", dur))
                # stop early if a foundational stage fails
                if sid in ("parse", "profile", "harmonise", "dataset2"):
                    print("A foundational stage failed; stopping. Later stages "
                          "depend on its output.\n")
                    break
        except FileNotFoundError:
            dur = time.time() - t0
            print(f"        !! script not found: {cmd[0]}\n")
            results.append((sid, desc, "missing", dur))

    total = time.time() - t_total

    # ---------------------------------------------------------------- summary
    print("=" * 74)
    print("  SUMMARY")
    print("=" * 74)
    print(f"  {'stage':10s} {'status':10s} {'time':>10s}   description")
    print("  " + "-" * 70)
    for sid, desc, status, dur in results:
        print(f"  {sid:10s} {status:10s} {human(dur):>10s}   {desc}")
    print("  " + "-" * 70)
    ok = sum(1 for _, _, s, _ in results if s == "ok")
    bad = [sid for sid, _, s, _ in results if s in ("FAILED", "missing",
                                                    "no-output")]
    print(f"  {ok}/{len(results)} stages completed cleanly")
    if bad:
        print(f"  attention needed: {', '.join(bad)}")
    print(f"\n  TOTAL RUNTIME : {human(total)}  ({total/60:.1f} minutes)")
    print("=" * 74)

    # write a small runtime log
    try:
        with open("pipeline_runtime.txt", "w") as f:
            f.write("Active Lives London — pipeline runtime\n")
            f.write(f"mode: {'quick' if args.quick else 'full'}\n\n")
            for sid, desc, status, dur in results:
                f.write(f"{sid:10s} {status:10s} {human(dur):>10s}  {desc}\n")
            f.write(f"\nTOTAL: {human(total)} ({total/60:.1f} min)\n")
        print("  (runtime log written to pipeline_runtime.txt)")
    except Exception:
        pass

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
