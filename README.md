# Forecasting Sport and Physical Activity Across London's Boroughs

A reproducible, demography-driven framework that forecasts physical activity,
club membership and volunteering across London's 32 boroughs, built on eight
years of Sport England's **Active Lives Adult Survey** (2015–16 to 2022–23,
135,497 respondents).

Prepared for a London-based non-profit sports organisation.

---

> **Note on contributions and commit history.** Because the four workstreams
> had little interdependency, the git history shows relatively few commits,
> most made by a single member. This reflects how the team worked rather than
> the distribution of effort: all four members independently developed and
> tested separate modelling approaches to establish which performed best. We
> then took the strongest approach forward, chosen on a combination of runtime,
> accuracy and error rates. Every member contributed substantially to the
> project.

---

## Table of contents

- [What this project does](#what-this-project-does)
- [Quick start (one command)](#quick-start-one-command)
- [What you need in place first](#what-you-need-in-place-first)
- [Project layout](#project-layout)
- [How the method works](#how-the-method-works)
- [The pipeline, stage by stage](#the-pipeline-stage-by-stage)
- [Outputs: where to find what](#outputs-where-to-find-what)
- [Key results at a glance](#key-results-at-a-glance)
- [Honesty of labelling](#honesty-of-labelling)
- [Adding a new year of data](#adding-a-new-year-of-data)
- [Runtime](#runtime)
- [Troubleshooting](#troubleshooting)
- [Glossary](#glossary)

---

## What this project does

The goal is to anticipate how participation in sport, physical activity and
volunteering will change across London over the coming decade, and to
understand how demographic, environmental and behavioural factors interact
across the capital's boroughs.

The project delivers:

- **A cleaned, harmonised dataset** built from eight structurally inconsistent
  survey years, validated against Sport England's own published figures.
- **A propensity model** of who is physically active, and how confident we can
  be in it.
- **Borough-level forecasts** with uncertainty intervals, plus ten-year
  scenarios.
- **Answers to all four areas of the brief** — overall participation,
  demographic insight, indoor vs outdoor activity, and volunteering.
- **A full report and presentation**, and a **reproducible pipeline** that
  absorbs future survey years automatically.

---

## Quick start (one command)

From the project root:

**Windows**

```bat
run_project.bat
```

**macOS / Linux**

```bash
bash run_project.sh
```

That single command **installs the dependencies and then runs the entire
pipeline** — every stage, in order — and prints a timed summary at the end.
Come back in roughly 40–60 minutes to a complete set of datasets, models,
forecasts, figures and a model scorecard.

For a faster pass (smaller bootstrap; identical point estimates, rougher
confidence intervals):

```bat
run_project.bat --quick
```

That's the whole thing. Everything below is reference.

---

## What you need in place first

Two input folders, next to the run script:

| Folder | Contents |
| --- | --- |
| `dictionaries/` | The eight Active Lives data-dictionary files (one per survey year). |
| `csvs/` | The eight annual Active Lives CSV extracts for London. |

Everything else — the `parsed/` folder, the datasets, all output folders — is
**created by the pipeline**. You do not prepare them.

You need **Python 3.9 or newer**. The run script installs the Python packages
for you; if you'd rather install them yourself:

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib pyarrow \
    python-docx openpyxl python-pptx
```

---

## Project layout

The repository has two states: **before** you run the pipeline (the files you
start with) and **after** (the same files plus every output folder the run
generates). Both are shown below.

### Before running — the original repository

These are the files and folders that ship with the project. Nothing here is
generated; this is what you begin with.

```
project/
├── run_project.bat            ← Windows: install dependencies + run everything
├── run_project.sh             ← macOS/Linux: install dependencies + run everything
├── README.md                  ← this file
│
├── csvs/                      ← YOU provide: 8 annual Active Lives CSV extracts
├── dictionaries/              ← YOU provide: 8 matching data-dictionary files
│
├── LaTeX/                     ← the written dissertation report
│   ├── DataScience_UoB_MSc_thesis(2).tex   ← the report source
│   ├── dissertation.cls       ← document class / formatting
│   ├── sample_bibtex.bib      ← bibliography
│   ├── Team_04_Group_Report.pdf      ← group report
│   ├── logo_uob_color.pdf     ← university logo used on the title page
│   └── figures/               ← all figures used in the report
│
└── scripts/                   ← all pipeline code
    ├── run_all.py             ← runs every stage in order, with timing
    ├── viz_theme.py           ← shared chart style (imported, never run directly)
    │
    ├── parse_dictionaries.py       ← stage 0: read the dictionaries
    ├── profile_waves.py            ← stage 0: profile the years, flag breaks
    ├── harmonise_waves.py          ← stage 1: build the individual dataset
    ├── 04_build_dataset2.py        ← stage 1: build the cell panel
    ├── 05_propensity_model.py      ← stage 2: the propensity model
    ├── 06_composition_vs_place.py  ← stage 2: composition vs place
    ├── 07_target_feasibility.py    ← stage 2: which targets are forecastable
    ├── 08_multi_target_ablation.py ← stage 2: the provision gradient
    ├── 09_forecast.py              ← stage 3: the forecasting engine
    ├── 10_visualise.py             ← stage 4: exploratory figures
    ├── 11_stageA_participation.py  ← brief section 1: participation
    ├── 12_stageB_demographics.py   ← brief section 2: demographics
    ├── 13_stageC_indoor_outdoor.py ← brief section 3: indoor vs outdoor
    ├── 14_stageD_volunteering.py   ← brief section 4: volunteering
    ├── 15_model_scorecard.py       ← model metrics table
    ├── 16_benchmarks.py            ← validation: benchmark floors
    ├── 17_backtest.py              ← validation: rolling-origin back-test
    ├── 18_corroboration.py         ← validation: right-reason check
    ├── 19_explainer_figures.py     ← plain-language method figures
    ├── 20_evidence_figures.py      ← evidence-the-forecast-works figures
    ├── 21_sensitivity.py           ← robustness: shrinkage + interval sources
    └── 22_requested_graphs.py      ← explanatory line graphs
```

### After running — the folders the pipeline generates

Running `run_project.bat` (or `run_project.sh`) creates all of the following.
Every folder below is **generated output** — none of it exists before the run,
and the whole set can be deleted and regenerated at any time.

```
project/
│   … (all of the original files above, unchanged) …
│
├── dataset1_individual.parquet   ← the cleaned people-level dataset (135,497 rows)
├── dataset2_cells.parquet        ← the aggregated borough × demographic panel
│
├── parsed/                    ← dictionary lookups + wave manifest        (stage 0)
├── model_out/                 ← model diagnostics, residuals, figures     (stage 2)
├── forecast_out/              ← forecasts, scenarios, borough figures     (stage 3)
├── viz_out/                   ← exploratory and prediction figures        (stage 4)
├── stageA_out/                ← brief section 1: participation figures/CSVs
├── stageB_out/                ← brief section 2: demographics figures/CSVs
├── stageC_out/                ← brief section 3: indoor vs outdoor
├── stageD_out/                ← brief section 4: volunteering
├── scorecard_out/             ← model_scorecard.csv / .md                 (stage 6)
├── benchmark_out/             ← classifier floors + borough rate panel    (stage 7)
├── backtest_out/              ← back-test results + interval calibration  (stage 7)
├── corroboration_out/         ← the right-reason scatter + CSV            (stage 7)
├── explainer_out/             ← plain-language method figures             (stage 8)
├── evidence_out/              ← predicted-vs-actual and skill figures     (stage 8)
├── sensitivity_out/           ← shrinkage sweep + interval-source figures (stage 9)
├── graphs_out/                ← the explanatory line graphs               (stage 9)
└── pipeline_runtime.txt       ← per-stage and total timing
```

> **Note.** The scripts also run fine sitting directly in the project root
> instead of a `scripts/` folder — the runner checks both. Keep `viz_theme.py`
> beside the stage scripts; it is imported by the figure code. The `LaTeX/`
> folder is independent of the pipeline: it holds the written report and is not
> read or modified by any script.

---

## How the method works

Forecasting from eight noisy annual snapshots is not a job a conventional
time-series or machine-learning model can do reliably: 32 boroughs over 8
years is only 256 data points, three of them disrupted by the pandemic. So the
framework **separates two things that are usually entangled**:

```
Forecast  =  who is active            ×  how the population is changing
             (learned from 135,497        (demographic projection)
              people, time-free)
          +  a small behavioural trend
```

- **The model** learns the relationship between a person's characteristics and
  their activity level — what a large survey is excellent for. It never
  extrapolates in time.
- **The population projection** carries the forecast forward, because
  demographic change is far more predictable than behaviour.

The pay-off is that every projected change splits cleanly into a
**composition effect** (the population changed) and a **behaviour effect**
(people of a given type changed) — two different things a single number would
hide.

The model itself is an **ordinal logistic regression** (Inactive → Fairly
Active → Active). It was benchmarked head-to-head against gradient boosting
(LightGBM); the complex model gained essentially nothing, so the simpler,
interpretable one was chosen. Validation is temporal — trained on the early
years, tuned on the second-to-last, and tested once on the final year — plus a
leave-one-borough-out test proving the model predicts boroughs it has never
seen as well as ones it trained on.

---

## The pipeline, stage by stage

Run automatically by `run_all.py`, in this order:

| # | Script | What it does |
| --- | --- | --- |
| 0 | `parse_dictionaries.py` | Reads the 8 dictionaries; extracts variable and geography lookups. |
| 0 | `profile_waves.py` | Compares variable availability across years; flags structural breaks. |
| 1 | `harmonise_waves.py` | Recodes, maps geography to permanent ONS codes, fixes four data defects → **individual dataset**. |
| 1 | `04_build_dataset2.py` | Aggregates to a borough × year × demographic **cell panel** (reconciles exactly with the individual file). |
| 2 | `05_propensity_model.py` | Fits and compares models; overfitting diagnostics. |
| 2 | `06_composition_vs_place.py` | Splits between-borough variation into composition vs place. |
| 2 | `07_target_feasibility.py` | Tests which behaviours have enough signal to forecast. |
| 2 | `08_multi_target_ablation.py` | The provision gradient across fifteen behaviours. |
| 3 | `09_forecast.py` | **The forecasting engine**: London + borough forecasts, decomposition, scenarios. |
| 4 | `10_visualise.py` | Exploratory and prediction figures. |
| 5 | `11_stageA_participation.py` | Brief §1 — overall activity, clubs, activity types, free time. |
| 5 | `12_stageB_demographics.py` | Brief §2 — demographic forecasts, preferred activities, health. |
| 5 | `13_stageC_indoor_outdoor.py` | Brief §3 — indoor vs outdoor by level and by inner/outer London. |
| 5 | `14_stageD_volunteering.py` | Brief §4 — who volunteers, and in what roles. |
| 6 | `15_model_scorecard.py` | Full metrics table for both models. |
| 7 | `16_benchmarks.py` | Classifier floors (prevalence, single-feature) and forecast benchmarks (naïve, drift). |
| 7 | `17_backtest.py` | **Rolling-origin back-test**: forward forecasts at each origin, per borough; measures forecast skill and the interval calibration. |
| 7 | `09_forecast.py` *(again)* | Re-run so the deployed intervals pick up the calibration the back-test measured (see two-pass note below). |
| 7 | `18_corroboration.py` | Tests whether the forecast is accurate for the right reason (skill vs demographic explainability). |
| 8 | `19_explainer_figures.py` | Plain-language figures: how each method works. |
| 8 | `20_evidence_figures.py` | Evidence figures: predicted-vs-actual and skill, built on the back-test. |
| 9 | `21_sensitivity.py` | Robustness checks: the shrinkage λ sweep (forecast barely moves) and the interval-source decomposition (coefficient uncertainty is the smallest source). |
| 9 | `22_requested_graphs.py` | Three explanatory line graphs: the λ sensitivity line, the flagship forecast vs the naïve benchmark across years, and the back-test error comparison. |

**Two-pass ordering (important).** The forecasting engine (`09`) runs **twice**. The first run produces the central forecasts. The back-test (`17`) then measures how much year-to-year uncertainty the intervals were missing and writes `backtest_out/interval_calibration.json`. The second run of `09` reads that file and widens the deployed intervals so their 90% bands are honest. The central forecasts are identical on both runs — only the interval widths change. `run_all.py` handles this ordering automatically.

Three checks run automatically and will flag or halt the pipeline if they
fail: the activity classification must match Sport England's published
grouping; the cell panel must reconcile with the individual file; and every
forecast must begin at the last observed value.

---

## Outputs: where to find what

| You want… | Look in |
| --- | --- |
| The cleaned datasets | `dataset1_individual.parquet`, `dataset2_cells.parquet` |
| Model diagnostics & comparison figures | `model_out/` |
| Model metrics table (accuracy, F1, AUC, etc.) | `scorecard_out/model_scorecard.csv` and `.md` |
| London & borough forecasts, scenarios | `forecast_out/` (CSVs + `figX*` figures) |
| Overall activity trend (3 bands, projected) | `forecast_out/figX1_overall_band_trend.png` |
| Top-10 / least-5 boroughs | `forecast_out/figX2_*`, `figX3_*` |
| Brief §1–4 figures and tables | `stageA_out/` … `stageD_out/` |
| Exploratory & prediction visuals | `viz_out/` |
| Benchmark floors (classifier + forecast) | `benchmark_out/classifier_floors.csv` |
| Back-test results & skill, interval calibration | `backtest_out/` (`backtest_by_origin.csv`, `backtest_skill.png`, `interval_calibration.json`) |
| Corroboration (right-reason evidence) | `corroboration_out/corroboration_scatter.png` |
| Explanatory figures (how it works) | `explainer_out/` |
| Evidence figures (that it works) | `evidence_out/` (`evid_pred_vs_actual.png` etc.) |
| Sensitivity & interval-source checks | `sensitivity_out/` (`sens_lambda.png`, `sens_variance_sources.png`) |
| Explanatory line graphs | `graphs_out/` (`req_sensitivity_line.png`, `req_baseline_vs_flagship.png`, `req_backtest.png`) |
| How long each stage took | `pipeline_runtime.txt` |

Figures prefixed `figX` are the standalone, themed charts (one message per
figure). Every figure carries a short caption describing what it shows.

---

## Key results at a glance

- **Activity is stable but polarising.** The share meeting the guideline held
  near 64% across eight years, but the partially active middle shrank from
  12.3% to 10.1% — the group closest to the guideline, and the cheapest to
  move across it.
- **Half the gap between boroughs is composition, half is place.** Demographics
  reproduce 48% of the variation between boroughs; the rest is the boroughs
  themselves — facilities, environment, infrastructure.
- **The provision gradient** — the more a behaviour depends on provision, the
  less demography explains it: walking 50%, cycling 29%, park use 18%,
  volunteering 15%. The behaviours least predictable from population are
  exactly those most open to influence through investment.
- **London's flat forecast hides two opposing forces.** Activity is projected
  at 64.7% by 2025–26 — a demographic drag of −0.37 points cancelling a
  behavioural gain of +0.69.
- **Inequality is wide and does not close.** A 24-point gap separates the most
  and least active boroughs, and nothing in the projection narrows it.

The model scorecard (real data): **test accuracy 0.67, AUC 0.70, train–CV
accuracy gap 0.002** — the tiny gap being the quantitative proof the model is
not overfitting.

**Forecast validation (real data).** The forecast is not merely produced but
tested. Three results establish it:

- **Skill +0.20** — in rolling-origin back-testing, the forecast reduces
  borough error by about a fifth versus assuming no change, at every clean
  origin (mean error 2.7pp vs 3.4pp for naïve).
- **Coverage 94%** — the 90% intervals were found over-confident (53%) and
  recalibrated against measured out-of-sample error; coverage now sits near
  90%, erring slightly conservative.
- **Correlation +0.74** — the forecast is accurate for the right reason: it
  predicts best on the boroughs whose activity is demographically explained.

Two robustness checks close the remaining questions a specialist would ask. The
shrinkage constant λ is shown not to matter — the London forecast moves just
**0.04 points** across its whole range, and even the most sensitive borough
moves only **1.3 points**. And the uncertainty range is decomposed into its
three sources, confirming the model's own coefficient uncertainty (**0.35 pp**)
is the smallest; year-to-year process variation (**5.66 pp**) dominates.

---

## Honesty of labelling

A principle runs through every output: **each question is answered at the
level the data can honestly support, and labelled accordingly.**

- **Forecast** — a projection with quantified uncertainty, where the data
  supports one (overall activity, club membership, activity types, demographic
  groups).
- **Description** — a robust profile of the current picture, where forecasting
  is not defensible: a question asked in only one year (free-time factors), a
  base that changed mid-series (volunteering), a rotating subsample
  (indoor/outdoor), or a variable missing from early years (health).

Presenting a solid description in place of a fragile forecast is a deliberate
strength, not a shortfall.

---

## Adding a new year of data

The pipeline is built to absorb future survey releases **with no code
changes**, through roughly the mid-2030s.

1. Drop the new year's CSV into `csvs/` and its dictionary into
   `dictionaries/`.
2. Re-run `run_project.bat` (or `run_project.sh`).

The wave numbering and the year mappings extend automatically. Three things to
know:

- **Each new year needs its own dictionary** (borough codes are reassigned as
  authorities merge elsewhere in England; the pipeline maps them correctly, but
  only with that year's dictionary).
- **A new structural break is flagged, not silently absorbed.** If the survey
  renames a column or changes a question's routing, the feasibility diagnostic
  reports it — a human then decides the fix, as was done for the four original
  defects.
- **The forecast improves with each added year.** More non-pandemic data firms
  up the behavioural trend, so the long-run scenarios become less speculative
  over time. This is a standing capability, not a one-off analysis. Each added
  year also gives the rolling-origin back-test one more origin, so the
  validation strengthens as the series grows.

---

## Runtime

A full clean run is roughly **40–60 minutes**, dominated by two bootstrap
steps (the model and the forecast). `--quick` reduces this to about
**15–20 minutes** by shrinking the bootstrap — the point estimates are
identical; only the confidence intervals are rougher.

Exact per-stage timings are written to `pipeline_runtime.txt` on every run.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError` for a package | The run script installs everything; if you ran a script directly, `pip install` the packages listed above. |
| `pyarrow` errors reading `.parquet` | `pip install pyarrow`. |
| A stage fails but you want the rest | The runner continues past a failed non-foundational stage and flags it in the summary. Re-run one stage with, e.g., `python scripts/run_all.py --from 09`. |
| Figures look unstyled or import `viz_theme` fails | Ensure `viz_theme.py` sits beside the stage scripts. |
| Free-time reason charts are missing | `harmonise_waves.py` must be re-run so the individual free-time columns are produced. |
| Word document contents page shows no page numbers | Open in Word and right-click the table of contents → *Update Field*. |
| Foundational stage (parse/harmonise) fails | Check the `dictionaries/` and `csvs/` folders exist and contain the expected files. The runner stops here on purpose, since later stages depend on the output. |

---

## Glossary

| Term | Meaning |
| --- | --- |
| **Active / Fairly Active / Inactive** | Activity bands: 150+, 30–149, and 0–29 weekly moderate-equivalent minutes. |
| **Composition effect** | The part of a projected change due to the population's demographic mix changing. |
| **Behaviour effect** | The part due to people of a given type changing their activity over time. |
| **Propensity model** | The time-free model relating a person's characteristics to their activity band. |
| **Provision gradient** | The finding that the more a behaviour depends on facilities or infrastructure, the less demography explains its geography. |
| **Leave-one-borough-out** | A validation predicting each borough from a model trained without it. |
| **AUC** | Area under the ROC curve; a model's ability to rank cases, where 0.5 is chance. |
| **Bootstrap** | Repeatedly refitting on resampled data to produce confidence intervals. |

---

*Every figure and number in this project can be traced back to a specific
stage of the pipeline, and every question in the brief is answered with either
a forecast or a stated reason it cannot be forecast. Nothing is forced past
what the data supports.*
