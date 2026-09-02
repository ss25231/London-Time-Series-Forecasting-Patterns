# Model scorecard

Temporal split: train waves [1, 2, 3, 4, 5, 6], validate wave 7, test wave 8. All metrics survey-weighted.

| Metric | Ordinal logistic regression | LightGBM |
|---|---|---|
| Test Accuracy | 0.6722 | 0.6715 |
| Test Precision (macro) | 0.4148 | 0.4159 |
| Test Recall (macro) | 0.4076 | 0.4032 |
| Test F1 (macro) | 0.3923 | 0.3870 |
| CV Accuracy | 0.6524 | 0.6524 |
| CV F1 (macro) | 0.3735 | 0.3770 |
| Train Accuracy | 0.6542 | 0.6603 |
| Train F1 (macro) | 0.3757 | 0.3801 |
| Runtime (min) | 0.1692 | 0.7351 |
| Train-CV Accuracy Gap | 0.0018 | 0.0080 |
| AUC (macro OvR) | 0.6592 | 0.6633 |
| AUC (Active vs rest) | 0.7043 | 0.7078 |
| LogLoss | 0.7976 | 0.7935 |

_Scorecard runtime: 0.91 min._
