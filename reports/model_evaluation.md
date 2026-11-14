# MediPredict AI - Model Evaluation & Technical Audit Report

**Document Status:** Complete & Verified  
**Training Date:** 2026-11-11T08:08:02.337779Z  
**Selected Model:** `HistGradientBoosting (Calibrated)`  
**Operating Decision Threshold:** `0.17`  

---

## 1. Executive Summary & Purpose

MediPredict AI frames hospital appointment no-show prediction as an operational binary classification problem to prioritize human reminder reviews. It is **not** a clinical diagnostic system and does not predict medical urgency or deny care.

All features are restricted to **point-of-scheduling availability** (booking time). Any clinical diagnosis fields (`Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`) and post-booking events (`SMS_received`) are strictly excluded by policy.

---

## 2. Chronological Split Strategy

To prevent temporal leakage and respect real-world operations where the model predicts future appointments from past booking records, the dataset was split chronologically by `ScheduledDay`:

| Partition | Samples | Time Window | No-Show Rate |
| :--- | :--- | :--- | :--- |
| **Train (70%)** | 77,366 | 2016-04-10 to 2016-05-22 | 20.7% |
| **Validation (15%)** | 16,578 | 2016-05-22 to 2016-05-31 | 20.96% |
| **Held-Out Test (15%)** | 16,580 | 2016-05-31 to 2016-06-09 | 20.73% |

*Note: Validation split was used for candidate architecture comparison, hyperparameter selection, and threshold tuning. The held-out test split remained strictly untouched until final evaluation.*

---

## 3. Model Architecture Comparison (Validation Partition)

| Candidate Model | Val ROC-AUC | Val PR-AUC | Optimal Threshold | Val Precision | Val Recall | Val F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy (Baseline)** | 0.5000 | 0.2096 | 0.050 | 0.2096 | 1.0000 | 0.3465 |
| **Logistic Regression** | 0.6955 | 0.3365 | 0.530 | 0.2983 | 0.7677 | 0.4296 |
| **Random Forest** | 0.6948 | 0.3370 | 0.520 | 0.2879 | 0.8474 | 0.4298 |
| **HistGradientBoosting** | 0.6942 | 0.3387 | 0.450 | 0.2810 | 0.9004 | 0.4283 |

---

## 4. Final Held-Out Test Evaluation

Evaluating `HistGradientBoosting (Calibrated)` on the untouched test partition (16,580 records):

### Performance at Operating Threshold (0.17) vs Default (0.50)

| Metric | Selected Operating Threshold (0.17) | Default Threshold (0.50) |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.6969** | 0.6969 |
| **PR-AUC** | **0.3336** | 0.3336 |
| **Precision** | **0.2828** | 0.6471 |
| **Recall (Sensitivity)** | **0.9031** | 0.0032 |
| **F1 Score** | **0.4307** | 0.0064 |
| **Brier Score (Calibration)** | **0.1509** | 0.1509 |

### Test Confusion Matrix
- **True Positives (Correctly prioritized no-shows):** 3,104
- **False Positives (Attended, flagged for review):** 7,873
- **True Negatives (Correctly identified attended):** 5,270
- **False Negatives (Missed no-shows):** 333

---

## 5. Threshold Selection Justification

In appointment operations, a default threshold of `0.50` often misses a large proportion of actual no-shows due to class imbalance (~20% positive class rate). By tuning the decision threshold to `0.17` on the validation partition, the system achieves a balanced operational trade-off: capturing significant no-show volume (Recall: 90.3%) while maintaining actionable precision (28.3%) for administrative staff without overwhelming staff capacity.

---

## 6. Responsible Use & Governance Boundaries

1. **Human-in-the-Loop:** Predictions must serve as an informational review aid only. Staff decide whether an appointment warrants an extra reminder call.
2. **Prohibited Automations:** No automated cancellation, rescheduling, or patient deprioritization may occur based on these scores.
3. **Non-Clinical Mandate:** No clinical diagnosis or medical triage is performed.
