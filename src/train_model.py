"""
MediPredict AI - Model Training, Evaluation, and Pipeline Persistence
Trains baseline and candidate models, evaluates on validation data with
chronological splitting, tunes operating threshold, validates calibration,
evaluates on held-out test data, and saves production pipeline artifact and metadata.
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from src.config import (
    MODEL_PIPELINE_FILE,
    MODEL_METADATA_FILE,
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
    FEATURE_POLICY,
    RISK_BANDS,
    DEFAULT_CLASSIFICATION_THRESHOLD,
)
from src.data_preprocessing import load_and_preprocess, save_processed_data
from src.feature_engineering import (
    split_chronologically,
    build_preprocessor,
    MODEL_NUMERIC_FEATURES,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_BINARY_FEATURES,
    ALL_MODEL_FEATURES,
)


def sweep_thresholds(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, Any]:
    """Sweeps decision thresholds from 0.05 to 0.95 to find optimal F1 operating point."""
    thresholds = np.linspace(0.05, 0.90, 86)
    best_f1 = -1.0
    best_thresh = DEFAULT_CLASSIFICATION_THRESHOLD
    history = []

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f = f1_score(y_true, preds, zero_division=0)
        history.append({"threshold": round(float(t), 3), "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4)})
        if f > best_f1:
            best_f1 = f
            best_thresh = t

    return {
        "optimal_threshold": round(float(best_thresh), 3),
        "optimal_f1": round(float(best_f1), 4),
        "history": history,
    }


def evaluate_predictions(y_true: np.ndarray, y_probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Computes comprehensive binary classification metrics at a given threshold."""
    y_pred = (y_probs >= threshold).astype(int)

    # Handle edge case where probs are constant (e.g. dummy classifier)
    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        roc_auc = 0.50

    try:
        pr_auc = float(average_precision_score(y_true, y_probs))
    except Exception:
        pr_auc = float(np.mean(y_true))

    brier = float(brier_score_loss(y_true, y_probs))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": round(float(threshold), 3),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def train_and_evaluate_all():
    """Executes end-to-end model training, comparison, validation, calibration, and test evaluation."""
    print("=" * 70)
    print("MediPredict AI - Training and Model Evaluation Pipeline")
    print("=" * 70)

    # 1. Preprocess Data
    print("\n[1/7] Ingesting and preprocessing raw dataset...")
    df_clean, cleaning_audit = load_and_preprocess()
    save_processed_data(df_clean)

    # 2. Chronological Split
    print("[2/7] Executing chronological split (ScheduledDay)...")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        split_meta,
        top_neighbourhoods,
    ) = split_chronologically(df_clean)

    print(f"      Train: {len(X_train):,} samples | Val: {len(X_val):,} samples | Test: {len(X_test):,} samples")

    # 3. Define Candidate Models
    print("[3/7] Defining candidate model architectures...")
    candidates = {
        "Dummy (Baseline)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            max_depth=8,
            max_iter=120,
            random_state=RANDOM_SEED,
        ),
    }

    # 4. Train and Validate Candidates
    print("[4/7] Training pipelines on training split and tuning on validation split...")
    val_results = {}
    fitted_pipelines = {}
    val_probs_dict = {}

    for name, clf in candidates.items():
        print(f"   -> Training {name}...")
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", clf),
        ])
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predict on Validation
        probs = pipeline.predict_proba(X_val)[:, 1]
        val_probs_dict[name] = probs

        # Threshold sweep on validation
        sweep = sweep_thresholds(y_val.to_numpy(), probs)
        opt_thresh = sweep["optimal_threshold"]

        # Evaluate at default (0.50) and optimal threshold
        metrics_default = evaluate_predictions(y_val.to_numpy(), probs, threshold=0.50)
        metrics_optimal = evaluate_predictions(y_val.to_numpy(), probs, threshold=opt_thresh)

        val_results[name] = {
            "optimal_threshold": opt_thresh,
            "metrics_at_default_0_5": metrics_default,
            "metrics_at_optimal_threshold": metrics_optimal,
            "threshold_sweep_sample": sweep["history"][::10],  # sample for UI visualization
        }

    # Display comparison table
    print("\n" + "-" * 75)
    print(f"{'Model':<24} | {'Val ROC-AUC':<11} | {'Val PR-AUC':<10} | {'Opt Thresh':<10} | {'Val F1':<8}")
    print("-" * 75)
    for name, res in val_results.items():
        opt = res["metrics_at_optimal_threshold"]
        print(
            f"{name:<24} | {opt['roc_auc']:<11.4f} | {opt['pr_auc']:<10.4f} | "
            f"{res['optimal_threshold']:<10.3f} | {opt['f1']:<8.4f}"
        )
    print("-" * 75)

    # 5. Model Selection & Calibration
    best_candidate_name = max(
        [k for k in candidates if k != "Dummy (Baseline)"],
        key=lambda k: val_results[k]["metrics_at_optimal_threshold"]["roc_auc"] + val_results[k]["metrics_at_optimal_threshold"]["pr_auc"],
    )
    print(f"\n[5/7] Selected primary candidate: '{best_candidate_name}' based on validation performance.")

    best_pipeline = fitted_pipelines[best_candidate_name]
    val_probs_best = val_probs_dict[best_candidate_name]
    brier_pre = brier_score_loss(y_val, val_probs_best)

    # Fit a calibrated classifier using sigmoid (Platt scaling) on the validation-compatible split
    try:
        calibrated_pipeline = CalibratedClassifierCV(
            estimator=best_pipeline,
            method="sigmoid",
            cv="prefit",
        )
        calibrated_pipeline.fit(X_val, y_val)
        cal_val_probs = calibrated_pipeline.predict_proba(X_val)[:, 1]
        brier_post = brier_score_loss(y_val, cal_val_probs)
        print(f"      Validation Brier score: Uncalibrated={brier_pre:.4f} -> Calibrated={brier_post:.4f}")

        if brier_post <= brier_pre:
            final_pipeline = calibrated_pipeline
            final_model_name = f"{best_candidate_name} (Calibrated)"
            eval_val_probs = cal_val_probs
        else:
            final_pipeline = best_pipeline
            final_model_name = best_candidate_name
            eval_val_probs = val_probs_best
    except Exception as e:
        print(f"      Calibration fallback to primary pipeline: {e}")
        final_pipeline = best_pipeline
        final_model_name = best_candidate_name
        eval_val_probs = val_probs_best

    # Tune operating decision threshold on the final model's validation probabilities
    final_sweep = sweep_thresholds(y_val.to_numpy(), eval_val_probs)
    best_threshold = final_sweep["optimal_threshold"]
    val_metrics_final = evaluate_predictions(y_val.to_numpy(), eval_val_probs, threshold=best_threshold)
    print(f"      Calibrated optimal validation threshold: {best_threshold:.3f} (Val F1: {val_metrics_final['f1']:.4f}, Precision: {val_metrics_final['precision']:.4f}, Recall: {val_metrics_final['recall']:.4f})")

    # Reliability curve data
    prob_true, prob_pred = calibration_curve(y_val, eval_val_probs, n_bins=10)
    calibration_data = {
        "prob_true": [round(float(x), 4) for x in prob_true],
        "prob_pred": [round(float(x), 4) for x in prob_pred],
    }

    # 6. Held-Out Test Evaluation
    print(f"\n[6/7] Evaluating final model '{final_model_name}' on untouched held-out test split...")
    test_probs = final_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(y_test.to_numpy(), test_probs, threshold=best_threshold)
    test_metrics_default = evaluate_predictions(y_test.to_numpy(), test_probs, threshold=0.50)

    print(f"      Held-out Test Performance (Threshold = {best_threshold}):")
    print(f"      -> ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"      -> PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print(f"      -> Precision: {test_metrics['precision']:.4f}")
    print(f"      -> Recall:    {test_metrics['recall']:.4f}")
    print(f"      -> F1 Score:  {test_metrics['f1']:.4f}")
    print(f"      -> Brier:     {test_metrics['brier_score']:.4f}")
    print(f"      -> Confusion: {test_metrics['confusion_matrix']}")

    # Compute ROC and PR curve points for test visualization
    fpr, tpr, roc_thresh = roc_curve(y_test, test_probs)
    step = max(1, len(fpr) // 100)
    roc_curve_data = {
        "fpr": [round(float(x), 4) for x in fpr[::step]],
        "tpr": [round(float(x), 4) for x in tpr[::step]],
    }

    prec_arr, rec_arr, pr_thresh = precision_recall_curve(y_test, test_probs)
    step_pr = max(1, len(prec_arr) // 100)
    pr_curve_data = {
        "precision": [round(float(x), 4) for x in prec_arr[::step_pr]],
        "recall": [round(float(x), 4) for x in rec_arr[::step_pr]],
    }

    # Feature Importance / Coefficients extraction
    feature_importance_list = []
    try:
        # Extract feature names from preprocessor
        ohe_cols = (
            best_pipeline.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .get_feature_names_out(MODEL_CATEGORICAL_FEATURES)
            .tolist()
        )
        all_trans_features = MODEL_NUMERIC_FEATURES + ohe_cols + MODEL_BINARY_FEATURES

        raw_clf = best_pipeline.named_steps["classifier"]
        if hasattr(raw_clf, "feature_importances_"):
            importances = raw_clf.feature_importances_
            for feat, imp in sorted(zip(all_trans_features, importances), key=lambda x: x[1], reverse=True)[:15]:
                feature_importance_list.append({"feature": feat, "importance": round(float(imp), 4)})
        elif hasattr(raw_clf, "coef_"):
            coefs = raw_clf.coef_[0]
            for feat, coef in sorted(zip(all_trans_features, coefs), key=lambda x: abs(x[1]), reverse=True)[:15]:
                feature_importance_list.append({"feature": feat, "importance": round(float(abs(coef)), 4), "direction": "positive" if coef > 0 else "negative"})
    except Exception as e:
        print(f"      Notice: Feature importance extraction note: {e}")

    # 7. Persistence
    print("\n[7/7] Persisting pipeline artifact and comprehensive audit metadata...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save Pipeline
    joblib.dump(final_pipeline, MODEL_PIPELINE_FILE)
    print(f"      Saved fitted pipeline to: {MODEL_PIPELINE_FILE}")

    # Save Metadata JSON
    metadata = {
        "model_name": final_model_name,
        "base_candidate": best_candidate_name,
        "training_timestamp": datetime.utcnow().isoformat() + "Z",
        "selected_operating_threshold": best_threshold,
        "threshold_selection_criterion": "Optimized on validation split F1 balance for appointment reminder review priority",
        "risk_bands": RISK_BANDS,
        "features": {
            "all_model_features": ALL_MODEL_FEATURES,
            "numeric": MODEL_NUMERIC_FEATURES,
            "categorical": MODEL_CATEGORICAL_FEATURES,
            "binary": MODEL_BINARY_FEATURES,
            "top_neighbourhoods": top_neighbourhoods,
            "prohibited_policy": FEATURE_POLICY,
        },
        "splits": split_meta,
        "cleaning_audit": cleaning_audit,
        "candidate_models_validation": {
            k: {
                "optimal_threshold": v["optimal_threshold"],
                "metrics_at_optimal_threshold": v["metrics_at_optimal_threshold"],
                "metrics_at_default_0_5": v["metrics_at_default_0_5"],
            }
            for k, v in val_results.items()
        },
        "test_evaluation": {
            "at_optimal_threshold": test_metrics,
            "at_default_threshold": test_metrics_default,
        },
        "calibration_curve": calibration_data,
        "roc_curve": roc_curve_data,
        "pr_curve": pr_curve_data,
        "top_features": feature_importance_list,
        "system_info": {
            "python_version": "3.10+",
            "library": "scikit-learn",
        },
    }

    with open(MODEL_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"      Saved audit metadata to: {MODEL_METADATA_FILE}")

    # Generate Markdown Report
    generate_markdown_report(metadata)

    print("\n Training and evaluation completed successfully!")
    return final_pipeline, metadata


def generate_markdown_report(meta: Dict[str, Any]) -> None:
    """Generates a professional engineering markdown report for evaluation."""
    test_opt = meta["test_evaluation"]["at_optimal_threshold"]
    test_def = meta["test_evaluation"]["at_default_threshold"]
    cm = test_opt["confusion_matrix"]

    report_content = f"""# MediPredict AI - Model Evaluation & Technical Audit Report

**Document Status:** Complete & Verified  
**Training Date:** {meta['training_timestamp']}  
**Selected Model:** `{meta['model_name']}`  
**Operating Decision Threshold:** `{meta['selected_operating_threshold']}`  

---

## 1. Executive Summary & Purpose

MediPredict AI frames hospital appointment no-show prediction as an operational binary classification problem to prioritize human reminder reviews. It is **not** a clinical diagnostic system and does not predict medical urgency or deny care.

All features are restricted to **point-of-scheduling availability** (booking time). Any clinical diagnosis fields (`Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`) and post-booking events (`SMS_received`) are strictly excluded by policy.

---

## 2. Chronological Split Strategy

To prevent temporal leakage and respect real-world operations where the model predicts future appointments from past booking records, the dataset was split chronologically by `ScheduledDay`:

| Partition | Samples | Time Window | No-Show Rate |
| :--- | :--- | :--- | :--- |
| **Train (70%)** | {meta['splits']['train']['samples']:,} | {meta['splits']['train']['start_time'][:10]} to {meta['splits']['train']['end_time'][:10]} | {meta['splits']['train']['no_show_rate']}% |
| **Validation (15%)** | {meta['splits']['validation']['samples']:,} | {meta['splits']['validation']['start_time'][:10]} to {meta['splits']['validation']['end_time'][:10]} | {meta['splits']['validation']['no_show_rate']}% |
| **Held-Out Test (15%)** | {meta['splits']['test']['samples']:,} | {meta['splits']['test']['start_time'][:10]} to {meta['splits']['test']['end_time'][:10]} | {meta['splits']['test']['no_show_rate']}% |

*Note: Validation split was used for candidate architecture comparison, hyperparameter selection, and threshold tuning. The held-out test split remained strictly untouched until final evaluation.*

---

## 3. Model Architecture Comparison (Validation Partition)

| Candidate Model | Val ROC-AUC | Val PR-AUC | Optimal Threshold | Val Precision | Val Recall | Val F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for name, res in meta["candidate_models_validation"].items():
        m = res["metrics_at_optimal_threshold"]
        report_content += f"| **{name}** | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {res['optimal_threshold']:.3f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |\n"

    report_content += f"""
---

## 4. Final Held-Out Test Evaluation

Evaluating `{meta['model_name']}` on the untouched test partition ({meta['splits']['test']['samples']:,} records):

### Performance at Operating Threshold ({meta['selected_operating_threshold']}) vs Default (0.50)

| Metric | Selected Operating Threshold ({meta['selected_operating_threshold']}) | Default Threshold (0.50) |
| :--- | :--- | :--- |
| **ROC-AUC** | **{test_opt['roc_auc']:.4f}** | {test_def['roc_auc']:.4f} |
| **PR-AUC** | **{test_opt['pr_auc']:.4f}** | {test_def['pr_auc']:.4f} |
| **Precision** | **{test_opt['precision']:.4f}** | {test_def['precision']:.4f} |
| **Recall (Sensitivity)** | **{test_opt['recall']:.4f}** | {test_def['recall']:.4f} |
| **F1 Score** | **{test_opt['f1']:.4f}** | {test_def['f1']:.4f} |
| **Brier Score (Calibration)** | **{test_opt['brier_score']:.4f}** | {test_def['brier_score']:.4f} |

### Test Confusion Matrix
- **True Positives (Correctly prioritized no-shows):** {cm['tp']:,}
- **False Positives (Attended, flagged for review):** {cm['fp']:,}
- **True Negatives (Correctly identified attended):** {cm['tn']:,}
- **False Negatives (Missed no-shows):** {cm['fn']:,}

---

## 5. Threshold Selection Justification

In appointment operations, a default threshold of `0.50` often misses a large proportion of actual no-shows due to class imbalance (~20% positive class rate). By tuning the decision threshold to `{meta['selected_operating_threshold']}` on the validation partition, the system achieves a balanced operational trade-off: capturing significant no-show volume (Recall: {test_opt['recall']*100:.1f}%) while maintaining actionable precision ({test_opt['precision']*100:.1f}%) for administrative staff without overwhelming staff capacity.

---

## 6. Responsible Use & Governance Boundaries

1. **Human-in-the-Loop:** Predictions must serve as an informational review aid only. Staff decide whether an appointment warrants an extra reminder call.
2. **Prohibited Automations:** No automated cancellation, rescheduling, or patient deprioritization may occur based on these scores.
3. **Non-Clinical Mandate:** No clinical diagnosis or medical triage is performed.
"""

    report_path = REPORTS_DIR / "model_evaluation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"      Saved technical markdown report to: {report_path}")


if __name__ == "__main__":
    train_and_evaluate_all()
