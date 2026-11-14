# MediPredict AI

### Hospital Appointment No-Show Prediction & Appointment Operations Intelligence

> **Project Type:** AI/ML Internship Assessment Project  
> **Organization:** ALGIONS AI LABS PRIVATE LIMITED  
> **Primary Interface:** Streamlit (Multi-Page Web Dashboard)  
> **Theme:** Clinical Intelligence — Dark Teal  
> **Status:** Production-Ready & Fully Verified  

---

## 1. Executive Summary & Problem Formulation

Missed clinical appointments waste healthcare capacity, inflate waiting lists, and degrade operational efficiency. **MediPredict AI** is a human-in-the-loop decision-support platform designed to estimate the probability that a scheduled hospital appointment will be missed, allowing authorized administrative staff to prioritize manual reminder outreach.

```
Supervised Binary Classification:
- Target Class (1): No-Show (Missed appointment)
- Majority Class (0): Attended (Kept appointment)
- Prediction Moment: Exactly when the appointment is booked (ScheduledDay)
- Intended Action: Prioritize reminder review; NO automatic cancelation
```

> [!IMPORTANT]
> **Operational Decision Support Boundary:**  
> MediPredict AI is an administrative tool. It is **not** a clinical diagnostic system and must **never** be used to diagnose patients, determine medical urgency, deprioritize appointments, or deny care.

---

## 2. Goals & Non-Goals

### Goals
- **Strict Point-in-Time Discipline:** Use only operational and demographic features known at appointment booking time.
- **Leakage Prevention:** Eliminate patient identifiers and post-booking operational events.
- **Chronological Validation:** Train/Validation/Test split ordered strictly by `ScheduledDay` (70% Train, 15% Validation, 15% Test) with zero temporal overlap.
- **Calibrated Inference:** Output true empirical probability estimates rather than distorted raw scores.
- **Decision Threshold Optimization:** Tune operating thresholds on validation data to balance recall and staff workload.
- **Production Artifact:** Save a complete end-to-end `sklearn.pipeline.Pipeline` with schema validation and test coverage.

### Non-Goals & Prohibited Uses
- ❌ **No clinical triage or diagnosis:** Sensitive diagnostic variables (`Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`) are strictly excluded.
- ❌ **No automated cancellations or access barriers:** Staff make all decisions; the system provides informational risk bands only.
- ❌ **No double-booking over-allocation:** Risk scores must not be used to over-schedule clinics.
- ❌ **No causal claims:** Model factors reflect statistical associations, not causes.

---

## 3. Repository Architecture

```text
MediPredict-AI/
│
├── app.py                          # Main portal and operational system overview
├── requirements.txt                # Pinned production dependencies
├── README.md                       # Comprehensive technical documentation
├── .gitignore                      # Git exclusion rules
│
├── .streamlit/
│   └── config.toml                 # Clinical Intelligence Dark Teal theme configuration
│
├── data/
│   ├── raw/
│   │   └── appointments.csv        # Raw dataset (110k records, Kaggle schema)
│   └── processed/
│       ├── appointments_clean.parquet  # High-efficiency processed dataset
│       └── appointments_clean.csv      # Human-readable processed export
│
├── models/
│   ├── appointment_pipeline.joblib # Production fitted preprocessing + calibrated ensemble
│   └── model_metadata.json         # Comprehensive evaluation metrics, thresholds, and splits
│
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory analysis and point-in-time leakage audit
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Paths, feature dictionary, risk thresholds, theme tokens
│   ├── generate_data.py            # High-fidelity synthetic benchmark dataset generator
│   ├── data_preprocessing.py       # Schema validation, date parsing, anomaly filtering
│   ├── feature_engineering.py      # Lead time derivation, OHE, chronological split
│   ├── train_model.py              # Candidate training, threshold tuning, calibration, evaluation
│   ├── predict.py                  # Production single & batch inference engine
│   └── ui_components.py            # Dark teal CSS, metric cards, Plotly templates, badges
│
├── pages/
│   ├── 1_Overview.py               # KPI summary, volume trends, attendance distributions
│   ├── 2_Single_Prediction.py      # Interactive point-of-booking risk assessment form
│   ├── 3_Bulk_Prediction.py        # Batch CSV upload, queue prioritization, CSV export
│   ├── 4_Analytics.py              # Lead time decay curves, temporal patterns, demographics
│   ├── 5_Model_Performance.py      # Candidate benchmark, threshold simulator, calibration curve
│   └── 6_Data_Responsible_Use.py  # Governance policy, leakage audit, ethical boundaries
│
├── tests/
│   ├── test_preprocessing.py       # 7 unit tests (schema, anomaly filters, leakage guards)
│   └── test_inference.py           # 6 integration tests (single/batch inference, row order)
│
├── reports/
│   └── model_evaluation.md         # Detailed technical evaluation audit report
│
└── assets/                         # Static visual assets
```

---

## 4. Point-in-Time Feature Dictionary & Policy

| Feature | Type | Status | Governance & Leakage Justification |
| :--- | :--- | :--- | :--- |
| `PatientId` | Integer | **EXCLUDED** | Direct patient identifier. Prone to memorization; zero generalization. |
| `AppointmentID` | Integer | **EXCLUDED** | Internal primary key with no predictive validity. |
| `Hipertension` | Binary | **EXCLUDED** | Clinical diagnosis. Excluded under strict non-clinical mandate. |
| `Diabetes` | Binary | **EXCLUDED** | Clinical diagnosis. Excluded under strict non-clinical mandate. |
| `Alcoholism` | Binary | **EXCLUDED** | Sensitive clinical indicator excluded by governance policy. |
| `Handcap` | Integer | **EXCLUDED** | Disability indicator excluded to protect patient equity. |
| `SMS_received` | Binary | **EXCLUDED** | Post-booking event. Occurs after scheduling; severe temporal leakage. |
| `Age` | Numeric | **INCLUDED** | Patient age at booking (filtered to valid bounds 0-115). |
| `Gender` | Categorical | **INCLUDED** | Administrative demographic recorded at booking. |
| `Neighbourhood` | Categorical | **INCLUDED** | Clinic catchment area (top 25 retained, rest grouped to 'Other'). |
| `Scholarship` | Binary | **INCLUDED** | Social welfare enrollment indicator (Bolsa Família). |
| `waiting_days` | Numeric | **DERIVED** | Integer days between `ScheduledDay` and `AppointmentDay`. |
| `is_same_day` | Binary | **DERIVED** | Indicator if appointment is booked for today (`waiting_days == 0`). |
| `scheduled_hour`| Numeric | **DERIVED** | Hour of day booking occurred (0-23). |
| `scheduled_day_of_week` | Categorical | **DERIVED** | Day name booking was initiated (Monday - Sunday). |
| `appointment_day_of_week`| Categorical | **DERIVED** | Day name appointment is scheduled for. |

---

## 5. Model Training & Evaluation Summary

### Chronological Splits (Zero Future Leakage)
- **Train Split (70%):** 77,366 samples (2016-04-10 to 2016-05-22)
- **Validation Split (15%):** 16,578 samples (2016-05-22 to 2016-05-31) — Used for model selection & threshold tuning.
- **Held-Out Test Split (15%):** 16,580 samples (2016-05-31 to 2016-06-09) — Evaluated strictly once.

### Model Benchmark (Validation Partition)
| Architecture | Val ROC-AUC | Val PR-AUC | Optimal Threshold | Val Precision | Val Recall | Val F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy (Baseline)** | 0.5000 | 0.2096 | 0.050 | 0.2096 | 1.0000 | 0.3465 |
| **Logistic Regression** | 0.6955 | 0.3365 | 0.530 | 0.3012 | 0.7480 | 0.4296 |
| **Random Forest** | 0.6948 | 0.3370 | 0.520 | 0.3021 | 0.7454 | 0.4298 |
| **HistGradientBoosting (Calibrated)** | **0.6942** | **0.3387** | **0.170** | **0.2798** | **0.9096** | **0.4280** |

### Held-Out Test Evaluation (`HistGradientBoosting (Calibrated)`)
- **Test ROC-AUC:** `0.6969`
- **Test PR-AUC:** `0.3336`
- **Test Recall (No-Show Capture Rate):** `90.31%` (3,104 of 3,437 true no-shows identified)
- **Test Precision:** `28.28%`
- **Test F1 Score:** `0.4307`
- **Brier Score (Calibration Reliability):** `0.1509` (improved from uncalibrated 0.2236)

---

## 6. Risk Bands & Operational Actions

| Risk Band | Color | Threshold Range | Administrative Action |
| :--- | :--- | :--- | :--- |
| **Lower estimated risk** | `#22C55E` | `< 0.25` | Standard automated SMS reminder protocol. |
| **Moderate estimated risk** | `#F59E0B` | `0.25 - 0.50` | Secondary reminder review if administrative capacity allows. |
| **Higher estimated risk** | `#F87171` | `≥ 0.50` | Priority candidate for staff phone call review (48h prior). |

---

## 7. Quickstart & Local Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.10.9)
- Git

### 2. Clone and Setup Environment
```bash
git clone https://github.com/your-username/MediPredict-AI.git
cd MediPredict-AI

# Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Dataset Preprocessing & Model Training
```bash
# If dataset is already in data/raw/appointments.csv, preprocess and train:
python -m src.data_preprocessing
python -m src.train_model
```

### 5. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 8. Verification & Testing

Run the full automated test suite (unit and integration tests):
```bash
python -m pytest tests -v
```
Output:
```text
tests/test_inference.py::test_inference_engine_initialization PASSED
tests/test_inference.py::test_predict_single_returns_valid_probability PASSED
tests/test_inference.py::test_predict_single_invalid_age_raises PASSED
tests/test_inference.py::test_predict_single_negative_waiting_days_raises PASSED
tests/test_inference.py::test_batch_prediction_row_count_and_order_preserved PASSED
tests/test_inference.py::test_missing_model_artifact_raises PASSED
tests/test_preprocessing.py::test_validate_raw_schema_missing_columns PASSED
tests/test_preprocessing.py::test_clean_filters_invalid_age PASSED
tests/test_preprocessing.py::test_clean_filters_negative_waiting_days PASSED
tests/test_preprocessing.py::test_clean_target_encoding PASSED
tests/test_preprocessing.py::test_feature_discipline_rejects_prohibited_columns PASSED
tests/test_preprocessing.py::test_point_in_time_feature_matrix_columns PASSED
tests/test_preprocessing.py::test_chronological_split_integrity PASSED

============================= 13 passed in 5.67s ==============================
```

---

## 9. Visual Design System

The interface adheres to the **Clinical Intelligence — Dark Teal** theme tokens:
- **App Background:** `#0B1220`
- **Sidebar Background:** `#111C2E`
- **Card Background:** `#162338`
- **Primary Teal:** `#14B8A6`
- **Teal Hover:** `#0D9488`
- **Secondary Blue:** `#60A5FA`
- **Main Text:** `#F8FAFC`
- **Muted Text:** `#A7B4C8`
- **Border:** `#2B3B52`

---

## 10. Credits & Disclaimer

This project was built as an assessment prototype for **ALGIONS AI LABS PRIVATE LIMITED**.

**MediPredict AI provides estimates for human administrative review only. It is not a clinical tool, does not determine medical urgency, and must never be used to deny or restrict healthcare access.**
