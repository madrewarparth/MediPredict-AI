"""
MediPredict AI - Realistic Synthetic Data Generator
Generates high-fidelity, de-identified synthetic appointment records mirroring
the schema, distributions, and behavioral relationships of the Kaggle
Medical Appointment No-Shows dataset (Vitoria, Brazil).
"""

import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.config import DATA_RAW_DIR, RAW_DATA_FILE

# High-frequency neighbourhoods from the Vitoria, Brazil municipal health system
NEIGHBOURHOODS = [
    "JARDIM DA PENHA",
    "MARIA ORTIZ",
    "RESISTÊNCIA",
    "JARDIM CAMBURI",
    "ITARARÉ",
    "CENTRO",
    "TABUAZEIRO",
    "SANTA MARTHA",
    "JESUS DE NAZARETH",
    "BONFIM",
    "SANTO ANTÔNIO",
    "CARATOÍRA",
    "JABOUR",
    "SÃO PEDRO",
    "ILHA DO PRÍNCIPE",
    "NOVA PALESTINA",
    "DA PENHA",
    "BELA VISTA",
    "SANTOS DUMONT",
    "MARUÍPE",
    "SÃO CRISTÓVÃO",
    "PRAIA DO CANTO",
    "GOIABEIRAS",
    "ANDORINHAS",
    "CONQUISTA",
    "GURIGICA",
    "ILHA DAS CAIEIRAS",
    "SÃO JOSÉ",
    "SANTA CLARA",
    "FONTE GRANDE",
]


def generate_synthetic_appointments(
    n_samples: int = 110527, random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic appointment records with realistic demographic,
    temporal, and attendance patterns.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Unique IDs
    appointment_ids = 5600000 + np.arange(n_samples)
    # Simulate repeat patients (~62,000 unique patients for 110k appointments)
    n_patients = int(n_samples * 0.56)
    patient_pool = rng.integers(100000000, 999999999, size=n_patients)
    patient_ids = rng.choice(patient_pool, size=n_samples)

    # 2. Gender (~65% Female, ~35% Male)
    gender = rng.choice(["F", "M"], size=n_samples, p=[0.65, 0.35])

    # 3. Age (bimodal: pediatric + adults peaking around 35-55)
    n_pediatric = int(n_samples * 0.15)
    n_adult = n_samples - n_pediatric
    age_raw = np.concatenate([
        rng.normal(loc=8, scale=5, size=n_pediatric),
        rng.normal(loc=42, scale=18, size=n_adult),
    ])
    rng.shuffle(age_raw)
    age = np.clip(np.round(age_raw), 0, 102).astype(int)

    # 4. Neighbourhoods
    # Log-normal distribution of patient volume across clinics
    weights = np.exp(-np.linspace(0, 2.5, len(NEIGHBOURHOODS)))
    probs = weights / weights.sum()
    neighbourhood = rng.choice(NEIGHBOURHOODS, size=n_samples, p=probs)

    # 5. Socioeconomic Scholarship (Bolsa Família, ~9.8%)
    scholarship = rng.choice([0, 1], size=n_samples, p=[0.902, 0.098])

    # 6. Sensitive clinical variables (included in raw data schema for realism,
    # but strictly EXCLUDED from model features per project policy)
    hipertension = np.where(age > 40, rng.choice([0, 1], size=n_samples, p=[0.68, 0.32]), 0)
    diabetes = np.where(age > 45, rng.choice([0, 1], size=n_samples, p=[0.82, 0.18]), 0)
    alcoholism = rng.choice([0, 1], size=n_samples, p=[0.97, 0.03])
    handcap = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.978, 0.018, 0.003, 0.0007, 0.0003])

    # 7. Timestamps and Waiting Time (Lead Time)
    # Start date for scheduling: 2016-04-10
    base_schedule_date = datetime(2016, 4, 10, 7, 0, 0)
    schedule_span_days = 60

    schedule_offsets = rng.uniform(0, schedule_span_days, size=n_samples)
    scheduled_days = []
    appointment_days = []
    waiting_days_list = []

    # Waiting time distribution:
    # 35% same day (0 days)
    # 25% short (1-7 days)
    # 22% medium (8-21 days)
    # 18% long (22-60 days)
    wait_type = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.35, 0.25, 0.22, 0.18])

    for i in range(n_samples):
        sched_sec = int(schedule_offsets[i] * 86400)
        sched_dt = base_schedule_date + timedelta(seconds=sched_sec)
        # Business scheduling hours: 07:00 to 19:00
        sched_dt = sched_dt.replace(
            hour=int(rng.integers(7, 20)),
            minute=int(rng.integers(0, 60)),
            second=int(rng.integers(0, 60)),
        )

        wt = wait_type[i]
        if wt == 0:
            wait = 0
        elif wt == 1:
            wait = int(rng.integers(1, 8))
        elif wt == 2:
            wait = int(rng.integers(8, 22))
        else:
            wait = int(rng.integers(22, 65))

        app_dt = (sched_dt + timedelta(days=wait)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        # Avoid Sunday appointments (clinics closed on Sunday)
        if app_dt.weekday() == 6:
            app_dt += timedelta(days=1)
            wait += 1

        scheduled_days.append(sched_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        appointment_days.append(app_dt.strftime("%Y-%m-%dT00:00:00Z"))
        waiting_days_list.append(wait)

    waiting_days_arr = np.array(waiting_days_list)

    # 8. Post-booking event: SMS received (~32% overall, but only sent for appointments > 2 days ahead)
    sms_received = np.where(
        waiting_days_arr >= 3,
        rng.choice([0, 1], size=n_samples, p=[0.45, 0.55]),
        0,
    )

    # 9. Realistic No-Show probability function (ground truth probabilistic generation)
    # Same-day bookings have ~4.5% no-show rate.
    # Long wait times increase no-show rate up to ~32%.
    # Younger patients (18-30) have higher no-show rates (~24%) than seniors 60+ (~15%).
    # Scholarship slightly increases no-show (~24% vs 19%).
    logits = (
        -1.45  # Base intercept calibrated to ~20.2% positive class rate
        + 0.040 * np.log1p(waiting_days_arr) * 4.0
        - 0.012 * (age - 37)
        + 0.35 * scholarship
        - 1.8 * (waiting_days_arr == 0).astype(float)
        + rng.normal(0, 0.45, size=n_samples)  # Unobserved real-world variance
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    # Calibrate overall mean no-show rate to ~20.2%
    is_noshow = rng.uniform(0, 1, size=n_samples) < probs
    noshow_str = np.where(is_noshow, "Yes", "No")

    df = pd.DataFrame({
        "PatientId": patient_ids,
        "AppointmentID": appointment_ids,
        "Gender": gender,
        "ScheduledDay": scheduled_days,
        "AppointmentDay": appointment_days,
        "Age": age,
        "Neighbourhood": neighbourhood,
        "Scholarship": scholarship,
        "Hipertension": hipertension,
        "Diabetes": diabetes,
        "Alcoholism": alcoholism,
        "Handcap": handcap,
        "SMS_received": sms_received,
        "No-show": noshow_str,
    })

    # Inject a tiny number of realistic raw data anomalies to exercise data quality checks
    # E.g., 2 records with age = -1, 2 records where ScheduledDay is after AppointmentDay
    df.loc[12, "Age"] = -1
    df.loc[45, "Age"] = -1
    df.loc[88, "AppointmentDay"] = "2016-04-01T00:00:00Z"  # Negative waiting days

    return df


def ensure_dataset(force: bool = False):
    """Checks for existing dataset, or generates synthetic data if missing or forced."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not force and RAW_DATA_FILE.exists() and RAW_DATA_FILE.stat().st_size > 1000:
        print(f"Dataset already exists at: {RAW_DATA_FILE} ({RAW_DATA_FILE.stat().st_size / 1e6:.2f} MB)")
        return RAW_DATA_FILE

    print("Generating high-fidelity synthetic benchmark dataset (110,527 records)...")
    df = generate_synthetic_appointments(n_samples=110527, random_seed=42)
    df.to_csv(RAW_DATA_FILE, index=False)
    print(f"Successfully generated and saved {len(df):,} records to {RAW_DATA_FILE}")
    return RAW_DATA_FILE


if __name__ == "__main__":
    force_gen = "--force" in sys.argv
    ensure_dataset(force=force_gen)
