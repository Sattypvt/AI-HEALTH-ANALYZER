import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

FEATURES   = ["age", "bmi", "bp", "glucose", "cholesterol", "smoking"]
TARGETS    = ["heart", "diabetes", "hypertension"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "health_model.pkl")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")


# ─── TRAIN & SAVE ─────────────────────────────────────────────────────────────

def train_and_save() -> dict:
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURES].values
    bundle = {"models": {}, "scalers": {}}

    for target in TARGETS:
        y      = df[target].values
        scaler = StandardScaler()
        Xs     = scaler.fit_transform(X)
        clf    = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(Xs, y)
        bundle["models"][target]  = clf
        bundle["scalers"][target] = scaler

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    return bundle


def load_bundle() -> dict:
    if not os.path.exists(MODEL_PATH):
        return train_and_save()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ─── HEALTH SCORE ─────────────────────────────────────────────────────────────

def compute_health_score(bmi, bp, glucose, cholesterol, smoking, sym_count=0) -> int:
    score = 100
    if bmi < 18.5 or bmi >= 35: score -= 20
    elif bmi >= 30:              score -= 12
    elif bmi >= 25:              score -= 6
    if bp >= 160:    score -= 20
    elif bp >= 140:  score -= 12
    elif bp >= 130:  score -= 6
    if glucose >= 180:   score -= 20
    elif glucose >= 140: score -= 12
    elif glucose >= 110: score -= 6
    if cholesterol >= 260: score -= 15
    elif cholesterol >= 230: score -= 8
    elif cholesterol >= 200: score -= 4
    if smoking == 1: score -= 15
    score -= sym_count * 3
    return max(0, min(100, int(score)))


def score_category(score: int) -> tuple[str, str]:
    if score >= 80: return "Excellent", "#2a9d8f"
    if score >= 60: return "Moderate",  "#e9a820"
    return "Poor", "#d62828"


# ─── PREDICT ──────────────────────────────────────────────────────────────────

def predict(age, bmi, bp, glucose, cholesterol, smoking, symptoms: dict) -> dict:
    bundle  = load_bundle()
    models  = bundle["models"]
    scalers = bundle["scalers"]

    X         = np.array([[age, bmi, bp, glucose, cholesterol, smoking]])
    sym_count = sum(symptoms.values())
    results   = {}

    for target in TARGETS:
        scaler = scalers[target]
        clf    = models[target]
        prob   = float(clf.predict_proba(scaler.transform(X))[0][1])

        # Symptom boost
        if symptoms.get("chest_pain")          and target == "heart":        prob = min(1.0, prob + 0.12)
        if symptoms.get("shortness_of_breath") and target == "heart":        prob = min(1.0, prob + 0.08)
        if symptoms.get("frequent_urination")  and target == "diabetes":     prob = min(1.0, prob + 0.12)
        if symptoms.get("blurred_vision")      and target == "diabetes":     prob = min(1.0, prob + 0.08)
        if symptoms.get("headache")            and target == "hypertension": prob = min(1.0, prob + 0.12)
        if symptoms.get("fatigue"):             prob = min(1.0, prob + 0.05)

        results[target] = {
            "probability": round(prob * 100, 1),
            "is_high":     prob >= 0.5,
            "emergency":   prob >= 0.75,
            "label":       "High Risk" if prob >= 0.5 else "Low Risk",
            "color":       "#d62828" if prob >= 0.5 else "#2a9d8f",
        }

    health_score            = compute_health_score(bmi, bp, glucose, cholesterol, smoking, sym_count)
    category, cat_color     = score_category(health_score)
    critical                = [t.replace("_", " ").title() for t in TARGETS if results[t]["emergency"]]

    return {
        "results":        results,
        "health_score":   health_score,
        "category":       category,
        "cat_color":      cat_color,
        "emergency":      len(critical) > 0,
        "critical":       critical,
        "inputs": {
            "age": age, "bmi": bmi, "bp": bp,
            "glucose": glucose, "cholesterol": cholesterol,
            "smoking": "Yes" if smoking == 1 else "No",
        },
    }