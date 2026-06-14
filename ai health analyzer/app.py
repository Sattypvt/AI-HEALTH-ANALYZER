import csv
import os
from datetime import datetime

from flask import Flask, redirect, render_template, request, session, url_for

from util.predictor import predict, train_and_save

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)
app.secret_key = "health-analyzer-secret"

HIST_FILE = "history.csv"


# ─── HISTORY HELPERS ─────────────────────────────────────────────────────────

def save_history(row: dict):
    exists = os.path.isfile(HIST_FILE)
    with open(HIST_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_history() -> list:
    if not os.path.isfile(HIST_FILE):
        return []
    with open(HIST_FILE, newline="") as f:
        return list(csv.DictReader(f))


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def run_predict():
    form = request.form

    age         = int(form["age"])
    bmi         = float(form["bmi"])
    bp          = int(form["bp"])
    glucose     = int(form["glucose"])
    cholesterol = int(form["cholesterol"])
    smoking     = int(form["smoking"])

    symptoms = {
        "chest_pain":          "chest_pain"          in form,
        "fatigue":             "fatigue"              in form,
        "frequent_urination":  "frequent_urination"   in form,
        "headache":            "headache"             in form,
        "blurred_vision":      "blurred_vision"       in form,
        "shortness_of_breath": "shortness_of_breath"  in form,
    }

    data = predict(age, bmi, bp, glucose, cholesterol, smoking, symptoms)
    data["symptoms"] = {k: ("Yes" if v else "No") for k, v in symptoms.items()}
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    save_history({
        "date":         data["timestamp"],
        "health_score": data["health_score"],
        "heart_pct":    data["results"]["heart"]["probability"],
        "diabetes_pct": data["results"]["diabetes"]["probability"],
        "hyper_pct":    data["results"]["hypertension"]["probability"],
    })

    session["result"] = data
    return redirect(url_for("result"))


@app.route("/result")
def result():
    data = session.get("result")
    if not data:
        return redirect(url_for("index"))
    return render_template("result.html", data=data)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/history")
def history():
    records = list(reversed(read_history()))
    return render_template("index.html", history=records)


@app.route("/history/clear")
def clear_history():
    if os.path.isfile(HIST_FILE):
        os.remove(HIST_FILE)
    return redirect(url_for("index"))


@app.route("/retrain")
def retrain():
    train_and_save()
    return "Model retrained successfully.", 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)