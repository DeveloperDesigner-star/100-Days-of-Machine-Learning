"""
CampusX 100 Days of Machine Learning — Day 3 Practice
Official playlist Videos 8, 9, and 10
Topics:
  8. Real-life applications of Machine Learning
  9. Machine Learning Development Life Cycle (MLDLC)
 10. Data Engineer vs Data Analyst vs Data Scientist vs ML Engineer

This is a beginner-friendly, self-contained practice file.
Requirements: Python 3. No third-party packages required.

Run:
    python CampusX_ML_Day_03_Practice.py

The examples are educational demonstrations, not production-ready systems.
"""

from collections import Counter
from statistics import mean
from math import sqrt


def moving_average_forecast(values, window=3):
    """Forecast the next value using the mean of the most recent window."""
    if window < 1:
        raise ValueError("window must be at least 1")
    if not values:
        raise ValueError("values must not be empty")
    recent = values[-window:]
    return mean(recent)


def classify_review(text):
    """
    Tiny rule-based sentiment baseline.
    This is NOT a trained ML model and cannot understand context, sarcasm,
    negation, spelling variations, or nuanced language.
    """
    positive_words = {
        "good", "great", "excellent", "love", "useful", "easy", "fast",
        "helpful", "amazing", "perfect", "reliable"
    }
    negative_words = {
        "bad", "poor", "hate", "slow", "broken", "difficult", "awful",
        "useless", "error", "unreliable", "terrible"
    }
    tokens = [token.strip(".,!?;:()[]{}\"'").lower()
              for token in text.split()]
    positive_score = sum(token in positive_words for token in tokens)
    negative_score = sum(token in negative_words for token in tokens)

    if positive_score > negative_score:
        label = "positive"
    elif negative_score > positive_score:
        label = "negative"
    else:
        label = "neutral/uncertain"

    return {
        "label": label,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "warning": "Simple word-list baseline, not a trained ML model."
    }


def flag_anomalies(values, threshold=2.0):
    """
    Flag values far from the mean using a simple standard-deviation rule.
    For learning only; real anomaly detection needs domain context and
    careful handling of skewed data, small samples, and outliers.
    """
    if len(values) < 2:
        return []
    center = mean(values)
    variance = sum((x - center) ** 2 for x in values) / len(values)
    std_dev = sqrt(variance)
    if std_dev == 0:
        return []
    return [
        {"index": i, "value": value, "distance_in_std_devs": abs(value - center) / std_dev}
        for i, value in enumerate(values)
        if abs(value - center) / std_dev >= threshold
    ]


def recommend_top_items(user_preferences, catalog, top_n=3):
    """
    Simple content-score recommender.
    user_preferences and catalog map item names to sets of tags.
    Items are ranked by the number of matching tags.
    This is a toy example, not a production recommendation engine.
    """
    preference_tags = set(user_preferences)
    ranked = []
    for item_name, tags in catalog.items():
        score = len(preference_tags.intersection(set(tags)))
        ranked.append((item_name, score))
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    return ranked[:top_n]


def clean_student_records(records):
    """
    Minimal data-quality checks for a CGPA dataset.
    Expected record format:
        {"study_hours": 3.5, "attendance_pct": 85, "previous_gpa": 3.2, "cgpa": 3.4}
    Removes records with missing values and out-of-range CGPA/attendance.
    Real projects should investigate why records are missing and avoid
    silently deleting data without checking potential bias.
    """
    cleaned = []
    for row in records:
        required = ("study_hours", "attendance_pct", "previous_gpa", "cgpa")
        if any(key not in row or row[key] is None for key in required):
            continue
        if not (0 <= row["attendance_pct"] <= 100):
            continue
        if not (0 <= row["previous_gpa"] <= 4):
            continue
        if not (0 <= row["cgpa"] <= 4):
            continue
        if row["study_hours"] < 0:
            continue
        cleaned.append(dict(row))
    return cleaned


def predict_cgpa_baseline(records, study_hours, attendance_pct, previous_gpa):
    """
    Transparent educational baseline: use the average CGPA among records
    with the nearest previous GPA. This is not a validated ML model.
    Do not use such a toy prediction for academic decisions.
    """
    if not records:
        raise ValueError("Need at least one cleaned record")
    nearest = min(records, key=lambda row: abs(row["previous_gpa"] - previous_gpa))
    # Tiny illustrative adjustment only, deliberately capped to the valid scale.
    adjustment = 0.015 * (study_hours - 3) + 0.002 * (attendance_pct - 75)
    prediction = nearest["cgpa"] + adjustment
    return max(0.0, min(4.0, prediction))


def role_guide():
    """Print a concise comparison of common data/ML roles."""
    roles = {
        "Data Engineer": "Builds and maintains data pipelines, storage, and data infrastructure.",
        "Data Analyst": "Queries data, creates reports/dashboards, and communicates insights.",
        "Data Scientist": "Uses statistics, experimentation, analysis, and predictive modeling.",
        "ML Engineer": "Integrates, deploys, monitors, and maintains ML systems in production.",
    }
    for role, description in roles.items():
        print(f"- {role}: {description}")


def run_demo():
    print("=" * 68)
    print("CAMPUSX DAY 3 — MACHINE LEARNING PRACTICE")
    print("=" * 68)

    sales = [100, 110, 105, 120, 125]
    print("\n1) Simple forecasting (moving average)")
    print(f"Recent sales: {sales}")
    print(f"Next-value baseline: {moving_average_forecast(sales, window=3):.1f}")

    print("\n2) Sentiment baseline")
    for review in (
        "This app is great and useful",
        "The app is slow and broken",
        "The app arrived yesterday"
    ):
        print(f"{review!r} -> {classify_review(review)['label']}")

    print("\n3) Simple anomaly demonstration")
    values = [10, 11, 9, 10, 12, 11, 80]
    print(f"Values: {values}")
    print(f"Potential anomalies: {flag_anomalies(values, threshold=1.8)}")

    print("\n4) Recommendation baseline")
    catalog = {
        "Movie A": {"comedy", "family"},
        "Movie B": {"science-fiction", "adventure"},
        "Movie C": {"documentary", "science"},
        "Movie D": {"adventure", "fantasy"},
    }
    print(recommend_top_items({"adventure", "science-fiction"}, catalog, top_n=3))

    print("\n5) CGPA dataset quality checks and toy baseline")
    raw_records = [
        {"study_hours": 3, "attendance_pct": 80, "previous_gpa": 3.0, "cgpa": 3.1},
        {"study_hours": 5, "attendance_pct": 92, "previous_gpa": 3.4, "cgpa": 3.6},
        {"study_hours": 2, "attendance_pct": 65, "previous_gpa": 2.4, "cgpa": 2.5},
        {"study_hours": 4, "attendance_pct": 88, "previous_gpa": 3.2, "cgpa": 3.3},
        {"study_hours": 2, "attendance_pct": 120, "previous_gpa": 2.7, "cgpa": 2.8},
        {"study_hours": None, "attendance_pct": 70, "previous_gpa": 2.6, "cgpa": 2.7},
    ]
    cleaned = clean_student_records(raw_records)
    print(f"Raw records: {len(raw_records)} | Valid records: {len(cleaned)}")
    prediction = predict_cgpa_baseline(
        cleaned, study_hours=4, attendance_pct=85, previous_gpa=3.1
    )
    print(f"Illustrative CGPA baseline: {prediction:.2f} / 4.00")
    print("Reminder: this toy baseline is not validated for real student decisions.")

    print("\n6) Data/ML job roles")
    role_guide()

    print("\n7) Day 3 lifecycle checklist")
    lifecycle = [
        "Define problem and success metric",
        "Collect data with permission and appropriate safeguards",
        "Clean data and investigate missing/invalid values",
        "Explore patterns (EDA)",
        "Engineer/select useful features",
        "Split data appropriately; train baseline and candidate models",
        "Evaluate on held-out data using suitable metrics",
        "Deploy only after validation and stakeholder review",
        "Monitor quality, drift, errors, fairness, and cost",
        "Maintain, retrain, or roll back when evidence calls for it",
    ]
    for step_no, step in enumerate(lifecycle, start=1):
        print(f"{step_no}. {step}")

    print("\nPractice completed. Change the sample data and rerun the file.")


if __name__ == "__main__":
    run_demo()
