"""
CampusX — 100 Days of Machine Learning
Day 2 Practice File: Videos 4–7
Topics:
  1) Batch vs online learning
  2) Instance-based vs model-based learning
  3) ML challenges: data quality, representativeness, overfitting,
     underfitting, irrelevant features, and data drift

This file uses small synthetic educational examples.
It is not a production ML pipeline.
"""

from __future__ import annotations

from math import sqrt
from typing import Iterable, Sequence


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return Euclidean distance between two same-length numeric points."""
    if len(a) != len(b):
        raise ValueError("Points must have the same number of features.")
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def knn_predict(
    training_x: Sequence[Sequence[float]],
    training_y: Sequence[str],
    new_x: Sequence[float],
    k: int = 3,
) -> str:
    """
    Tiny KNN classifier demonstrating instance-based learning.
    It stores examples and predicts using the labels of nearest examples.
    For real use, scale numeric features and validate k properly.
    """
    if not training_x or len(training_x) != len(training_y):
        raise ValueError("Training features and labels must be non-empty and aligned.")
    if not 1 <= k <= len(training_x):
        raise ValueError("k must be between 1 and the number of training rows.")

    ranked = sorted(
        (
            (euclidean_distance(row, new_x), label)
            for row, label in zip(training_x, training_y)
        ),
        key=lambda pair: pair[0],
    )
    votes: dict[str, int] = {}
    for _, label in ranked[:k]:
        votes[label] = votes.get(label, 0) + 1
    # Stable tie-break: highest vote count, then alphabetical label.
    return sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]


def simple_linear_regression_predict(
    x_train: Sequence[float],
    y_train: Sequence[float],
    x_new: float,
) -> float:
    """
    Fit a simple one-feature least-squares line and predict y for x_new.
    This compact implementation is for learning; scikit-learn is preferred
    for real workflows and robust validation.
    """
    if len(x_train) != len(y_train) or len(x_train) < 2:
        raise ValueError("Provide at least two aligned training observations.")

    x_mean = sum(x_train) / len(x_train)
    y_mean = sum(y_train) / len(y_train)
    denominator = sum((x - x_mean) ** 2 for x in x_train)
    if denominator == 0:
        raise ValueError("All x values are identical; a line cannot be fitted.")
    slope = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_train, y_train)
    ) / denominator
    intercept = y_mean - slope * x_mean
    return intercept + slope * x_new


def describe_update_strategies() -> None:
    """Print the key difference between batch and online learning."""
    print("\\n=== Batch vs Online Learning ===")
    print("Batch: train on a collected dataset in a separate training cycle.")
    print("Online: update incrementally as new examples/mini-batches arrive.")
    print("Safety: check data quality and monitor performance before trusting updates.")


def describe_ml_challenges() -> None:
    """Print short definitions of common ML challenges."""
    challenges = {
        "Insufficient data": "Too few useful examples for reliable learning.",
        "Non-representative data": "Training sample does not reflect intended users.",
        "Poor-quality data": "Missing, incorrect, duplicated, or mislabeled values.",
        "Irrelevant features": "Inputs add little signal or distract the model.",
        "Overfitting": "Training performance is strong, unseen performance is weak.",
        "Underfitting": "The model performs poorly on both training and test data.",
        "Data drift": "Input patterns or real-world relationships change over time.",
    }
    print("\\n=== Machine Learning Challenges ===")
    for name, meaning in challenges.items():
        print(f"- {name}: {meaning}")


def data_quality_checklist(rows: Iterable[dict]) -> None:
    """
    Simple illustrative check for missing values.
    This does not replace schema validation, range checks, or domain review.
    """
    rows = list(rows)
    print("\\n=== Basic Data Quality Check ===")
    if not rows:
        print("Dataset is empty.")
        return

    keys = sorted({key for row in rows for key in row})
    for key in keys:
        missing = sum(key not in row or row[key] is None for row in rows)
        print(f"{key}: {missing} missing value(s) across {len(rows)} row(s)")
    print("Also check impossible values, duplicates, inconsistent units, and labels.")


def quiz_scenarios() -> None:
    """Quick self-test based on Day 2 Video 7."""
    scenarios = [
        ("99% training accuracy and 60% test accuracy", "Overfitting"),
        ("Poor results on both training and test sets", "Underfitting"),
        ("Only urban data but deployment includes rural users", "Non-representative data"),
        ("Missing and incorrectly entered marks", "Inspect and clean data"),
        ("Recommendations worsen as preferences change", "Investigate data drift"),
    ]
    print("\\n=== Day 2 Self-Test: Answers ===")
    for scenario, answer in scenarios:
        print(f"Scenario: {scenario}\\nLikely answer: {answer}\\n")


def main() -> None:
    print("CampusX ML — Day 2 practice (Videos 4–7)")
    describe_update_strategies()

    # Instance-based example: each row is [attendance, assessment score].
    student_examples = [
        [95.0, 88.0],
        [90.0, 82.0],
        [60.0, 55.0],
        [65.0, 58.0],
        [80.0, 75.0],
    ]
    student_outcomes = ["Likely pass", "Likely pass", "Needs support",
                        "Needs support", "Likely pass"]
    new_student = [78.0, 72.0]
    prediction = knn_predict(student_examples, student_outcomes, new_student, k=3)
    print("\\n=== Instance-Based Learning: KNN ===")
    print(f"New example [attendance, assessment] = {new_student}")
    print(f"Illustrative KNN prediction = {prediction}")
    print("KNN compares the new example with stored examples; scale features in real projects.")

    # Model-based example: learn a line from synthetic x/y data.
    sizes = [500, 700, 900, 1100, 1300]
    prices = [120, 160, 205, 245, 290]  # synthetic price units
    predicted_price = simple_linear_regression_predict(sizes, prices, 1000)
    print("\\n=== Model-Based Learning: Linear Regression ===")
    print(f"Estimated target for size 1000 = {predicted_price:.2f} synthetic units")
    print("Regression learns a general relationship instead of searching every stored row.")

    # Data-quality example.
    sample_rows = [
        {"attendance": 95, "marks": 88},
        {"attendance": 82, "marks": None},
        {"attendance": 101, "marks": 73},  # out-of-range value: inspect!
    ]
    data_quality_checklist(sample_rows)
    print("Reminder: the simple check above detects missing values, not all invalid values.")

    describe_ml_challenges()
    quiz_scenarios()


if __name__ == "__main__":
    main()
