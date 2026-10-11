"""CampusX 100 Days of Machine Learning — Day 1 Practice
Install: python -m pip install scikit-learn
All examples are invented toy data, not real-world forecasts.
"""
from sklearn.linear_model import LinearRegression


def traditional_programming_example(number: int) -> str:
    """Explicit rules decide whether a number is even or odd."""
    return "even" if number % 2 == 0 else "odd"


def train_house_price_model():
    """Train regression on illustrative house-size and price examples."""
    X = [[500], [700], [900], [1100], [1300]]  # feature: size
    y = [2.0, 2.8, 3.6, 4.4, 5.2]             # target: price, Rs. millions
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_study_hours_model():
    """Mini-project using invented study-hour and marks examples."""
    X = [[2], [4], [6], [8]]  # feature: study hours
    y = [40, 55, 70, 85]      # target: marks
    model = LinearRegression()
    model.fit(X, y)
    return model


def self_review():
    print("\nDAY 1 SELF-REVIEW")
    print("1. Study hours = feature X; exam marks = target y.")
    print("2. House-price prediction = regression (numeric target).")
    print("3. Spam/not spam = classification (category target).")
    print("4. Grouping unlabelled students = unsupervised learning.")
    print("5. Some labelled + more unlabelled examples = semi-supervised learning.")
    print("6. Actions and task-defined rewards = reinforcement learning.")
    print("7. Deep Learning is within ML; ML is within AI.")
    print("8. Unseen-data testing estimates generalisation; it does not automatically")
    print("   improve the model.")


def main():
    print("CAMPUSX ML — DAY 1 PRACTICE")
    print("-" * 36)
    for n in [4, 7, 10]:
        print(f"{n} is {traditional_programming_example(n)}")

    house_model = train_house_price_model()
    size = 1000
    price = house_model.predict([[size]])[0]
    print("\nIllustrative house-price regression:")
    print(f"House size: {size} sq. ft.")
    print(f"Predicted price: Rs. {price:.2f} million")
    print("(Toy data only; not a real property valuation.)")

    student_model = train_study_hours_model()
    hours = 5
    marks = student_model.predict([[hours]])[0]
    print("\nInvented study-hours mini-project:")
    print(f"Study hours: {hours}")
    print(f"Predicted marks: {marks:.1f}")
    print("(Practice data only; real marks depend on many factors.)")
    print("\nTry changing the input hours:")
    for h in [3, 5, 7]:
        print(f"{h} hours -> {student_model.predict([[h]])[0]:.1f} predicted marks")
    self_review()


if __name__ == "__main__":
    main()
