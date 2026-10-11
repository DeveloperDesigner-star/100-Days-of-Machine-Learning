"""CampusX 100 Days of Machine Learning — Day 2 practice.
Topics: batch vs online learning; instance-based vs model-based learning.
Optional dependencies: pip install scikit-learn numpy
Run: python CampusX_ML_Day_02_Complete_Practice.py
"""
def nearest_neighbor_prediction(examples, new_hours):
    """Return target value and closest (hours, marks) training example."""
    closest = min(examples, key=lambda pair: abs(pair[0] - new_hours))
    return closest[1], closest

def demo_instance_based():
    print("\\n=== Instance-based learning: nearest example ===")
    data = [(2, 40), (4, 60), (6, 80)]
    hours = 4.5
    marks, neighbor = nearest_neighbor_prediction(data, hours)
    print("Training examples (hours, marks):", data)
    print("New student's study hours:", hours)
    print("Closest stored example:", neighbor)
    print("Simple 1-nearest-neighbor prediction:", marks)
    print("Teaching demo only; real KNN may use multiple neighbours and a distance metric.")

def demo_model_based():
    print("\\n=== Model-based learning: linear regression ===")
    try:
        import numpy as np
        from sklearn.linear_model import LinearRegression
    except ImportError:
        print("Optional packages missing. Install with: pip install scikit-learn numpy")
        return
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([35, 40, 50, 55, 65], dtype=float)
    model = LinearRegression().fit(X, y)
    prediction = model.predict(np.array([[4.5]]))[0]
    print("Learned slope:", round(float(model.coef_[0]), 2))
    print("Learned intercept:", round(float(model.intercept_), 2))
    print("Predicted marks at 4.5 study hours:", round(float(prediction), 2))
    print("A real model must be evaluated on suitable unseen data.")

def explain_modes():
    print("\\n=== Quick comparison ===")
    print("Batch learning: train on collected data in a separate training cycle.")
    print("Online learning: update incrementally from incoming data or mini-batches.")
    print("Instance-based: compare a new case with stored examples.")
    print("Model-based: learn a general pattern or relationship.")
    print("Online does not necessarily mean updating after every individual record.")

def mini_exercise():
    print("\\n=== Mini exercise / answers for revision ===")
    print("1. Retrain a forecast model monthly -> batch learning.")
    print("2. Update a model in small chunks as data arrives -> online learning.")
    print("3. Compare new case with similar stored cases -> instance-based learning.")
    print("4. Learn coefficients for a prediction equation -> model-based learning.")

if __name__ == "__main__":
    explain_modes()
    demo_instance_based()
    demo_model_based()
    mini_exercise()
