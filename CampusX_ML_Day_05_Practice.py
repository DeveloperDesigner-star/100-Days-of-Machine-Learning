"""
CampusX 100 Days of Machine Learning — Day 5 Practice
Topics: problem framing, end-to-end workflow, CSV/pandas, X/y,
train/test split, model evaluation, missing values, data leakage.
Install: pip install pandas scikit-learn
"""
from pathlib import Path
import pandas as pd

# Create a small, illustrative dataset and save it as CSV.
sample = {
    "area": [1000, 1500, 2000, 1200, 1800, 900, 2200, 1600],
    "bedrooms": [2, 3, 4, 2, 3, 2, 4, 3],
    "age": [5, 3, 2, 8, 4, 10, 1, 6],
    "price": [8000000, 12000000, 17000000, 9000000,
              14000000, 7000000, 19000000, 13000000],
}
df_sample = pd.DataFrame(sample)
csv_path = Path(__file__).with_name("houses.csv")
df_sample.to_csv(csv_path, index=False)

# 1. Load and inspect the CSV.
df = pd.read_csv(csv_path)
print("First rows:\\n", df.head())
print("Shape (rows, columns):", df.shape)  # (8, 4)
print("Columns:", df.columns.tolist())
print("\\nData types and non-null counts:")
df.info()
print("\\nMissing values per column:\\n", df.isnull().sum())
print("\\nNumerical summary:\\n", df.describe())

# 2. Separate features and target.
X = df[["area", "bedrooms", "age"]]
y = df["price"]
print("\\nFeatures X:\\n", X.head())
print("\\nTarget y:\\n", y.head())

# 3. Train a basic regression model and evaluate on held-out rows.
try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Actual prices:", y_test.to_list())
    print("Predicted prices:", [round(float(p), 2) for p in predictions])
    print("MAE (PKR):", round(mean_absolute_error(y_test, predictions), 2))
    print("RMSE (PKR):", round(mean_squared_error(y_test, predictions) ** 0.5, 2))
    print("R²:", round(r2_score(y_test, predictions), 3))
    print("Reminder: this tiny invented dataset is for practice, not real predictions.")
except ImportError:
    print("Install scikit-learn to run the optional model section: pip install scikit-learn")

# 4. Missing-value demonstration.
df_missing = df.copy()
df_missing.loc[2, "age"] = None
print("\\nMissing-value demo:\\n", df_missing.isnull().sum())

# 5. Exercises.
print("""
PRACTICE TASKS
1. Add a location column.
2. Is spam/not-spam prediction regression or classification?
3. Print rows where area > 1500.
4. Why must price not be included in X when y is price?
5. Give two examples of data leakage.
6. What would df.shape return for 1,000 rows and 5 columns?
Answer key:
1. Add df["location"] = ["Burewala", ...] with one value per row.
2. Classification: the output is a label.
3. print(df[df["area"] > 1500])
4. It would expose the answer as an input and cause leakage.
5. Using post-event information, or fitting preprocessing on all data
   before a split when it learns from the test set.
6. (1000, 5)
""")
