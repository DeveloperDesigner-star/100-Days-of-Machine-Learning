"""
CampusX 100 Days of Machine Learning — Day 4 Practice
Topics: tensors, shape, rank, matrix validation, Anaconda, Jupyter, Google Colab.
Standard-library only. Run: python CampusX_ML_Day_04_Practice.py
"""

import platform
import sys


def shape_of_nested_list(value):
    """Return shape for a regular rectangular nested list; scalar -> ()."""
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    sub_shapes = [shape_of_nested_list(item) for item in value]
    if any(s != sub_shapes[0] for s in sub_shapes):
        raise ValueError("Ragged structure: nested items have inconsistent shapes.")
    return (len(value),) + sub_shapes[0]


def validate_matrix(matrix, expected_rows=None, expected_cols=None):
    if not isinstance(matrix, list):
        raise TypeError("A matrix must be a list of rows.")
    if not matrix:
        rows, cols = 0, 0
    else:
        if not all(isinstance(row, list) for row in matrix):
            raise TypeError("Every matrix row must be a list.")
        rows, cols = len(matrix), len(matrix[0])
        if any(len(row) != cols for row in matrix):
            raise ValueError("Matrix rows have different lengths.")
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {rows}.")
    if expected_cols is not None and cols != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, got {cols}.")
    return rows, cols


def show(label, data):
    shape = shape_of_nested_list(data)
    print(f"{label}: shape={shape}, rank={len(shape)}")


def main():
    print("=" * 60)
    print("CAMPUSX ML — DAY 4 PRACTICE")
    print("=" * 60)
    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform(), "\\n")

    scalar = 7
    vector = [10, 20, 30]
    matrix = [[1, 2, 3], [4, 5, 6]]
    cube = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    print("BASIC STRUCTURES")
    print("Scalar:", scalar, "-> shape=(), rank=0")
    show("Vector", vector)
    show("Matrix", matrix)
    show("3D nested list", cube)
    print()

    print("IMAGE SHAPES")
    rgb_shape = (32, 64, 64, 3)
    print("32 RGB images, 64x64 pixels, channels-last:", rgb_shape)
    print("Rank:", len(rgb_shape), "(batch, height, width, channels)")
    gray_shape = (12, 28, 28)
    print("12 grayscale images, 28x28 pixels:", gray_shape)
    print()

    print("STUDENT MARKS MATRIX")
    marks = [[(student + subject) % 101 for subject in range(5)]
             for student in range(100)]
    rows, cols = validate_matrix(marks, 100, 5)
    print(f"Shape=({rows}, {cols}); {rows} students, {cols} subjects")
    print("Number of entries:", rows * cols)
    print("First student's demonstration marks:", marks[0])
    print()

    print("RECTANGULAR DATA CHECK")
    good = [[1, 2], [3, 4], [5, 6]]
    print("Valid example shape:", shape_of_nested_list(good))
    try:
        shape_of_nested_list([[1, 2], [3]])
    except ValueError as exc:
        print("Correctly detected irregular shape:", exc)
    print()

    print("TOOL QUICK GUIDE")
    print("Python       = programming language")
    print("Anaconda     = distribution/environment and package tools")
    print("Jupyter      = interactive notebook interface/application")
    print("Google Colab = hosted notebook service in a browser")
    print()

    print("QUICK ANSWERS")
    print("8 rows x 4 columns -> shape (8, 4), rank 2, 32 elements")
    print("100 students x 5 subjects -> shape (100, 5), 500 entries")
    print("A tensor's rank describes dimensions; it does NOT describe accuracy.")
    print("\\nDone. Nested lists here are teaching examples, not optimized tensors.")


if __name__ == "__main__":
    main()
