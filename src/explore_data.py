# src/explore_data.py

import pandas as pd

# Load both datasets, and tag each row with which type of wine it is.
# This lets us combine them later if we want, while still knowing which is which.
red_wine = pd.read_csv("data/raw/winequality-red.csv", sep=";")
red_wine["wine_type"] = "red"

white_wine = pd.read_csv("data/raw/winequality-white.csv", sep=";")
white_wine["wine_type"] = "white"

# --- Check 1: Are there any missing values? ---
# .isnull() marks each cell True/False if it's empty, .sum() counts Trues per column
print("Missing values in RED wine:")
print(red_wine.isnull().sum())

print("\nMissing values in WHITE wine:")
print(white_wine.isnull().sum())

# --- Check 2: Are there duplicate rows? ---
# .duplicated() flags rows that are exact copies of an earlier row
print("\nDuplicate rows in RED wine:", red_wine.duplicated().sum())
print("Duplicate rows in WHITE wine:", white_wine.duplicated().sum())

# --- Check 3: What does the quality score distribution look like? ---
# .value_counts() counts how many rows fall into each unique value,
# .sort_index() puts them in order (3, 4, 5, 6, 7, 8...) instead of by frequency
print("\nRED wine quality distribution:")
print(red_wine["quality"].value_counts().sort_index())

print("\nWHITE wine quality distribution:")
print(white_wine["quality"].value_counts().sort_index())

# --- Check 4: Basic statistics for every numeric column ---
# .describe() gives mean, std, min, max, and quartiles — good for spotting outliers
print("\nRED wine summary statistics:")
print(red_wine.describe())