# src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Step 1: Load the data ---
red_wine = pd.read_csv("data/raw/winequality-red.csv", sep=";")

# --- Step 2: Remove duplicate rows ---
# Duplicates can make our model look artificially good, since the same
# example could end up in both training and test sets.
before = len(red_wine)
red_wine = red_wine.drop_duplicates()
after = len(red_wine)
print(f"Removed {before - after} duplicate rows. {after} rows remain.")

# --- Step 3: Split into features (X) and target (y) ---
# X = everything the model uses to make a prediction (the 11 chemistry columns)
# y = the thing we're trying to predict (the quality score)
X = red_wine.drop(columns=["quality"])
y = red_wine["quality"]

print("\nFeature columns (X):", list(X.columns))
print("Target (y) sample values:", y.head().tolist())

# --- Step 4: Split into train and test sets ---
# We train the model on 80% of the data, and hold back 20% it has never
# seen, to check if it actually learned something generalizable.
# stratify=y keeps the same quality-score proportions in both splits,
# which matters a lot here since some quality scores are rare.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)} rows")
print(f"Test set size: {len(X_test)} rows")

# --- Step 5: Scale the features ---
# Different columns have very different numeric ranges (e.g. "total sulfur
# dioxide" can be 0-300, while "density" is around 0.99-1.0). Scaling puts
# every column on the same footing so no single feature dominates just
# because its numbers happen to be bigger.
scaler = StandardScaler()

# fit_transform on TRAINING data only: the scaler "learns" the mean/spread
# from training data, then applies that transformation.
X_train_scaled = scaler.fit_transform(X_train)

# transform (not fit_transform) on TEST data: we reuse the same scaling
# rules learned from training, we never let the test set influence them.
# This prevents "data leakage."
X_test_scaled = scaler.transform(X_test)

print("\nExample of scaled training data (first row):")
print(X_train_scaled[0])