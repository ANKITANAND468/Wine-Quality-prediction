# src/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.combine import SMOTETomek

# --- Steps 1-5: same preprocessing as before ---
red_wine = pd.read_csv("data/raw/winequality-red.csv", sep=";")
red_wine = red_wine.drop_duplicates()

X = red_wine.drop(columns=["quality"])
y = red_wine["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # test set: scaled only, never resampled

# --- Step 6: Apply SMOTE-Tomek to the TRAINING data only ---
# SMOTETomek does two things:
#  1. SMOTE: creates synthetic examples of rare quality scores
#  2. Tomek links: removes some overlapping/ambiguous majority-class
#     examples near the boundary, sharpening the decision boundary
#
# k_neighbors controls how many nearby same-class points SMOTE looks at
# to generate synthetic samples. Our rarest class only has a handful of
# examples in the training set, so we cap k_neighbors accordingly —
# otherwise SMOTE will error out asking for more neighbors than exist.
min_class_count = y_train.value_counts().min()
k = min(5, min_class_count - 1)

print(f"Smallest class has {min_class_count} training examples, using k_neighbors={k}")

smote_tomek = SMOTETomek(random_state=42, smote=None) if k < 1 else SMOTETomek(random_state=42)
if k >= 1:
    from imblearn.over_sampling import SMOTE
    smote_tomek = SMOTETomek(random_state=42, smote=SMOTE(random_state=42, k_neighbors=k))

X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_scaled, y_train)

print(f"\nBefore SMOTE-Tomek: {len(y_train)} training rows")
print(f"After SMOTE-Tomek: {len(y_train_resampled)} training rows")
print("\nClass distribution AFTER resampling:")
print(y_train_resampled.value_counts().sort_index())

# --- Step 7: Train the model on the RESAMPLED, balanced training data ---
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
    # note: no class_weight="balanced" needed anymore — SMOTE already
    # balanced the classes directly in the data
)

model.fit(X_train_resampled, y_train_resampled)

# --- Step 8: Predict on the ORIGINAL, untouched test set ---
y_pred = model.predict(X_test_scaled)

# --- Step 9: Evaluate ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.3f}")

print("\nDetailed report (per quality score):")
print(classification_report(y_test, y_pred, zero_division=0))



# --- Step 10: Save the model and scaler to disk ---
import joblib

# joblib is the standard tool for saving scikit-learn objects to disk.
# It serializes the entire trained object (all 300 trees, their learned
# splits, everything) into a single file.
joblib.dump(model, "models/red_wine_model.pkl")
joblib.dump(scaler, "models/red_wine_scaler.pkl")

print("\nModel and scaler saved to the 'models/' folder.")