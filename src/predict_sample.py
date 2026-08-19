# src/predict_sample.py

import joblib
import pandas as pd

# --- Step 1: Load the saved model and scaler back into memory ---
# This is the reverse of joblib.dump() — it reconstructs the exact
# trained objects we saved earlier, no retraining needed.
model = joblib.load("models/red_wine_model.pkl")
scaler = joblib.load("models/red_wine_scaler.pkl")

print("Model and scaler loaded successfully.")

# --- Step 2: Create one new, made-up wine sample ---
# These are 11 chemistry values in the SAME ORDER as the original
# training columns. Real values inspired from a typical red wine.
new_wine = pd.DataFrame([{
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}])

print("\nNew wine input:")
print(new_wine)

# --- Step 3: Scale it using the SAME scaler from training ---
# Critical: we use .transform(), NOT .fit_transform(). The scaler
# already learned its rules from training data — we just apply them.
new_wine_scaled = scaler.transform(new_wine)

# --- Step 4: Predict ---
prediction = model.predict(new_wine_scaled)
print(f"\nPredicted quality score: {prediction[0]}")

# --- Step 5 (bonus): See the model's confidence per class ---
# predict_proba gives a probability for EACH possible quality score,
# not just the single winning one. Useful for understanding how
# confident (or unsure) the model actually is.
probabilities = model.predict_proba(new_wine_scaled)
classes = model.classes_

print("\nConfidence per quality score:")
for cls, prob in zip(classes, probabilities[0]):
    print(f"  Quality {cls}: {prob:.1%}")