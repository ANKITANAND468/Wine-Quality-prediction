# app.py

import streamlit as st
import pandas as pd
import joblib

# --- Page setup ---
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")
st.title("🍷 Red Wine Quality Predictor")
st.write(
    "Enter a wine's chemistry measurements below, and the model will "
    "predict its quality score (3-8) along with confidence levels."
)

# --- Load the saved model and scaler ---
# @st.cache_resource tells Streamlit to load these ONCE and reuse them,
# instead of reloading from disk every single time the user interacts
# with the app (which would be slow).
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("models/red_wine_model.pkl")
    scaler = joblib.load("models/red_wine_scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# --- Input widgets ---
# st.slider creates a draggable slider. Arguments: label, min, max, default.
# We picked ranges based on the summary statistics we saw in Stage 2.
st.header("Wine chemistry inputs")

col1, col2 = st.columns(2)  # split inputs into two side-by-side columns

with col1:
    fixed_acidity = st.slider("Fixed acidity", 4.0, 16.0, 7.4)
    volatile_acidity = st.slider("Volatile acidity", 0.1, 1.6, 0.70)
    citric_acid = st.slider("Citric acid", 0.0, 1.0, 0.0)
    residual_sugar = st.slider("Residual sugar", 0.5, 16.0, 1.9)
    chlorides = st.slider("Chlorides", 0.01, 0.62, 0.076)
    free_sulfur_dioxide = st.slider("Free sulfur dioxide", 1.0, 72.0, 11.0)

with col2:
    total_sulfur_dioxide = st.slider("Total sulfur dioxide", 6.0, 289.0, 34.0)
    density = st.slider("Density", 0.990, 1.004, 0.9978, format="%.4f")
    pH = st.slider("pH", 2.7, 4.0, 3.51)
    sulphates = st.slider("Sulphates", 0.3, 2.0, 0.56)
    alcohol = st.slider("Alcohol (%)", 8.0, 15.0, 9.4)

# --- Predict button ---
if st.button("Predict quality"):
    # Build a one-row DataFrame from the slider values, matching the
    # exact column names/order the model was trained on.
    new_wine = pd.DataFrame([{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
    }])

    # Scale using the SAME scaler from training, then predict
    new_wine_scaled = scaler.transform(new_wine)
    prediction = model.predict(new_wine_scaled)[0]
    probabilities = model.predict_proba(new_wine_scaled)[0]

    # --- Display results ---
    st.subheader(f"Predicted quality: {prediction} / 8")

    st.write("Confidence per quality score:")
    prob_df = pd.DataFrame({
        "Quality score": model.classes_,
        "Confidence": probabilities
    })
    st.bar_chart(prob_df.set_index("Quality score"))