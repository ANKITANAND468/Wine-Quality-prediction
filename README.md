# 🍷 Wine Quality Prediction

A machine learning web app that predicts red wine quality (score 3-8) from 11 physicochemical measurements, built with scikit-learn and deployed with Streamlit.

## Live demo
https://wine-quality-prediction-bbu22vrfv2hqu3tx7wweh2.streamlit.app/

## Overview

This project trains a Random Forest classifier on the UCI Wine Quality dataset (red wine, 1,599 samples), using SMOTE-Tomek resampling to address severe class imbalance in the quality scores. The methodology follows the approach outlined in *"Wine Quality Prediction with Ensemble Trees: A Unified, Leak-Free Comparative Study"* (Chen, 2026).

## Pipeline

1. **Data loading & exploration** — checked for missing values, duplicates, and class distribution
2. **Preprocessing** — removed 240 duplicate rows, split 80/20 train-test with stratification, scaled features with StandardScaler
3. **Class imbalance handling** — applied SMOTE-Tomek to the training set only, to avoid data leakage
4. **Model training** — Random Forest (300 trees)
5. **Evaluation** — accuracy, precision, recall, F1-score per class
6. **Deployment** — interactive Streamlit app where users input wine chemistry and get a real-time prediction with confidence scores

## Results

- Accuracy: ~0.56-0.60
- Weighted F1: ~0.57-0.59
- Best performance on common quality scores (5, 6, 7); rare extreme scores (3, 8) remain difficult to predict due to very few real-world examples (a known limitation also observed in the reference paper)

## Tech stack

- Python, pandas, scikit-learn, imbalanced-learn
- Streamlit (web app)
- joblib (model persistence)

## Run locally

\`\`\`bash
git clone <your-repo-url>
cd wine-quality-prediction
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Project structure

\`\`\`
├── app.py                  # Streamlit web app
├── src/
│   ├── load_data.py
│   ├── explore_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict_sample.py
├── models/                 # saved trained model + scaler
├── data/raw/                   # dataset (not tracked in git if large)
└── requirements.txt
