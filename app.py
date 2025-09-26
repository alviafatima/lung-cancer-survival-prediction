import streamlit as st
import joblib
import tensorflow as tf
import pandas as pd

# Load pre-trained models 
scaler = joblib.load("scaler.pkl")
xgb_model = joblib.load("xgb_model.pkl")
nn_model = tf.keras.models.load_model("nn_lung_survival.keras", compile=False)
feature_columns = joblib.load("feature_columns.pkl")



# Preprocessing function

def preprocess_patient(patient_data, feature_columns, scaler):
    """Convert user input into model-ready format."""
    df = pd.DataFrame([patient_data])

    # Map cancer stage to numeric values
    stage_map = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}
    df["cancer_stage"] = df["cancer_stage"].map(stage_map).fillna(0)

    # Categorize BMI
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    df["bmi_category"] = df["bmi"].apply(categorize_bmi)

    # One-hot encode categorical features
    categorical_cols = ["gender", "treatment_type", "smoking_status", "bmi_category", "family_history"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    # Ensure alignment with training columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    # Scale numeric features
    X_scaled = scaler.transform(df)
    return X_scaled



# Prediction function

def predict_patient(patient_data):
    """Return predictions from both XGBoost and Neural Network models."""
    X = preprocess_patient(patient_data, feature_columns, scaler)

    # XGBoost prediction
    prob_xgb = xgb_model.predict_proba(X)[:, 1][0]
    pred_xgb = int(prob_xgb >= 0.5)

    # Neural Network prediction
    prob_nn = nn_model.predict(X).ravel()[0]
    pred_nn = int(prob_nn >= 0.5)

    return {
        "XGBoost Prediction": pred_xgb,
        "XGBoost Probability": float(prob_xgb),
        "Neural Network Prediction": pred_nn,
        "Neural Network Probability": float(prob_nn)
    }



# Streamlit user interface

st.title("🩺 Lung Cancer Survival Prediction")
st.write("Input patient information below to predict survival probability.")

# Collect user inputs
age = st.number_input("Age", 18, 100, 60)
gender = st.selectbox("Gender", ["Male", "Female"])
cancer_stage = st.selectbox("Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
family_history = st.selectbox("Family History of Cancer?", ["Yes", "No"])
smoking_status = st.selectbox("Smoking Status", ["Smoker", "Non-Smoker"])
bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
other_cancer = st.selectbox("Other Cancer", [0, 1])
treatment_type = st.selectbox("Treatment Type", ["Chemotherapy", "Radiation", "Surgery", "Immunotherapy"])
time_since_diagnosis = st.number_input("Days since diagnosis", 0, 5000, 365)
treatment_duration = st.number_input("Treatment duration (days)", 0, 2000, 180)

# Make prediction when user clicks button
if st.button("Predict"):
    patient_info = {
        "age": age,
        "gender": gender,
        "cancer_stage": cancer_stage,
        "family_history": family_history,
        "smoking_status": smoking_status,
        "bmi": bmi,
        "other_cancer": other_cancer,
        "treatment_type": treatment_type,
        "time_since_diagnosis": time_since_diagnosis,
        "treatment_duration": treatment_duration
    }

    results = predict_patient(patient_info)

    # Display results
    st.subheader("🔮 Prediction Results")
    st.write(f"**XGBoost Prediction:** {'Survived' if results['XGBoost Prediction'] else 'Not Survived'} "
             f"(Probability: {results['XGBoost Probability']:.2f})")
    st.write(f"**Neural Network Prediction:** {'Survived' if results['Neural Network Prediction'] else 'Not Survived'} "
             f"(Probability: {results['Neural Network Probability']:.2f})")





