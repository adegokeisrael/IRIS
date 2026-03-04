import streamlit as st
import pandas as pd
import joblib

# ============================================================
# LOAD SAVED MODEL
# ============================================================

model = joblib.load("random_forest_iris_model.pkl")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

st.title("🌸 Iris Flower Classification App")
st.write("Predict the species of an Iris flower using a Random Forest model.")

# ============================================================
# CREATE FORM (WITHOUT WITH STATEMENT)
# ============================================================

form = st.form("iris_form")

form.subheader("Enter Flower Measurements")

sepal_length = form.number_input(
    "Sepal Length (cm)",
    min_value=4.0,
    max_value=8.0,
    value=5.1
)

sepal_width = form.number_input(
    "Sepal Width (cm)",
    min_value=2.0,
    max_value=4.5,
    value=3.5
)

petal_length = form.number_input(
    "Petal Length (cm)",
    min_value=1.0,
    max_value=7.0,
    value=1.4
)

petal_width = form.number_input(
    "Petal Width (cm)",
    min_value=0.1,
    max_value=2.5,
    value=0.2
)

submit_button = form.form_submit_button("Predict")

# ============================================================
# PREDICTION
# ============================================================

if submit_button:

    input_data = pd.DataFrame({
        "SepalLengthCm": [sepal_length],
        "SepalWidthCm": [sepal_width],
        "PetalLengthCm": [petal_length],
        "PetalWidthCm": [petal_width]
    })

    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    st.subheader("Prediction Result")
    st.success(f"🌼 Predicted Species: {prediction[0]}")

    st.subheader("Prediction Probability")
    st.write(probabilities)

