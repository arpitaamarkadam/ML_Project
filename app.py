import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("advertising_sales_model.pkl")

# Page title
st.set_page_config(
    page_title="Advertising Sales Prediction",
    page_icon="📊"
)

st.title("📊 Advertising Sales Prediction")
st.write("Predict product sales based on advertising expenditure.")

st.divider()

# User inputs
st.subheader("Enter Advertising Details")

tv = st.number_input(
    "TV Advertising Budget",
    min_value=0.0,
    value=100.0
)

radio = st.number_input(
    "Radio Advertising Budget",
    min_value=0.0,
    value=20.0
)

newspaper = st.number_input(
    "Newspaper Advertising Budget",
    min_value=0.0,
    value=20.0
)

# Prediction button
if st.button("Predict Sales"):

    input_data = pd.DataFrame({
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [newspaper]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Sales: {prediction[0]:.2f}"
    )

    st.info(
        "Interpretation: Higher predicted sales indicate that "
        "the given advertising investment may generate higher sales."
    )