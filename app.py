import pickle
from pathlib import Path
import pandas as pd
import streamlit as st

# Configure the browser tab
st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Placement Predictor")
st.markdown("Enter student academic metrics below to predict campus placement likelihood.")

# Base directory for reliable file loading regardless of execution cwd
BASE_DIR = Path(__file__).resolve().parent

# 1. Load the trained Pipeline (StandardScaler + LogisticRegression)
@st.cache_resource
def load_pipeline():
    model_path = BASE_DIR / "model.pkl"
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_pipeline()

# 2. Input features
st.subheader("Student Metrics")
col1, col2 = st.columns(2)
with col1:
    feature_1 = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.5,
        step=0.1,
        help="Cumulative Grade Point Average (0.0 - 10.0)"
    )
with col2:
    feature_2 = st.number_input(
        "IQ / Test Score",
        min_value=0.0,
        max_value=250.0,
        value=110.0,
        step=1.0,
        help="Cognitive / aptitude test score"
    )

# 3. Predict action
if st.button("Predict Placement", type="primary", use_container_width=True):
    # Pass inputs directly with feature names into the pipeline
    input_data = pd.DataFrame([[feature_1, feature_2]], columns=["cgpa", "iq"])

    # Predict using the Pipeline (auto-scales and predicts)
    prediction = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)[0]
    placed_prob = probabilities[1] * 100

    st.write("---")

    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        if prediction == 1:
            st.success("### 🎉 Placed")
            st.caption("High probability of campus selection.")
        else:
            st.error("### ⚠️ Not Placed")
            st.caption("Needs score improvement to qualify.")

    with col_metric2:
        st.metric(label="Placement Probability", value=f"{placed_prob:.1f}%")
        st.progress(min(max(int(placed_prob), 0), 100))

