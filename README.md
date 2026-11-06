# 🎓 Student Placement Prediction Model

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning classification project that predicts student campus placement outcomes based on academic performance (**CGPA**) and cognitive aptitude (**IQ**). Includes a fully executed, reproducible Jupyter Notebook and an interactive **Streamlit** web application.

---

## 📌 Project Overview

- **Dataset**: 100 historical student records with features `cgpa`, `iq`, and target `placement` (`1` = Placed, `0` = Not Placed).
- **Algorithm**: Logistic Regression with standardized feature scaling.
- **Architecture**: Encapsulated inside a single Scikit-Learn `Pipeline` (`StandardScaler` + `LogisticRegression`) to eliminate data leakage and guarantee consistent preprocessing during inference.
- **Evaluation**: 5-Fold Stratified Cross-Validation, Confusion Matrix, Precision, Recall, and F1 Score.
- **Web App**: Interactive Streamlit interface providing instant classification and confidence/probability gauge.

---

## 📊 Model Pipeline & Performance

The core machine learning workflow is designed as a unified Scikit-Learn pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])
```

### Evaluation Results (Stratified 80/20 Split)

| Metric | Score | Details |
| :--- | :---: | :--- |
| **5-Fold Cross-Validation** | **91.25%** | Evaluated across 5 stratified training folds |
| **Test Accuracy** | **90.00%** | Evaluated on 20 unseen holdout test samples |
| **Precision** | **90.00%** | True positive precision rate |
| **Recall** | **90.00%** | Sensitivity on placed candidates |
| **F1-Score** | **90.00%** | Harmonic mean of precision & recall |

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/balganeshchettiyar/placement-prediction-model.git
cd placement-prediction-model
```

### 2. Set Up Environment & Install Dependencies
```bash
# Recommended: Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run the Interactive Web App
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to interact with the model.

---

## 📁 Repository Structure

```
placement-prediction-model/
├── app.py               # Streamlit web application with confidence score gauge
├── main.ipynb           # Model development, EDA, training & evaluation (with outputs)
├── model.pkl            # Serialized Scikit-Learn Pipeline (StandardScaler + LogisticRegression)
├── placement.csv        # Student placement dataset
├── requirements.txt     # Python package requirements
├── .gitignore           # Git ignore rules for virtual environments & checkpoints
└── README.md            # Comprehensive project documentation
```

---

## 💡 Key Features of this Implementation

1. **Zero Data Leakage**: Standardized using a unified `Pipeline` so test/inference features are scaled strictly using parameters learned from training data.
2. **Deterministic & Reproducible**: Uses fixed `random_state=42` and stratified sampling.
3. **Streamlit UI Enhancements**:
   - Dynamic path resolution ensuring the model loads regardless of current working directory.
   - Resource caching via `@st.cache_resource` for instant response times.
   - Live probability estimation and visual progress bar.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
