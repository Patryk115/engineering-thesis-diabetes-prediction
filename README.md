# Type 2 Diabetes Risk Predictor

> **B.Eng. Engineering Thesis Project**
> An end-to-end Machine Learning pipeline and interactive web application for early assessment of Type 2 Diabetes risk based on demographic and behavioral data.

## Project Overview

This project serves as an "early warning system" designed to identify individuals at high risk of developing Type 2 Diabetes. It analyzes behavioral and demographic data from the 2023 BRFSS (Behavioral Risk Factor Surveillance System) dataset provided by the CDC.

Instead of providing a final medical diagnosis, the application evaluates user inputs and encourages high-risk individuals to seek professional medical consultation.

---

## Key Engineering Challenges Solved

* **Large Dataset Handling:** Processed over 400,000 records extracted directly from government SAS XPT files using memory-efficient Pandas chunking.
* **Inverse Causality Elimination:** Conducted rigorous feature engineering to identify and remove variables suffering from inverse causality (e.g., recognizing that "frequent cholesterol checks" are a result of illness, not a cause), ensuring the model learns true risk factors.
* **Handling Class Imbalance ("Safety First" Strategy):** With only 14% positive cases in the dataset, the Logistic Regression model was optimized to maximize Recall (Sensitivity). In medical screening, a False Positive (false alarm) is significantly safer than a False Negative (missing a sick patient).

---

## Tech Stack

* **Programming Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Joblib
* **Frontend & Visualization:** Streamlit, Plotly, Matplotlib, Seaborn

---

## Application Preview

<img width="1919" height="949" alt="image" src="https://github.com/user-attachments/assets/501dcc8d-98bf-4d19-8a0d-3499ca1f0948" />

---

## How to Run the App Locally

If you want to run this application on your own machine, follow these steps:

**1. Clone the repository:**
```bash
git clone https://github.com/Patryk115/engineering-thesis-diabetes-prediction.git
cd engineering-thesis-diabetes-prediction
```

**2. Create and activate a virtual environment (Recommended):**

*On Windows:*
```bash
python -m venv .venv
.venv\Scripts\activate
```

*On macOS/Linux:*
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install required dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application:**
```bash
streamlit run diabetes_prediction_app.py
```
> The app will automatically open in your default web browser at `http://localhost:8501`.

---

## Repository Structure

* `diabetes_prediction_app.py` - The main Streamlit web application.
* `diabetes_prediction_and_analysis.py` - The core ML script used for training and evaluating the model.
* `extract_brfss_features.py` - Script for extracting and chunking data from raw SAS XPT files.
* `*.joblib` - Serialized artifacts (Logistic Regression model, scaler, and expected columns) allowing the app to run inference in real-time.
* `requirements.txt` - List of Python dependencies.

---

## Author
* **Patryk Kindra** [LinkedIn](https://www.linkedin.com/in/patryk-kindra-228a97365/)
