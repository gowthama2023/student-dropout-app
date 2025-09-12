# app.py
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ===============================
# 1) Load trained lightweight model
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("xgboost_dropout_small.pkl")

model = load_model()

# Expected feature order (from training script)
expected_order = [
    "Curricular units 2nd sem (approved)",
    "Tuition fees up to date",
    "Curricular units 1st sem (approved)",
    "Course",
    "Age at enrollment",
    "Scholarship holder"
]

# Label mapping (binary classification)
label_map = {
    0: "Graduate",
    1: "Dropout"
}

# ===============================
# 2) Streamlit UI
# ===============================
st.set_page_config(page_title="Student Dropout Prediction", layout="centered")

st.title("🎓 AI-based Student Dropout Prediction & Counselling System")
st.write("Enter student details to predict dropout risk and get tailored counselling suggestions.")

# Input fields (based on top 6 SHAP features)
course = st.number_input("Course Code", min_value=0, max_value=9999, value=1, help="Numeric code for course chosen")
tuition = st.selectbox("Tuition Fees Up to Date", [0, 1], help="1 = Yes, 0 = No")
curr_units_1 = st.number_input("Curricular Units 1st Sem (Approved)", min_value=0, max_value=20, value=5)
curr_units_2 = st.number_input("Curricular Units 2nd Sem (Approved)", min_value=0, max_value=20, value=5)
age = st.number_input("Age at Enrollment", min_value=16, max_value=60, value=20)
scholarship = st.selectbox("Scholarship Holder", [0, 1], help="1 = Yes, 0 = No")

# Convert inputs to DataFrame with correct names
input_data = pd.DataFrame({
    "Curricular units 2nd sem (approved)": [curr_units_2],
    "Tuition fees up to date": [tuition],
    "Curricular units 1st sem (approved)": [curr_units_1],
    "Course": [course],
    "Age at enrollment": [age],
    "Scholarship holder": [scholarship]
})

# Ensure correct column order
input_data = input_data[expected_order]

# ===============================
# 3) Prediction
# ===============================
if st.button("🔮 Predict Dropout Risk"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    # Show results with label mapping
    st.subheader("✅ Prediction Result")
    st.write(f"**Predicted Class:** {label_map[pred]}")
    st.write("**Probabilities:**")
    st.json({
        "Graduate": float(prob[0]),
        "Dropout": float(prob[1])
    })

    # ===============================
    # 4) Show Explainability & Suggestions only if Dropout
    # ===============================
    if pred == 1:  # Dropout
        # SHAP Explainability
        st.subheader("🔍 Why this prediction?")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)

        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(bbox_inches="tight")

        # Counselling Suggestions
        st.subheader("💡 Counselling Recommendations")
        suggestions = []
        if tuition == 0:
            suggestions.append("📌 Encourage the student to update tuition fee payments; financial aid or payment plans may help.")
        if curr_units_1 < 4:
            suggestions.append("📌 Provide extra tutoring or academic mentoring for 1st semester courses.")
        if curr_units_2 < 4:
            suggestions.append("📌 Monitor performance in 2nd semester and suggest study groups or remedial sessions.")
        if age > 30:
            suggestions.append("📌 Older students may face work–study balance issues; offer flexible schedules or counselling.")
        if scholarship == 0:
            suggestions.append("📌 Explore scholarship opportunities or need-based grants to reduce financial stress.")

        if not suggestions:
            st.write("📌 General counselling is advised, but no major risk indicators found.")
        else:
            for s in suggestions:
                st.write(s)
    else:
        st.success("🎉 Student is predicted to Graduate — no counselling needed.")
