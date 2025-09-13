# app.py
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ===============================
# 1) Load Model
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("xgboost_dropout_small.pkl")

model = load_model()

expected_order = [
    "Curricular units 2nd sem (approved)",
    "Tuition fees up to date",
    "Curricular units 1st sem (approved)",
    "Course",
    "Age at enrollment",
    "Scholarship holder"
]
label_map = {0: "Graduate", 1: "Dropout"}

# ===============================
# 2) Page Config + CSS
# ===============================
st.set_page_config(page_title="Dropout Prediction System", layout="wide")

st.markdown("""
    <style>
        .main-title {text-align:center;font-size:2.5em;font-weight:bold;color:#2c3e50;margin-bottom:10px;}
        .subtitle {text-align:center;font-size:1.2em;color:#555;margin-bottom:25px;}
        .big-button {display:flex;justify-content:center;margin:30px 0;}
        .prediction-box {padding:20px;border-radius:12px;text-align:center;font-size:22px;font-weight:bold;}
        .graduate {background-color:#e8f9e9;color:#1e7e34;border:2px solid #28a745;}
        .dropout {background-color:#fdeaea;color:#b71c1c;border:2px solid #dc3545;}
    </style>
""", unsafe_allow_html=True)

# ===============================
# 3) Navigation State
# ===============================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ===============================
# 4) Home Page
# ===============================
if st.session_state.page == "home":
    st.markdown('<p class="main-title">🎓 AI-Powered Dropout Prediction & Counselling</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict at-risk students early, understand why, and provide tailored support.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("📊 Dataset", "UCI Student Data")
    with col2: st.metric("🤖 Model", "XGBoost")
    with col3: st.metric("📈 Accuracy", "≈ 85%")

    st.write("---")
    st.write("""
    ### 🔍 Project Overview
    This system uses **AI-powered predictive analytics** to:
    - Flag students at risk of dropping out  
    - Explain contributing factors (via SHAP)  
    - Suggest **tailored counselling** strategies  
    """)

    st.write("---")
    st.markdown('<div class="big-button">', unsafe_allow_html=True)
    if st.button("🚀 Start Prediction"):
        st.session_state.page = "predict"
        st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# 5) Prediction Page
# ===============================
elif st.session_state.page == "predict":
    st.sidebar.header("📝 Student Details")

    # 🔹 Sample Profiles
    sample_profiles = {
        "📉 High Risk": {"course": 101, "tuition": 0, "curr_units_1": 2, "curr_units_2": 3, "age": 28, "scholarship": 0},
        "⚖️ Medium Risk": {"course": 205, "tuition": 1, "curr_units_1": 4, "curr_units_2": 5, "age": 24, "scholarship": 0},
        "📈 Low Risk": {"course": 305, "tuition": 1, "curr_units_1": 8, "curr_units_2": 10, "age": 20, "scholarship": 1}
    }
    profile_choice = st.sidebar.selectbox("📂 Load Sample Profile", ["Custom Input"] + list(sample_profiles.keys()))

    # Load defaults
    defaults = sample_profiles.get(profile_choice, {"course": 1, "tuition": 1, "curr_units_1": 5, "curr_units_2": 5, "age": 20, "scholarship": 0})

    course = st.sidebar.number_input("📘 Course Code", min_value=0, max_value=9999, value=defaults["course"])
    tuition = st.sidebar.selectbox("💰 Tuition Fees Up to Date", [0, 1], index=defaults["tuition"])
    curr_units_1 = st.sidebar.number_input("📚 1st Sem Units Approved", min_value=0, max_value=20, value=defaults["curr_units_1"])
    curr_units_2 = st.sidebar.number_input("📚 2nd Sem Units Approved", min_value=0, max_value=20, value=defaults["curr_units_2"])
    age = st.sidebar.number_input("🎂 Age at Enrollment", min_value=16, max_value=60, value=defaults["age"])
    scholarship = st.sidebar.selectbox("🎓 Scholarship Holder", [0, 1], index=defaults["scholarship"])

    # DataFrame
    input_data = pd.DataFrame({
        "Curricular units 2nd sem (approved)": [curr_units_2],
        "Tuition fees up to date": [tuition],
        "Curricular units 1st sem (approved)": [curr_units_1],
        "Course": [course],
        "Age at enrollment": [age],
        "Scholarship holder": [scholarship]
    })[expected_order]

    st.markdown('<p class="main-title">📊 Dropout Risk Prediction</p>', unsafe_allow_html=True)

    if st.sidebar.button("🔮 Predict Dropout Risk"):
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Prediction", "🔍 Explainability", "💡 Counselling"])

        with tab1:
            st.subheader("Prediction Result")
            if pred == 1:
                st.markdown(f'<div class="prediction-box dropout">⚠️ Predicted: Dropout</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box graduate">🎉 Predicted: Graduate</div>', unsafe_allow_html=True)

            st.write("### Probability Breakdown")
            st.progress(float(prob[1]))
            st.write(f"**Graduate:** {prob[0]*100:.1f}%")
            st.write(f"**Dropout:** {prob[1]*100:.1f}%")

        with tab2:
            st.subheader("Why this prediction?")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_data)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            st.pyplot(bbox_inches="tight")

        with tab3:
            st.subheader("Tailored Counselling Recommendations")
            suggestions = []
            if tuition == 0:
                suggestions.append("💰 Update tuition fee payments; explore financial aid or installment options.")
            if curr_units_1 < 4:
                suggestions.append("📚 Provide tutoring or mentoring for 1st semester courses.")
            if curr_units_2 < 4:
                suggestions.append("📚 Suggest remedial sessions or study groups for 2nd semester.")
            if age > 30:
                suggestions.append("🕒 Offer flexible schedules for mature students balancing work/study.")
            if scholarship == 0:
                suggestions.append("🎓 Encourage applying for scholarships or grants to reduce financial stress.")

            if not suggestions:
                st.success("✅ Student appears low-risk. Continue regular academic monitoring.")
            else:
                for s in suggestions:
                    st.info(s)

    st.sidebar.write("---")
    if st.sidebar.button("🏠 Back to Home"):
        st.session_state.page = "home"
        st.experimental_rerun()
