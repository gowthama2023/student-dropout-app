# app.py
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ===============================
# 0) Initialize Session State
# ===============================
if "page" not in st.session_state:
    st.session_state.page = "home"

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
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Orbitron', sans-serif;
        }

        /* Cyberpunk Background Animation */
        @keyframes matrix {
            0% { background-position: 0 0; }
            100% { background-position: 0 100%; }
        }
        .matrix-bg {
            background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364);
            background-size: 100% 200%;
            animation: matrix 15s linear infinite;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
        }

        /* Main Title Styling */
        .cyber-title {
            font-size: 3em;
            font-weight: bold;
            color: #00ffe7;
            text-shadow: 0 0 10px #00ffe7, 0 0 20px #00ffe7, 0 0 30px #00ffe7;
            animation: glow 2s ease-in-out infinite alternate;
        }

        /* Subtitle Styling */
        .cyber-subtitle {
            font-size: 1.5em;
            font-weight: 500;
            color: #fff;
            margin-top: 10px;
            text-shadow: 0 0 5px #ff00ff, 0 0 10px #ff00ff;
            animation: flicker 3s infinite alternate;
        }

        /* Glow Animation */
        @keyframes glow {
            from { text-shadow: 0 0 10px #00ffe7, 0 0 20px #00ffe7, 0 0 30px #00ffe7; }
            to   { text-shadow: 0 0 20px #ff00ff, 0 0 40px #ff00ff, 0 0 60px #ff00ff; }
        }

        /* Flicker Animation */
        @keyframes flicker {
            0%   { opacity: 1; }
            45%  { opacity: 0.8; }
            55%  { opacity: 0.6; }
            70%  { opacity: 1; }
            100% { opacity: 0.9; }
        }

        .big-button {display:flex;justify-content:center;margin:30px 0;}
        .prediction-box {
            padding:20px;
            border-radius:12px;
            text-align:center;
            font-size:22px;
            font-weight:bold;
            margin-bottom:20px;
        }
        .graduate {
            background:rgba(0,255,231,0.1);
            color:#00ffe7;
            border:2px solid #00ffe7;
            box-shadow:0 0 15px #00ffe7;
        }
        .dropout {
            background:rgba(255,0,127,0.1);
            color:#ff007f;
            border:2px solid #ff007f;
            box-shadow:0 0 15px #ff007f;
        }
        .neon-card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 0 20px rgba(0,255,231,0.4);
        }
    </style>
""", unsafe_allow_html=True)

# ===============================
# 3) Custom Visualizations
# ===============================
def plot_probability_pie(prob):
    fig = go.Figure(data=[go.Pie(
        labels=["Graduate 🎓", "Dropout ⚠️"],
        values=[prob[0], prob[1]],
        hole=0.4,
        marker=dict(colors=["#00ffe7", "#ff007f"]),
        textinfo="label+percent"
    )])
    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    return fig

def plot_dropout_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob[1]*100,
        title={"text": "Dropout Risk 🔮", "font": {"size": 20, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": "#ff007f"},
            "bgcolor": "black",
            "borderwidth": 2,
            "bordercolor": "#00ffe7",
            "steps": [
                {"range": [0, 30], "color": "green"},
                {"range": [30, 70], "color": "orange"},
                {"range": [70, 100], "color": "red"}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    return fig

def plot_top_features(shap_values, input_data, top_n=5):
    values = shap_values.values[0]
    feature_names = input_data.columns
    top_idx = abs(values).argsort()[-top_n:][::-1]

    fig = go.Figure(go.Bar(
        x=[values[i] for i in top_idx],
        y=[feature_names[i] for i in top_idx],
        orientation="h",
        marker=dict(color="#00ffe7")
    ))
    fig.update_layout(
        title="Top Feature Impacts ⚡",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(t=40, b=20, l=80, r=20)
    )
    return fig

# ===============================
# 4) Home Page
# ===============================
if st.session_state.page == "home":
    st.markdown("""
        <div class="matrix-bg">
            <div class="cyber-title">🎓 AI-Powered Dropout Prediction & Counselling</div>
            <div class="cyber-subtitle">Predict at-risk students early, understand why, and provide tailored support</div>
        </div>
    """, unsafe_allow_html=True)

    st.write("---")

    # Stats Section
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("📊 Dataset", "UCI Student Data")
    with col2: st.metric("🤖 Model", "XGBoost")
    with col3: st.metric("📈 Accuracy", "≈ 89%")

    st.write("---")

    # Centered Project Overview
    st.markdown("""
        <div class="neon-card" style="text-align:center;font-size:1.2em;color:#eee;">
         🔍 Project Overview  
        This system uses AI-powered predictive analytics to:  
        - Flag students at risk of dropping out  
        - Explain contributing factors (via SHAP)  
        - Suggest tailored counselling strategies  
        </div>
    """, unsafe_allow_html=True)

    st.write("---")
    st.markdown('<div class="big-button">', unsafe_allow_html=True)
    if st.button("🚀 Start Prediction"):
        st.session_state.page = "predict"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# 5) Prediction Page
# ===============================
elif st.session_state.page == "predict":
    st.sidebar.header("📝 Student Details")

    sample_profiles = {
        "📉 High Risk": {"course": 101, "tuition": 0, "curr_units_1": 2, "curr_units_2": 3, "age": 28, "scholarship": 0},
        "⚖️ Medium Risk": {"course": 205, "tuition": 1, "curr_units_1": 4, "curr_units_2": 5, "age": 24, "scholarship": 0},
        "📈 Low Risk": {"course": 900, "tuition": 1, "curr_units_1": 8, "curr_units_2": 10, "age": 20, "scholarship": 1}
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

    st.markdown('<p class="cyber-title">📊 Dropout Risk Prediction</p>', unsafe_allow_html=True)

    if st.sidebar.button("🔮 Predict Dropout Risk"):
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Prediction", "🔍 Explainability", "💡 Counselling"])

        with tab1:
            if pred == 1:
                st.markdown(f'<div class="prediction-box dropout">⚠️ Predicted: Dropout</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box graduate">🎉 Predicted: Graduate</div>', unsafe_allow_html=True)

            colA, colB = st.columns(2)
            with colA:
                st.plotly_chart(plot_probability_pie(prob), use_container_width=True)
            with colB:
                st.plotly_chart(plot_dropout_gauge(prob), use_container_width=True)

        with tab2:
            st.subheader("Why this prediction?")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_data)
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            st.pyplot(bbox_inches="tight")

            st.plotly_chart(plot_top_features(shap_values, input_data), use_container_width=True)

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
        st.rerun()
