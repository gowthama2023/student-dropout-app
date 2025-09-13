# ===============================
# 4) Home Page
# ===============================
if st.session_state.page == "home":
    st.markdown("""
        <style>
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
        </style>
    """, unsafe_allow_html=True)

    # Title Section with Animation
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
    with col3: st.metric("📈 Accuracy", "≈ 89%")  # ✅ Updated

    st.write("---")

    # Centered Project Overview
    st.markdown("""
        <div style="text-align:center;font-size:1.2em;color:#eee;padding:15px;">
        ### 🔍 Project Overview  
        This system uses **AI-powered predictive analytics** to:  
        - Flag students at risk of dropping out  
        - Explain contributing factors (via SHAP)  
        - Suggest **tailored counselling** strategies  
        </div>
    """, unsafe_allow_html=True)

    st.write("---")
    st.markdown('<div class="big-button">', unsafe_allow_html=True)
    if st.button("🚀 Start Prediction"):
        st.session_state.page = "predict"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
