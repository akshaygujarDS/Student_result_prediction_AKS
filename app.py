import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

# Page configuration
st.set_page_config(
    page_title="Student Result Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for pure white background, simple card layout, and soft shadows
st.markdown("""
    <style>
    /* Global App Background - Clean White */
    .stApp {
        background-color: #ffffff;
        color: #1f2937;
    }

    /* Main Title Styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-size: 0.95rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 1.8rem;
    }

    /* Vertical Form Card with Soft Shadow */
    .vertical-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 28px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
        margin-bottom: 20px;
    }

    /* Input Field Labels */
    .stNumberInput label {
        color: #334155 !important;
        font-weight: 600 !important;
    }

    /* Styled Action Button */
    div.stButton > button {
        width: 100%;
        background-color: #2563eb;
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        transition: all 0.2s ease-in-out;
        cursor: pointer;
    }

    div.stButton > button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
        transform: translateY(-1px);
    }

    /* Result Boxes */
    .result-pass {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(34, 197, 94, 0.15);
        margin-top: 15px;
    }

    .result-fail {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.15);
        margin-top: 15px;
    }

    .status-text {
        font-size: 2.8rem;
        font-weight: 900;
        margin: 5px 0;
    }

    .pass-color { color: #16a34a; }
    .fail-color { color: #dc2626; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

# Application Header
st.markdown('<div class="main-title">🎓 Student Result Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter marks below to check student Pass or Fail status</div>', unsafe_allow_html=True)

# Main Form Container
st.markdown('<div class="vertical-card">', unsafe_allow_html=True)

st.subheader("📝 Enter Marks (0 - 100)")

# Vertical Stack Inputs
hindi = st.number_input("Hindi", min_value=0, max_value=100, value=75, step=1)
english = st.number_input("English", min_value=0, max_value=100, value=80, step=1)
science = st.number_input("Science", min_value=0, max_value=100, value=85, step=1)
maths = st.number_input("Maths", min_value=0, max_value=100, value=90, step=1)
history = st.number_input("History", min_value=0, max_value=100, value=70, step=1)
geography = st.number_input("Geography / Geograpgy", min_value=0, max_value=100, value=78, step=1)

st.markdown('</div>', unsafe_allow_html=True)

# Prediction Logic
if st.button("Calculate Result"):
    
    raw_inputs = {
        'Hindi': hindi,
        'English': english,
        'Science': science,
        'Maths': maths,
        'History': history,
        'Geography': geography,
        'Geograpgy': geography
    }

    # Format input data based on model requirements
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        input_data = pd.DataFrame([{col: raw_inputs.get(col, 0) for col in expected_cols}])
    elif hasattr(model, "n_features_in_"):
        num_features = model.n_features_in_
        val_list = [hindi, english, science, maths, history, geography]
        
        if len(val_list) < num_features:
            val_list.extend([0] * (num_features - len(val_list)))
        elif len(val_list) > num_features:
            val_list = val_list[:num_features]
            
        input_data = np.array([val_list])
    else:
        input_data = pd.DataFrame([raw_inputs])

    with st.spinner("Evaluating result..."):
        time.sleep(0.3)
        try:
            raw_pred = model.predict(input_data)[0]
            
            # Map prediction output: 1 -> PASS, 0 -> FAIL
            if str(raw_pred).strip() in ['1', '1.0', 'Pass', 'PASS']:
                status = "PASS"
                card_style = "result-pass"
                text_style = "pass-color"
                st.balloons()
            else:
                status = "FAIL"
                card_style = "result-fail"
                text_style = "fail-color"

            # Render output
            st.markdown(
                f"""
                <div class="{card_style}">
                    <div style="font-weight: 600; color: #475569;">Predicted Result</div>
                    <div class="status-text {text_style}">{status}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        except Exception as err:
            st.error(f"Prediction error: {err}")
