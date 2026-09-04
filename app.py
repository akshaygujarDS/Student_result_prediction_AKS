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

# Custom CSS for UI styling, dark gradient background, vertical layout cards, and shadows
st.markdown("""
    <style>
    /* Global App Background */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        color: #f0f6fc;
    }
    
    /* Title Styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    
    .sub-title {
        font-size: 1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Vertical Card Layout Styling */
    .vertical-card {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.6), 0 0 15px rgba(56, 189, 248, 0.1);
        margin-bottom: 24px;
    }

    /* Styled Predict Button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 50%, #9333ea 100%);
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 0.85rem 1.5rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
        transition: all 0.3s ease-in-out;
        cursor: pointer;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.7);
        background: linear-gradient(90deg, #4338ca 0%, #6d28d9 50%, #7e22ce 100%);
    }

    div.stButton > button:active {
        transform: translateY(1px);
    }

    /* Output Card Box */
    .result-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.3);
        margin-top: 20px;
    }

    .result-score {
        font-size: 3rem;
        font-weight: 900;
        color: #34d399;
        margin: 10px 0;
    }
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

# Title Header
st.markdown('<div class="main-title">🎓 Student Result Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter individual subject marks below to generate model output</div>', unsafe_allow_html=True)

# Main Form Container (Clean Vertical Stack)
st.markdown('<div class="vertical-card">', unsafe_allow_html=True)

st.subheader("📝 Subject Marks (0 - 100)")

# Vertical Stack Inputs
hindi = st.number_input("Hindi", min_value=0, max_value=100, value=75, step=1)
english = st.number_input("English", min_value=0, max_value=100, value=80, step=1)
science = st.number_input("Science", min_value=0, max_value=100, value=85, step=1)
maths = st.number_input("Maths", min_value=0, max_value=100, value=90, step=1)
history = st.number_input("History", min_value=0, max_value=100, value=70, step=1)
geography = st.number_input("Geography / Geograpgy", min_value=0, max_value=100, value=78, step=1)

st.markdown('</div>', unsafe_allow_html=True)

# Predict Trigger
if st.button("🚀 Calculate & Predict Result"):
    
    # Feature dictionary matching expected names
    raw_inputs = {
        'Hindi': hindi,
        'English': english,
        'Science': science,
        'Maths': maths,
        'History': history,
        'Geography': geography,
        'Geograpgy': geography
    }

    # Inspect model requirements dynamically to avoid feature count mismatch errors
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        # Build input dataframe using exact columns model expects
        input_data = pd.DataFrame([{col: raw_inputs.get(col, 0) for col in expected_cols}])
    elif hasattr(model, "n_features_in_"):
        num_features = model.n_features_in_
        val_list = [hindi, english, science, maths, history, geography]
        
        # Adjust array length if model expects more or fewer features
        if len(val_list) < num_features:
            val_list.extend([0] * (num_features - len(val_list)))
        elif len(val_list) > num_features:
            val_list = val_list[:num_features]
            
        input_data = np.array([val_list])
    else:
        input_data = pd.DataFrame([raw_inputs])

    with st.spinner("Processing input and running KNN model..."):
        time.sleep(0.5)
        try:
            prediction = model.predict(input_data)[0]
            st.balloons()
            
            st.markdown(
                f"""
                <div class="result-box">
                    <h3 style="margin:0; color: #94a3b8;">Predicted Result</h3>
                    <div class="result-score">{prediction}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as err:
            st.error(f"Prediction failed due to feature mismatch: {err}")
            if hasattr(model, "n_features_in_"):
                st.info(f"Model expects {model.n_features_in_} features.")
            if hasattr(model, "feature_names_in_"):
                st.info(f"Model expects these exact column names: {list(model.feature_names_in_)}")
