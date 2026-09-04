import streamlit as st
import pickle
import numpy as np
import time

# Page configuration
st.set_page_config(
    page_title="Academic Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling, cards, and button ripple/hover effects
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    /* Header Styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Cards */
    .input-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }

    /* Styled Predict Button with Glow & Pulse Effect */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
        transition: all 0.3s ease;
        cursor: pointer;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.7);
        background: linear-gradient(90deg, #4f46e5 0%, #9333ea 100%);
    }

    div.stButton > button:active {
        transform: translateY(1px) scale(0.99);
    }

    /* Result Box */
    .result-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        animation: fadeIn 0.8s ease-in-out;
    }

    .result-score {
        font-size: 3rem;
        font-weight: 900;
        color: #34d399;
    }

    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load model from model.pkl"""
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading `model.pkl`: {e}")
    st.stop()

# Header Section
st.markdown('<div class="main-title">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter subject marks to calculate and predict the overall score.</div>', unsafe_allow_html=True)

# Main Form Container
with st.container():
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.subheader("📝 Enter Subject Scores (0 - 100)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hindi = st.number_input("Hindi", min_value=0, max_value=100, value=75, step=1)
        english = st.number_input("English", min_value=0, max_value=100, value=80, step=1)
        science = st.number_input("Science", min_value=0, max_value=100, value=85, step=1)
        
    with col2:
        maths = st.number_input("Maths", min_value=0, max_value=100, value=90, step=1)
        history = st.number_input("History", min_value=0, max_value=100, value=70, step=1)
        geography = st.number_input("Geography", min_value=0, max_value=100, value=78, step=1)
        
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction Action
if st.button("🚀 Predict Total Score"):
    # Input validation / array prep
    input_data = np.array([[hindi, english, science, maths, history, geography]])
    
    # Visual effects during processing
    with st.spinner("Analyzing marks and running model prediction..."):
        time.sleep(0.8) # Small delay to show off animation effect
        prediction = model.predict(input_data)[0]

    # Show celebratory effects
    st.balloons()
    
    # Render Result
    st.markdown(
        f"""
        <div class="result-box">
            <h3 style="margin: 0; color: #94a3b8; font-weight: 600;">Predicted Output</h3>
            <div class="result-score">{prediction}</div>
            <p style="margin: 0; color: #cbd5e1;">Model: K-Neighbors Classifier</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
