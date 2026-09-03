import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better mobile responsiveness
st.markdown("""
<style>
    /* Mobile-first responsive design */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    @media (max-width: 768px) {
        .stButton button {
            width: 100% !important;
            font-size: 18px !important;
            padding: 0.75rem !important;
        }
        
        .stSelectbox, .stSlider, .stNumberInput {
            margin-bottom: 0.5rem;
        }
        
        .stMetric {
            padding: 0.5rem;
        }
        
        .row-widget.stColumns {
            flex-direction: column !important;
        }
        
        .stSelectbox select, .stNumberInput input {
            font-size: 16px !important;
            padding: 12px !important;
        }
        
        h1 {
            font-size: 24px !important;
        }
        h2 {
            font-size: 20px !important;
        }
        h3 {
            font-size: 18px !important;
        }
        
        .element-container {
            padding: 0.25rem 0;
        }
    }
    
    @media (min-width: 769px) and (max-width: 1024px) {
        .stButton button {
            font-size: 16px !important;
            padding: 0.6rem 1.2rem !important;
        }
    }
    
    .risk-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .risk-low {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    
    .risk-moderate {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    
    .risk-elevated {
        background-color: #ffe5d0;
        border: 2px solid #fd7e14;
    }
    
    .risk-high {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .stButton button {
        touch-action: manipulation;
        cursor: pointer;
    }
    
    /* Result cards */
    .result-success {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-danger {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Define the model architecture
class OptunaDiabetesANN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_2, 1)
        )

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_model_artifacts():
    """Load the trained model, scaler, and configuration."""
    try:
        config_path = Path("models/model_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        scaler = joblib.load("models/scaler.joblib")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = OptunaDiabetesANN(
            input_size=config["input_size"],
            hidden_size_1=config["hidden_size_1"],
            hidden_size_2=config["hidden_size_2"],
            dropout=config["dropout"]
        )
        model.load_state_dict(torch.load("models/diabetes_ann_model.pth", map_location=device))
        model.to(device)
        model.eval()
        
        return model, scaler, config, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


model, scaler, config, device = load_model_artifacts()

# Feature information
feature_info = {
    "HighBP": {"label": "High Blood Pressure", "type": "binary", "options": ["No", "Yes"]},
    "HighChol": {"label": "High Cholesterol", "type": "binary", "options": ["No", "Yes"]},
    "CholCheck": {"label": "Cholesterol Check", "type": "binary", "options": ["No", "Yes"]},
    "BMI": {"label": "Body Mass Index (BMI)", "type": "float", "min": 10.0, "max": 60.0, "default": 25.0},
    "Smoker": {"label": "Smoker", "type": "binary", "options": ["No", "Yes"]},
    "Stroke": {"label": "Had a Stroke", "type": "binary", "options": ["No", "Yes"]},
    "HeartDiseaseorAttack": {"label": "Heart Disease/Attack", "type": "binary", "options": ["No", "Yes"]},
    "PhysActivity": {"label": "Physical Activity", "type": "binary", "options": ["No", "Yes"]},
    "Fruits": {"label": "Eats Fruits Daily", "type": "binary", "options": ["No", "Yes"]},
    "Veggies": {"label": "Eats Vegetables Daily", "type": "binary", "options": ["No", "Yes"]},
    "HvyAlcoholConsump": {"label": "Heavy Alcohol Use", "type": "binary", "options": ["No", "Yes"]},
    "AnyHealthcare": {"label": "Has Healthcare", "type": "binary", "options": ["No", "Yes"]},
    "NoDocbcCost": {"label": "Couldn't See Doctor Due to Cost", "type": "binary", "options": ["No", "Yes"]},
    "GenHlth": {"label": "General Health", "type": "select", "options": ["Excellent", "Very Good", "Good", "Fair", "Poor"]},
    "MentHlth": {"label": "Poor Mental Health (Days)", "type": "int", "min": 0, "max": 30, "default": 0},
    "PhysHlth": {"label": "Poor Physical Health (Days)", "type": "int", "min": 0, "max": 30, "default": 0},
    "DiffWalk": {"label": "Difficulty Walking", "type": "binary", "options": ["No", "Yes"]},
    "Sex": {"label": "Gender", "type": "binary", "options": ["Female", "Male"]},
    "Age": {"label": "Age Group", "type": "select", "options": ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]},
    "Education": {"label": "Education", "type": "select", "options": ["Never attended", "Elementary", "Some high school", "High school grad", "Some college", "College grad"]},
    "Income": {"label": "Income Level", "type": "select", "options": ["< $10K", "$10-15K", "$15-20K", "$20-25K", "$25-35K", "$35-50K", "$50-75K", "> $75K"]},
}

# Mapping dictionaries
gen_health_map = {"Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5}
age_map = {
    "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
    "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
    "70-74": 11, "75-79": 12, "80+": 13
}
education_map = {
    "Never attended": 1, "Elementary": 2, "Some high school": 3,
    "High school grad": 4, "Some college": 5, "College grad": 6
}
income_map = {
    "< $10K": 1, "$10-15K": 2, "$15-20K": 3, "$20-25K": 4,
    "$25-35K": 5, "$35-50K": 6, "$50-75K": 7, "> $75K": 8
}


def predict_diabetes(patient_data):
    """Make a prediction using the trained model."""
    feature_columns = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
    ]
    
    patient_df = pd.DataFrame([patient_data])[feature_columns]
    patient_scaled = scaler.transform(patient_df)
    patient_tensor = torch.tensor(patient_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(patient_tensor)
        probability = torch.sigmoid(output).item()
    
    threshold = config.get("threshold", 0.5)
    prediction = int(probability >= threshold)
    
    return probability, prediction


# ---- APP UI ----
# Header
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h1>🩺 Diabetes Risk Predictor</h1>
    <p style="font-size: 1.1rem; color: #666;">
        Enter your health information for a personalized diabetes risk assessment
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for results
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'results' not in st.session_state:
    st.session_state.results = None

# Input Section
st.markdown("### 📋 Your Health Information")

# Use expandable sections for better mobile experience
with st.expander("👤 Demographics & General Health", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        user_inputs = {}
        user_inputs["Sex"] = 1 if st.selectbox("Gender", feature_info["Sex"]["options"]) == "Male" else 0
        age_text = st.selectbox("Age Group", feature_info["Age"]["options"])
        user_inputs["Age"] = age_map[age_text]
        edu_text = st.selectbox("Education", feature_info["Education"]["options"])
        user_inputs["Education"] = education_map[edu_text]
    
    with col2:
        income_text = st.selectbox("Income", feature_info["Income"]["options"])
        user_inputs["Income"] = income_map[income_text]
        gen_health_text = st.selectbox("General Health", feature_info["GenHlth"]["options"])
        user_inputs["GenHlth"] = gen_health_map[gen_health_text]

with st.expander("📏 Physical Health Indicators", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        user_inputs["BMI"] = st.slider(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.5,
            help="BMI = weight(kg) / height(m)²"
        )
        user_inputs["MentHlth"] = st.number_input(
            "Days of Poor Mental Health",
            min_value=0,
            max_value=30,
            value=0,
            step=1
        )
    
    with col2:
        user_inputs["PhysHlth"] = st.number_input(
            "Days of Poor Physical Health",
            min_value=0,
            max_value=30,
            value=0,
            step=1
        )

with st.expander("❤️ Health Conditions & Lifestyle", expanded=True):
    binary_features = [
        "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk"
    ]
    
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(binary_features):
        col_target = col1 if i < len(binary_features) // 2 else col2
        value = col_target.selectbox(
            feature_info[feature]["label"],
            feature_info[feature]["options"],
            key=f"binary_{feature}"
        )
        user_inputs[feature] = 1 if value == "Yes" else 0

# Predict button
st.markdown("---")
predict_button = st.button(
    "🔍 Assess Diabetes Risk",
    use_container_width=True,
    type="primary"
)

# Results Section - Appears directly below the button
if predict_button:
    st.session_state.show_results = True
    with st.spinner("🔄 Analyzing your health data..."):
        try:
            probability, prediction = predict_diabetes(user_inputs)
            st.session_state.results = {
                'probability': probability,
                'prediction': prediction
            }
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.session_state.show_results = False

# Display results if available
if st.session_state.show_results and st.session_state.results:
    probability = st.session_state.results['probability']
    prediction = st.session_state.results['prediction']
    
    st.markdown("---")
    st.markdown("## 📊 Your Results")
    
    # Result card based on prediction
    if prediction == 1:
        st.markdown("""
        <div class="result-danger">
            <h2 style="color: #dc3545; margin: 0;">⚠️ Diabetes Risk Detected</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                Based on your health indicators, the model suggests you may be at risk.
            </p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.95rem; color: #666;">
                Please consult a healthcare professional for proper diagnosis and guidance.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-success">
            <h2 style="color: #28a745; margin: 0;">✅ No Diabetes Risk Detected</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                Based on your health indicators, the model does not detect diabetes risk.
            </p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.95rem; color: #666;">
                Continue maintaining a healthy lifestyle.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics in responsive layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Risk Probability",
            f"{probability * 100:.1f}%",
            help="The model's confidence score"
        )
    
    with col2:
        # Risk level with color
        if probability < 0.3:
            risk_level = "Low 🟢"
        elif probability < 0.5:
            risk_level = "Moderate 🟡"
        elif probability < 0.7:
            risk_level = "Elevated 🟠"
        else:
            risk_level = "High 🔴"
        
        st.metric(
            "Risk Level",
            risk_level,
            help="Risk classification based on probability"
        )
    
    with col3:
        st.metric(
            "Threshold",
            f"{config.get('threshold', 0.5):.2f}",
            help="Classification threshold used"
        )
    
    # Detailed interpretation
    with st.expander("📖 How to Interpret Your Results"):
        st.markdown("""
        ### Understanding Your Risk Assessment
        
        **Risk Probability:** The percentage indicates how confident the model is that you have diabetes.
        
        **Risk Levels:**
        - **🟢 Low (0-30%):** Low probability of diabetes. Continue healthy habits.
        - **🟡 Moderate (30-50%):** Some risk factors detected. Consider lifestyle improvements.
        - **🟠 Elevated (50-70%):** Multiple risk factors present. Consult a healthcare provider.
        - **🔴 High (70-100%):** High probability of diabetes. Seek medical attention promptly.
        
        ### Model Performance
        - **Accuracy:** 74.16%
        - **ROC-AUC Score:** 82.98%
        - **F1-Score:** 77.57%
        
        ### Important Note
        This is a **screening tool** and not a medical diagnosis. Always consult a qualified healthcare professional.
        """)
    
    # Input summary
    with st.expander("📊 View Your Input Summary"):
        display_data = {}
        for feature, value in user_inputs.items():
            label = feature_info[feature]["label"]
            if feature_info[feature]["type"] == "binary":
                display_data[label] = "Yes" if value == 1 else "No"
            elif feature == "GenHlth":
                display_data[label] = [k for k, v in gen_health_map.items() if v == value][0]
            elif feature == "Age":
                display_data[label] = [k for k, v in age_map.items() if v == value][0]
            elif feature == "Education":
                display_data[label] = [k for k, v in education_map.items() if v == value][0]
            elif feature == "Income":
                display_data[label] = [k for k, v in income_map.items() if v == value][0]
            else:
                display_data[label] = value
        
        summary_df = pd.DataFrame([display_data]).T
        summary_df.columns = ["Value"]
        st.dataframe(summary_df, use_container_width=True)

# Info section (always visible)
with st.expander("ℹ️ About This Tool", expanded=False):
    st.markdown("""
    ### About This Diabetes Risk Predictor
    
    This tool uses a deep learning model trained on health indicators from the BRFSS 2015 dataset.
    
    **Model Architecture:**
    - Artificial Neural Network with 2 hidden layers
    - 21 health and demographic features
    - Trained on 70,692 observations
    
    **Performance Metrics:**
    - Accuracy: 74.16%
    - F1-Score: 77.57%
    - ROC-AUC: 82.98%
    
    **⚠️ Disclaimer:**
    This application is for **educational purposes only** and should NOT be used as a medical diagnostic tool.
    """)

# Sidebar
with st.sidebar:
    st.header("ℹ️ Quick Guide")
    
    st.markdown("""
    **How to use:**
    1. Fill in your health information
    2. Click "Assess Diabetes Risk"
    3. Review your results below
    
    **Risk Levels:**
    - 🟢 Low (0-30%)
    - 🟡 Moderate (30-50%)
    - 🟠 Elevated (50-70%)
    - 🔴 High (70-100%)
    """)
    
    st.divider()
    
    st.caption(f"""
    **Model Configuration**
    - Threshold: {config.get('threshold', 0.5)}
    - Hidden Layer 1: {config.get('hidden_size_1', 96)} neurons
    - Hidden Layer 2: {config.get('hidden_size_2', 32)} neurons
    - Dropout Rate: {config.get('dropout', 0.2)}
    """)

# Footer
st.markdown("""
<div style="text-align: center; padding: 1rem 0; margin-top: 2rem; border-top: 1px solid #ddd;">
    <small>Made with ❤️ | For educational purposes only | Not a medical device</small>
</div>
""", unsafe_allow_html=True)