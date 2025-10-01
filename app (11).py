import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import InferenceClient
import requests
import json

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AHD Copilot", layout="wide", page_icon="🧠")

st.title("🧠 Advanced HIV Disease (AHD) Copilot")
st.markdown("""
This tool supports clinicians in **detecting Advanced HIV Disease (AHD)**,  
exploring analytics, and interacting with **guideline chatbot**.  
""")

# -------------------------------
# Load Model
# -------------------------------
try:
    deploy = joblib.load("ahd_model_C_hybrid_fixed.pkl")
    model = deploy['model']
    feature_names = deploy['feature_names']
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load model: {e}")
    model_loaded = False

# -------------------------------
# Hugging Face Chatbot Setup - ROBUST VERSION
# -------------------------------
def initialize_hf_client():
    """Initialize Hugging Face client with multiple fallback options"""
    try:
        HF_TOKEN = st.secrets["huggingface"]["token"]
        
        # List of reliable models that work with Inference API
        model_options = [
            "microsoft/DialoGPT-medium",  # Conversational AI
            "gpt2",  # Basic GPT-2
            "google/flan-t5-base",  # Instruction-following model
            "facebook/blenderbot-400M-distill",  # Chat model
            "distilgpt2"  # Lightweight GPT-2
        ]
        
        client = None
        working_model = None
        
        # Try each model until one works
        for model_name in model_options:
            try:
                st.info(f"🔄 Trying model: {model_name}")
                client = InferenceClient(token=HF_TOKEN)
                
                # Test the model with a simple prompt
                test_response = client.text_generation(
                    prompt="Hello",
                    model=model_name,
                    max_new_tokens=10
                )
                
                working_model = model_name
                st.success(f"✅ Successfully initialized with model: {working_model}")
                break
                
            except Exception as e:
                st.warning(f"❌ Model {model_name} failed: {str(e)[:100]}...")
                continue
        
        if client and working_model:
            return client, working_model
        else:
            st.error("❌ All model attempts failed. Using fallback mode.")
            return None, None
            
    except Exception as e:
        st.error(f"❌ Failed to initialize Hugging Face client: {e}")
        return None, None

# Initialize the client
client, MODEL_NAME = initialize_hf_client()

# -------------------------------
# Alternative: Direct API Call Approach
# -------------------------------
def query_hf_api(prompt, model_name="microsoft/DialoGPT-medium"):
    """Alternative method using direct API calls"""
    try:
        HF_TOKEN = st.secrets["huggingface"]["token"]
        API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.3,
                "do_sample": True
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if isinstance(result, list) and 'generated_text' in result[0]:
            return result[0]['generated_text']
        elif isinstance(result, dict) and 'generated_text' in result:
            return result['generated_text']
        else:
            return str(result)
            
    except Exception as e:
        raise Exception(f"API call failed: {e}")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Analytics", "💬 Guideline Chatbot"])

# -------------------------------
# TAB 1: Dashboard (Prediction) - UNCHANGED
# -------------------------------
with tab1:
    st.subheader("📊 Patient Risk Prediction Dashboard")

    if model_loaded:
        st.sidebar.header("📝 Patient Information")
        age = st.sidebar.number_input("Age at Reporting", min_value=0, max_value=120, value=35)
        weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=60.0)
        height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=220, value=165)
        cd4 = st.sidebar.number_input("Latest CD4 Count", min_value=0, max_value=2000, value=350)
        vl = st.sidebar.number_input("Latest Viral Load (copies/ml)", min_value=0, max_value=10000000, value=1000)
        months_rx = st.sidebar.slider("Months of Prescription", 0, 6, 3)
        who_stage = st.sidebar.selectbox("Last WHO Stage", [1, 2, 3, 4])
        cd4_risk = st.sidebar.selectbox("CD4 Risk Category", ["Severe", "Moderate", "Normal", "Unknown"])
        sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
        st.sidebar.markdown("---")

        # Derived features (unchanged)
        bmi = weight / ((height / 100) ** 2) if height > 0 else 0
        cd4_missing = 0 if cd4 > 0 else 1
        vl_missing = 0 if vl > 0 else 1
        vl_suppressed = 1 if vl < 1000 else 0

        cd4_risk_Severe = 1 if cd4_risk == "Severe" else 0
        cd4_risk_Moderate = 1 if cd4_risk == "Moderate" else 0
        cd4_risk_Normal = 1 if cd4_risk == "Normal" else 0

        Last_WHO_Stage_2 = 1 if who_stage == 2 else 0
        Last_WHO_Stage_3 = 1 if who_stage == 3 else 0
        Last_WHO_Stage_4 = 1 if who_stage == 4 else 0
        Sex_M = 1 if sex.lower().startswith("m") else 0

        Active_in_PMTCT_Missing = 0
        Cacx_Screening_Missing = 0
        Refill_Date_Missing = 0

        input_data_dict = {
            'Age at reporting': age,
            'Weight': weight,
            'Height': height,
            'BMI': bmi,
            'Latest CD4 Result': cd4,
            'CD4_Missing': cd4_missing,
            'Last VL Result': vl,
            'VL_Suppressed': vl_suppressed,
            'VL_Missing': vl_missing,
            'Months of Prescription': months_rx,
            'cd4_risk_Moderate': cd4_risk_Moderate,
            'cd4_risk_Normal': cd4_risk_Normal,
            'cd4_risk_Severe': cd4_risk_Severe,
            'Last_WHO_Stage_2': Last_WHO_Stage_2,
            'Last_WHO_Stage_3': Last_WHO_Stage_3,
            'Last_WHO_Stage_4': Last_WHO_Stage_4,
            'Active_in_PMTCT_Missing': Active_in_PMTCT_Missing,
            'Cacx_Screening_Missing': Cacx_Screening_Missing,
            'Refill_Date_Missing': Refill_Date_Missing,
            'Sex_M': Sex_M
        }

        X_input = pd.DataFrame([input_data_dict])[feature_names].astype(float)

        if st.sidebar.button("🔍 Predict AHD Risk"):
            pred = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0][1]

            st.metric("AHD Risk", "Yes" if int(pred) == 1 else "No")
            st.metric("Risk Probability", f"{proba:.2%}")

            st.progress(proba)
            if proba > 0.75:
                st.error("⚠️ High Risk – Consider immediate clinical review.")
            elif proba > 0.45:
                st.warning("🟠 Moderate Risk – Monitor closely.")
            else:
                st.success("🟢 Low Risk – Continue routine care.")

            with st.expander("Input Features Used"):
                st.write(X_input.T)

# -------------------------------
# TAB 2: Analytics (Sample) - UNCHANGED
# -------------------------------
with tab2:
    st.subheader("📈 Analytics Overview")

    # Dummy data for visualization
    data = pd.DataFrame({
        "CD4 Count": np.random.randint(50, 600, 100),
        "Viral Load": np.random.randint(20, 50000, 100),
        "Risk": np.random.choice(["Low", "Moderate", "High"], 100)
    })

    col1, col2 = st.columns(2)

    with col1:
        st.write("📊 CD4 Count Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data["CD4 Count"], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("📊 Viral Load vs CD4")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x="CD4 Count", y="Viral Load", hue="Risk", ax=ax)
        st.pyplot(fig)

# -------------------------------
# TAB 3: Chatbot - COMPLETELY FIXED VERSION
# -------------------------------
with tab3:
    st.subheader("💬 Guideline Chatbot")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about HIV/AHD guidelines..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # Enhanced medical context prompt
                medical_prompt = f"""As a medical AI assistant specializing in HIV and Advanced HIV Disease (AHD), provide accurate information about:

- HIV treatment guidelines
- AHD diagnosis and management
- CD4 count interpretation
- Viral load monitoring
- WHO staging
- ART regimens
- Prevention strategies

Question: {prompt}

Provide a concise, clinically relevant answer:"""

                # Try multiple approaches
                reply_text = None
                
                # Approach 1: Use InferenceClient with explicit model parameter
                if client and MODEL_NAME:
                    try:
                        response = client.text_generation(
                            prompt=medical_prompt,
                            model=MODEL_NAME,  # Explicitly specify model
                            max_new_tokens=300,
                            temperature=0.3,
                            do_sample=True
                        )
                        reply_text = response.strip()
                    except Exception as e:
                        st.warning(f"Client approach failed, trying API direct: {e}")
                
                # Approach 2: Direct API call
                if not reply_text:
                    try:
                        reply_text = query_hf_api(medical_prompt, "microsoft/DialoGPT-medium")
                    except Exception as e:
                        st.warning(f"API direct approach failed: {e}")
                
                # Approach 3: Final fallback
                if not reply_text:
                    reply_text = f"""I'm currently experiencing technical difficulties with my AI model. 

For accurate HIV/AHD guidelines, please consult:
- WHO Consolidated HIV Guidelines
- National HIV Treatment Guidelines
- CDC HIV Clinical Guidelines

For your question about "{prompt}", please refer to the latest WHO clinical guidelines for HIV treatment and AHD management."""

                # Clean up response
                if medical_prompt in reply_text:
                    reply_text = reply_text.replace(medical_prompt, "").strip()
                
                message_placeholder.markdown(reply_text)
                st.session_state.messages.append({"role": "assistant", "content": reply_text})
                
            except Exception as e:
                error_msg = f"""I apologize, but I'm having technical issues at the moment. 

Error details: {str(e)}

For immediate assistance with HIV/AHD guidelines, please consult:
- Your institutional protocol
- WHO HIV guidelines
- National Ministry of Health guidelines"""
                
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat and model selection
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("Reinitialize Chatbot"):
            st.session_state.client, st.session_state.MODEL_NAME = initialize_hf_client()
            st.rerun()

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>© 2025 | Built with ❤️ by <b>Idah Anyango</b></div>", unsafe_allow_html=True)
