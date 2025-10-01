import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import InferenceClient

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
# Hugging Face Chatbot Setup - FIXED
# -------------------------------
try:
    HF_TOKEN = st.secrets["huggingface"]["token"]
    
    # Try different models - starting with a more reliable one
    # Option 1: Use a model that's known to work well with the inference API
    MODEL_NAME = "microsoft/DialoGPT-medium"  # Good for conversational AI
    # MODEL_NAME = "gpt2"  # Basic GPT-2
    # MODEL_NAME = "google/flan-t5-small"  # Alternative if others fail
    
    client = InferenceClient(token=HF_TOKEN)
    st.success(f"✅ Hugging Face client initialized with model: {MODEL_NAME}")
    
except Exception as e:
    client = None
    MODEL_NAME = None
    st.warning("⚠️ Hugging Face client failed to initialize.")
    st.error(f"Details: {e}")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Analytics", "💬 Guideline Chatbot"])

# -------------------------------
# TAB 1: Dashboard (Prediction)
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

        # Derived features
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
# TAB 2: Analytics (Sample)
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
# TAB 3: Chatbot - IMPROVED VERSION
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
            
            if client and MODEL_NAME:
                try:
                    # Enhanced prompt for medical context
                    medical_prompt = f"""You are a medical AI assistant specializing in HIV and Advanced HIV Disease (AHD) guidelines. 
                    Provide accurate, helpful information about HIV treatment, prevention, and AHD management.
                    
                    Question: {prompt}
                    
                    Answer:"""
                    
                    # Call Hugging Face API
                    response = client.text_generation(
                        prompt=medical_prompt,
                        max_new_tokens=300,
                        temperature=0.3,  # Lower temperature for more consistent responses
                        do_sample=True
                    )
                    
                    # Clean up the response
                    reply_text = response.strip()
                    
                    # Remove the prompt from response if it's included
                    if medical_prompt in reply_text:
                        reply_text = reply_text.replace(medical_prompt, "").strip()
                    
                    message_placeholder.markdown(reply_text)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": reply_text})
                    
                except Exception as e:
                    error_msg = f"❌ Chatbot failed to respond: {e}"
                    message_placeholder.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
                    # Debug information
                    with st.expander("Debug Information"):
                        st.write(f"Model: {MODEL_NAME}")
                        st.write(f"Error details: {str(e)}")
            else:
                error_msg = "❌ Chatbot service is currently unavailable. Please check your Hugging Face token and model configuration."
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>© 2025 | Built with ❤️ by <b>Idah Anyango</b></div>", unsafe_allow_html=True)
