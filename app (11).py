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
# Hugging Face Chatbot Setup
# -------------------------------
try:
    HF_TOKEN = st.secrets["huggingface"]["token"]
    client = InferenceClient(api_key=HF_TOKEN, model="gtp2")
except Exception as e:
    client = None
    st.warning("⚠️ Hugging Face token not found. Chatbot tab may not work.")

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
# -------------------------------
# TAB 3: Chatbot (Smart Fallback)
# -------------------------------
with tab3:
    st.subheader("💬 Guideline Chatbot")

    # Fallback responses dictionary
    fallback_responses = {
        "what is ahd": "Advanced HIV Disease (AHD) is defined by a CD4 count below 200 cells/mm³ or WHO stage 3 or 4 conditions.",
        "cd4 threshold": "The CD4 threshold for AHD is typically <200 cells/mm³.",
        "art eligibility": "All individuals with AHD should initiate ART promptly, regardless of CD4 count.",
        "who stage 3": "WHO Stage 3 includes conditions like severe weight loss, chronic diarrhea, and persistent fever.",
        "who stage 4": "WHO Stage 4 includes severe opportunistic infections such as cryptococcal meningitis and extrapulmonary TB.",
        "tb screening": "TB screening is essential for all PLHIV. Use symptom-based screening and consider Xpert MTB/RIF testing.",
        "cryptococcal screening": "Screen all patients with CD4 <100 for cryptococcal antigen before ART initiation.",
        "fluconazole use": "Fluconazole is used for pre-emptive treatment of cryptococcal antigen-positive patients.",
        "cotrimoxazole prophylaxis": "Cotrimoxazole is recommended for all AHD patients to prevent opportunistic infections.",
        "adherence support": "Enhanced adherence counseling is critical for patients with AHD starting or restarting ART."
    }

    st.markdown("💡 Try asking about: `CD4 threshold`, `ART eligibility`, `WHO stage 3`, `TB screening`, `fluconazole use`")

    user_input = st.text_input("Ask a question about HIV/AHD guidelines:")
    if st.button("Send") and user_input:
        user_input_lower = user_input.lower().strip()

        response = None
        source = "demo"

        # Try Hugging Face model first
        if client:
            try:
                prompt = f"Answer the following question clearly:\n{user_input}"
                response = client.text2text_generation(
                    prompt=prompt,
                    max_new_tokens=200
                )
                source = "model"
            except Exception:
                response = None

        # Fallback if model fails or client is None
        if not response:
            response = fallback_responses.get(
                user_input_lower,
                "❓ Sorry, I couldn't find a guideline for that. Try asking about CD4 thresholds, ART eligibility, or WHO staging."
            )

        st.markdown(f"**Assistant ({'LLM' if source == 'model' else 'Demo'}):** {response}")

        
# Footer
# -------------------------------
st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>© 2025 | Built with ❤️ by <b>Idah Anyango</b></div>", unsafe_allow_html=True)



