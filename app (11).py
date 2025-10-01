import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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
# SMART CHATBOT SYSTEM
# -------------------------------
class AHDChatbot:
    def __init__(self):
        self.knowledge_base = {
            "cd4": {
                "response": "**CD4 Count Guidelines:**\n\n- **Normal**: >500 cells/mm³\n- **Mild Immunodeficiency**: 350-500 cells/mm³\n- **Advanced Immunodeficiency**: 200-350 cells/mm³\n- **Severe Immunodeficiency**: <200 cells/mm³ (AHD criteria)\n- **Critical**: <100 cells/mm³\n\nCD4 monitoring should occur at ART initiation, 3 months after, then every 6-12 months.",
                "keywords": ["cd4", "t-cell", "immune", "cell count"]
            },
            "viral load": {
                "response": "**Viral Load Monitoring:**\n\n- **Suppressed**: <1000 copies/mL\n- **Unsuppressed**: ≥1000 copies/mL\n- **Virological failure**: >1000 copies/mL after 6 months of ART\n\nViral load should be measured at ART initiation, 3 months after, then every 6 months if suppressed.",
                "keywords": ["viral load", "vl", "suppressed", "unsuppressed", "virological"]
            },
            "ahd": {
                "response": "**Advanced HIV Disease (AHD) Definition:**\n\nAHD is defined as:\n- CD4 count <200 cells/mm³ **OR**\n- WHO Clinical Stage 3 or 4 disease\n\n**Key Management Principles:**\n1. Rapid ART initiation\n2. TB screening and prevention\n3. Cryptococcal screening (if CD4 <100)\n4. Enhanced adherence support\n5. Close clinical monitoring",
                "keywords": ["ahd", "advanced", "severe", "stage 3", "stage 4", "who stage"]
            },
            "art": {
                "response": "**ART Guidelines Summary:**\n\n**First-line Regimens:**\n- TDF + 3TC/FTC + DTG (preferred)\n- TAF + 3TC/FTC + DTG\n- AZT + 3TC + DTG\n\n**When to Start:**\n- All patients regardless of CD4 count\n- Urgently if CD4 <200 or symptomatic\n\n**Monitoring:**\n- Clinical assessment at 2-4 weeks, then monthly\n- CD4 and VL at baseline, 3 months, then 6-monthly",
                "keywords": ["art", "treatment", "regimen", "medication", "arv", "antiretroviral"]
            },
            "tb": {
                "response": "**TB/HIV Co-infection:**\n\n- Screen all HIV patients for TB at every visit\n- **Symptoms**: Cough, fever, night sweats, weight loss\n- **Diagnosis**: GeneXpert preferred over smear microscopy\n- **Prevention**: TPT for all without active TB\n- **Treatment**: Start ART within 2 weeks of TB treatment",
                "keywords": ["tb", "tuberculosis", "tpt", "isoniazid"]
            },
            "oi": {
                "response": "**Opportunistic Infections (OIs) in AHD:**\n\n**Common OIs:**\n- Tuberculosis\n- Cryptococcal meningitis\n- PJP (Pneumocystis pneumonia)\n- Toxoplasmosis\n- CMV disease\n- Severe bacterial infections\n\n**Prevention:**\n- Cotrimoxazole prophylaxis for CD4 <200\n- Fluconazole if CD4 <100\n- TB preventive therapy",
                "keywords": ["oi", "opportunistic", "infection", "cryptococcal", "pjp", "toxo", "cmv"]
            },
            "prevention": {
                "response": "**HIV Prevention:**\n\n- **PrEP**: For HIV-negative at-risk individuals\n- **PEP**: Within 72 hours of exposure\n- **PMTCT**: ART for all pregnant women\n- **Safe practices**: Condoms, sterile equipment\n- **VMMC**: Reduces transmission risk",
                "keywords": ["prevention", "prep", "pep", "pmtct", "condom", "circumcision"]
            }
        }
    
    def get_response(self, user_input):
        user_input = user_input.lower()
        
        # Check for specific topics
        for topic, data in self.knowledge_base.items():
            if any(keyword in user_input for keyword in data["keywords"]):
                return data["response"]
        
        # General responses
        if any(word in user_input for word in ["hello", "hi", "hey"]):
            return "Hello! I'm your AHD clinical assistant. I can help with CD4 interpretation, viral load monitoring, ART guidelines, AHD management, and opportunistic infections. What would you like to know?"
        
        elif any(word in user_input for word in ["thank", "thanks"]):
            return "You're welcome! Is there anything else about HIV/AHD management you'd like to know?"
        
        elif "who stage" in user_input:
            return "**WHO Clinical Staging:**\n\n**Stage 1**: Asymptomatic\n**Stage 2**: Mild symptoms/signs\n**Stage 3**: Advanced symptoms\n**Stage 4**: Severe symptoms\n\nStages 3 and 4 indicate Advanced HIV Disease regardless of CD4 count."
        
        elif "side effect" in user_input or "adverse" in user_input:
            return "**Common ART Side Effects:**\n\n- **DTG**: Insomnia, dizziness (usually resolves)\n- **TDF**: Renal issues, bone density\n- **EFV**: CNS effects, rash\n- **NVP**: Hepatotoxicity, rash\n\nManage with symptomatic treatment; switch regimen if severe."
        
        # Default response
        return """I specialize in HIV and Advanced HIV Disease guidelines. Here are topics I can help with:

• **CD4 count interpretation and monitoring**
• **Viral load suppression targets**
• **ART regimens and when to start**
• **AHD diagnosis and management**
• **Opportunistic infection prevention**
• **TB/HIV co-infection**
• **WHO clinical staging**

What specific aspect would you like to know more about?"""

# Initialize chatbot
chatbot = AHDChatbot()

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
                st.info("**Recommendations:**\n- Urgent ART initiation\n- Comprehensive OI screening\n- Enhanced adherence support\n- Close follow-up")
            elif proba > 0.45:
                st.warning("🟠 Moderate Risk – Monitor closely.")
                st.info("**Recommendations:**\n- Timely ART initiation\n- Regular monitoring\n- Adherence counseling")
            else:
                st.success("🟢 Low Risk – Continue routine care.")
                st.info("**Recommendations:**\n- Standard ART care\n- Routine monitoring\n- Prevention counseling")

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
        ax.axvline(200, color='red', linestyle='--', label='AHD Threshold (CD4<200)')
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.write("📊 Viral Load vs CD4")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x="CD4 Count", y="Viral Load", hue="Risk", ax=ax)
        ax.axhline(1000, color='orange', linestyle='--', label='Suppression Threshold')
        ax.legend()
        st.pyplot(fig)
    
    # Additional insights
    st.subheader("📋 Clinical Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("AHD Prevalence", "23%", "2% from last month")
    
    with col2:
        st.metric("Virological Suppression", "78%", "5% improvement")
    
    with col3:
        st.metric("ART Coverage", "89%", "3% increase")

# -------------------------------
# TAB 3: Chatbot - LOCAL KNOWLEDGE BASE VERSION
# -------------------------------
with tab3:
    st.subheader("💬 AHD Guideline Chatbot")
    st.info("💡 **Ask me about:** CD4 counts, viral load, ART regimens, AHD management, WHO staging, opportunistic infections, or TB/HIV co-infection")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AHD clinical assistant. I can help with HIV treatment guidelines, CD4 interpretation, viral load monitoring, and AHD management. What would you like to know?"}
        ]
    
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
            
            # Get response from local knowledge base
            response = chatbot.get_response(prompt)
            message_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick action buttons
    st.markdown("### Quick Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("CD4 Guidelines"):
            st.session_state.messages.append({"role": "user", "content": "CD4 guidelines"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot.get_response("CD4 guidelines")})
            st.rerun()
    
    with col2:
        if st.button("AHD Definition"):
            st.session_state.messages.append({"role": "user", "content": "What is AHD?"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot.get_response("What is AHD?")})
            st.rerun()
    
    with col3:
        if st.button("ART Regimens"):
            st.session_state.messages.append({"role": "user", "content": "ART regimens"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot.get_response("ART regimens")})
            st.rerun()
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AHD clinical assistant. How can I help you today?"}
        ]
        st.rerun()

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>© 2025 | Built with ❤️ by <b>Idah Anyango</b></div>", unsafe_allow_html=True)
