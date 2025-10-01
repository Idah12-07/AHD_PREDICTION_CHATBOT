import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
# IMPROVED CHATBOT SYSTEM
# -------------------------------
class AHDChatbot:
    def __init__(self):
        self.knowledge_base = {
            # Basic Definitions
            "what is ahd": {
                "response": """**Advanced HIV Disease (AHD) Definition:**

AHD refers to HIV infection that has progressed to a severe stage, specifically defined as:
- **CD4 count below 200 cells/mm³** **OR**
- **WHO Clinical Stage 3 or 4 disease** regardless of CD4 count

**Key Characteristics:**
• Severe immunodeficiency
• High risk of opportunistic infections
• Requires urgent intervention and close monitoring""",
                "keywords": ["what is ahd", "define ahd", "ahd definition", "meaning of ahd"]
            },
            "what is hiv": {
                "response": """**HIV (Human Immunodeficiency Virus) Definition:**

HIV is a virus that attacks the body's immune system, specifically the CD4 cells (T cells), which help the immune system fight off infections.

**Key Facts:**
- **Transmission**: Through bodily fluids (blood, semen, vaginal fluids, breast milk)
- **Progression**: Without treatment, leads to AIDS
- **Treatment**: ART (Antiretroviral Therapy) can control the virus
- **Prevention**: Condoms, PrEP, PEP, safe injection practices""",
                "keywords": ["what is hiv", "define hiv", "hiv definition", "what does hiv mean"]
            },
            "what is cd4": {
                "response": """**CD4 Cell Definition:**

CD4 cells (also called T-helper cells) are white blood cells that play a crucial role in the immune system. They coordinate the body's response to infections.

**In HIV Context:**
- **Normal range**: 500-1500 cells/mm³
- **HIV impact**: Virus destroys CD4 cells
- **Monitoring**: CD4 count indicates immune system health
- **AHD threshold**: <200 cells/mm³ indicates advanced disease""",
                "keywords": ["what is cd4", "define cd4", "cd4 definition", "what are cd4 cells"]
            },
            "what is viral load": {
                "response": """**Viral Load Definition:**

Viral load refers to the amount of HIV virus in a milliliter of blood.

**Clinical Significance:**
- **Measurement**: Copies per milliliter (copies/mL)
- **Suppressed**: <1000 copies/mL (treatment success)
- **Unsuppressed**: ≥1000 copies/mL (needs intervention)
- **Monitoring**: Key indicator of treatment effectiveness""",
                "keywords": ["what is viral load", "define viral load", "viral load definition"]
            },
            "what is art": {
                "response": """**ART (Antiretroviral Therapy) Definition:**

ART is the combination of medications used to treat HIV infection. It doesn't cure HIV but controls the virus, allowing people to live long, healthy lives.

**Key Principles:**
- **Combination therapy**: Usually 3 drugs from different classes
- **Adherence**: Must be taken consistently
- **Goals**: Suppress viral load, preserve CD4 cells, prevent transmission
- **Initiation**: Recommended for all people with HIV""",
                "keywords": ["what is art", "define art", "art definition", "what does art mean"]
            },
            
            # Clinical Guidelines
            "cd4": {
                "response": """**CD4 Count Guidelines:**

**Interpretation:**
- **>500 cells/mm³**: Normal immune function
- **350-500 cells/mm³**: Mild immunodeficiency  
- **200-350 cells/mm³**: Advanced immunodeficiency
- **<200 cells/mm³**: Severe immunodeficiency (AHD criteria)
- **<100 cells/mm³**: Critical risk for OIs

**Monitoring Frequency:**
- At ART initiation
- 3 months after starting ART
- Every 6-12 months if stable""",
                "keywords": ["cd4", "t-cell", "immune", "cell count", "cd4 count"]
            },
            "viral load": {
                "response": """**Viral Load Monitoring Guidelines:**

**Classification:**
- **Suppressed**: <1000 copies/mL
- **Unsuppressed**: ≥1000 copies/mL  
- **Virological failure**: >1000 copies/mL after 6 months of ART

**Monitoring Schedule:**
- At ART initiation
- 3 months after starting ART
- Every 6 months if suppressed
- Every 3 months if unsuppressed""",
                "keywords": ["viral load", "vl", "suppressed", "unsuppressed", "virological"]
            },
            "ahd management": {
                "response": """**AHD Management Protocol:**

**Urgent Actions:**
1. **Rapid ART initiation** (within 7 days, same day if possible)
2. **Comprehensive OI screening** (TB, cryptococcus, etc.)
3. **Preventive therapy** (Cotrimoxazole, fluconazole if CD4<100)
4. **Enhanced adherence support**

**Key Components:**
- Close clinical monitoring (2-4 week follow-up)
- Psychosocial support
- Treatment literacy
- Comorbidity management""",
                "keywords": ["ahd management", "manage ahd", "treat ahd", "ahd protocol"]
            },
            "art regimens": {
                "response": """**ART Regimen Guidelines:**

**Preferred First-line:**
- **TDF + 3TC/FTC + DTG** (Dolutegravir-based)
- **TAF + 3TC/FTC + DTG** (if renal/bone concerns)

**Alternative Options:**
- AZT + 3TC + DTG
- ABC + 3TC + DTG

**When to Start:**
- All patients regardless of CD4 count
- Urgently if CD4 <200 or symptomatic""",
                "keywords": ["art", "treatment", "regimen", "medication", "arv", "antiretroviral", "first-line"]
            },
            "tb screening": {
                "response": """**TB/HIV Co-infection Screening:**

**Screening at Every Visit:**
- **Symptoms**: Cough >2 weeks, fever, night sweats, weight loss
- **Diagnosis**: GeneXpert (preferred over smear)
- **Prevention**: TPT for all without active TB
- **Treatment**: Start ART within 2 weeks of TB treatment

**TPT Options:**
- 6H (isoniazid × 6 months)
- 3HP (isoniazid + rifapentine × 3 months)
- 1HP (isoniazid + rifapentine × 1 month)""",
                "keywords": ["tb", "tuberculosis", "tpt", "isoniazid", "screening"]
            },
            "oi prevention": {
                "response": """**Opportunistic Infection Prevention:**

**Cotrimoxazole Prevention:**
- **Indication**: CD4 <200 or WHO stage 3/4
- **Duration**: Until CD4 >200 for 6 months on ART

**Fluconazole Prevention:**
- **Indication**: CD4 <100 in cryptococcal meningitis endemic areas
- **Duration**: Until CD4 >200 for 6 months

**Other Key Measures:**
- TB preventive therapy
- Vaccinations (PCV, HPV, influenza)
- Safe water and food practices""",
                "keywords": ["oi", "opportunistic", "infection", "prevention", "cotrimoxazole", "fluconazole"]
            }
        }
    
    def get_response(self, user_input):
        user_input = user_input.lower().strip()
        
        # First, check for exact definition matches
        for topic, data in self.knowledge_base.items():
            if any(user_input == keyword for keyword in data["keywords"]):
                return data["response"]
        
        # Then check for keyword matches
        for topic, data in self.knowledge_base.items():
            if any(keyword in user_input for keyword in data["keywords"]):
                return data["response"]
        
        # Handle general questions
        return self._handle_general_questions(user_input)
    
    def _handle_general_questions(self, user_input):
        """Handle general questions and conversations"""
        user_input_lower = user_input.lower()
        
        # Greetings
        if any(word in user_input_lower for word in ["hello", "hi", "hey", "greetings"]):
            return """Hello! I'm your AHD clinical assistant. I can help you with:

• **Basic definitions** (HIV, AHD, CD4, viral load, ART)
• **Clinical guidelines** and protocols  
• **Treatment recommendations**
• **Monitoring schedules**
• **Prevention strategies**

What would you like to know about today?"""
        
        # Thanks
        elif any(word in user_input_lower for word in ["thank", "thanks"]):
            return "You're welcome! I'm here to help with any other HIV/AHD questions you have."
        
        # WHO staging
        elif "who stage" in user_input_lower:
            return """**WHO Clinical Staging System:**

**Stage 1**: Asymptomatic or persistent generalized lymphadenopathy
**Stage 2**: Moderate unexplained weight loss, recurrent respiratory infections, herpes zoster
**Stage 3**: Severe weight loss, chronic diarrhea, persistent fever, pulmonary TB, severe bacterial infections
**Stage 4**: HIV wasting syndrome, PCP, toxoplasmosis, cryptococcosis, extrapulmonary TB, Kaposi sarcoma

*Stages 3 and 4 indicate Advanced HIV Disease*"""
        
        # Side effects
        elif any(phrase in user_input_lower for phrase in ["side effect", "adverse", "complication"]):
            return """**Common ART Side Effects:**

**DTG (Dolutegravir):** Insomnia, dizziness, headache (usually resolve in weeks)
**TDF (Tenofovir):** Renal impairment, bone density loss
**EFV (Efavirenz):** CNS effects (dizziness, dreams), rash
**NVP (Nevirapine):** Hepatotoxicity, severe rash

**Management:** Most side effects improve with time. Consult clinician for persistent or severe effects."""
        
        # Default response for unrecognized queries
        return """I want to make sure I understand your question correctly. I specialize in HIV and Advanced HIV Disease topics like:

• **Basic concepts**: What is HIV? What is AHD? What are CD4 cells?
• **Clinical guidelines**: CD4 monitoring, viral load targets, ART regimens  
• **AHD management**: Screening, prevention, treatment protocols
• **WHO staging** and opportunistic infections

Could you rephrase your question or ask about one of these specific topics?"""

# Initialize chatbot
chatbot = AHDChatbot()

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

        # Derived features (same as before)
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

# -------------------------------
# TAB 3: Chatbot - IMPROVED VERSION
# -------------------------------
with tab3:
    st.subheader("💬 AHD Guideline Chatbot")
    st.info("💡 **Ask me about definitions:** 'What is HIV?', 'What is AHD?', 'What are CD4 cells?' or ask about clinical guidelines")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AHD clinical assistant. I can help with **basic definitions** and **clinical guidelines** for HIV and Advanced HIV Disease. What would you like to know?"}
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
            
            # Get response from improved chatbot
            response = chatbot.get_response(prompt)
            message_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick action buttons for definitions
    st.markdown("### Quick Definitions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("What is HIV?"):
            st.session_state.messages.append({"role": "user", "content": "What is HIV?"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot.get_response("What is HIV?")})
            st.rerun()
    
    with col2:
        if st.button("What is AHD?"):
            st.session_state.messages.append({"role": "user", "content": "What is AHD?"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot.get_response("What is AHD?")})
            st.rerun()
    
    with col3:
        if st.button("What is CD4?"):
            st.session_state.messages.append({"role": "user", "content": "What is CD4?"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot.get_response("What is CD4?")})
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
