import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AHD Copilot", layout="wide", page_icon="🧠")

st.title("🧠 Advanced HIV Disease (AHD) Copilot")
st.markdown("""
This tool supports clinicians in **detecting Advanced HIV Disease (AHD)**,  
exploring analytics, and interacting with **comprehensive HIV/AIDS expert chatbot**.  
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
# COMPREHENSIVE HIV/AIDS EXPERT CHATBOT
# -------------------------------
class HIVExpertChatbot:
    def __init__(self):
        self.current_year = 2025
        self.statistics = {
            "global": {
                "total_cases": "38.4 million",
                "new_infections_2023": "1.3 million",
                "aids_related_deaths_2023": "630,000",
                "art_coverage": "76%"
            },
            "africa": {
                "total_cases": "25.6 million",
                "new_infections_2023": "800,000",
                "aids_related_deaths_2023": "380,000",
                "art_coverage": "78%",
                "most_affected": "Eastern and Southern Africa"
            },
            "kenya": {
                "total_cases": "1.4 million",
                "prevalence_rate": "4.5%",
                "new_infections_2023": "32,000",
                "aids_related_deaths_2023": "19,000",
                "art_coverage": "85%",
                "mother_to_child_transmission": "8.7%"
            }
        }
        
        self.treatment_regimens = {
            "first_line": {
                "preferred": [
                    "TDF + 3TC/FTC + DTG (Dolutegravir)",
                    "TAF + 3TC/FTC + DTG",
                    "TDF + 3TC/FTC + EFV (Efavirenz)"
                ],
                "alternative": [
                    "AZT + 3TC + DTG",
                    "ABC + 3TC + DTG"
                ]
            },
            "second_line": {
                "options": [
                    "TDF/FTC + DTG (if failed on NNRTI-based regimen)",
                    "AZT + 3TC + ATV/r (Atazanavir/ritonavir)",
                    "TDF + 3TC + LPV/r (Lopinavir/ritonavir)"
                ]
            },
            "third_line": {
                "options": [
                    "DRV/r + DTG + optimized NRTI backbone",
                    "Newer agents: Bictegravir, Doravirine"
                ]
            }
        }

    def get_statistics(self, region="global"):
        """Get HIV statistics for different regions"""
        if region.lower() in self.statistics:
            stats = self.statistics[region.lower()]
            response = f"**HIV Statistics for {region.upper()} ({self.current_year} estimates):**\n\n"
            
            for key, value in stats.items():
                # Format key for display
                display_key = key.replace('_', ' ').title()
                response += f"• **{display_key}**: {value}\n"
            
            response += f"\n*Source: WHO/UNAIDS {self.current_year} estimates*"
            return response
        else:
            return f"Statistics for {region} not available. Try 'global', 'africa', or 'kenya'."

    def get_treatment_info(self, regimen_type="first_line"):
        """Get detailed treatment regimen information"""
        if regimen_type in self.treatment_regimens:
            regimen = self.treatment_regimens[regimen_type]
            response = f"**{regimen_type.replace('_', ' ').title()} ART Regimens:**\n\n"
            
            if "preferred" in regimen:
                response += "**Preferred Regimens:**\n"
                for option in regimen["preferred"]:
                    response += f"• {option}\n"
                response += "\n"
            
            if "alternative" in regimen:
                response += "**Alternative Regimens:**\n"
                for option in regimen["alternative"]:
                    response += f"• {option}\n"
                response += "\n"
            
            if "options" in regimen:
                for option in regimen["options"]:
                    response += f"• {option}\n"
            
            response += f"\n*Based on WHO {self.current_year} Consolidated Guidelines*"
            return response
        else:
            return "Regimen type not found. Try 'first_line', 'second_line', or 'third_line'."

    def interpret_prediction(self, prediction, probability, features):
        """Interpret model prediction with clinical insights"""
        risk_level = "High" if prediction == 1 else "Low"
        
        interpretation = f"**Prediction Interpretation:**\n\n"
        interpretation += f"• **Risk Level**: {risk_level}\n"
        interpretation += f"• **Probability**: {probability:.1%}\n"
        interpretation += f"• **Confidence**: {'High' if probability > 0.7 or probability < 0.3 else 'Moderate'}\n\n"
        
        # Feature-based insights
        interpretation += "**Key Contributing Factors:**\n"
        
        # CD4-based insights
        cd4 = features.get('Latest CD4 Result', 0)
        if cd4 < 200:
            interpretation += f"• **Critical CD4**: {cd4} cells/mm³ (AHD threshold <200)\n"
        elif cd4 < 350:
            interpretation += f"• **Low CD4**: {cd4} cells/mm³ (needs close monitoring)\n"
        
        # Viral Load insights
        vl = features.get('Last VL Result', 0)
        vl_suppressed = features.get('VL_Suppressed', 0)
        if not vl_suppressed and vl > 0:
            interpretation += f"• **Unsuppressed VL**: {vl:,} copies/mL\n"
        
        # WHO Stage insights
        who_stage_3 = features.get('Last_WHO_Stage_3', 0)
        who_stage_4 = features.get('Last_WHO_Stage_4', 0)
        if who_stage_4:
            interpretation += f"• **WHO Stage 4**: Severe symptoms present\n"
        elif who_stage_3:
            interpretation += f"• **WHO Stage 3**: Advanced symptoms present\n"
        
        # BMI insights
        bmi = features.get('BMI', 0)
        if bmi < 18.5:
            interpretation += f"• **Low BMI**: {bmi:.1f} (underweight)\n"
        elif bmi > 30:
            interpretation += f"• **High BMI**: {bmi:.1f} (obese)\n"
        
        # Clinical recommendations
        interpretation += "\n**Clinical Recommendations:**\n"
        if prediction == 1:
            interpretation += "• **Urgent ART initiation** (within 7 days)\n"
            interpretation += "• **Comprehensive OI screening** (TB, cryptococcus)\n"
            interpretation += "• **Cotrimoxazole preventive therapy**\n"
            interpretation += "• **Enhanced adherence counseling**\n"
            interpretation += "• **Close follow-up** (2-4 weeks)\n"
        else:
            interpretation += "• **Continue routine ART care**\n"
            interpretation += "• **Standard monitoring schedule**\n"
            interpretation += "• **Prevention counseling**\n"
            interpretation += "• **Regular viral load monitoring**\n"
        
        return interpretation

    def get_response(self, user_input):
        user_input = user_input.lower().strip()
        
        # Statistics queries
        if any(word in user_input for word in ["statistic", "prevalence", "rate", "number", "data"]):
            if "kenya" in user_input:
                return self.get_statistics("kenya")
            elif "africa" in user_input:
                return self.get_statistics("africa")
            elif "global" in user_input or "world" in user_input:
                return self.get_statistics("global")
            else:
                return self.get_statistics("global")

        # Treatment regimen queries
        elif any(word in user_input for word in ["treatment", "regimen", "art", "medication", "first-line", "second-line", "third-line"]):
            if "second" in user_input:
                return self.get_treatment_info("second_line")
            elif "third" in user_input:
                return self.get_treatment_info("third_line")
            else:
                return self.get_treatment_info("first_line")

        # Basic definitions and comprehensive HIV knowledge
        elif any(phrase in user_input for phrase in ["what is hiv", "define hiv"]):
            return self._get_hiv_definition()
        
        elif any(phrase in user_input for phrase in ["what is ahd", "define ahd"]):
            return self._get_ahd_definition()
        
        elif any(phrase in user_input for phrase in ["what is cd4", "define cd4"]):
            return self._get_cd4_definition()
        
        elif any(phrase in user_input for phrase in ["what is viral load", "define viral load"]):
            return self._get_viral_load_definition()
        
        elif any(phrase in user_input for phrase in ["what is art", "define art"]):
            return self._get_art_definition()

        # Prevention methods
        elif any(word in user_input for word in ["prevent", "prevention", "prep", "pep", "condom"]):
            return self._get_prevention_info()

        # Transmission
        elif any(word in user_input for word in ["transmit", "transmission", "spread", "catch"]):
            return self._get_transmission_info()

        # Symptoms
        elif any(word in user_input for word in ["symptom", "sign", "feel", "experience"]):
            return self._get_symptoms_info()

        # Testing
        elif any(word in user_input for word in ["test", "testing", "diagnose", "result"]):
            return self._get_testing_info()

        # Opportunistic infections
        elif any(word in user_input for word in ["oi", "opportunistic", "infection", "tb", "cryptococcus"]):
            return self._get_oi_info()

        # WHO staging
        elif "who stage" in user_input:
            return self._get_who_staging()

        # General greeting
        elif any(word in user_input for word in ["hello", "hi", "hey", "greetings"]):
            return self._get_greeting()

        # Default comprehensive response
        else:
            return self._get_comprehensive_response(user_input)

    def _get_hiv_definition(self):
        return """**HIV (Human Immunodeficiency Virus) - Comprehensive Overview**

**Definition:** HIV is a virus that attacks the body's immune system, specifically targeting CD4 cells (T-helper cells), which are crucial for fighting infections.

**Virology:**
- **Family**: Retroviridae
- **Types**: HIV-1 (most common worldwide), HIV-2 (primarily West Africa)
- **Structure**: Enveloped virus with RNA genome

**Pathophysiology:**
1. **Entry**: Binds to CD4 receptors and co-receptors (CCR5, CXCR4)
2. **Integration**: Reverse transcriptase converts RNA to DNA, integrates into host genome
3. **Replication**: Uses host cell machinery to produce new virus particles
4. **Immune Destruction**: Progressive loss of CD4 cells leading to immunodeficiency

**Natural History:**
- **Acute Infection**: 2-4 weeks post-exposure, flu-like symptoms
- **Clinical Latency**: Asymptomatic phase, 8-10 years without treatment
- **Symptomatic HIV**: Opportunistic infections appear
- **AIDS**: CD4 <200 or specific opportunistic diseases

**Global Impact**: Affected 38.4 million people worldwide in 2023"""

    def _get_ahd_definition(self):
        return """**Advanced HIV Disease (AHD) - Comprehensive Definition**

**Definition:** AHD refers to advanced stage of HIV infection characterized by severe immunodeficiency, defined as:
- CD4 cell count <200 cells/mm³ **OR**
- WHO Clinical Stage 3 or 4 disease **regardless of CD4 count**

**Epidemiology:**
- 15-20% of people starting ART present with AHD
- Higher mortality in first 3 months of treatment
- More common in late presenters and re-starters

**Clinical Features:**
- **Severe weight loss** (>10% body weight)
- **Chronic symptoms**: fever, diarrhea >1 month
- **Neurological manifestations**: cryptococcal meningitis, toxoplasmosis
- **Opportunistic infections**: TB, PJP, esophageal candidiasis
- **Malignancies**: Kaposi sarcoma, lymphoma

**Management Principles:**
1. **Rapid ART initiation** (within 7 days, same day if possible)
2. **Comprehensive OI package**: screening, prevention, treatment
3. **Enhanced adherence support**
4. **Close clinical monitoring** (2-4 week intervals)"""

    def _get_cd4_definition(self):
        return """**CD4 Cells - Comprehensive Explanation**

**Definition:** CD4 cells (cluster of differentiation 4) are T-helper lymphocytes that play a central role in adaptive immune responses.

**Normal Physiology:**
- **Normal range**: 500-1500 cells/mm³
- **Function**: Coordinate immune response, activate B cells and cytotoxic T cells
- **Lifecycle**: Produced in bone marrow, mature in thymus

**HIV Impact:**
- **Direct killing**: HIV replication destroys CD4 cells
- **Chronic activation**: Leads to exhaustion and apoptosis
- **Thymic impairment**: Reduced production of new cells

**Clinical Interpretation:**
- **>500**: Normal immune function
- **200-500**: Mild to moderate immunodeficiency
- **<200**: Severe immunodeficiency (AHD criteria)
- **<100**: Critical risk for opportunistic infections

**Monitoring Guidelines:**
- Baseline at diagnosis
- 3 months post-ART initiation
- Every 6-12 months when stable
- More frequent if unsuppressed viral load"""

    def _get_viral_load_definition(self):
        return """**Viral Load - Comprehensive Explanation**

**Definition:** Viral load measures the amount of HIV RNA in blood plasma, expressed as copies per milliliter (copies/mL).

**Clinical Significance:**
- **Treatment efficacy**: Primary marker of ART success
- **Transmission risk**: Higher VL increases transmission probability
- **Disease progression**: Predicts CD4 decline and clinical outcomes

**Interpretation:**
- **Suppressed**: <1000 copies/mL (treatment goal)
- **Undetectable**: <50 copies/mL (optimal suppression)
- **Unsuppressed**: ≥1000 copies/mL (needs intervention)
- **Virological failure**: >1000 copies/mL after 6 months ART

**Monitoring Schedule:**
- Baseline before ART
- 3 months post-ART initiation
- Every 6 months if suppressed
- Every 3 months if unsuppressed

**Public Health Impact:**
- **U=U**: Undetectable = Untransmittable
- **Treatment as prevention**: ART reduces transmission by 96%"""

    def _get_art_definition(self):
        return """**ART (Antiretroviral Therapy) - Comprehensive Overview**

**Definition:** ART refers to the combination of antiretroviral drugs used to treat HIV infection.

**Classes of ARVs:**
1. **NRTIs**: Nucleoside Reverse Transcriptase Inhibitors (TDF, 3TC, ABC)
2. **NNRTIs**: Non-Nucleoside Reverse Transcriptase Inhibitors (EFV, NVP)
3. **PIs**: Protease Inhibitors (LPV/r, ATV/r)
4. **INSTIs**: Integrase Strand Transfer Inhibitors (DTG, RAL)
5. **Entry Inhibitors**: Block viral entry (Maraviroc)
6. **Post-attachment Inhibitors** (Ibalizumab)

**Treatment Principles:**
- **Combination therapy**: Minimum 3 drugs from ≥2 classes
- **Adherence**: >95% adherence required for success
- **Lifelong treatment**: No current cure, but controllable
- **Rapid initiation**: Start ASAP after diagnosis

**Goals of Therapy:**
- Virological: Achieve and maintain viral suppression
- Immunological: CD4 recovery >500 cells/mm³
- Clinical: Prevent disease progression and OIs
- Quality of life: Normal life expectancy and function"""

    def _get_prevention_info(self):
        return """**HIV Prevention - Comprehensive Strategies**

**Biomedical Interventions:**
- **PrEP** (Pre-Exposure Prophylaxis): Daily TDF/FTC for HIV-negative at-risk individuals
- **PEP** (Post-Exposure Prophylaxis): 28-day ART course started within 72 hours of exposure
- **VMMC** (Voluntary Medical Male Circumcision): 60% reduction in female-to-male transmission
- **Treatment as Prevention**: ART reduces transmission risk by 96%

**Behavioral Interventions:**
- **Condom use**: Male and female condoms
- **Harm reduction**: Needle exchange programs for PWID
- **Partner reduction**: Limiting number of sexual partners
- **Testing and counseling**: Regular HIV testing

**Structural Interventions:**
- **PMTCT** (Prevention of Mother-to-Child Transmission): ART during pregnancy/breastfeeding
- **Blood safety**: Screening all blood products
- **Stigma reduction**: Community education and anti-discrimination laws
- **Legal frameworks**: Protecting rights of PLHIV

**Effectiveness:**
- Combined approaches provide >90% protection
- Tailored to individual risk factors and context"""

    def _get_transmission_info(self):
        return """**HIV Transmission - Comprehensive Guide**

**Established Routes:**
1. **Sexual Transmission** (75-85% of cases)
   - Unprotected anal/vaginal sex
   - Higher risk: receptive anal > insertive anal > vaginal
   - Factors: STIs, viral load, mucosal integrity

2. **Blood-borne Transmission**
   - Contaminated needles (PWID, healthcare)
   - Blood transfusions (rare with screening)
   - Organ transplantation

3. **Perinatal Transmission**
   - During pregnancy, delivery, or breastfeeding
   - Risk: 15-45% without intervention, <5% with ART

**Non-Transmission Routes:**
- **Casual contact**: Hugging, shaking hands
- **Airborne**: Coughing, sneezing
- **Saliva, tears, sweat**
- **Toilets, swimming pools**
- **Insect bites**

**Risk Reduction:**
- **ART**: Viral suppression eliminates sexual transmission
- **Condoms**: 80-95% reduction in transmission
- **PrEP**: >90% reduction when adherent
- **Medical procedures**: Sterile equipment, universal precautions"""

    def _get_symptoms_info(self):
        return """**HIV Symptoms - Comprehensive Overview**

**Acute HIV Infection (2-4 weeks post-exposure):**
- Fever, chills, rash
- Sore throat, mouth ulcers
- Fatigue, muscle aches
- Swollen lymph nodes
- Night sweats
- *50-90% experience symptoms*

**Clinical Latency Stage (Asymptomatic):**
- May last 8-10 years without treatment
- Progressive immune decline continues
- Some may have persistent generalized lymphadenopathy

**Symptomatic HIV (Moderate Immunodeficiency):**
- Recurrent respiratory infections
- Herpes zoster (shingles)
- Oral candidiasis (thrush)
- Chronic diarrhea
- Unexplained weight loss (<10%)
- Fever >1 month

**AIDS-Defining Conditions (Severe Immunodeficiency):**
- Opportunistic infections (PJP, toxoplasmosis, cryptococcosis)
- HIV wasting syndrome (>10% weight loss)
- Kaposi sarcoma, lymphomas
- HIV encephalopathy
- Extrapulmonary TB

**Important Notes:**
- Symptoms vary greatly between individuals
- Many people asymptomatic for years
- Testing is only way to know HIV status
- Early diagnosis improves outcomes"""

    def _get_testing_info(self):
        return """**HIV Testing - Comprehensive Guide**

**Testing Technologies:**
1. **Rapid Tests** (Point-of-care)
   - Results in 20 minutes
   - Fingerstick or oral fluid
   - >99% sensitivity/specificity

2. **4th Generation ELISA**
   - Detects both p24 antigen and antibodies
   - Window period: 2-3 weeks
   - Gold standard for diagnosis

3. **PCR/Viral Load**
   - Detects viral RNA
   - Window period: 10-14 days
   - Used for early infant diagnosis

**Testing Algorithms:**
- **Initial test**: Rapid or 4th gen ELISA
- **Confirmatory**: Different platform or Western blot
- **Differentiation**: HIV-1 vs HIV-2

**Testing Recommendations:**
- **Universal**: All adults/adolescents at least once
- **High-risk**: Every 3-6 months
- **Pregnant women**: Every pregnancy
- **Partners**: Before new sexual relationships

**Window Period Considerations:**
- **Antibody tests**: 3-12 weeks
- **4th gen tests**: 2-3 weeks  
- **PCR tests**: 10-14 days
- Retest after potential exposure window"""

    def _get_oi_info(self):
        return """**Opportunistic Infections - Comprehensive Guide**

**Common OIs in AHD:**

1. **Tuberculosis (TB)**
   - Most common OI globally
   - Screening: Symptom screen at every visit
   - Diagnosis: GeneXpert, culture
   - Treatment: RHZE regimen, early ART

2. **Cryptococcal Meningitis**
   - CD4 <100, high mortality
   - Screening: CrAg in blood if CD4 <100
   - Treatment: Amphotericin B + flucytosine

3. **Pneumocystis jirovecii Pneumonia (PJP)**
   - CD4 <200, subacute respiratory symptoms
   - Prophylaxis: Cotrimoxazole if CD4 <200
   - Treatment: High-dose cotrimoxazole

4. **Toxoplasmosis**
   - CD4 <100, CNS manifestations
   - Prophylaxis: Cotrimoxazole
   - Treatment: Pyrimethamine + sulfadiazine

5. **Esophageal Candidiasis**
   - Painful swallowing, oral thrush
   - Treatment: Fluconazole

**Prevention Strategy:**
- **Cotrimoxazole**: CD4 <200 or WHO stage 3/4
- **Fluconazole**: CD4 <100 in endemic areas
- **IPT**: TB preventive therapy
- **Vaccinations**: PCV, HPV, influenza"""

    def _get_who_staging(self):
        return """**WHO Clinical Staging System - Comprehensive**

**Stage 1:**
- Asymptomatic
- Persistent generalized lymphadenopathy

**Stage 2:**
- Moderate unexplained weight loss (<10%)
- Recurrent respiratory infections
- Herpes zoster (within 5 years)
- Angular cheilitis
- Recurrent oral ulceration
- Papular pruritic eruptions
- Seborrheic dermatitis
- Fungal nail infections

**Stage 3:**
- Unexplained severe weight loss (>10%)
- Unexplained chronic diarrhea (>1 month)
- Unexplained persistent fever (>1 month)
- Persistent oral candidiasis
- Oral hairy leukoplakia
- Pulmonary tuberculosis (current)
- Severe bacterial infections
- Acute necrotizing ulcerative stomatitis/gingivitis/periodontitis

**Stage 4 (AIDS-Defining):**
- HIV wasting syndrome
- Pneumocystis pneumonia
- Recurrent severe bacterial pneumonia
- Chronic herpes simplex infection
- Esophageal candidiasis
- Extrapulmonary tuberculosis
- Kaposi sarcoma
- CMV infection
- CNS toxoplasmosis
- HIV encephalopathy
- Cryptococcosis
- Disseminated non-tuberculous mycobacteria

**Clinical Utility:**
- Guides OI prophylaxis needs
- Determines ART urgency
- Predicts disease progression
- Resource-limited settings where CD4 unavailable"""

    def _get_greeting(self):
        return """👋 **Hello! I'm your Comprehensive HIV/AIDS Expert Assistant**

I can help you with **any topic related to HIV/AIDS**, including:

🔬 **Basic Science & Definitions**
• HIV virology and pathophysiology  
• CD4 cells and immune function
• Viral load dynamics
• Disease progression

💊 **Treatment & Medications**
• ART regimens (1st, 2nd, 3rd line)
• Drug classes and mechanisms
• Side effect management
• Adherence strategies

🛡️ **Prevention & Testing**
• Transmission routes and prevention
• PrEP, PEP, and condom use
• Testing technologies and algorithms
• Risk reduction counseling

📊 **Epidemiology & Statistics**
• Global, African, and Kenyan statistics
• Prevalence and incidence rates
• Demographic patterns
• Progress toward 95-95-95 targets

🏥 **Clinical Management**
• WHO staging system
• Opportunistic infections
• AHD diagnosis and management
• Comorbidities and coinfections

📈 **Data Interpretation**
• CD4 and viral load results
• Risk prediction explanations
• Clinical decision support

**What would you like to know about today?**"""

    def _get_comprehensive_response(self, user_input):
        return f"""🤔 **I want to provide you with the most accurate information about:** "{user_input}"

I specialize in **comprehensive HIV/AIDS knowledge** including:

• **Basic virology and immunology**
• **Treatment guidelines and ART regimens** 
• **Prevention strategies (PrEP, PEP, condoms)**
• **Testing and diagnosis**
• **WHO clinical staging**
• **Opportunistic infections**
• **Mother-to-child transmission**
• **Epidemiology and statistics**
• **Clinical management and guidelines**

**Could you please rephrase your question or ask about one of these specific HIV/AIDS topics?**

💡 **Try asking:**
• "What are the current first-line ART regimens?"
• "How does HIV transmission occur?"
• "What statistics are available for Kenya?"
• "Explain WHO staging system"
• "What is the difference between HIV and AIDS?"
• "How effective is PrEP for prevention?""""

# Initialize chatbot
chatbot = HIVExpertChatbot()

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Analytics", "💬 HIV Expert Chatbot"])

# -------------------------------
# TAB 1: Dashboard (Prediction) - ENHANCED
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

            # Display prediction results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("AHD Risk", "High Risk" if int(pred) == 1 else "Low Risk", 
                         delta="Clinical review needed" if int(pred) == 1 else "Routine care")
            with col2:
                st.metric("Risk Probability", f"{proba:.1%}")

            # Progress bar with interpretation
            st.progress(proba)
            if proba > 0.75:
                st.error("🔴 **High Risk** – Immediate clinical review recommended")
            elif proba > 0.45:
                st.warning("🟡 **Moderate Risk** – Close monitoring advised")
            else:
                st.success("🟢 **Low Risk** – Continue standard care")

            # Prediction interpretation
            st.subheader("🎯 Clinical Interpretation")
            interpretation = chatbot.interpret_prediction(pred, proba, input_data_dict)
            st.markdown(interpretation)

            # Enhanced features table
            st.subheader("📋 Input Features Analysis")
            
            # Create a more informative features table
            features_df = pd.DataFrame({
                'Feature': list(input_data_dict.keys()),
                'Value': list(input_data_dict.values()),
                'Clinical Significance': [
                    'Demographic factor' if 'Age' in k or 'Sex' in k else
                    'Nutritional indicator' if 'Weight' in k or 'Height' in k or 'BMI' in k else
                    'Critical immunologic marker' if 'CD4' in k else
                    'Virologic marker' if 'VL' in k else
                    'Treatment adherence indicator' if 'Months' in k else
                    'Disease severity indicator' if 'WHO' in k else
                    'Risk stratification' if 'risk' in k else
                    'Data quality indicator' if 'Missing' in k else 'Other'
                    for k in input_data_dict.keys()
                ]
            })
            
            # Display features in an expandable table with better formatting
            with st.expander("📊 Detailed Feature Analysis", expanded=True):
                st.dataframe(features_df, use_container_width=True, hide_index=True)
                
                # Add feature insights
                st.markdown("**🔍 Key Feature Insights:**")
                if cd4 < 200:
                    st.markdown(f"- **CD4 {cd4}**: Below AHD threshold (<200 cells/mm³)")
                if vl > 1000:
                    st.markdown(f"- **Viral Load {vl:,}**: Unsuppressed (≥1000 copies/mL)")
                if bmi < 18.5:
                    st.markdown(f"- **BMI {bmi:.1f}**: Underweight, consider nutritional support")
                if who_stage in [3, 4]:
                    st.markdown(f"- **WHO Stage {who_stage}**: Advanced disease presentation")

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
# TAB 3: COMPREHENSIVE HIV EXPERT CHATBOT
# -------------------------------
with tab3:
    st.subheader("💬 Comprehensive HIV/AIDS Expert Chatbot")
    st.info("🔬 **Ask me anything about HIV/AIDS** - Science, Treatment, Prevention, Statistics, Guidelines, Clinical Management")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": chatbot._get_greeting()}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask any HIV/AIDS question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔍 Researching comprehensive information...")
            
            # Get response from expert chatbot
            response = chatbot.get_response(prompt)
            message_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick action buttons
    st.markdown("### 🚀 Quick Access Topics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Kenya Statistics"):
            st.session_state.messages.append({"role": "user", "content": "HIV statistics Kenya 2025"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot.get_statistics("kenya")})
            st.rerun()
        
        if st.button("💊 ART Regimens"):
            st.session_state.messages.append({"role": "user", "content": "first line ART regimens"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot.get_treatment_info("first_line")})
            st.rerun()
    
    with col2:
        if st.button("🛡️ Prevention"):
            st.session_state.messages.append({"role": "user", "content": "HIV prevention methods"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot._get_prevention_info()})
            st.rerun()
        
        if st.button("🔬 HIV Definition"):
            st.session_state.messages.append({"role": "user", "content": "What is HIV?"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot._get_hiv_definition()})
            st.rerun()
    
    with col3:
        if st.button("🏥 WHO Staging"):
            st.session_state.messages.append({"role": "user", "content": "WHO clinical staging"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot._get_who_staging()})
            st.rerun()
        
        if st.button("🌍 Global Stats"):
            st.session_state.messages.append({"role": "user", "content": "global HIV statistics"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot.get_statistics("global")})
            st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": chatbot._get_greeting()}
        ]
        st.rerun()

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>© 2025 | Built with ❤️ by <b>Idah Anyango</b></div>", unsafe_allow_html=True)
