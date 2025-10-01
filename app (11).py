import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AHD Copilot", layout="wide", page_icon="🎗️")

st.title("🎗️ Advanced HIV Disease (AHD) Copilot")
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
# COMPREHENSIVE HIV/AIDS EXPERT CHATBOT CLASS
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
        
        interpretation += "**Key Contributing Factors:**\n"
        
        cd4 = features.get('Latest CD4 Result', 0)
        if cd4 < 200:
            interpretation += f"• **Critical CD4**: {cd4} cells/mm³ (AHD threshold <200)\n"
        elif cd4 < 350:
            interpretation += f"• **Low CD4**: {cd4} cells/mm³ (needs close monitoring)\n"
        
        vl = features.get('Last VL Result', 0)
        vl_suppressed = features.get('VL_Suppressed', 0)
        if not vl_suppressed and vl > 0:
            interpretation += f"• **Unsuppressed VL**: {vl:,} copies/mL\n"
        
        who_stage_3 = features.get('Last_WHO_Stage_3', 0)
        who_stage_4 = features.get('Last_WHO_Stage_4', 0)
        if who_stage_4:
            interpretation += f"• **WHO Stage 4**: Severe symptoms present\n"
        elif who_stage_3:
            interpretation += f"• **WHO Stage 3**: Advanced symptoms present\n"
        
        bmi = features.get('BMI', 0)
        if bmi < 18.5:
            interpretation += f"• **Low BMI**: {bmi:.1f} (underweight)\n"
        elif bmi > 30:
            interpretation += f"• **High BMI**: {bmi:.1f} (obese)\n"
        
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
        
        if any(word in user_input for word in ["statistic", "prevalence", "rate", "number", "data"]):
            if "kenya" in user_input:
                return self.get_statistics("kenya")
            elif "africa" in user_input:
                return self.get_statistics("africa")
            elif "global" in user_input or "world" in user_input:
                return self.get_statistics("global")
            else:
                return self.get_statistics("global")

        elif any(word in user_input for word in ["treatment", "regimen", "art", "medication", "first-line", "second-line", "third-line"]):
            if "second" in user_input:
                return self.get_treatment_info("second_line")
            elif "third" in user_input:
                return self.get_treatment_info("third_line")
            else:
                return self.get_treatment_info("first_line")

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

        elif any(word in user_input for word in ["prevent", "prevention", "prep", "pep", "condom"]):
            return self._get_prevention_info()

        elif any(word in user_input for word in ["transmit", "transmission", "spread", "catch"]):
            return self._get_transmission_info()

        elif any(word in user_input for word in ["symptom", "sign", "feel", "experience"]):
            return self._get_symptoms_info()

        elif any(word in user_input for word in ["test", "testing", "diagnose", "result"]):
            return self._get_testing_info()

        elif any(word in user_input for word in ["oi", "opportunistic", "infection", "tb", "cryptococcus"]):
            return self._get_oi_info()

        elif "who stage" in user_input:
            return self._get_who_staging()

        elif any(word in user_input for word in ["hello", "hi", "hey", "greetings"]):
            return self._get_greeting()

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

**HIV Impact:**
- **Direct killing**: HIV replication destroys CD4 cells
- **Chronic activation**: Leads to exhaustion and apoptosis

**Clinical Interpretation:**
- **>500**: Normal immune function
- **200-500**: Mild to moderate immunodeficiency
- **<200**: Severe immunodeficiency (AHD criteria)
- **<100**: Critical risk for opportunistic infections"""

    def _get_viral_load_definition(self):
        return """**Viral Load - Comprehensive Explanation**

**Definition:** Viral load measures the amount of HIV RNA in blood plasma, expressed as copies per milliliter (copies/mL).

**Clinical Significance:**
- **Treatment efficacy**: Primary marker of ART success
- **Transmission risk**: Higher VL increases transmission probability

**Interpretation:**
- **Suppressed**: <1000 copies/mL (treatment goal)
- **Undetectable**: <50 copies/mL (optimal suppression)
- **Unsuppressed**: ≥1000 copies/mL (needs intervention)

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

**Treatment Principles:**
- **Combination therapy**: Minimum 3 drugs from ≥2 classes
- **Adherence**: >95% adherence required for success
- **Lifelong treatment**: No current cure, but controllable
- **Rapid initiation**: Start ASAP after diagnosis"""

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
- **Testing and counseling**: Regular HIV testing"""

    def _get_transmission_info(self):
        return """**HIV Transmission - Comprehensive Guide**

**Established Routes:**
1. **Sexual Transmission** (75-85% of cases)
   - Unprotected anal/vaginal sex
   - Higher risk: receptive anal > insertive anal > vaginal

2. **Blood-borne Transmission**
   - Contaminated needles (PWID, healthcare)
   - Blood transfusions (rare with screening)

3. **Perinatal Transmission**
   - During pregnancy, delivery, or breastfeeding
   - Risk: 15-45% without intervention, <5% with ART

**Risk Reduction:**
- **ART**: Viral suppression eliminates sexual transmission
- **Condoms**: 80-95% reduction in transmission
- **PrEP**: >90% reduction when adherent"""

    def _get_symptoms_info(self):
        return """**HIV Symptoms - Comprehensive Overview**

**Acute HIV Infection (2-4 weeks post-exposure):**
- Fever, chills, rash
- Sore throat, mouth ulcers
- Fatigue, muscle aches
- Swollen lymph nodes
- Night sweats

**Clinical Latency Stage (Asymptomatic):**
- May last 8-10 years without treatment
- Progressive immune decline continues

**Symptomatic HIV (Moderate Immunodeficiency):**
- Recurrent respiratory infections
- Herpes zoster (shingles)
- Oral candidiasis (thrush)
- Chronic diarrhea
- Unexplained weight loss

**AIDS-Defining Conditions (Severe Immunodeficiency):**
- Opportunistic infections (PJP, toxoplasmosis, cryptococcosis)
- HIV wasting syndrome (>10% weight loss)
- Kaposi sarcoma, lymphomas"""

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

**Testing Recommendations:**
- **Universal**: All adults/adolescents at least once
- **High-risk**: Every 3-6 months
- **Pregnant women**: Every pregnancy"""

    def _get_oi_info(self):
        return """**Opportunistic Infections - Comprehensive Guide**

**Common OIs in AHD:**

1. **Tuberculosis (TB)**
   - Most common OI globally
   - Screening: Symptom screen at every visit

2. **Cryptococcal Meningitis**
   - CD4 <100, high mortality
   - Screening: CrAg in blood if CD4 <100

3. **Pneumocystis jirovecii Pneumonia (PJP)**
   - CD4 <200, subacute respiratory symptoms
   - Prophylaxis: Cotrimoxazole if CD4 <200

**Prevention Strategy:**
- **Cotrimoxazole**: CD4 <200 or WHO stage 3/4
- **Fluconazole**: CD4 <100 in endemic areas
- **IPT**: TB preventive therapy"""

    def _get_who_staging(self):
        return """**WHO Clinical Staging System - Comprehensive**

**Stage 1:** Asymptomatic, Persistent generalized lymphadenopathy

**Stage 2:** Moderate unexplained weight loss, Recurrent respiratory infections, Herpes zoster

**Stage 3:** Unexplained severe weight loss, Unexplained chronic diarrhea, Pulmonary tuberculosis

**Stage 4 (AIDS-Defining):** HIV wasting syndrome, Pneumocystis pneumonia, Extrapulmonary tuberculosis, Kaposi sarcoma

**Clinical Utility:**
- Guides OI prophylaxis needs
- Determines ART urgency
- Predicts disease progression"""

    def _get_greeting(self):
        return """👋 **Hello! I'm your Comprehensive HIV/AIDS Expert Assistant**

I can help you with **any topic related to HIV/AIDS**, including:

🔬 **Basic Science & Definitions**
• HIV virology and pathophysiology  
• CD4 cells and immune function
• Viral load dynamics

💊 **Treatment & Medications**
• ART regimens (1st, 2nd, 3rd line)
• Drug classes and mechanisms
• Side effect management

🛡️ **Prevention & Testing**
• Transmission routes and prevention
• PrEP, PEP, and condom use
• Testing technologies

📊 **Epidemiology & Statistics**
• Global, African, and Kenyan statistics
• Prevalence and incidence rates

🏥 **Clinical Management**
• WHO staging system
• Opportunistic infections
• AHD diagnosis and management

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
• **Epidemiology and statistics**

**Could you please rephrase your question or ask about one of these specific HIV/AIDS topics?**

💡 **Try asking:**
• "What are the current first-line ART regimens?"
• "How does HIV transmission occur?"
• "What statistics are available for Kenya?"
• "Explain WHO staging system"
• "What is the difference between HIV and AIDS?"
• "How effective is PrEP for prevention?" """

# -------------------------------
# ENHANCED ANALYTICS DASHBOARD CLASS
# -------------------------------
class ClinicAnalytics:
    def __init__(self):
        self.who_targets = {
            'viral_suppression': 90,
            'art_coverage': 95,
            'patient_retention': 90,
            'ahd_prevention': 85
        }
    
    def generate_sample_data(self, clinic_type="urban"):
        """Generate realistic synthetic clinic data"""
        np.random.seed(42)
        n_patients = 300
        
        if clinic_type == "urban":
            base_cd4 = 350
            suppression_rate = 0.85
            retention_rate = 0.88
            late_presenters = 0.12
        else:
            base_cd4 = 280
            suppression_rate = 0.62
            retention_rate = 0.72
            late_presenters = 0.25
        
        data = []
        for i in range(n_patients):
            age = np.random.normal(38, 12)
            age = max(18, min(80, age))
            
            cd4 = np.random.gamma(3, base_cd4/3)
            cd4 = max(50, min(1200, cd4))
            
            if np.random.random() < suppression_rate:
                viral_load = np.random.lognormal(2.5, 0.8)
            else:
                viral_load = np.random.lognormal(8, 1.5)
            viral_load = int(max(20, viral_load))
            
            if cd4 < 200:
                who_stage = np.random.choice([3, 4], p=[0.6, 0.4])
            elif cd4 < 350:
                who_stage = np.random.choice([2, 3], p=[0.7, 0.3])
            else:
                who_stage = np.random.choice([1, 2], p=[0.8, 0.2])
            
            if np.random.random() < late_presenters:
                months_art = np.random.randint(1, 6)
            else:
                months_art = np.random.randint(6, 60)
            
            missed_visits = np.random.poisson(0.3 if np.random.random() < retention_rate else 2.5)
            
            patient = {
                'Patient_ID': f'PAT{1000 + i}',
                'Age': int(age),
                'Gender': np.random.choice(['Male', 'Female'], p=[0.45, 0.55]),
                'CD4_Count': int(cd4),
                'Viral_Load': viral_load,
                'WHO_Stage': who_stage,
                'Months_on_ART': months_art,
                'Last_Visit_Date': (datetime.now() - timedelta(days=np.random.randint(0, 90))).strftime('%Y-%m-%d'),
                'ART_Regimen': np.random.choice(['TDF/3TC/DTG', 'TAF/FTC/DTG', 'AZT/3TC/EFV'], p=[0.6, 0.3, 0.1]),
                'Clinic_Location': 'Urban' if clinic_type == 'urban' else 'Rural',
                'Missed_Visits': missed_visits
            }
            data.append(patient)
        
        return pd.DataFrame(data)
    
    def analyze_clinic_data(self, df):
        """Comprehensive analysis of clinic data"""
        analysis = {}
        
        analysis['total_patients'] = len(df)
        analysis['avg_age'] = df['Age'].mean()
        analysis['gender_distribution'] = df['Gender'].value_counts(normalize=True)
        analysis['ahd_cases'] = (df['CD4_Count'] < 200).mean() * 100
        analysis['avg_cd4'] = df['CD4_Count'].mean()
        analysis['viral_suppression'] = (df['Viral_Load'] < 1000).mean() * 100
        analysis['undetectable'] = (df['Viral_Load'] < 50).mean() * 100
        analysis['who_stage_dist'] = df['WHO_Stage'].value_counts(normalize=True).sort_index()
        analysis['new_patients'] = (df['Months_on_ART'] < 6).mean() * 100
        analysis['experienced_patients'] = (df['Months_on_ART'] >= 12).mean() * 100
        analysis['avg_art_duration'] = df['Months_on_ART'].mean()
        analysis['good_retention'] = (df['Missed_Visits'] <= 1).mean() * 100
        analysis['poor_retention'] = (df['Missed_Visits'] > 2).mean() * 100
        
        return analysis
    
    def generate_insights(self, analysis, df):
        """Generate smart insights from analysis"""
        insights = []
        
        if analysis['ahd_cases'] > 20:
            insights.append({
                'type': '🚨 CRITICAL',
                'title': 'High AHD Prevalence',
                'message': f"{analysis['ahd_cases']:.1f}% of patients have Advanced HIV Disease",
                'recommendation': 'Implement same-day ART initiation and enhance community testing'
            })
        elif analysis['ahd_cases'] > 10:
            insights.append({
                'type': '⚠️ WARNING', 
                'title': 'Moderate AHD Cases',
                'message': f"{analysis['ahd_cases']:.1f}% AHD prevalence needs monitoring",
                'recommendation': 'Strengthen early detection and rapid linkage to care'
            })
        
        suppression_gap = self.who_targets['viral_suppression'] - analysis['viral_suppression']
        if suppression_gap > 20:
            insights.append({
                'type': '🚨 CRITICAL',
                'title': 'Low Viral Suppression',
                'message': f"Only {analysis['viral_suppression']:.1f}% suppression ({suppression_gap:.1f}% below target)",
                'recommendation': 'Enhanced adherence counseling and regimen review'
            })
        elif suppression_gap > 10:
            insights.append({
                'type': '⚠️ WARNING',
                'title': 'Suppression Below Target',
                'message': f"{analysis['viral_suppression']:.1f}% suppression needs improvement",
                'recommendation': 'Focus on patients with unsuppressed viral load'
            })
        
        if analysis['poor_retention'] > 20:
            insights.append({
                'type': '⚠️ WARNING',
                'title': 'Patient Retention Issues',
                'message': f"{analysis['poor_retention']:.1f}% of patients missing multiple visits",
                'recommendation': 'Implement appointment reminders and community follow-up'
            })
        
        young_patients = df[df['Age'] < 25]
        if len(young_patients) > 0:
            young_suppression = (young_patients['Viral_Load'] < 1000).mean() * 100
            if young_suppression < 70:
                insights.append({
                    'type': '🎯 TARGETED',
                    'title': 'Youth Engagement Challenge',
                    'message': f"Young patients (18-25) have only {young_suppression:.1f}% suppression",
                    'recommendation': 'Develop youth-friendly services and peer support programs'
                })
        
        if analysis['avg_cd4'] < 300:
            insights.append({
                'type': '📊 MONITOR',
                'title': 'CD4 Recovery Needs Attention',
                'message': f"Average CD4 count is {analysis['avg_cd4']:.0f} cells/mm³",
                'recommendation': 'Review patients with slow immune recovery'
            })
        
        return insights

# -------------------------------
# INITIALIZE COMPONENTS
# -------------------------------
chatbot = HIVExpertChatbot()
analytics_engine = ClinicAnalytics()

# -------------------------------
# CREATE TABS - THIS MUST COME AFTER ALL CLASS DEFINITIONS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Analytics", "💬 HIV Expert Chatbot"])

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

        if st.sidebar.button("🔍 Predict AHD Risk", type="primary"):
            pred = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0][1]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("AHD Risk", "High Risk" if int(pred) == 1 else "Low Risk", 
                         delta="Clinical review needed" if int(pred) == 1 else "Routine care")
            with col2:
                st.metric("Risk Probability", f"{proba:.1%}")

            st.progress(proba)
            if proba > 0.75:
                st.error("🔴 **High Risk** – Immediate clinical review recommended")
            elif proba > 0.45:
                st.warning("🟡 **Moderate Risk** – Close monitoring advised")
            else:
                st.success("🟢 **Low Risk** – Continue standard care")

            st.subheader("🎯 Clinical Interpretation")
            interpretation = chatbot.interpret_prediction(pred, proba, input_data_dict)
            st.markdown(interpretation)

            st.subheader("📋 Input Features Analysis")
            
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
            
            with st.expander("📊 Detailed Feature Analysis", expanded=True):
                st.dataframe(features_df, use_container_width=True, hide_index=True)
                
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
# TAB 2: Enhanced Analytics Dashboard
# -------------------------------
with tab2:
    st.subheader("🏥 Clinic Performance Analytics Dashboard")
    
    st.info("📊 **Understand Your Clinic's Performance** - Upload data or use sample data to discover insights and get actionable recommendations.")
    
    # File upload section
    st.markdown("### 📁 Data Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file with patient data", 
            type=['csv'],
            help="File should contain: Patient_ID, Age, Gender, CD4_Count, Viral_Load, WHO_Stage, Months_on_ART, etc."
        )
    
    with col2:
        st.markdown("### 🚀 Quick Demo")
        demo_option = st.selectbox(
            "Try sample data:",
            ["Select sample...", "Urban Excellence Clinic", "Rural Challenge Clinic"],
            label_visibility="collapsed"
        )
    
    # Data management
    current_data = None
    data_source = None
    
    if uploaded_file is not None:
        current_data = pd.read_csv(uploaded_file)
        data_source = "Uploaded File"
        st.success(f"✅ Successfully loaded {len(current_data)} patient records")
        
    elif demo_option != "Select sample...":
        clinic_type = "urban" if "Urban" in demo_option else "rural"
        current_data = analytics_engine.generate_sample_data(clinic_type)
        data_source = demo_option
        st.success(f"✅ Generated {len(current_data)} sample patient records from {demo_option}")
    
    # Main dashboard content
    if current_data is not None:
        st.markdown("---")
        
        with st.spinner("🔍 Analyzing clinic data..."):
            analysis = analytics_engine.analyze_clinic_data(current_data)
            insights = analytics_engine.generate_insights(analysis, current_data)
        
        # Key Metrics Dashboard
        st.markdown("### 📈 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_suppression = analysis['viral_suppression'] - analytics_engine.who_targets['viral_suppression']
            st.metric(
                "Viral Suppression", 
                f"{analysis['viral_suppression']:.1f}%",
                f"{delta_suppression:+.1f}%",
                delta_color="inverse" if delta_suppression < 0 else "normal"
            )
            st.caption(f"WHO Target: {analytics_engine.who_targets['viral_suppression']}%")
        
        with col2:
            st.metric(
                "AHD Prevalence", 
                f"{analysis['ahd_cases']:.1f}%",
                "Lower is better",
                delta_color="inverse"
            )
            st.caption("CD4 <200 cells/mm³")
        
        with col3:
            st.metric(
                "Avg CD4 Count", 
                f"{analysis['avg_cd4']:.0f}",
                "cells/mm³"
            )
            st.caption("Immune recovery")
        
        with col4:
            st.metric(
                "Patient Retention", 
                f"{analysis['good_retention']:.1f}%",
                "Regular attendees"
            )
            st.caption("Missed ≤1 visit")
        
        # Insights and Recommendations - IMPROVED VISIBILITY
        st.markdown("### 💡 Smart Insights & Recommendations")
        
        if insights:
            for insight in insights:
                # Use different background colors based on severity
                if 'CRITICAL' in insight['type']:
                    bg_color = '#ffebee'  # Light red
                    border_color = '#f44336'
                elif 'WARNING' in insight['type']:
                    bg_color = '#fff3e0'  # Light orange
                    border_color = '#ff9800'
                elif 'TARGETED' in insight['type']:
                    bg_color = '#e8f5e8'  # Light green
                    border_color = '#4caf50'
                else:
                    bg_color = '#e3f2fd'  # Light blue
                    border_color = '#2196f3'
                
                st.markdown(f"""
                <div style='
                    padding: 15px; 
                    border-radius: 10px; 
                    border-left: 5px solid {border_color}; 
                    background-color: {bg_color}; 
                    margin: 10px 0;
                    border: 1px solid {border_color}20;
                '>
                <h4 style='margin: 0 0 10px 0; color: {border_color};'>{insight['type']}: {insight['title']}</h4>
                <p style='margin: 0 0 10px 0; font-weight: bold;'>{insight['message']}</p>
                <p style='margin: 0; font-style: italic;'>💡 <strong>Recommendation:</strong> {insight['recommendation']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 Excellent! Your clinic is meeting or exceeding most performance targets!")
        
        # Rest of analytics content would go here...
        # ... [rest of the analytics tab content] ...

# -------------------------------
# TAB 3: COMPREHENSIVE HIV EXPERT CHATBOT - NOW PROPERLY PLACED
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
    
    # Quick action buttons - IMPROVED VISIBILITY
    st.markdown("### 🚀 Quick Access Topics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Kenya Statistics", use_container_width=True, type="secondary"):
            st.session_state.messages.append({"role": "user", "content": "HIV statistics Kenya 2025"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot.get_statistics("kenya")})
            st.rerun()
        
        if st.button("💊 ART Regimens", use_container_width=True, type="secondary"):
            st.session_state.messages.append({"role": "user", "content": "first line ART regimens"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot.get_treatment_info("first_line")})
            st.rerun()
    
    with col2:
        if st.button("🛡️ Prevention", use_container_width=True, type="secondary"):
            st.session_state.messages.append({"role": "user", "content": "HIV prevention methods"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot._get_prevention_info()})
            st.rerun()
        
        if st.button("🔬 HIV Definition", use_container_width=True, type="secondary"):
            st.session_state.messages.append({"role": "user", "content": "What is HIV?"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot._get_hiv_definition()})
            st.rerun()
    
    with col3:
        if st.button("🏥 WHO Staging", use_container_width=True, type="secondary"):
            st.session_state.messages.append({"role": "user", "content": "WHO clinical staging"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot._get_who_staging()})
            st.rerun()
        
        if st.button("🌍 Global Stats", use_container_width=True, type="secondary"):
            st.session_state.messages.append({"role": "user", "content": "global HIV statistics"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot.get_statistics("global")})
            st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="primary"):
        st.session_state.messages = [
            {"role": "assistant", "content": chatbot._get_greeting()}
        ]
        st.rerun()

# -------------------------------
# Single Footer
# -------------------------------
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>© 2025 | Built with ❤️ by <b>Idah Anyango</b> for Better HIV Care</div>", unsafe_allow_html=True)
