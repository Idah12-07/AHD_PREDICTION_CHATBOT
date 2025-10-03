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

# Custom CSS for red ribbon only
st.markdown("""
<style>
    .red-ribbon {
        background-color: #d32f2f;
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="red-ribbon"><h1>🎗️ Advanced HIV Disease (AHD) Copilot</h1></div>', unsafe_allow_html=True)
st.markdown("""
This tool supports clinicians in **detecting Advanced HIV Disease (AHD)**,  
exploring analytics, and interacting with **comprehensive HIV/AIDS expert chatbot**.  
""")

# -------------------------------
# Load Model - FIXED VERSION
# -------------------------------
try:
    deploy = joblib.load("ahd_model_C_hybrid_fixed.pkl")
    model = deploy['model']
    feature_names = deploy['feature_names']
    model_loaded = True
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"⚠️ Could not load model: {e}")
    st.info("🔧 Using demo mode for predictions")
    model_loaded = False
    model = None
    feature_names = []

# -------------------------------
# ENHANCED COMPREHENSIVE HIV/AIDS EXPERT CHATBOT CLASS
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

        self.ncd_integration = {
            "hypertension": {
                "prevalence_hiv": "35-40%",
                "management": "ACE inhibitors preferred, monitor drug interactions with ART",
                "screening": "Every clinical visit, control target <140/90 mmHg"
            },
            "diabetes": {
                "prevalence_hiv": "10-15%",
                "management": "Metformin first-line, watch for interactions with PIs",
                "screening": "Annual fasting glucose, HbA1c every 6 months"
            },
            "mental_health": {
                "depression_prevalence": "20-30%",
                "management": "SSRIs compatible with ART, avoid St. John's wort",
                "screening": "PHQ-9 at every clinical visit"
            }
        }

        self.myths_misconceptions = {
            "transmission": [
                "HIV CANNOT be transmitted through: kissing, hugging, shaking hands, sharing utensils, toilet seats, mosquitoes",
                "HIV CAN be transmitted through: unprotected sex, sharing needles, mother-to-child during pregnancy/birth/breastfeeding",
                "People on treatment with undetectable viral load CANNOT transmit HIV sexually (U=U)"
            ],
            "treatment": [
                "MYTH: You don't need ART if you feel fine - FACT: HIV damages immune system even without symptoms",
                "MYTH: ART is toxic and will make you sick - FACT: Modern ART has minimal side effects",
                "MYTH: You can stop treatment once viral load is undetectable - FACT: Treatment is lifelong"
            ],
            "prevention": [
                "MYTH: PrEP is only for gay men - FACT: PrEP works for everyone at risk",
                "MYTH: Condoms are 100% effective - FACT: Condoms are highly effective but not 100%",
                "MYTH: You can tell if someone has HIV by looking - FACT: HIV has no specific visible signs"
            ]
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

    def get_ncd_info(self, condition=None):
        """Get information about HIV and NCD comorbidities"""
        if condition and condition.lower() in self.ncd_integration:
            ncd = self.ncd_integration[condition.lower()]
            response = f"**HIV and {condition.title()} Comorbidity Management:**\n\n"
            
            for key, value in ncd.items():
                display_key = key.replace('_', ' ').title()
                response += f"• **{display_key}**: {value}\n"
            
            response += f"\n*Key Considerations:*\n"
            if condition.lower() == "hypertension":
                response += "- Avoid drug interactions between ART and antihypertensives\n"
                response += "- Monitor renal function with TDF-containing regimens\n"
                response += "- Target BP <140/90 mmHg in PLHIV\n"
            elif condition.lower() == "diabetes":
                response += "- PI-based regimens may increase diabetes risk\n"
                response += "- Monitor weight gain with newer INSTIs\n"
                response += "- Screen all PLHIV for diabetes annually\n"
            elif condition.lower() == "mental_health":
                response += "- Depression affects ART adherence significantly\n"
                response += "- Integrated mental health services improve outcomes\n"
                response += "- Screen all patients with PHQ-9 at each visit\n"
            
            return response
        else:
            response = "**HIV and Non-Communicable Diseases (NCDs):**\n\n"
            response += "Common NCDs in PLHIV:\n"
            for condition in self.ncd_integration.keys():
                response += f"• {condition.title()}\n"
            response += "\nAsk about specific conditions for detailed management guidelines."
            return response

    def get_myths_info(self, category=None):
        """Get information about HIV myths and misconceptions"""
        if category and category.lower() in self.myths_misconceptions:
            myths = self.myths_misconceptions[category.lower()]
            response = f"**HIV Myths & Facts - {category.title()}: **\n\n"
            
            for myth in myths:
                response += f"• {myth}\n"
            
            return response
        else:
            response = "**Common HIV Myths and Misconceptions:**\n\n"
            response += "**Categories:**\n"
            for category in self.myths_misconceptions.keys():
                response += f"• {category.title()}\n"
            response += "\nAsk about specific categories for detailed myth-busting information."
            return response

    def get_mental_health_info(self):
        """Comprehensive mental health information for PLHIV"""
        return """**Mental Health and HIV - Comprehensive Guide**

**Common Mental Health Conditions in PLHIV:**
• **Depression**: 20-30% prevalence - screen with PHQ-9
• **Anxiety disorders**: 15-20% prevalence - screen with GAD-7
• **HIV-associated neurocognitive disorders (HAND)**: 15-50%
• **Substance use disorders**: Higher prevalence than general population
• **PTSD**: Common after HIV diagnosis

**Screening Recommendations:**
• **PHQ-9**: At every clinical visit for depression
• **GAD-7**: For anxiety symptoms
• **MMSE**: For cognitive impairment if symptoms present
• **AUDIT**: For alcohol use disorders

**Treatment Approaches:**
• **Integrated care**: Mental health services within HIV clinics
• **Pharmacotherapy**: SSRIs (compatible with ART)
• **Psychotherapy**: CBT, supportive therapy, group therapy
• **Peer support**: Support groups for PLHIV

**Key Considerations:**
• Mental health affects ART adherence and outcomes
• Stigma prevents help-seeking behavior
• Integrated services improve both mental health and HIV outcomes"""

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
            interpretation += "• **Mental health screening** (PHQ-9, anxiety)\n"
            interpretation += "• **NCD screening** (hypertension, diabetes)\n"
        else:
            interpretation += "• **Continue routine ART care**\n"
            interpretation += "• **Standard monitoring schedule**\n"
            interpretation += "• **Prevention counseling**\n"
            interpretation += "• **Regular viral load monitoring**\n"
            interpretation += "• **Annual NCD screening**\n"
            interpretation += "• **Mental health assessment**\n"
        
        return interpretation

    def get_response(self, user_input):
        user_input = user_input.lower().strip()
        
        # Enhanced natural language understanding
        # Statistics queries
        if any(word in user_input for word in ["statistic", "prevalence", "rate", "number", "data", "how many", "cases"]):
            if "kenya" in user_input or "nairobi" in user_input:
                return self.get_statistics("kenya")
            elif "africa" in user_input or "african" in user_input:
                return self.get_statistics("africa")
            elif "global" in user_input or "world" in user_input or "international" in user_input:
                return self.get_statistics("global")
            else:
                return self.get_statistics("global")

        # Treatment queries
        elif any(word in user_input for word in ["treatment", "regimen", "art", "medication", "drug", "first-line", "second-line", "third-line", "arv"]):
            if "second" in user_input:
                return self.get_treatment_info("second_line")
            elif "third" in user_input:
                return self.get_treatment_info("third_line")
            else:
                return self.get_treatment_info("first_line")

        # NCD and comorbidity queries
        elif any(word in user_input for word in ["ncd", "comorbidity", "hypertension", "blood pressure", "diabetes", "sugar", "mental health", "depression", "anxiety", "psych"]):
            if "hypertension" in user_input or "blood pressure" in user_input or "bp" in user_input:
                return self.get_ncd_info("hypertension")
            elif "diabetes" in user_input or "sugar" in user_input:
                return self.get_ncd_info("diabetes")
            elif "mental" in user_input or "depression" in user_input or "anxiety" in user_input or "psych" in user_input:
                return self.get_mental_health_info()
            else:
                return self.get_ncd_info()

        # Myths and misconceptions
        elif any(word in user_input for word in ["myth", "misconception", "false", "wrong", "believe", "think", "rumor", "stigma"]):
            if "transmit" in user_input or "spread" in user_input or "catch" in user_input:
                return self.get_myths_info("transmission")
            elif "treatment" in user_input or "art" in user_input or "med" in user_input:
                return self.get_myths_info("treatment")
            elif "prevent" in user_input or "prevention" in user_input or "condom" in user_input:
                return self.get_myths_info("prevention")
            else:
                return self.get_myths_info()

        # PMTCT queries
        elif any(word in user_input for word in ["pmtct", "pregnant", "pregnancy", "mother", "child", "vertical transmission", "breastfeed", "delivery"]):
            return self._get_pmtct_info()

        # TB-HIV coinfection
        elif any(word in user_input for word in ["tb", "tuberculosis", "coinfection", "lung"]):
            return self._get_tb_hiv_info()

        # Mental health specific
        elif any(word in user_input for word in ["depression", "anxiety", "mental", "psychology", "stress", "trauma"]):
            return self.get_mental_health_info()

        # Basic definitions
        elif any(phrase in user_input for phrase in ["what is hiv", "define hiv", "hiv means", "hiv definition"]):
            return self._get_hiv_definition()
        
        elif any(phrase in user_input for phrase in ["what is ahd", "define ahd", "ahd means", "advanced hiv"]):
            return self._get_ahd_definition()
        
        elif any(phrase in user_input for phrase in ["what is cd4", "define cd4", "cd4 means", "cd4 cells"]):
            return self._get_cd4_definition()
        
        elif any(phrase in user_input for phrase in ["what is viral load", "define viral load", "viral load means", "vl"]):
            return self._get_viral_load_definition()
        
        elif any(phrase in user_input for phrase in ["what is art", "define art", "art means", "antiretroviral"]):
            return self._get_art_definition()

        elif any(word in user_input for word in ["prevent", "prevention", "prep", "pep", "condom", "safe sex"]):
            return self._get_prevention_info()

        elif any(word in user_input for word in ["transmit", "transmission", "spread", "catch", "get hiv"]):
            return self._get_transmission_info()

        elif any(word in user_input for word in ["symptom", "sign", "feel", "experience", "show"]):
            return self._get_symptoms_info()

        elif any(word in user_input for word in ["test", "testing", "diagnose", "result", "positive", "negative"]):
            return self._get_testing_info()

        elif any(word in user_input for word in ["oi", "opportunistic", "infection", "cryptococcus", "pjp", "toxo"]):
            return self._get_oi_info()

        elif "who stage" in user_input or "staging" in user_input:
            return self._get_who_staging()

        elif any(word in user_input for word in ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"]):
            return "Hello! I'm your HIV/AIDS expert assistant. How can I help you with HIV-related questions today?"

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
4. **Close clinical monitoring** (2-4 week intervals)
5. **Integrated mental health and NCD screening**"""

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
- **Testing and counseling**: Regular HIV testing

**Structural Interventions:**
- **Stigma reduction programs**
- **Legal protections**
- **Economic empowerment**
- **Comprehensive sex education**"""

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

**NO Transmission Through:**
- Kissing, hugging, shaking hands
- Sharing utensils, toilet seats
- Mosquito bites, sweat, tears
- Swimming pools, air

**Risk Reduction:**
- **ART**: Viral suppression eliminates sexual transmission
- **Condoms**: 80-95% reduction in transmission
- **PrEP**: >90% reduction when adherent
- **Medical male circumcision**: 60% reduction in male acquisition"""

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
- Kaposi sarcoma, lymphomas
- HIV-associated neurocognitive disorders"""

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

3. **Viral Load PCR**
   - For infant diagnosis and treatment monitoring
   - Not for routine diagnosis

**Testing Recommendations:**
- **Universal**: All adults/adolescents at least once
- **High-risk**: Every 3-6 months
- **Pregnant women**: Every pregnancy
- **Partner testing**: Encourage mutual disclosure"""

    def _get_oi_info(self):
        return """**Opportunistic Infections - Comprehensive Guide**

**Common OIs in AHD:**

1. **Tuberculosis (TB)**
   - Most common OI globally
   - Screening: Symptom screen at every visit
   - Prevention: Isoniazid preventive therapy (IPT)

2. **Cryptococcal Meningitis**
   - CD4 <100, high mortality
   - Screening: CrAg in blood if CD4 <100
   - Treatment: Amphotericin B + flucytosine

3. **Pneumocystis jirovecii Pneumonia (PJP)**
   - CD4 <200, subacute respiratory symptoms
   - Prophylaxis: Cotrimoxazole if CD4 <200

4. **Toxoplasmosis**
   - CD4 <100, CNS symptoms
   - Prophylaxis: Cotrimoxazole

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
- Predicts disease progression
- Informs prognosis and monitoring frequency"""

    def _get_pmtct_info(self):
        return """**Prevention of Mother-to-Child Transmission (PMTCT) - Comprehensive Guide**

**Four-Pronged Approach:**
1. **Primary prevention** of HIV in women
2. **Prevent unintended pregnancies** in HIV+ women
3. **Prevent transmission** to infants
4. **Provide treatment and support** to HIV+ mothers and families

**ART in Pregnancy:**
- **Preferred**: TDF + 3TC/FTC + DTG
- **Start ASAP** regardless of CD4 or gestational age
- **Continue throughout** pregnancy, delivery, and breastfeeding

**Infant Prophylaxis:**
- **High risk**: NVP for 6-12 weeks
- **Low risk**: NVP for 6 weeks
- **Breastfeeding**: Continue maternal ART, infant prophylaxis if high risk

**Delivery Planning:**
- **Viral load <1000**: Vaginal delivery appropriate
- **Viral load ≥1000**: Consider C-section at 38 weeks
- **Avoid invasive procedures** if unknown status

**Breastfeeding:**
- **Recommend**: Exclusive breastfeeding for 6 months
- **Continue**: ART throughout breastfeeding period
- **Wean gradually** over 1 month when transitioning"""

    def _get_tb_hiv_info(self):
        return """**TB-HIV Coinfection Management - Comprehensive Guide**

**Epidemiology:**
- **HIV increases TB risk** 15-20 times
- **Leading cause of death** in PLHIV
- **Global burden**: 8% of TB cases are HIV-positive

**Screening:**
- **At every visit**: WHO 4-symptom screen (cough, fever, night sweats, weight loss)
- **If any symptom**: GeneXpert MTB/RIF preferred
- **All TB patients**: Routine HIV testing

**Diagnosis Challenges:**
- **Atypical presentations** in advanced HIV
- **Higher rates** of extrapulmonary TB
- **Lower sensitivity** of sputum smear

**Treatment:**
- **Start ART** within 2 weeks of TB treatment (all CD4 counts)
- **Watch for**: Immune reconstitution inflammatory syndrome (IRIS)
- **Drug interactions**: Rifampicin reduces PI levels

**Prevention:**
- **IPT** for all PLHIV without active TB
- **Duration**: 6-36 months depending on setting
- **TPT** (TB preventive therapy) reduces mortality by 37%"""

    def _get_comprehensive_response(self, user_input):
        return f"""I understand you're asking about: "{user_input}"

I want to provide you with accurate, evidence-based information about HIV/AIDS. 

I specialize in comprehensive HIV topics including:
• **HIV treatment guidelines** and ART regimens
• **Prevention strategies** (PrEP, PEP, condoms, U=U)
• **Testing and diagnosis** approaches  
• **WHO clinical staging** and management
• **Opportunistic infections** and prevention
• **PMTCT** and pediatric HIV care
• **TB-HIV coinfection** management
• **HIV and NCDs** (hypertension, diabetes, mental health)
• **Mental health integration** in HIV care
• **Myths and misconceptions** about HIV
• **Epidemiology and statistics**

Could you please rephrase your question or ask about one of these specific HIV/AIDS topics? I'm here to provide you with the most current, evidence-based information."""

# -------------------------------
# ENHANCED ANALYTICS DASHBOARD CLASS WITH COMPREHENSIVE RECOMMENDATIONS
# -------------------------------
class ClinicAnalytics:
    def __init__(self):
        self.who_targets = {
            'viral_suppression': 90,
            'art_coverage': 95,
            'patient_retention': 90,
            'ahd_prevention': 85
        }
    
    def validate_and_clean_data(self, df):
        """Comprehensive data validation and cleaning"""
        validation_report = {
            'original_shape': df.shape,
            'missing_values': {},
            'outliers': {},
            'data_quality_issues': [],
            'cleaning_applied': [],
            'final_shape': None
        }
        
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Check for required columns
        required_columns = ['Patient_ID', 'Age', 'Gender', 'CD4_Count', 'Viral_Load', 'WHO_Stage']
        missing_required = [col for col in required_columns if col not in cleaned_df.columns]
        
        if missing_required:
            validation_report['data_quality_issues'].append(
                f"CRITICAL: Missing required columns: {missing_required}"
            )
            # If critical columns missing, return original with warning
            if 'CD4_Count' in missing_required or 'Viral_Load' in missing_required:
                return cleaned_df, validation_report
        
        # Handle missing values
        for column in cleaned_df.columns:
            missing_count = cleaned_df[column].isnull().sum()
            missing_pct = (missing_count / len(cleaned_df)) * 100
            
            validation_report['missing_values'][column] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            
            if missing_pct > 0:
                if missing_pct > 30:
                    validation_report['data_quality_issues'].append(
                        f"HIGH missing values in {column}: {missing_pct:.1f}%"
                    )
                elif missing_pct > 10:
                    validation_report['data_quality_issues'].append(
                        f"MODERATE missing values in {column}: {missing_pct:.1f}%"
                    )
                
                # Imputation strategy based on column type
                if column in ['CD4_Count', 'Viral_Load', 'Age', 'Months_on_ART']:
                    # For numerical clinical data, use median
                    median_val = cleaned_df[column].median()
                    cleaned_df[column].fillna(median_val, inplace=True)
                    validation_report['cleaning_applied'].append(
                        f"Imputed {column} with median: {median_val:.2f}"
                    )
                elif column in ['Gender', 'ART_Regimen', 'WHO_Stage']:
                    # For categorical, use mode
                    mode_val = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown'
                    cleaned_df[column].fillna(mode_val, inplace=True)
                    validation_report['cleaning_applied'].append(
                        f"Imputed {column} with mode: {mode_val}"
                    )
        
        # Handle outliers for numerical columns
        numerical_columns = ['Age', 'CD4_Count', 'Viral_Load', 'Months_on_ART']
        for column in numerical_columns:
            if column in cleaned_df.columns:
                Q1 = cleaned_df[column].quantile(0.25)
                Q3 = cleaned_df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = cleaned_df[(cleaned_df[column] < lower_bound) | (cleaned_df[column] > upper_bound)]
                outlier_count = len(outliers)
                
                validation_report['outliers'][column] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(cleaned_df)) * 100,
                    'bounds': [lower_bound, upper_bound]
                }
                
                if outlier_count > 0:
                    # Cap outliers instead of removing
                    cleaned_df[column] = np.where(cleaned_df[column] < lower_bound, lower_bound, cleaned_df[column])
                    cleaned_df[column] = np.where(cleaned_df[column] > upper_bound, upper_bound, cleaned_df[column])
                    
                    validation_report['cleaning_applied'].append(
                        f"Capped {outlier_count} outliers in {column} to IQR bounds"
                    )
        
        # Validate data ranges
        if 'CD4_Count' in cleaned_df.columns:
            invalid_cd4 = cleaned_df[(cleaned_df['CD4_Count'] < 0) | (cleaned_df['CD4_Count'] > 3000)]
            if len(invalid_cd4) > 0:
                validation_report['data_quality_issues'].append(
                    f"Invalid CD4 values: {len(invalid_cd4)} records outside 0-3000 range"
                )
                # Cap to valid range
                cleaned_df['CD4_Count'] = cleaned_df['CD4_Count'].clip(0, 3000)
        
        if 'Viral_Load' in cleaned_df.columns:
            invalid_vl = cleaned_df[cleaned_df['Viral_Load'] < 0]
            if len(invalid_vl) > 0:
                validation_report['data_quality_issues'].append(
                    f"Invalid Viral Load values: {len(invalid_vl)} negative records"
                )
                cleaned_df['Viral_Load'] = cleaned_df['Viral_Load'].clip(0)
        
        if 'Age' in cleaned_df.columns:
            invalid_age = cleaned_df[(cleaned_df['Age'] < 0) | (cleaned_df['Age'] > 120)]
            if len(invalid_age) > 0:
                validation_report['data_quality_issues'].append(
                    f"Invalid Age values: {len(invalid_age)} records outside 0-120 range"
                )
                cleaned_df['Age'] = cleaned_df['Age'].clip(0, 120)
        
        validation_report['final_shape'] = cleaned_df.shape
        
        return cleaned_df, validation_report
    
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
    
    def generate_comprehensive_recommendations(self, analysis, df, data_quality_report=None):
        """Generate comprehensive, actionable recommendations with priority scoring"""
        recommendations = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': [],
            'data_quality': []
        }
        
        # Data Quality Recommendations
        if data_quality_report and data_quality_report['data_quality_issues']:
            critical_issues = [issue for issue in data_quality_report['data_quality_issues'] if 'CRITICAL' in issue]
            if critical_issues:
                recommendations['high_priority'].append({
                    'title': 'Address Critical Data Quality Issues',
                    'description': f"Found {len(critical_issues)} critical data quality issues affecting analysis reliability",
                    'actions': [
                        "Implement data validation protocols during entry",
                        "Train staff on complete and accurate data collection",
                        "Set up automated data quality checks",
                        "Conduct regular data audits"
                    ],
                    'timeline': 'Immediate (1-2 weeks)',
                    'resources': 'Data manager, IT support, staff training',
                    'expected_impact': 'High - improves decision-making accuracy'
                })
        
        # Clinical Performance Recommendations
        # AHD Prevalence
        if analysis['ahd_cases'] > 20:
            recommendations['high_priority'].append({
                'title': 'Reduce High AHD Prevalence',
                'description': f"{analysis['ahd_cases']:.1f}% of patients have Advanced HIV Disease - exceeds acceptable threshold",
                'actions': [
                    "Implement same-day ART initiation protocol",
                    "Enhance community testing and linkage programs",
                    "Train staff on AHD package implementation",
                    "Set up AHD patient tracking system",
                    "Conduct root cause analysis for late presentation"
                ],
                'timeline': 'Urgent (2-4 weeks)',
                'resources': 'Clinical team, community health workers, tracking system',
                'expected_impact': 'High - reduces early mortality by 40%'
            })
        elif analysis['ahd_cases'] > 10:
            recommendations['medium_priority'].append({
                'title': 'Monitor Moderate AHD Cases',
                'description': f"{analysis['ahd_cases']:.1f}% AHD prevalence needs close monitoring",
                'actions': [
                    "Strengthen early detection systems",
                    "Implement rapid linkage to care",
                    "Enhance patient education on early testing",
                    "Monitor time-to-ART initiation"
                ],
                'timeline': 'Short-term (1-2 months)',
                'resources': 'Clinical staff, educational materials',
                'expected_impact': 'Medium - prevents progression to severe AHD'
            })
        
        # Viral Suppression
        suppression_gap = self.who_targets['viral_suppression'] - analysis['viral_suppression']
        if suppression_gap > 20:
            recommendations['high_priority'].append({
                'title': 'Address Low Viral Suppression',
                'description': f"Only {analysis['viral_suppression']:.1f}% suppression ({suppression_gap:.1f}% below WHO target)",
                'actions': [
                    "Implement enhanced adherence counseling",
                    "Conduct regimen review for side effects",
                    "Establish differentiated service delivery models",
                    "Set up peer support programs",
                    "Implement pharmacy refill tracking"
                ],
                'timeline': 'Immediate (2-4 weeks)',
                'resources': 'Adherence counselors, peer educators, pharmacy staff',
                'expected_impact': 'High - 20-30% improvement in suppression rates'
            })
        elif suppression_gap > 10:
            recommendations['medium_priority'].append({
                'title': 'Improve Viral Suppression Rates',
                'description': f"{analysis['viral_suppression']:.1f}% suppression below target, needs improvement",
                'actions': [
                    "Focus on patients with unsuppressed viral load",
                    "Implement targeted adherence support",
                    "Review ART regimens for optimization",
                    "Enhance patient education on adherence"
                ],
                'timeline': 'Short-term (1-3 months)',
                'resources': 'Clinical team, educational resources',
                'expected_impact': 'Medium - 15-20% improvement possible'
            })
        
        # Patient Retention
        if analysis['poor_retention'] > 20:
            recommendations['high_priority'].append({
                'title': 'Improve Patient Retention',
                'description': f"{analysis['poor_retention']:.1f}% of patients missing multiple visits - high risk of loss to follow-up",
                'actions': [
                    "Implement appointment reminder system (SMS/calls)",
                    "Establish community outreach and tracking",
                    "Develop flexible clinic hours",
                    "Set up patient navigation services",
                    "Conduct exit interviews for lost patients"
                ],
                'timeline': 'Short-term (1-2 months)',
                'resources': 'Community health workers, SMS system, navigation staff',
                'expected_impact': 'High - 25-40% reduction in loss to follow-up'
            })
        elif analysis['poor_retention'] > 10:
            recommendations['medium_priority'].append({
                'title': 'Enhance Retention Strategies',
                'description': f"{analysis['poor_retention']:.1f} retention issues detected",
                'actions': [
                    "Implement basic appointment reminders",
                    "Train staff on retention strategies",
                    "Monitor missed appointment patterns",
                    "Develop patient feedback system"
                ],
                'timeline': 'Medium-term (2-4 months)',
                'resources': 'Clinic staff, basic reminder system',
                'expected_impact': 'Medium - 15-25% improvement in retention'
            })
        
        # Population-Specific Recommendations
        young_patients = df[df['Age'] < 25]
        if len(young_patients) > 0:
            young_suppression = (young_patients['Viral_Load'] < 1000).mean() * 100
            if young_suppression < 70:
                recommendations['medium_priority'].append({
                    'title': 'Address Youth Engagement Challenges',
                    'description': f"Young patients (18-25) have only {young_suppression:.1f}% viral suppression",
                    'actions': [
                        "Develop youth-friendly services",
                        "Implement peer educator programs",
                        "Create social media engagement strategies",
                        "Provide mental health support for youth",
                        "Establish youth support groups"
                    ],
                    'timeline': 'Medium-term (3-6 months)',
                    'resources': 'Youth coordinators, peer educators, mental health support',
                    'expected_impact': 'Medium - 20-30% improvement in youth outcomes'
                })
        
        # CD4 Recovery
        if analysis['avg_cd4'] < 300:
            recommendations['medium_priority'].append({
                'title': 'Address Slow Immune Recovery',
                'description': f"Average CD4 count is {analysis['avg_cd4']:.0f} cells/mm³ - indicates suboptimal immune recovery",
                'actions': [
                    "Review patients with slow CD4 recovery",
                    "Optimize ART regimens if needed",
                    "Address comorbidities affecting recovery",
                    "Enhance nutritional support",
                    "Monitor for treatment failure"
                ],
                'timeline': 'Medium-term (2-4 months)',
                'resources': 'Clinical team, nutritionist, lab support',
                'expected_impact': 'Medium - improves long-term outcomes'
            })
        
        # Positive Performance Recognition
        if analysis['viral_suppression'] >= self.who_targets['viral_suppression']:
            recommendations['low_priority'].append({
                'title': 'Maintain Excellent Viral Suppression',
                'description': f"Congratulations! {analysis['viral_suppression']:.1f}% exceeds WHO 90% target",
                'actions': [
                    "Continue current successful strategies",
                    "Document and share best practices",
                    "Monitor for any emerging challenges",
                    "Celebrate team success"
                ],
                'timeline': 'Ongoing',
                'resources': 'Minimal additional resources needed',
                'expected_impact': 'Maintenance of excellent performance'
            })
        
        if analysis['ahd_cases'] < 5:
            recommendations['low_priority'].append({
                'title': 'Sustain Low AHD Prevalence',
                'description': f"Only {analysis['ahd_cases']:.1f}% AHD cases - excellent early detection performance",
                'actions': [
                    "Continue strong testing and linkage programs",
                    "Maintain community engagement",
                    "Monitor for any changes in presentation patterns",
                    "Share successful strategies with other clinics"
                ],
                'timeline': 'Ongoing',
                'resources': 'Current program maintenance',
                'expected_impact': 'Sustained low AHD rates'
            })
        
        return recommendations
    
    def generate_insights(self, analysis, df, data_quality_report=None):
        """Generate smart insights from analysis with data quality considerations"""
        insights = []
        
        # Data quality insights
        if data_quality_report and data_quality_report['data_quality_issues']:
            critical_issues = [issue for issue in data_quality_report['data_quality_issues'] if 'CRITICAL' in issue]
            if critical_issues:
                insights.append({
                    'type': '🚨 DATA QUALITY',
                    'title': 'Critical Data Quality Issues',
                    'message': f"Found {len(critical_issues)} critical data quality issues that may affect analysis accuracy",
                    'recommendation': 'Review data collection processes and ensure complete required fields'
                })
        
        # Clinical insights
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
        
        # Positive insights for good performance
        if analysis['viral_suppression'] >= self.who_targets['viral_suppression']:
            insights.append({
                'type': '✅ EXCELLENT',
                'title': 'Viral Suppression Target Achieved',
                'message': f"Congratulations! {analysis['viral_suppression']:.1f}% exceeds WHO target",
                'recommendation': 'Maintain current strategies and share best practices'
            })
        
        if analysis['ahd_cases'] < 5:
            insights.append({
                'type': '✅ EXCELLENT',
                'title': 'Low AHD Prevalence',
                'message': f"Only {analysis['ahd_cases']:.1f}% AHD cases - excellent early detection",
                'recommendation': 'Continue strong testing and linkage programs'
            })
        
        return insights

# -------------------------------
# INITIALIZE COMPONENTS
# -------------------------------
chatbot = HIVExpertChatbot()
analytics_engine = ClinicAnalytics()

# -------------------------------
# CREATE TABS
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
            # Validate input data
            validation_errors = []
            if cd4 <= 0:
                validation_errors.append("CD4 count should be positive")
            if vl < 0:
                validation_errors.append("Viral load cannot be negative")
            if age <= 0:
                validation_errors.append("Age must be positive")
            
            if validation_errors:
                for error in validation_errors:
                    st.error(f"⚠️ {error}")
            else:
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
# TAB 2: Enhanced Analytics Dashboard with Comprehensive Recommendations
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
            help="File should contain: Patient_ID, Age, Gender, CD4_Count, Viral_Load, WHO_Stage, etc."
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
    data_quality_report = None
    
    if uploaded_file is not None:
        try:
            current_data = pd.read_csv(uploaded_file)
            data_source = "Uploaded File"
            
            # Validate and clean data
            with st.spinner("🔍 Validating and cleaning data..."):
                current_data, data_quality_report = analytics_engine.validate_and_clean_data(current_data)
            
            st.success(f"✅ Successfully loaded {len(current_data)} patient records")
            
            # Show data quality summary
            if data_quality_report and data_quality_report['data_quality_issues']:
                st.warning(f"⚠️ Found {len(data_quality_report['data_quality_issues'])} data quality issues")
                
                with st.expander("📋 Data Quality Report Details"):
                    st.write("**Original Data Shape:**", data_quality_report['original_shape'])
                    st.write("**Final Data Shape:**", data_quality_report['final_shape'])
                    
                    if data_quality_report['missing_values']:
                        st.subheader("Missing Values Summary")
                        missing_df = pd.DataFrame(data_quality_report['missing_values']).T
                        st.dataframe(missing_df)
                    
                    if data_quality_report['outliers']:
                        st.subheader("Outliers Detected")
                        outliers_df = pd.DataFrame(data_quality_report['outliers']).T
                        st.dataframe(outliers_df)
                    
                    if data_quality_report['data_quality_issues']:
                        st.subheader("Data Quality Issues")
                        for issue in data_quality_report['data_quality_issues']:
                            st.write(f"- {issue}")
                    
                    if data_quality_report['cleaning_applied']:
                        st.subheader("Cleaning Actions Applied")
                        for action in data_quality_report['cleaning_applied']:
                            st.write(f"- {action}")
                            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
        
    elif demo_option != "Select sample...":
        clinic_type = "urban" if "Urban" in demo_option else "rural"
        current_data = analytics_engine.generate_sample_data(clinic_type)
        data_source = demo_option
        st.success(f"✅ Generated {len(current_data)} sample patient records from {demo_option}")
    
    # Main dashboard content
    if current_data is not None:
        st.markdown("---")
        
        # Perform analysis
        with st.spinner("🔍 Analyzing clinic data..."):
            analysis = analytics_engine.analyze_clinic_data(current_data)
            insights = analytics_engine.generate_insights(analysis, current_data, data_quality_report)
            recommendations = analytics_engine.generate_comprehensive_recommendations(analysis, current_data, data_quality_report)
        
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
        
        # Insights and Recommendations
        st.markdown("### 💡 Smart Insights & Recommendations")
        
        if insights:
            for insight in insights:
                if 'CRITICAL' in insight['type']:
                    st.markdown(f"""
                    <div style='background-color: #ffebee; border-left: 6px solid #d32f2f; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                        <h4>{insight['type']}: {insight['title']}</h4>
                        <p><strong>{insight['message']}</strong></p>
                        <p>💡 <strong>Recommendation:</strong> {insight['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif 'WARNING' in insight['type']:
                    st.markdown(f"""
                    <div style='background-color: #fff3e0; border-left: 6px solid #ff9800; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                        <h4>{insight['type']}: {insight['title']}</h4>
                        <p><strong>{insight['message']}</strong></p>
                        <p>💡 <strong>Recommendation:</strong> {insight['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background-color: #e8f5e8; border-left: 6px solid #4caf50; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                        <h4>{insight['type']}: {insight['title']}</h4>
                        <p><strong>{insight['message']}</strong></p>
                        <p>💡 <strong>Recommendation:</strong> {insight['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("🎉 Excellent! Your clinic is meeting or exceeding most performance targets!")
        
        # Comprehensive Recommendations Section
        st.markdown("### 🎯 Comprehensive Action Plan")
        
        # High Priority Recommendations
        if recommendations['high_priority']:
            st.markdown("#### 🔴 High Priority Actions (Address Immediately)")
            for i, rec in enumerate(recommendations['high_priority'], 1):
                with st.expander(f"{i}. {rec['title']}", expanded=True):
                    st.write(f"**Description:** {rec['description']}")
                    st.write("**Recommended Actions:**")
                    for action in rec['actions']:
                        st.write(f"- {action}")
                    st.write(f"**Timeline:** {rec['timeline']}")
                    st.write(f"**Resources Needed:** {rec['resources']}")
                    st.write(f"**Expected Impact:** {rec['expected_impact']}")
        
        # Medium Priority Recommendations
        if recommendations['medium_priority']:
            st.markdown("#### 🟡 Medium Priority Actions (Address Soon)")
            for i, rec in enumerate(recommendations['medium_priority'], 1):
                with st.expander(f"{i}. {rec['title']}"):
                    st.write(f"**Description:** {rec['description']}")
                    st.write("**Recommended Actions:**")
                    for action in rec['actions']:
                        st.write(f"- {action}")
                    st.write(f"**Timeline:** {rec['timeline']}")
                    st.write(f"**Resources Needed:** {rec['resources']}")
                    st.write(f"**Expected Impact:** {rec['expected_impact']}")
        
        # Low Priority Recommendations
        if recommendations['low_priority']:
            st.markdown("#### 🟢 Low Priority Actions (Maintain Excellence)")
            for i, rec in enumerate(recommendations['low_priority'], 1):
                with st.expander(f"{i}. {rec['title']}"):
                    st.write(f"**Description:** {rec['description']}")
                    st.write("**Recommended Actions:**")
                    for action in rec['actions']:
                        st.write(f"- {action}")
                    st.write(f"**Timeline:** {rec['timeline']}")
                    st.write(f"**Resources Needed:** {rec['resources']}")
                    st.write(f"**Expected Impact:** {rec['expected_impact']}")
        
        # Export Section with Comprehensive Report
        st.markdown("---")
        st.markdown("### 📥 Export Comprehensive Analysis Report")
        
        # Generate comprehensive report
        report_text = f"""
COMPREHENSIVE CLINIC PERFORMANCE ANALYSIS REPORT
================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Data Source: {data_source}
Total Patients Analyzed: {len(current_data)}

EXECUTIVE SUMMARY:
------------------
Overall Clinic Performance: {'EXCELLENT' if not recommendations['high_priority'] else 'NEEDS IMPROVEMENT'}
Key Strengths: {len(recommendations['low_priority'])} areas of excellence
Priority Improvements: {len(recommendations['high_priority'])} urgent actions needed

KEY PERFORMANCE INDICATORS:
---------------------------
• Viral Suppression Rate: {analysis['viral_suppression']:.1f}% (Target: 90%) - {'✓ MET' if analysis['viral_suppression'] >= 90 else '✗ NOT MET'}
• AHD Prevalence: {analysis['ahd_cases']:.1f}% (Target: <10%) - {'✓ MET' if analysis['ahd_cases'] < 10 else '✗ NOT MET'}
• Average CD4 Count: {analysis['avg_cd4']:.0f} cells/mm³
• Patient Retention: {analysis['good_retention']:.1f}% (Target: >90%) - {'✓ MET' if analysis['good_retention'] >= 90 else '✗ NOT MET'}

DATA QUALITY ASSESSMENT:
------------------------
• Data Completeness: {100 - (current_data.isnull().sum().sum() / (len(current_data) * len(current_data.columns)) * 100):.1f}%
• Data Quality Issues: {len(data_quality_report['data_quality_issues']) if data_quality_report else 0}

PRIORITY RECOMMENDATIONS:
=========================

HIGH PRIORITY ACTIONS (Address within 2-4 weeks):
-------------------------------------------------
"""
        
        for i, rec in enumerate(recommendations['high_priority'], 1):
            report_text += f"\n{i}. {rec['title']}"
            report_text += f"\n   Description: {rec['description']}"
            report_text += f"\n   Timeline: {rec['timeline']}"
            report_text += f"\n   Expected Impact: {rec['expected_impact']}"
            report_text += f"\n   Key Actions:"
            for action in rec['actions'][:3]:  # Include top 3 actions
                report_text += f"\n     - {action}"
            report_text += "\n"
        
        report_text += f"""
MEDIUM PRIORITY ACTIONS (Address within 1-3 months):
----------------------------------------------------
"""
        
        for i, rec in enumerate(recommendations['medium_priority'], 1):
            report_text += f"\n{i}. {rec['title']}"
            report_text += f"\n   Description: {rec['description']}"
            report_text += f"\n   Timeline: {rec['timeline']}"
            report_text += f"\n   Expected Impact: {rec['expected_impact']}"
            report_text += f"\n   Key Actions:"
            for action in rec['actions'][:2]:  # Include top 2 actions
                report_text += f"\n     - {action}"
            report_text += "\n"
        
        report_text += f"""
AREAS OF EXCELLENCE (Maintain and share best practices):
--------------------------------------------------------
"""
        
        for i, rec in enumerate(recommendations['low_priority'], 1):
            report_text += f"\n{i}. {rec['title']}"
            report_text += f"\n   Description: {rec['description']}"
            report_text += "\n"
        
        report_text += f"""
IMPLEMENTATION ROADMAP:
-----------------------
Week 1-2: {len([r for r in recommendations['high_priority'] if 'Immediate' in r.get('timeline', '')])} immediate actions
Month 1: {len(recommendations['high_priority'])} high priority projects
Month 2-3: {len(recommendations['medium_priority'])} medium priority initiatives
Ongoing: {len(recommendations['low_priority'])} maintenance activities

RESOURCE REQUIREMENTS:
----------------------
• Staff Training: {sum(1 for r in recommendations['high_priority'] + recommendations['medium_priority'] if 'training' in r.get('resources', '').lower())} programs needed
• System Improvements: {sum(1 for r in recommendations['high_priority'] + recommendations['medium_priority'] if 'system' in r.get('resources', '').lower())} implementations
• Community Engagement: {sum(1 for r in recommendations['high_priority'] + recommendations['medium_priority'] if 'community' in r.get('resources', '').lower())} initiatives

EXPECTED OUTCOMES:
------------------
• Viral Suppression Improvement: {max(0, 90 - analysis['viral_suppression']):.1f}% gap to close
• AHD Reduction: {max(0, analysis['ahd_cases'] - 10):.1f}% reduction needed
• Retention Improvement: {max(0, 90 - analysis['good_retention']):.1f}% gap to close

---
Report generated by AHD Copilot Analytics Dashboard
For clinical decision support and quality improvement planning
Contact: HIV Program Manager for implementation support
"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download Comprehensive Report",
                data=report_text,
                file_name=f"clinic_comprehensive_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # Convert DataFrame to CSV for download
            csv = current_data.to_csv(index=False)
            st.download_button(
                label="📊 Download Analyzed Data (CSV)",
                data=csv,
                file_name=f"clinic_analyzed_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        # No data loaded - show instructions
        st.markdown("---")
        st.markdown("""
        ### 🚀 How to Use This Dashboard
        
        1. **Upload Your Data**: Click 'Browse files' to upload your clinic's CSV data
        2. **Try Sample Data**: Select a sample clinic to see how the analysis works
        3. **Get Insights**: The dashboard will automatically analyze and provide recommendations
        
        ### 📋 Expected Data Format
        Your CSV file should include these columns:
        ```
        Patient_ID, Age, Gender, CD4_Count, Viral_Load, WHO_Stage, 
        Months_on_ART, ART_Regimen, Missed_Visits
        ```
        """)

# -------------------------------
# TAB 3: ENHANCED HIV EXPERT CHATBOT
# -------------------------------
with tab3:
    st.subheader("💬 HIV/AIDS Expert Chatbot")
    st.info("🔬 **Your comprehensive HIV clinical decision support assistant**")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your HIV/AIDS expert assistant. I can help with treatment guidelines, prevention strategies, clinical management, mental health integration, myths clarification, and much more. What would you like to know?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input at the top for better UX
    if prompt := st.chat_input("Ask any HIV-related question in your own words..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            # Get response from expert chatbot
            response = chatbot.get_response(prompt)
            message_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick action buttons moved to bottom
    st.markdown("---")
    st.markdown("### 💡 Quick Access Topics")
    
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
        
        if st.button("🩺 NCDs & HIV", use_container_width=True, type="secondary"):
            st.session_state.messages.append({"role": "user", "content": "HIV and NCDs"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot.get_ncd_info()})
            st.rerun()
    
    with col2:
        if st.button("🛡️ Prevention", use_container_width=True, type="secondary"):
            st.session_state.messages.append({"role": "user", "content": "HIV prevention methods"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot._get_prevention_info()})
            st.rerun()
        
        if st.button("🤰 PMTCT", use_container_width=True, type="secondary"):
            st.session_state.messages.append({"role": "user", "content": "PMTCT guidelines"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot._get_pmtct_info()})
            st.rerun()
        
        if st.button("🦠 TB-HIV", use_container_width=True, type="secondary"):
            st.session_state.messages.append({"role": "user", "content": "TB HIV coinfection"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot._get_tb_hiv_info()})
            st.rerun()
    
    with col3:
        if st.button("🏥 WHO Staging", use_container_width=True, type="secondary"):
            st.session_state.messages.append({"role": "user", "content": "WHO clinical staging"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot._get_who_staging()})
            st.rerun()
        
        if st.button("❌ Myths & Facts", use_container_width=True, type="secondary"):
            st.session_state.messages.append({"role": "user", "content": "HIV myths and misconceptions"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot.get_myths_info()})
            st.rerun()
        
        if st.button("🧠 Mental Health", use_container_width=True, type="secondary"):
            st.session_state.messages.append({"role": "user", "content": "mental health and HIV"})
            st.session_state.messages.append({"role": "assistant", "content": chatbot.get_mental_health_info()})
            st.rerun()
    
    # Clear chat button at the bottom
    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True, type="primary"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your HIV/AIDS expert assistant. I can help with treatment guidelines, prevention strategies, clinical management, mental health integration, myths clarification, and much more. What would you like to know?"}
        ]
        st.rerun()

# -------------------------------
# Single Footer
# -------------------------------
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>© 2025 | Built with ❤️ by <b>Idah Anyango</b> for Better HIV Care</div>", unsafe_allow_html=True)
