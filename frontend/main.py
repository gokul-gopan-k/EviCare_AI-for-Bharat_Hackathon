import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime
import hashlib
from clinical_graph import ClinicalKnowledgeEngine
# imported utilities
from s3_utils import (load_data_from_s3,
                       upload_to_s3, log_action,
                         append_patient_to_s3,
                         get_audit_log_as_csv, 
                         get_patient_history)

from boto3.dynamodb.conditions import Key, Attr
# Backend & PDF Server Configuration 


PUBLIC_IP = "3.236.137.75"

if os.getenv("DOCKER_RUNNING"):
    BACKEND_URL = "http://backend:8000/api/recommend"
else:
    BACKEND_URL = "http://127.0.0.1:8000/api/recommend"
PDF_SERVER_URL = f"http://{PUBLIC_IP}:8000/static/"


if os.getenv("DOCKER_RUNNING"):
    BASE_URL = "http://backend:8000/api"
    HEALTH_URL = "http://backend:8000/health"
else:
    BASE_URL = "http://127.0.0.1:8000/api"
    HEALTH_URL = "http://127.0.0.1:8000/health"


# Initialize the Clinical Knowledge Engine
engine = ClinicalKnowledgeEngine()

# --- Streamlit Page Config ---
st.set_page_config(layout="wide", page_title="EviCare Dashboard")
st.title("EviCare")
# Map guideline names to local filenames
PDF_MAP = {
    "ICMR Guidelines for Management of Type 2 Diabetes": "ICMR.diabetesGuidelines.2018",
    "WHO package of essential noncommunicable (PEN) disease interventions for primary health care": "WHO PEN package of essential NCD interventions 2021",
    "Diagnosis and management of type 2 diabetes (HEARTS-D)": "WHO-UCN-NCD-20.1-eng"
}

def get_patient_hash(data_dict):
    """Creates a unique MD5 hash of the patient's clinical values."""
    encoded = str(data_dict).encode()
    return hashlib.md5(encoded).hexdigest()

@st.cache_data(show_spinner=False)
def get_cached_recommendation(payload, p_hash):
    """Calls the backend only if the patient data hash is new."""
    res = requests.post(BACKEND_URL, json=payload, timeout=45)
    if res.status_code == 200:
        return res.json()
    return None


# Custom CSS
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    .stButton>button { width: 100%; border-radius: 20px; }
    .main { background-color: #f8f9fa; }
    .chunk-box { background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 5px; border-left: 5px solid #007bff; }        
    </style>
""", unsafe_allow_html=True)


# --- Sidebar: Admin Tools ---
with st.sidebar:
    
    with st.expander("System Audit Logs"):
        if st.button("Prepare Log Download"):
            audit_csv = get_audit_log_as_csv()
            if audit_csv:
                st.download_button(
                    label="📥 Download Cloud Audit Log (CSV)",
                    data=audit_csv,
                    file_name=f"evicare_dynamo_audit_{datetime.now().strftime('%Y-%m-%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.caption("No audit logs found in DynamoDB.")
    st.divider()
    with st.expander("➕ Add New Patient"):
        with st.form("new_patient_form"):
            # Personal Info
            new_id = st.text_input("Patient ID")
            new_name = st.text_input("Name")
            new_age = st.number_input("Age", 1, 120, 45)
            
            new_sex = st.selectbox("Sex", ["Male", "Female", "Other"])    
            st.markdown("**Symptoms**")
            new_diag = st.text_input("Diagnosis", "Type 2 Diabetes, Hypertension")
            # Clinical Vitals
            st.markdown("**Lab Results**")
           
            new_hba1c = st.number_input("HbA1c", 4.0, 15.0, 7.0)
            new_glucose = st.number_input("Fasting Glucose", 50, 400, 120)
            new_bps = st.number_input("BP Systolic", 80, 200, 130)
            new_bpd = st.number_input("BP Diastolic", 40, 120, 80)

            # Labs
      
            new_ldl = st.number_input("LDL", 50, 300, 100)
            new_hdl = st.number_input("HDL", 10, 100, 45)
            new_creat = st.number_input("Creatinine", 0.1, 10.0, 1.0)
            new_egfr = st.number_input("eGFR", 5, 150, 90)

            # Lifestyle & Meds
            st.markdown("---")
            new_meds = st.text_input("Medications (comma separated)", "Metformin, Amlodipine")
            new_exercise = st.number_input("Exercise (min/day)", 0, 300, 30)
            new_diet = st.text_input("Diet", "Low sugar")
            new_symptoms = st.text_area("Symptoms", "None")

            submit = st.form_submit_button("Add to Cloud List")
            
            if submit and new_id and new_name:
                # Construct the full dictionary matching your exact format
                new_data = {
                    "PatientID": new_id,
                    "Name": new_name,
                    "Age": new_age,
                    "Sex": new_sex,
                    "Diagnosis": new_diag,
                    "HbA1c": new_hba1c,
                    "Fasting_Glucose": new_glucose,
                    "BP_Systolic": new_bps,
                    "BP_Diastolic": new_bpd,
                    "LDL": new_ldl,
                    "HDL": new_hdl,
                    "Creatinine": new_creat,
                    "eGFR": new_egfr,
                    "Medications": [m.strip() for m in new_meds.split(",")] if new_meds else [],
                    "Exercise_min_per_day": new_exercise,
                    "Diet": new_diet,
                    "Symptoms": new_symptoms
                }

                #  Add last date for versioning
                new_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                if append_patient_to_s3(new_data):
                    st.success(f"✅ Patient {new_id} ({new_name}) added to Cloud!")
                    st.rerun()
    st.divider()
    st.subheader("System Status")
    try:
        health_res = requests.get(HEALTH_URL, timeout=2)
        if health_res.status_code == 200:
            health_data = health_res.json()

            status = health_data.get("status", "unknown")
            metrics = health_data.get("resilience_metrics", {})
            
            if status == "healthy":
                st.success("🟢 System: Optimal")
            else:
                st.warning("🟡 System: Degraded (Fallback)")
                
            if metrics.get("fallback_count", 0) > 0:
                st.caption(f"⚠️ Active Fallbacks: {metrics['fallback_count']}")
        else:
            st.error("🔴 System: Offline")
    except Exception:
        st.error("🔴 Connection Failed")

# --- Sidebar: Patient Upload & Selection ---
with st.sidebar:
    # Load from S3
    df = load_data_from_s3()
    
    # default to none
    selected_pid = None

    if df is not None and not df.empty:
        patient_ids = ["-- Select Patient --"] + df['PatientID'].unique().tolist()
        selected_pid = st.radio("Select Patient ID", patient_ids)
        if selected_pid != "-- Select Patient --":
            selected_row = df[df['PatientID'] == selected_pid].iloc[0]

            # Reset session state when switching patients
            if "last_pid" not in st.session_state or st.session_state.last_pid != selected_pid:
                st.session_state.last_pid = selected_pid
                for key in list(st.session_state.keys()):
                    if key.startswith("show_mod_"):
                        del st.session_state[key]
        else:
            st.info("💡 No patients in cloud. Use 'Add New Patient' above to start.")
            st.stop()

# --- Main Layout ---
col_dash, col_detail = st.columns([1, 1], gap="large")

# --- LEFT: Dashboard ---
with col_dash:
    if selected_pid is None or selected_pid == "-- Select Patient --":
        st.info("Please select a patient from the sidebar to view details.")
    else:
        selected_row = df[df['PatientID'] == selected_pid].iloc[0]
        st.subheader(f" Patient {selected_pid}")
        with st.container(border=True):
            st.write(f"### {selected_row['Name']}")
            st.markdown("**Clinical Summary**")
            with st.container(border=True):
                summary_text = engine.generate_kg_summary(selected_row)
                if "Knowledge Graph Analysis:" in summary_text:
                    main_summary, kg_part = summary_text.split("Knowledge Graph Analysis:", 1)
                    if "Recent labs and vitals:" in kg_part:
                        kg_content, labs_part = kg_part.split("Recent labs and vitals:", 1)
                        summary_display = main_summary + "\nRecent labs and vitals:" + labs_part
                    else:
                        kg_content = kg_part
                        summary_display = main_summary
                else:
                    summary_display = summary_text
                    kg_content = None

                st.markdown(summary_display)

            if kg_content:
                with st.expander("🌐 View Knowledge Graph Pathophysiology"):
                    st.info(kg_content.strip().replace("•", "\n\n• "))

            # Critical Data Points & Lab Status
            st.markdown("**Critical Data Points & Risk Factors**")
            risks = engine.extract_critical_risks(selected_row)
            st.markdown("**Lab Status**")
            l1, l2, l3,l4 = st.columns(4)
            def display_lab(col, label, value, high_limit, low_limit=0):
                status = "Normal"
                color = "green"
                if value > high_limit: status, color = "High", "red"
                elif value < low_limit: status, color = "Low", "blue"
                col.metric(label, value)
                # col.markdown(f"Status: **:{color}  \n [{status}]**")
                col.markdown(
                    f"<div style='font-weight:600;'>Status:</div>"
                    f"<div style='color:{color}; font-weight:600;'>{status}</div>",
                    unsafe_allow_html=True
                )
            display_lab(l1, "HbA1c  \n (%)", selected_row['HbA1c'], 7.0)
            display_lab(l2, "BP Systolic  \n (mmHg)", selected_row['BP_Systolic'], 140)
            display_lab(l4, "BP Diastolic  \n (mmHg)", selected_row['BP_Diastolic'], 90)
            display_lab(l3, "eGFR  \n (mL/min/1.73m²)", selected_row['eGFR'], 200, 60)
            
            st.divider()
            for risk in risks:
                st.warning(risk)

            # Symptoms & Medications
            st.markdown("**Historical Clinical Context**")
            history = get_patient_history(selected_pid)
            h_col1, h_col2 = st.columns(2)

            if history:
                # Extract date from the timestamp or date field
                last_visit = history.get('date', 'Unknown')
                # Use the 'ai_recommendation' or 'modified_recommendation' as the protocol
                prev_proto = history.get('modified_recommendation') or history.get('ai_recommendation', 'Standard Care')
                status = history.get('status', 'Logged')

                with h_col1:
                    st.caption("📅 Last Clinical Review")
                    st.write(f"**{last_visit}**")
                with h_col2:
                    st.caption(f"📋 Previous Protocol ({status})")
                    st.write(f"**{prev_proto}**")
            else:
                st.info("No previous cloud-logged encounters for this patient.")

            st.divider()

        # AI Recommendation Trigger
        if st.button("Generate AI Recommendations ", type="primary"):

            payload = {
                "raw_summary": engine.generate_kg_summary(selected_row),
                "hba1c": float(selected_row.get('HbA1c', 0)),
                "bp_systolic": int(selected_row.get('BP_Systolic', 0)),
                "bp_diastolic": int(selected_row.get('BP_Diastolic', 0))
            }
            
            # Generate the hash based on the payload values
            p_hash = get_patient_hash(payload)
            with st.spinner("Analyzing Medical Guidelines..."):
                try:
                    res = get_cached_recommendation(payload, p_hash)
                    if res:
                        st.session_state['api_res'] = res
                        st.session_state['current_pid'] = selected_pid

                        st.rerun()
                    
                    else:
                        st.error("🚨 Failed to fetch recommendation. The LLM service may be overloaded or the backend is offline.")
                
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The clinical analysis is taking longer than expected.")
                except Exception as e:
                    st.error(f"Connection Error: {str(e)}")

# --- RIGHT: Recommendation Detail ---
with col_detail:
    st.subheader("Recommendation Detail Screen")
    if 'api_res' in st.session_state and st.session_state.get('current_pid') == selected_pid:
        data = st.session_state['api_res']
        for i, rec in enumerate(data['recommendations']):
            with st.container(border=True):
                st.markdown(f"#### Recommendation {i+1}")
                st.markdown(f"#### {rec['title']}")

                c1, c2 = st.columns([1,1])
                conf = rec.get('reliability_score',0)
                c1.write(f"**Confidence Score:** {conf}%")
                c1.progress(max(0.0, min(float(conf)/100,1.0)))
                color = "green" if conf>=70 else "orange" if conf>=40 else "red"
                c1.markdown(f"Status: **:{color}[{'High' if color=='green' else 'Moderate' if color=='orange' else 'Low'}]**")
                
                c2.caption(f"**Source:** {rec['citation_source']}")

                with st.container(border=True):
                    st.write(f"**Explanation:** {rec['reasoning']}")

                with st.expander("🔍 View Retrieved Evidence Chunks"):
                    chunks = rec.get('source_chunks', [])
                    for chunk in chunks:
                        text = chunk.get('text',"")
                        meta = chunk.get('metadata',{})
                        guideline_key = meta.get('guideline')
                        section_text = meta.get('section','General')
                        st.markdown(f"<div class='chunk-box'>{chunk['text']}</div>", unsafe_allow_html=True)
                        filename = PDF_MAP.get(guideline_key,"ICMR.diabetesGuidelines.2018")
                        # Check if text exists and is long enough to be a valid search
                        pdf_url = f"{PDF_SERVER_URL}{filename}.pdf"
                        st.link_button(f"📖 Open Source:", url=pdf_url)

                # Action Buttons
                b1, b2, b3 = st.columns(3)
                if b1.button("Accept", key=f"acc_{i}", use_container_width=True):
                    log_action(selected_pid, rec['title'], "Accept")
                    st.success("Recommendation accepted and saved!")

                if b2.button("Modify", key=f"mod_btn_{i}", use_container_width=True):
                    st.session_state[f"show_mod_{i}"] = True

                if b3.button("Reject", key=f"rej_{i}", use_container_width=True):
                    log_action(selected_pid, rec['title'], "Reject")
                    st.error("Rejected & Saved")

                if st.session_state.get(f"show_mod_{i}"):
                    mod_text = st.text_input("Enter modified recommendation:", value=rec['title'], key=f"txt_{i}")
                    if st.button("Confirm Modification", key=f"conf_{i}"):
                        log_action(selected_pid, rec['title'], "Modify", mod_text)
                        st.session_state[f"show_mod_{i}"] = False
                        st.success("Modified & Saved")
    else:
        st.info("Click 'Generate' to see AI-backed clinical recommendations.")