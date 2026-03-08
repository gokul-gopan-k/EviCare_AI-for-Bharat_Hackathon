# s3_utils.py
import boto3
import pandas as pd
import io
import os
from datetime import datetime
import streamlit as st
import json

# Add to imports
import json

# Initialize DynamoDB
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
AUDIT_TABLE = "EviCare_Audit_Logs" # Ensure this table exists in AWS


# --- AWS S3 Client ---
s3 = boto3.client('s3')
BUCKET_NAME = "evicare-knowledge-base-2026"
EXCEL_KEY = "patient_data.xlsx"


# -----------------------------
# S3 and dynamodb Helper Functions
# -----------------------------
def load_data_from_s3():
    """Fetches the master patient list directly from S3 Data Lake."""
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=EXCEL_KEY)
        return pd.read_excel(io.BytesIO(obj['Body'].read()))
    except Exception as e:
        st.sidebar.error(f"S3 Connection Error: {e}")
        return None


def upload_to_s3(file_obj):
    """Syncs manual uploads back to S3 to maintain a single source of truth."""
    try:
        s3.upload_fileobj(file_obj, BUCKET_NAME, EXCEL_KEY)
        return True
    except Exception as e:
        st.sidebar.error(f"S3 Sync Failed: {e}")
        return False

def get_patient_history(patient_id):
    """Queries DynamoDB for the latest recorded action for a specific patient."""
    try:
        table = dynamodb.Table(AUDIT_TABLE)
        # We query by patient_id. Note: This requires a Global Secondary Index (GSI) 
        # or a scan with a filter if patient_id isn't your primary key.
        # For a hackathon, a FilterExpression is easiest:
        response = table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr('patient_id').eq(str(patient_id))
        )
        items = response.get('Items', [])
        
        if not items:
            return None
            
        # Sort by timestamp to get the absolute latest visit
        items.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return items[0] 
    except Exception as e:
        print(f"History Lookup Error: {e}")
        return None



def log_action(patient_id, ai_rec, status, modified_text=""):
    """Saves recommendation actions to DynamoDB with current date."""
    try:
        table = dynamodb.Table(AUDIT_TABLE)
        now = datetime.now()
        table.put_item(
            Item={
                "action_id": f"{patient_id}_{datetime.now().timestamp()}",
                "date": now.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": now.isoformat(timespec="seconds"),
                "patient_id": str(patient_id),
                "ai_recommendation": ai_rec,
                "status": status,
                "modified_recommendation": modified_text
            }
        )
    except Exception as e:
        st.error(f"DynamoDB Log Error: {e}")

def append_patient_to_s3(new_row_dict):
    """Appends a single new patient row to the existing S3 Excel file."""
    try:
        df_full = load_data_from_s3()
        # Keep only the latest record per PatientID
        df = df_full.sort_values("last_updated").groupby("PatientID", as_index=False).last()
        if df is not None:
            # Handle list to string conversion for Excel storage
            if isinstance(new_row_dict.get("Medications"), list):
                new_row_dict["Medications"] = ", ".join(new_row_dict["Medications"])
            
            # Use pd.concat for modern pandas compatibility
            new_df = pd.concat([df, pd.DataFrame([new_row_dict])], ignore_index=True)
            
            # Save back to S3
            buffer = io.BytesIO()
            new_df.to_excel(buffer, index=False)
            buffer.seek(0)
            s3.upload_fileobj(buffer, BUCKET_NAME, EXCEL_KEY)
            return True
        return False
    except Exception as e:
        st.error(f"S3 Append Error: {e}")
        return False

@st.cache_data(ttl=600)  # Cache for 10 mins so you don't hit DynamoDB too often
def get_audit_log_as_csv():
    """Scans the DynamoDB table and returns a CSV string."""
    try:
        table = dynamodb.Table(AUDIT_TABLE)
        response = table.scan()
        data = response.get('Items', [])
        
        # Handle pagination if table is very large (>1MB)
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            data.extend(response.get('Items', []))
            
        if not data:
            return None
            
        # Convert to DataFrame and then to CSV
        df = pd.DataFrame(data)
        return df.to_csv(index=False).encode('utf-8')
    except Exception as e:
        st.error(f"Error fetching DynamoDB logs: {e}")
        return None