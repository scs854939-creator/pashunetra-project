import os
import requests
import streamlit as st

# ==========================================
# CONFIGURATION & BACKEND URL RESOLUTION
# ==========================================
st.set_page_config(
    page_title="PashuNetra (పశునేత్ర)",
    page_icon="🐄",
    layout="wide"
)

# Fetch backend URL dynamically from Streamlit Secrets or environment variables.
# Defaults to localhost if running locally without secrets.
BACKEND_URL = st.secrets.get("BACKEND_URL", "http://127.0.0.1:8000")


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def push_logs_to_server(sync_payload):
    """
    Sends offline/local diagnostic logs to the deployed backend server.
    Handles network errors gracefully if the backend is unreachable.
    """
    endpoint = f"{BACKEND_URL.rstrip('/')}/api/sync"
    
    try:
        response = requests.post(
            endpoint, 
            json=sync_payload, 
            timeout=10
        )
        if response.status_code in (200, 201):
            return True, response.json().get("message", "Sync successful!")
        else:
            return False, f"Server responded with status code: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return False, f"Could not connect to backend server at {BACKEND_URL}. Check if the service is online."
    except requests.exceptions.Timeout:
        return False, "Connection timed out while attempting to reach the server."
    except Exception as e:
        return False, f"An unexpected error occurred: {str(e)}"


# ==========================================
# UI & NAVIGATION
# ==========================================
st.title("🐄 PashuNetra (పశునేత్ర)")
st.caption("Multi-modal Diagnostic & Edge Synchronization Portal")

# Tab Navigation
tab1, tab2, tab3 = st.tabs(["🔍 Mobile Lesion Scanner", "📊 Outbreak Analytics", "🔄 Sync History"])

with tab1:
    st.header("Mobile Lesion Scanner")
    uploaded_file = st.file_uploader("Upload Lesion Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.success("Image processed locally.")

st.divider()

# ==========================================
# EDGE SYNCHRONIZATION MANAGER
# ==========================================
st.subheader("🌐 Edge Synchronization Manager")

# Example local logs pending synchronization
pending_logs = st.session_state.get("pending_logs", [
    {
        "tag_id": "AP-13100",
        "location": "Raghumanda",
        "symptoms": "Hoof lesion / appetite loss",
        "timestamp": "2026-09-17 10:30:00"
    }
])

col1, col2 = st.columns([1, 3])

with col1:
    st.metric(label="Pending Sync Logs", value=len(pending_logs))
    sync_clicked = st.button("⚡ Push Logs to Server", type="primary")

if sync_clicked:
    if not pending_logs:
        st.info("No pending logs to synchronize.")
    else:
        with st.spinner(f"Synchronizing with backend ({BACKEND_URL})..."):
            success, result_msg = push_logs_to_server({"logs": pending_logs})
            
            if success:
                st.success(f"✅ {result_msg}")
                st.session_state["pending_logs"] = []  # Clear pending logs after success
            else:
                st.error(f"❌ {result_msg}")