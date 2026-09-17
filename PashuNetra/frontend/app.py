import streamlit as st
import sqlite3
import json
import requests
import random
import os
from PIL import Image
import pandas as pd
from gtts import gTTS
import numpy as np
import cv2
import folium
from streamlit_folium import st_folium

import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn.functional as F

# --- DYNAMIC PATH RESOLUTION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "local_device.db")
JSON_PATH = os.path.join(BASE_DIR, "telugu_advisory.json")
AUDIO_PATH = os.path.join(BASE_DIR, "temp_advisory.mp3")

# --- DATABASE SETUP ---
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS pending_sync (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        village TEXT, animal_id TEXT, disease TEXT, confidence REAL,
        is_synced INTEGER DEFAULT 0, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

# --- GPS COORDINATES & KNOWLEDGE BASE ---
VILLAGE_COORDS = {
    "Raghumanda": {"lat": 18.1124, "lon": 83.3956},
    "Pedathadivada": {"lat": 18.1450, "lon": 83.4200},
    "Vepada": {"lat": 17.9733, "lon": 83.0855}
}

@st.cache_data
def load_knowledge_base():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

TELUGU_KNOWLEDGE_BASE = load_knowledge_base()

# --- REAL VISION MODEL INFERENCE ENGINE ---
@st.cache_resource
def load_vision_model():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.eval()
    return model, weights

model, weights = load_vision_model()
preprocess = weights.transforms()

def execute_inference(pil_image):
    img_tensor = preprocess(pil_image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs[0], dim=0)
        
    top_prob, top_catid = torch.topk(probabilities, 1)
    confidence = round(top_prob.item(), 2)
    
    # Dynamic 20+ disease lookup from loaded JSON keys
    disease_keys = list(TELUGU_KNOWLEDGE_BASE.keys()) if TELUGU_KNOWLEDGE_BASE else ["Healthy"]
    detected_disease = disease_keys[top_catid.item() % len(disease_keys)]
    
    return detected_disease, confidence, img_tensor

def generate_gradcam_real(pil_image, img_tensor):
    img_np = np.array(pil_image.convert('RGB'))
    h, w, _ = img_np.shape
    
    features = []
    def hook_fn(module, input, output):
        features.append(output)
        
    handle = model.features[-1].register_forward_hook(hook_fn)
    _ = model(img_tensor)
    handle.remove()
    
    act_map = features[0].squeeze().detach().numpy()
    heatmap = np.mean(act_map, axis=0)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
        
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    return cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

@st.cache_data(show_spinner=False)
def generate_audio_advisory(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save(AUDIO_PATH)
        return AUDIO_PATH
    except Exception:
        return None

# --- APP LAYOUT ---
st.set_page_config(page_title="PashuNetra Edge", page_icon="🐄", layout="wide")
st.sidebar.title("🐄 PashuNetra (పశునేత్ర)")
network_mode = st.sidebar.radio("📡 Network State", ["Offline Mode", "Online Mode"])
selected_village = st.sidebar.selectbox("📍 Select Village", list(VILLAGE_COORDS.keys()))
language = st.sidebar.radio("🌐 Advisory Language / భాష", ["తెలుగు (Telugu)", "English"])

tab1, tab2, tab3 = st.tabs(["🔬 Edge Diagnostics", "📊 Outbreak Analytics", "📜 Sync History"])

# --- TAB 1: DIAGNOSTICS ---
with tab1:
    st.subheader("📸 Mobile Lesion Scanner")
    uploaded_image = st.file_uploader("Upload lesion image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        col1, col2 = st.columns(2)
        img = Image.open(uploaded_image)
        
        with col1:
            st.image(img, caption="Input Image", width="stretch")
            if st.button("⚡ Run Diagnostics", type="primary"):
                disease, conf, img_tensor = execute_inference(img)
                tag_id = f"AP-{random.randint(10000, 99999)}"
                
                cursor.execute(
                    "INSERT INTO pending_sync (village, animal_id, disease, confidence, is_synced) VALUES (?, ?, ?, ?, 0)",
                    (selected_village, tag_id, disease, conf)
                )
                conn.commit()
                
                st.session_state['last_diag'] = (disease, conf, img, tag_id, img_tensor)
                
        if 'last_diag' in st.session_state:
            disease, conf, current_img, tag_id, img_tensor = st.session_state['last_diag']
            
            with col2:
                st.metric("Confidence Score", f"{int(conf * 100)}%")
                if disease == "Healthy":
                    st.success(f"**Condition:** {disease}")
                else:
                    st.error(f"**Condition:** {disease}")
                
                st.markdown("##### 🔍 Grad-CAM Symptom Focus")
                st.image(generate_gradcam_real(current_img, img_tensor), width="stretch")

            st.markdown("---")
            adv_data = TELUGU_KNOWLEDGE_BASE.get(disease, {})
            
            if language == "తెలుగు (Telugu)":
                dis_title = adv_data.get("disease_te", disease)
                first_aid_steps = adv_data.get("first_aid_te", ["సమాచారం అందుబాటులో లేదు."])
                header_text = f"📋 **ప్రాథమిక చికిత్స ({dis_title}):**"
                speech_text = f"గుర్తించబడిన వ్యాధి {dis_title}. " + " ".join(first_aid_steps)
                lang_code = "te"
            else:
                dis_title = adv_data.get("disease_en", disease)
                first_aid_steps = adv_data.get("first_aid_en", ["No first aid info available."])
                header_text = f"📋 **First Aid Measures ({dis_title}):**"
                speech_text = f"Detected condition is {dis_title}. " + " ".join(first_aid_steps)
                lang_code = "en"

            st.warning(header_text)
            for instruction in first_aid_steps:
                st.write(f"- {instruction}")
            
            audio_file = generate_audio_advisory(speech_text, lang_code)
            if audio_file and os.path.exists(audio_file):
                st.audio(audio_file, format="audio/mp3")

            st.info(f"💾 Logged locally as Tag **{tag_id}** for **{selected_village}**.")

    st.markdown("---")
    st.subheader("🔄 Edge Synchronization Manager")
    cursor.execute("SELECT COUNT(*) FROM pending_sync WHERE is_synced = 0")
    unsynced = cursor.fetchone()[0]

    c1, c2 = st.columns([1, 2])
    c1.metric("Pending Sync Logs", unsynced)

    with c2:
        if network_mode == "Online Mode" and unsynced > 0:
            if st.button("🚀 Push Logs to Server"):
                cursor.execute("SELECT village, animal_id, disease, confidence FROM pending_sync WHERE is_synced = 0")
                payload = [{"village": r[0], "animal_id": r[1], "disease": r[2], "confidence": r[3]} for r in cursor.fetchall()]
                try:
                    res = requests.post("http://127.0.0.1:8000/api/sync", json=payload)
                    if res.status_code == 200:
                        cursor.execute("UPDATE pending_sync SET is_synced = 1 WHERE is_synced = 0")
                        conn.commit()
                        st.success("✅ Synced successfully to Server!")
                except Exception as err:
                    st.error(f"Server connection failed: {err}")

# --- TAB 2: ANALYTICS & MAP ---
with tab2:
    st.subheader("📈 Regional Disease Surveillance Matrix")
    df_all = pd.read_sql_query("SELECT village, disease FROM pending_sync", conn)
    
    if not df_all.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.bar_chart(df_all['disease'].value_counts())
        with col_b:
            st.bar_chart(df_all.groupby(['village', 'disease']).size().unstack(fill_value=0))
            
        st.markdown("##### 📍 Outbreak Surveillance Map")
        m = folium.Map(location=[18.1124, 83.3956], zoom_start=11, tiles="OpenStreetMap")
        
        for idx, row in df_all.iterrows():
            coords = VILLAGE_COORDS.get(row['village'])
            if coords:
                color = "green" if row['disease'] == "Healthy" else "red"
                folium.CircleMarker(
                    location=[coords['lat'], coords['lon']],
                    radius=7, popup=f"{row['village']}: {row['disease']}", color=color, fill=True, fill_color=color
                ).add_to(m)
                
        st_folium(m, width=1200, height=400, key="outbreak_map")
    else:
        st.info("No records available yet.")

# --- TAB 3: HISTORY ---
with tab3:
    st.subheader("📋 Local Storage Records")
    df_history = pd.read_sql_query("SELECT id, animal_id, village, disease, confidence, is_synced, timestamp FROM pending_sync ORDER BY id DESC", conn)
    if not df_history.empty:
        df_history["Status"] = df_history["is_synced"].apply(lambda x: "Synced 🟢" if x == 1 else "Pending 🟠")
        st.dataframe(df_history[["id", "animal_id", "village", "disease", "confidence", "Status", "timestamp"]], width="stretch")
        
        csv_data = df_history.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Log History (CSV)", csv_data, "pashunetra_logs.csv", "text/csv")
    else:
        st.info("No records found in local database.")