import os
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="PashuNetra Central Sync Server", version="1.0")

# --- DYNAMIC PATH RESOLUTION ---
# Ensures central_records.db is always created/opened in the same directory as main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "central_records.db")

# --- DATABASE SETUP ---
def init_central_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS central_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            village TEXT,
            animal_id TEXT,
            disease TEXT,
            confidence REAL,
            synced_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database table on server startup
init_central_db()

# --- DATA MODELS ---
class SyncPayload(BaseModel):
    village: str
    animal_id: str
    disease: str
    confidence: float

# --- API ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "online", "message": "PashuNetra Central Backend Engine is running."}

@app.post("/api/sync")
def receive_sync(records: List[SyncPayload]):
    if not records:
        return {"status": "success", "message": "No records to sync", "synced_count": 0}
        
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        for item in records:
            cursor.execute(
                "INSERT INTO central_logs (village, animal_id, disease, confidence) VALUES (?, ?, ?, ?)",
                (item.village, item.animal_id, item.disease, item.confidence)
            )
        conn.commit()
        conn.close()
        return {"status": "success", "synced_count": len(records)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database synchronization error: {str(e)}")
