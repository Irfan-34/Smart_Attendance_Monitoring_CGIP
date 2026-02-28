"""
FastAPI Server for Smart Attendance System
Handles Database connections (SQLite + ChromaDB) and Face Processing logic via API.
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import chromadb
import cv2
import numpy as np
import os
from datetime import datetime
from deepface import DeepFace
from anti_spoofing import check_liveness_heuristic

# Ensure relative paths resolve
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Smart Attendance API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For Streamlit frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Database Initialization
# -------------------------------------------------------------------

# 1. SQLite (For relational data: Students & Attendance Logs)
DB_PATH = os.path.join(_SCRIPT_DIR, "dataset", "attendance.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_sqlite():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            confidence REAL,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    ''')
    conn.commit()
    conn.close()

init_sqlite()

# 2. ChromaDB (Vector DB for storing 512-D ArcFace Embeddings)
CHROMA_PATH = os.path.join(_SCRIPT_DIR, "dataset", "chromadb")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    collection = chroma_client.get_collection(name="face_embeddings")
except Exception:
    collection = chroma_client.create_collection(
        name="face_embeddings",
        metadata={"hnsw:space": "cosine"} # Use Cosine Distance for Facial Matching
    )

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def extract_embedding(image_bytes: bytes):
    """Takes raw image bytes, converts to OpenCV format, and extracts ArcFace embedding."""
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image.")
        
    try:
        objs = DeepFace.represent(
            img_path=img, 
            model_name="ArcFace", 
            detector_backend="retinaface",
            enforce_detection=True
        )
        if len(objs) == 0:
            raise ValueError("No face detected.")
        if len(objs) > 1:
            raise ValueError("Multiple faces detected in registration image.")
            
        return objs[0]["embedding"]
    except Exception as e:
        raise ValueError(f"Face extraction failed: {str(e)}")


# -------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------

@app.post("/register")
async def register_student(
    student_id: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        image_bytes = await file.read()
        try:
            embedding = extract_embedding(image_bytes)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO students (id, name) VALUES (?, ?)", (student_id, name))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            raise HTTPException(status_code=400, detail="Student ID already exists.")
        finally:
            conn.close()
            
        collection.add(
            embeddings=[embedding],
            documents=[name],
            ids=[student_id]
        )
        
        return {"status": "success", "message": f"Successfully registered {name}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/recognize")
async def recognize_attendance(file: UploadFile = File(...)):
    """Accepts an image, finds closest match in ChromaDB, and marks SQLite attendance."""
    try:
        image_bytes = await file.read()
        
        # 1. Check Liveness (Anti-Spoofing)
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        is_live, liveness_msg = check_liveness_heuristic(img)
        
        if not is_live:
            return {"status": "spoof", "message": f"🚨 Security Alert: {liveness_msg}"}

        # 2. Extract Embeddings (Multi-face support)
        try:
            # We use enforce_detection=True but it returns a list of faces
            objs = DeepFace.represent(
                img_path=img, 
                model_name="ArcFace", 
                detector_backend="retinaface",
                enforce_detection=True
            )
            print(f"DEBUG: DeepFace detected {len(objs)} faces.")
        except ValueError as e:
            print(f"DEBUG Error: {e}")
            return {"status": "error", "message": f"Face detection failed: {str(e)}"}
            
        if len(objs) == 0:
            return {"status": "error", "message": "No faces detected in the image."}

        # Initialize results array
        processed_faces = []
        
        # Connect to SQLite with a wider timeout for concurrent writes
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10.0)
            cursor = conn.cursor()
        except sqlite3.Error as e:
            return {"status": "error", "message": f"Database connection failed: {str(e)}"}
            
        today = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H:%M:%S')

        for idx, face_obj in enumerate(objs):
            try:
                live_embedding = face_obj.get("embedding", None)
                if not live_embedding:
                    continue
                    
                print(f"DEBUG: Processing face {idx+1}/{len(objs)}")
                
                # Query ChromaDB for this specific face
                results = collection.query(
                    query_embeddings=[live_embedding],
                    n_results=1
                )
                
                if not results.get("ids") or not results["ids"][0]:
                    processed_faces.append({"status": "unknown", "message": "Unknown face."})
                    continue

                distance = results["distances"][0][0]
                student_id = results["ids"][0][0]
                student_name = results["documents"][0][0]

                # ArcFace Cosine Threshold
                COSINE_THRESHOLD = 0.60
                if distance > COSINE_THRESHOLD:
                    processed_faces.append({
                        "status": "unknown", 
                        "message": f"Guessed {student_name} but confidence was too low ({distance:.3f})"
                    })
                    continue

                # Check if already marked today
                cursor.execute("SELECT id FROM attendance WHERE student_id = ? AND date = ?", (student_id, today))
                if cursor.fetchone():
                    processed_faces.append({
                        "status": "duplicate", 
                        "student": {"id": student_id, "name": student_name},
                        "message": "Already present."
                    })
                    continue

                # Log new attendance
                cursor.execute(
                    "INSERT INTO attendance (student_id, date, time, confidence) VALUES (?, ?, ?, ?)", 
                    (student_id, today, current_time, float(distance))
                )
                processed_faces.append({
                    "status": "success",
                    "student": {"id": student_id, "name": student_name},
                    "distance": distance
                })
            except Exception as inner_e:
                print(f"DEBUG Error processing face {idx}: {inner_e}")
                processed_faces.append({"status": "error", "message": "Failed to process this face."})
                
        try:
            conn.commit()
        except sqlite3.Error as e:
            print(f"DEBUG SQLite Commit Error: {e}")
        finally:
            conn.close()
        
        # Return aggregate results
        return {
            "status": "success", 
            "faces_detected": len(objs),
            "results": processed_faces
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/attendance")
def get_attendance_logs(date: str = None):
    """Returns attendance logs. If date is provided (YYYY-MM-DD), filters by date."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if date:
        cursor.execute('''
            SELECT a.student_id, s.name, a.date, a.time, a.confidence 
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.date = ?
            ORDER BY a.time DESC
        ''', (date,))
    else:
         cursor.execute('''
            SELECT a.student_id, s.name, a.date, a.time, a.confidence 
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            ORDER BY a.date DESC, a.time DESC
            LIMIT 100
        ''')       
    
    rows = cursor.fetchall()
    conn.close()
    
    logs = [
        {"student_id": r[0], "name": r[1], "date": r[2], "time": r[3], "distance": r[4]}
        for r in rows
    ]
    return {"logs": logs}

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "2.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
