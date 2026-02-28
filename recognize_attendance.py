"""
Real-Time Face Recognition & Attendance Module - DeepFace Upgrade
Recognizes students by comparing live webcam face embeddings (via ArcFace)
against stored .npy embeddings in the database using Cosine Similarity.
"""

import cv2
import os
import csv
import sys
from datetime import datetime
import time
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Ensure relative paths resolve
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
# ArcFace Threshold for Cosine Similarity: 
# Values below 0.68 are generally considered a match for ArcFace.
COSINE_THRESHOLD = 0.60  
RECOGNITION_COOLDOWN = 60 # seconds before same student can be re-logged


def open_webcam_with_timeout(camera_index=0, timeout_seconds=5):
    try:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            cap.release()
            return None
        ret, _ = cap.read()
        if not ret:
            cap.release()
            return None
        return cap
    except Exception as e:
        print(f"Exception while opening webcam: {e}")
        return None


def load_embeddings():
    """Loads all saved .npy embeddings into memory for fast comparison."""
    db_path = os.path.join(_SCRIPT_DIR, 'dataset', 'embeddings')
    embeddings_dict = {}
    
    if os.path.exists(db_path):
        for f in os.listdir(db_path):
            if f.endswith('.npy'):
                # filename format: "001_John_Doe.npy"
                name_parts = f.replace('.npy', '').split('_', 1)
                student_id = name_parts[0]
                student_name = name_parts[1].replace('_', ' ') if len(name_parts) > 1 else 'Unknown'
                
                vector = np.load(os.path.join(db_path, f))
                embeddings_dict[f"{student_id} - {student_name}"] = vector
                
    return embeddings_dict


def mark_attendance(student_id, student_name):
    """Mark attendance if not already marked today."""
    attendance_file = os.path.join(_SCRIPT_DIR, 'attendance', 'attendance.csv')
    os.makedirs(os.path.join(_SCRIPT_DIR, 'attendance'), exist_ok=True)

    today = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')

    if os.path.exists(attendance_file):
        try:
            with open(attendance_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3 and row[0] == student_id and row[2] == today:
                        return False
        except Exception:
            pass

    try:
        file_empty = not os.path.exists(attendance_file) or os.path.getsize(attendance_file) == 0
        with open(attendance_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if file_empty:
                writer.writerow(['Student ID', 'Name', 'Date', 'Time'])
            writer.writerow([student_id, student_name, today, current_time])
        print(f"\n[✅] Attendance marked for {student_name} ({student_id}) at {current_time}")
        return True
    except Exception as e:
        print(f"\n⚠️  Error marking attendance: {str(e)}")
        return False


def start_attendance():
    """Start real-time face recognition and attendance system using DeepFace."""
    
    # 1. Load the database into memory
    print("\n" + "-"*60)
    print("🚀 LOADING VECTOR DATABASE...")
    embeddings_dict = load_embeddings()
    
    if not embeddings_dict:
         print("❌ Error: No valid embeddings found! Please register a student first.")
         return
         
    print(f"[OK] Loaded {len(embeddings_dict)} face embeddings.")
    print("-" * 60)
    
    # 2. Start webcam
    print("📹 Starting Deep Learning Attendance System...")
    sys.stdout.flush()
    cap = open_webcam_with_timeout()
    
    if cap is None:
        print("❌ Error: Could not access webcam!")
        return

    attendance_logged = {}   # student_id → timestamp last logged
    
    # We'll use OpenCV's Haar just for FAST bounding box drawing
    # while DeepFace handles the heavy embedding lifting in the background
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            
            # Fast tracking visualization (Haar)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fast_faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            for (x, y, w, h) in fast_faces:
                 cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                 
            # Display Instructions
            cv2.putText(frame, "Press 'R' to Recognise Face | Press 'Q' to Exit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('Deep Learning Smart Attendance', frame)

            key = cv2.waitKey(1) & 0xFF
            
            # We trigger recognition on 'R' keypress to save CPU load 
            # (Running ArcFace every frame is too heavy for standard CPUs)
            if key == ord('r') or key == ord('R'):
                 cv2.putText(frame, "Scanning...", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                 cv2.imshow('Deep Learning Smart Attendance', frame)
                 cv2.waitKey(1)
                 
                 try:
                     # Extract live embedding using RetinaFace detector + ArcFace model
                     live_objs = DeepFace.represent(
                         img_path=frame, 
                         model_name="ArcFace", 
                         detector_backend="retinaface",
                         enforce_detection=True
                     )
                     
                     if len(live_objs) == 0:
                         print("⚠️ No face detected. Try again.")
                         continue
                         
                     # For each detected live face, compare against database
                     for obj in live_objs:
                         live_embedding = np.array(obj["embedding"])
                         box = obj["facial_area"]
                         x, y, w, h = box['x'], box['y'], box['w'], box['h']
                         
                         best_match_name = "Unknown"
                         best_distance = float('inf')
                         
                         # Vector Search (1-to-N comparison)
                         for name, stored_embedding in embeddings_dict.items():
                             dist = cosine(live_embedding, stored_embedding)
                             if dist < best_distance:
                                 best_distance = dist
                                 best_match_name = name
                                 
                         # Check threshold
                         if best_distance < COSINE_THRESHOLD:
                             student_id, student_name = best_match_name.split(' - ')
                             print(f"🎯 Recognized: {student_name} (ID: {student_id}) - Distance: {best_distance:.3f}")
                             
                             now = time.time()
                             last_logged = attendance_logged.get(student_id, 0)
                             if now - last_logged > RECOGNITION_COOLDOWN:
                                 mark_attendance(student_id, student_name)
                                 attendance_logged[student_id] = now
                                 
                             # Draw Green Box for success
                             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                             cv2.putText(frame, f"{student_name} (OK)", (x, y-10), 
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                             
                         else:
                             print(f"❓ Face not recognized. Best match distance was {best_distance:.3f} (Need < {COSINE_THRESHOLD})")
                             # Draw Red Box for unknown
                             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                             cv2.putText(frame, "Unknown", (x, y-10), 
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                         
                         # Show the updated frame for 2 seconds
                         cv2.imshow('Deep Learning Smart Attendance', frame)
                         cv2.waitKey(2000)
                         
                 except ValueError as e:
                     print(f"⚠️ Face detection failed: {e}")

            elif key == ord('q') or key == ord('Q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n" + "-"*60)
    print("[OK] Architecture Stopped")


if __name__ == "__main__":
    os.chdir(_SCRIPT_DIR)
    start_attendance()
