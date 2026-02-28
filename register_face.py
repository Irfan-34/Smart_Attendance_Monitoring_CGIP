"""
Face Registration Module - DeepFace Upgrade
Captures a single high-quality face image, extracts its 512-dimensional embedding using ArcFace,
and saves the vector to the database.
"""

import cv2
import os
import time
import numpy as np
from deepface import DeepFace

# Ensure relative paths resolve
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def register_student():
    print("\n" + "-"*60)
    print("[*] DEEP LEARNING STUDENT REGISTRATION")
    print("-" * 60)

    student_id = input("\n[*] Enter Student ID (e.g., 001, 002): ").strip()
    if not student_id:
        print("❌ Student ID cannot be empty!")
        return

    student_name = input("📛 Enter Full Name (First and Last): ").strip()
    if not student_name:
        print("❌ Student Name cannot be empty!")
        return

    print("\n" + "-"*60)
    print(f"[*] Registration Details:")
    print(f"   Student ID: {student_id}")
    print(f"   Name: {student_name}")
    print("-"*60)

    # We now save embeddings directly to a central folder database
    db_path = os.path.join(_SCRIPT_DIR, 'dataset', 'embeddings')
    os.makedirs(db_path, exist_ok=True)
    
    # Path where we'll save the embedding
    embedding_file = os.path.join(db_path, f"{student_id}_{student_name.replace(' ', '_')}.npy")
    
    if os.path.exists(embedding_file):
         confirm = input(f"\n⚠️  Student ID {student_id} already exists. Overwrite? (y/n): ").strip().lower()
         if confirm != 'y':
             print("Registration cancelled.")
             return

    print("\n[CAM] Accessing webcam to extract face embedding...")
    print("  Look straight at the camera. Press 'SPACE' to capture.")
    print("  Press 'Q' to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not access webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    embedding_extracted = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to read frame from webcam!")
                break

            frame = cv2.flip(frame, 1)

            # Draw UI
            cv2.putText(frame, f"Registering: {student_name} ({student_id})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture face", (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('DeepFace Registration', frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == 32: # SPACE BAR
                print("\n[⏳] Extracting face embedding using ArcFace. Please wait...")
                cv2.putText(frame, "Extracting Embedding...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cv2.imshow('DeepFace Registration', frame)
                cv2.waitKey(1)
                
                try:
                    # Represent the face using ArcFace (returns a list of dictionaries, we just need the first face's embedding)
                    # We use mtcnn or retinaface for detection as it's far superior to haar cascades
                    embedding_objs = DeepFace.represent(
                        img_path=frame, 
                        model_name="ArcFace", 
                        detector_backend="retinaface",
                        enforce_detection=True
                    )
                    
                    if len(embedding_objs) == 0:
                        print("⚠️ No face detected. Try again.")
                        continue
                        
                    if len(embedding_objs) > 1:
                        print("⚠️ Multiple faces detected! Please ensure only one person is in the frame.")
                        continue
                        
                    embedding = embedding_objs[0]["embedding"]
                    
                    # Save as numpy array
                    np.save(embedding_file, np.array(embedding))
                    embedding_extracted = True
                    break
                    
                except ValueError as e:
                    print(f"⚠️ Face detection failed: {e}. Please ensure good lighting and look at the camera.")

            elif key == ord('q') or key == ord('Q'):
                print("\n[*] Capture cancelled by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if embedding_extracted:
        print(f"\n[✅] Success! Extracted 512-D ArcFace embedding for {student_name} ({student_id}).")
        print(f"💾 Saved to: {embedding_file}\n")


if __name__ == "__main__":
    os.chdir(_SCRIPT_DIR)
    register_student()
