"""
Debug script to diagnose face recognition issues
Shows what the system is detecting and confidence levels
"""

import cv2
import os

# Ensure relative paths resolve from the script's own directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def apply_clahe(gray_img):
    """Normalize lighting with CLAHE (matches preprocessing used during registration)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)


def debug_recognition():
    """
    Debug face recognition in real-time
    Shows:
    - If webcam works
    - If faces are detected
    - Confidence levels from the model
    """
    
    # Check if model exists
    model_file = os.path.join(_SCRIPT_DIR, 'trainer', 'trainer.yml')
    if not os.path.exists(model_file):
        print("❌ Model file not found!")
        return
    
    # Load cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    if face_cascade.empty():
        print("❌ Could not load Haar Cascade classifier!")
        return
    
    # Load model
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_file)
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load labels
    label_dict = {}
    label_file = os.path.join(_SCRIPT_DIR, 'trainer', 'labels.txt')
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    label, name = line.split(':', 1)
                    label_dict[int(label)] = name.strip()
    
    print(f"[OK] Labels loaded: {label_dict}")
    
    # Open webcam
    print("\n🎥 Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not access webcam!")
        return
    
    print("[OK] Webcam opened successfully")
    print("\n[CAM] Analyzing faces...")
    print("=" * 60)
    print("Move your face in front of the camera")
    print("Press 'Q' to exit")
    print("=" * 60 + "\n")
    
    frame_count = 0
    max_confidence = 100
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading frame!")
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=4,
                minSize=(20, 20)
            )
            
            frame_count += 1
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Extract and process face (match registration: CLAHE first, then resize)
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = apply_clahe(face_roi)
                    face_roi = cv2.resize(face_roi, (100, 100))
                    
                    # Get prediction
                    label, confidence = recognizer.predict(face_roi)
                    
                    # Get name
                    if label in label_dict:
                        name = label_dict[label]
                        status = "[OK] RECOGNIZED" if confidence < 62 else "⚠️  LOW CONFIDENCE"
                    else:
                        name = "Unknown Student"
                        status = "❌ NOT IN DATABASE"
                    
                    # Print debug info
                    print(f"\n[Frame {frame_count}] Face Detected!")
                    print(f"  Label ID: {label}")
                    print(f"  Name: {name}")
                    print(f"  Confidence: {confidence:.2f} (lower is better)")
                    print(f"  Status: {status}")
                    print(f"  Face Size: {w}x{h} pixels")
                    
                    # Draw on frame
                    threshold = 62
                    color = (0, 255, 0) if confidence < threshold else (0, 165, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    text = f"{name} ({confidence:.1f})"
                    cv2.putText(
                        frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )
            else:
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"\n[Frame {frame_count}] ⚠️  No faces detected - check lighting and camera angle")
            
            # Add text to frame
            cv2.putText(
                frame,
                "Press 'Q' to exit",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Show frame
            cv2.imshow('Face Recognition Debug', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 60)
        print("Debug session ended")

if __name__ == "__main__":
    os.chdir(_SCRIPT_DIR)
    debug_recognition()
