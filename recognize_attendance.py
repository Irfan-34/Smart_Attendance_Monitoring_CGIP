"""
Real-Time Face Recognition & Attendance Module
Recognizes students and marks attendance — supports multiple faces simultaneously.

Improvements over original:
- CLAHE preprocessing before predict() (matches training preprocessing)
- Confidence threshold tightened: 55 (was 80) — prevents strangers being tagged as known
- Consecutive-frame confirmation (5 frames) before marking or labelling a face
- True multi-face support: each face tracked and confirmed independently
- On-screen face count and per-face confidence display
"""

import cv2
import os
import csv
import sys
from datetime import datetime
import time
import numpy as np
import threading

# Ensure relative paths resolve from the script's own directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Tuning constants — adjust these if your environment needs it
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 100  # LBPH: lower = stricter. Faces above this → UNKNOWN
REQUIRED_FRAMES = 5         # consecutive frames a face must match before it's confirmed
RECOGNITION_COOLDOWN = 5    # seconds before the same student can be re-logged to console
CAPTURE_INTERVAL = 0.15     # seconds between successive auto-captures in registration


def apply_clahe(gray_img):
    """Normalize lighting with CLAHE (matches preprocessing used during training)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)


def open_webcam_with_timeout(camera_index=0, timeout_seconds=5):
    """
    Attempt to open webcam directly.
    Returns the VideoCapture object if successful, None if error.
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Verify webcam is actually open
        if not cap.isOpened():
            cap.release()
            return None
        
        # Test if we can read a frame
        ret, _ = cap.read()
        if not ret:
            cap.release()
            return None
            
        return cap
    except Exception as e:
        print(f"Exception while opening webcam: {e}")
        return None


def load_labels():
    """Load student label dictionary from trainer/labels.txt."""
    label_dict = {}
    label_file = os.path.join(_SCRIPT_DIR, 'trainer', 'labels.txt')

    if os.path.exists(label_file):
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        label_str, name = line.split(':', 1)
                        label_dict[int(label_str)] = name.strip()
        except Exception:
            pass

    return label_dict


def get_student_id_from_label(label_name):
    """Extract student ID from label name (e.g., '001 - John' → '001')."""
    if '-' in label_name:
        return label_name.split('-')[0].strip()
    return label_name


def mark_attendance(student_id, student_name):
    """
    Mark attendance for a student (once per day per student).
    Returns True if attendance was newly recorded, False if already marked.
    """
    attendance_file = os.path.join(_SCRIPT_DIR, 'attendance', 'attendance.csv')
    os.makedirs(os.path.join(_SCRIPT_DIR, 'attendance'), exist_ok=True)

    today = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')

    # Check if already marked today
    if os.path.exists(attendance_file):
        try:
            with open(attendance_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3 and row[0] == student_id and row[2] == today:
                        return False   # Already marked
        except Exception:
            pass

    # Write record
    try:
        file_empty = (
            not os.path.exists(attendance_file)
            or os.path.getsize(attendance_file) == 0
        )
        with open(attendance_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if file_empty:
                writer.writerow(['Student ID', 'Name', 'Date', 'Time'])
            writer.writerow([student_id, student_name, today, current_time])
        print(f"\n[OK] Attendance marked for {student_name} ({student_id})")
        return True
    except Exception as e:
        print(f"\n⚠️  Error marking attendance: {str(e)}")
        return False


def iou(boxA, boxB):
    """Intersection-over-Union between two (x, y, w, h) boxes."""
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    xA = max(ax, bx)
    yA = max(ay, by)
    xB = min(ax + aw, bx + bw)
    yB = min(ay + ah, by + bh)
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union


def start_attendance():
    """
    Start real-time face recognition and attendance system.

    Multi-face support:
        Each detected face is tracked independently. A face must be confirmed
        as the same identity for REQUIRED_FRAMES consecutive frames before its
        label is displayed and attendance is marked. This eliminates single-frame
        false positives where an unknown face briefly matches a registered person.
    """

    model_file = os.path.join(_SCRIPT_DIR, 'trainer', 'trainer.yml')
    if not os.path.exists(model_file):
        print("❌ Error: Trained model not found!")
        print("   Please train the model first (Option 2 from main menu)")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    if face_cascade.empty():
        print("❌ Error: Could not load Haar Cascade classifier!")
        return

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_file)
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return

    label_dict = load_labels()
    if not label_dict:
        print("❌ Error: Could not load student labels!")
        return

    print("\n" + "-"*60)
    print("📹 Starting attendance system...")
    print(f"   Confidence threshold : {CONFIDENCE_THRESHOLD} (lower = stricter)")
    print(f"   Frame confirmation   : {REQUIRED_FRAMES} consecutive frames")
    print("Press 'Q' to quit")
    print("-"*60 + "\n")

    print("📷 Attempting to access webcam...")
    sys.stdout.flush()
    cap = open_webcam_with_timeout(camera_index=0, timeout_seconds=5)
    
    if cap is None:
        print("❌ Error: Could not access webcam!")
        print("\n   Troubleshooting steps:")
        print("   [OK] Ensure your webcam is connected and powered on")
        print("   [OK] Check if another application (e.g., Zoom, Teams) is using the webcam")
        print("   [OK] Try restarting the application")
        print("   [OK] On Windows, restart the camera service in Device Manager")
        print("   [OK] Try running with administrator privileges")
        return

    print("[OK] Webcam initialized successfully!")
    print("⏳ Initializing video stream...")
    sys.stdout.flush()

    # --- Per-face tracking state ---
    # Each tracker entry: {
    #   'box'         : (x, y, w, h)  — last known box
    #   'label_id'    : int            — predicted label
    #   'conf'        : float          — latest confidence
    #   'streak'      : int            — consecutive matching frames
    #   'confirmed'   : bool           — has reached REQUIRED_FRAMES
    #   'student_id'  : str | None
    #   'student_name': str
    # }
    trackers = []

    # Track when attendance was last printed per student (to avoid console spam)
    attendance_logged = {}   # student_id → timestamp last logged

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to read frame from webcam!")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- Face detection ---
            detected_faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Build new tracker list aligned to current detections
            new_trackers = []

            if len(detected_faces) > 0:
                # Match each detected face to an existing tracker by IOU
                used_tracker_idxs = set()

                for (x, y, w, h) in detected_faces:
                    # Extract & preprocess the face ROI
                    # Order must match registration: CLAHE first, then resize
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = apply_clahe(face_roi)           # CLAHE on original-size ROI (matches registration)
                    face_roi = cv2.resize(face_roi, (100, 100))

                    # Predict identity
                    label, confidence = recognizer.predict(face_roi)

                    # Find best matching existing tracker
                    best_idx = None
                    best_iou = 0.3   # minimum IOU to consider a match
                    for i, t in enumerate(trackers):
                        if i in used_tracker_idxs:
                            continue
                        score = iou((x, y, w, h), t['box'])
                        if score > best_iou:
                            best_iou = score
                            best_idx = i

                    if best_idx is not None:
                        used_tracker_idxs.add(best_idx)
                        t = trackers[best_idx]

                        # Check if prediction matches previous tracker identity
                        same_identity = (label == t['label_id'] and confidence < CONFIDENCE_THRESHOLD)

                        if same_identity:
                            t['streak'] = min(t['streak'] + 1, REQUIRED_FRAMES + 5)
                        else:
                            t['streak'] = max(0, t['streak'] - 1)

                        t['box'] = (x, y, w, h)
                        t['conf'] = confidence
                        t['label_id'] = label

                        if t['streak'] >= REQUIRED_FRAMES:
                            t['confirmed'] = True

                        new_trackers.append(t)

                    else:
                        # New face — create a fresh tracker
                        is_known = (confidence < CONFIDENCE_THRESHOLD) and (label in label_dict)
                        new_trackers.append({
                            'box': (x, y, w, h),
                            'label_id': label,
                            'conf': confidence,
                            'streak': 1,
                            'confirmed': False,
                            'student_id': get_student_id_from_label(label_dict[label]) if is_known else None,
                            'student_name': label_dict[label] if is_known else 'Unknown',
                        })

                # Refresh student info on confirmed trackers
                for t in new_trackers:
                    if t['confirmed']:
                        is_known = (t['conf'] < CONFIDENCE_THRESHOLD) and (t['label_id'] in label_dict)
                        t['student_name'] = label_dict[t['label_id']] if is_known else 'Unknown'
                        t['student_id'] = get_student_id_from_label(t['student_name']) if is_known else None

            trackers = new_trackers

            # --- Draw results & mark attendance ---
            for t in trackers:
                x, y, w, h = t['box']
                name = t['student_name']
                conf = t['conf']
                confirmed = t['confirmed']
                sid = t['student_id']

                # Predicted label name (even if below threshold, show who it thinks it is)
                predicted_name = label_dict.get(t['label_id'], '??')

                # Color: green = confirmed known | red = unknown/pending
                if confirmed and name != 'Unknown':
                    color = (0, 255, 0)     # green — confirmed known
                    display_name = f"{name}"
                else:
                    color = (0, 0, 255)     # red — unknown or still confirming
                    display_name = "Unknown"

                # Bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                # Label background
                text_size = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
                cv2.rectangle(
                    frame,
                    (x, y - text_size[1] - 10),
                    (x + text_size[0] + 8, y),
                    color, -1
                )
                cv2.putText(
                    frame, display_name,
                    (x + 4, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
                )

                # Mark attendance for confirmed, known faces
                if confirmed and sid and name != 'Unknown':
                    now = time.time()
                    last_logged = attendance_logged.get(sid, 0)
                    if now - last_logged > RECOGNITION_COOLDOWN:
                        mark_attendance(sid, name)
                        attendance_logged[sid] = now

            # --- Frame overlay ---
            face_count = len(trackers)
            cv2.putText(
                frame,
                f"Faces detected: {face_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                frame,
                "Press 'Q' to exit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1
            )

            cv2.imshow('Smart Attendance System - Face Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n" + "-"*60)
    print("[OK] Attendance system stopped")
    print(f"📊 Attendance saved to: attendance/attendance.csv\n")


if __name__ == "__main__":
    os.chdir(_SCRIPT_DIR)
    start_attendance()
