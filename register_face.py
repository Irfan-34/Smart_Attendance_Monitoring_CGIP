"""
Face Registration Module
Captures and stores multiple face images for each student with lighting normalization.

Improvements:
- 60 samples (was 30) for a richer training dataset
- CLAHE preprocessing on every saved image for lighting consistency
- Auto-capture mode — no need to press SPACE repeatedly
- Pose-variation prompts: eyes level, left tilt, right tilt for better generalization
"""

import cv2
import os
import time

# Ensure relative paths resolve from the script's own directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def apply_clahe(gray_img):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to normalize lighting."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)


def register_student():
    """
    Register a new student by capturing face images.

    Process:
    - Takes student ID and name as input
    - Creates a folder with student ID
    - Auto-captures 60 face samples (20 per pose) using webcam
    - Applies CLAHE to every saved image for consistent lighting
    """

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    if face_cascade.empty():
        print("❌ Error: Could not load Haar Cascade classifier!")
        return

    # Get student information
    print("\n" + "-"*60)
    print("[*] STUDENT REGISTRATION FORM")
    print("-"*60)

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

    # Create dataset directory for this student
    dataset_path = os.path.join(_SCRIPT_DIR, 'dataset', student_id)

    # Check if student already exists
    if os.path.exists(dataset_path):
        confirm = input(f"\n⚠️  Student ID {student_id} already exists. Overwrite? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Registration cancelled.")
            return
        # Remove old images
        for f in os.listdir(dataset_path):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                os.remove(os.path.join(dataset_path, f))
    else:
        os.makedirs(dataset_path, exist_ok=True)

    # Create a text file to store student info
    info_file = os.path.join(dataset_path, 'info.txt')
    with open(info_file, 'w') as f:
        f.write(f"Student ID: {student_id}\n")
        f.write(f"Name: {student_name}\n")

    # Pose phases: each phase has a prompt and target sample count
    samples_needed = 60
    samples_per_pose = samples_needed // 3
    pose_prompts = [
        "Look straight at the camera",
        "Tilt head slightly LEFT",
        "Tilt head slightly RIGHT",
    ]

    print("\n" + "-"*60)
    print(f"[CAM] Auto-capturing samples for {student_name}")
    print("  Auto-capture is ON — just hold each pose when prompted.")
    print("  Press 'Q' to quit early.")
    print("-"*60 + "\n")

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not access webcam!")
        print("   Try these solutions:")
        print("   1. Close any other application using the webcam")
        print("   2. Check if webcam is properly connected")
        print("   3. Check camera permissions in Windows settings")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    count = 0
    current_pose = 0
    pose_count = 0  # samples captured for this pose
    # Delay between auto-captures to avoid duplicate frames (seconds)
    capture_interval = 0.15
    last_capture_time = 0

    try:
        while count < samples_needed:
            ret, frame = cap.read()

            if not ret:
                print("❌ Error: Failed to read frame from webcam!")
                break

            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=(60, 60)
            )

            # Determine current pose prompt
            pose_text = pose_prompts[current_pose] if current_pose < len(pose_prompts) else "Look straight"

            # --- Auto-capture logic ---
            now = time.time()
            if len(faces) > 0 and (now - last_capture_time) >= capture_interval:
                # Pick the largest face
                (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                face_roi = gray[y:y+h, x:x+w]

                # Apply CLAHE for lighting normalization
                face_roi = apply_clahe(face_roi)

                # Resize to standard training size
                face_roi = cv2.resize(face_roi, (100, 100))

                image_path = os.path.join(dataset_path, f"{student_id}_{count}.jpg")
                cv2.imwrite(image_path, face_roi)

                count += 1
                pose_count += 1
                last_capture_time = now

                # Advance to next pose phase after enough samples
                if pose_count >= samples_per_pose and current_pose < len(pose_prompts) - 1:
                    current_pose += 1
                    pose_count = 0
                    print(f"  ➡️  Next pose: {pose_prompts[current_pose]}")

            # ---- Overlay UI ----
            if len(faces) > 0:
                (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                status_text = f"Face detected  [{count}/{samples_needed}]"
                status_color = (0, 255, 0)
            else:
                status_text = "No face detected — move closer or adjust lighting"
                status_color = (0, 0, 255)

            # Sample counter
            cv2.putText(frame, f"Samples: {count}/{samples_needed}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Status
            cv2.putText(frame, status_text, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

            # Pose prompt
            cv2.putText(frame, f"Pose: {pose_text}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

            # Instructions
            cv2.putText(frame, "Auto-capturing  |  Press Q to exit",
                        (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow(f'Registering: {student_name} ({student_id})', frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[*] Capture cancelled by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if count > 0:
        print(f"\n[OK] Registered {student_name} ({student_id}) — {count} samples saved.")
        print(f"📁 Images saved in: dataset/{student_id}\n")
    else:
        print(f"\n⚠️  No samples captured for {student_name}.")


if __name__ == "__main__":
    os.chdir(_SCRIPT_DIR)
    register_student()
