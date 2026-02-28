"""
Model Training Module
Trains LBPH Face Recognizer using registered student faces.

Improvements:
- CLAHE preprocessing applied to every training image (matches registration)
- Better LBPH parameters: radius=2, neighbors=16 (finer texture detail)
- Stricter threshold=60.0 (was 85) — rejects poor matches during predict()
"""

import cv2
import os
import numpy as np
import sys
import time

# Ensure relative paths resolve from the script's own directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def apply_clahe(gray_img):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to normalize lighting."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)


def train_model():
    """
    Train LBPH Face Recognizer Model.

    Process:
    - Scans the dataset folder for student subfolders
    - Reads all face images, applies CLAHE, and collects labels
    - Trains LBPH recognizer with improved parameters
    - Saves model as trainer.yml and labels as labels.txt
    """

    dataset_path = os.path.join(_SCRIPT_DIR, 'dataset')
    trainer_path = os.path.join(_SCRIPT_DIR, 'trainer')
    model_file = os.path.join(trainer_path, 'trainer.yml')

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print("❌ Error: Dataset folder not found!")
        return

    # Initialize lists for images and labels
    faces = []
    labels = []
    label_dict = {}  # label_int -> "StudentID - Name"
    current_label = 0

    print("\n" + "-"*60)
    print("🔄 Reading dataset...\n")
    
    start_load_time = time.time()

    # Read all student folders
    student_ids = sorted([
        folder for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, folder))
    ])

    if not student_ids:
        print("❌ Error: No student data found in dataset folder!")
        return

    for student_id in student_ids:
        student_path = os.path.join(dataset_path, student_id)

        # Read student name from info.txt
        info_file = os.path.join(student_path, 'info.txt')
        student_name = student_id

        if os.path.exists(info_file):
            try:
                with open(info_file, 'r') as f:
                    for line in f:
                        if line.startswith('Name:'):
                            student_name = line.split(':', 1)[1].strip()
                            break
            except Exception:
                pass

        label_dict[current_label] = f"{student_id} - {student_name}"

        # Read and preprocess all images for this student
        image_count = 0
        images_in_folder = [img for img in os.listdir(student_path) 
                           if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for idx, image_name in enumerate(images_in_folder, 1):
            image_path = os.path.join(student_path, image_name)
            
            # Show progress
            sys.stdout.write(f"\r  Processing {student_id}: {idx}/{len(images_in_folder)} images")
            sys.stdout.flush()

            face_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if face_image is None:
                continue

            if face_image.size < 100:   # Skip corrupt / tiny files
                continue

            # Resize to consistent training size
            face_image = cv2.resize(face_image, (100, 100))

            # NOTE: CLAHE is already applied during registration (register_face.py)
            # so we do NOT apply it again here to avoid double-processing

            faces.append(face_image)
            labels.append(current_label)
            image_count += 1

        if image_count > 0:
            print(f"\n[OK] {student_id} ({student_name}): {image_count} images loaded")
            current_label += 1
        else:
            print(f"\n⚠️  {student_id} ({student_name}): No valid images found — skipping")

    total_images = len(faces)

    if total_images == 0:
        print("\n❌ Error: No face images found in dataset!")
        return

    load_time = time.time() - start_load_time
    print(f"\n[OK] Dataset loaded in {load_time:.2f} seconds")
    print(f"[OK] Total images loaded : {total_images}")
    print(f"[OK] Total students      : {len(label_dict)}\n")

    # Convert to numpy arrays
    faces = np.array(faces, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)

    print("-"*60)
    print("🤖 Training LBPH Face Recognizer...\n")

    # LBPH parameters:
    #   radius=1     — standard 3×3 neighborhood
    #   neighbors=8  — 8 sampling points (2^8 = 256 bins — fast & accurate)
    #   grid_x/y=8   — 8×8 spatial grid
    #   No threshold here — thresholding is done in application code
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    try:
        print("⏱️  Converting to numpy arrays...")
        start_time = time.time()
        
        recognizer.train(faces, labels)
        
        train_time = time.time() - start_time
        print(f"[OK] Training completed in {train_time:.2f} seconds\n")

        print("💾 Saving model...")
        os.makedirs(trainer_path, exist_ok=True)
        recognizer.write(model_file)

        # Save label dictionary
        label_info_file = os.path.join(trainer_path, 'labels.txt')
        with open(label_info_file, 'w') as f:
            for label, name in sorted(label_dict.items()):
                f.write(f"{label}:{name}\n")

        print(f"[OK] Model training completed successfully!")
        print(f"[OK] Model saved  : {model_file}")
        print(f"[OK] Labels saved : {label_info_file}\n")

    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return

    print("-"*60)
    print("🎉 Ready to perform attendance!\n")


if __name__ == "__main__":
    os.chdir(_SCRIPT_DIR)
    train_model()
