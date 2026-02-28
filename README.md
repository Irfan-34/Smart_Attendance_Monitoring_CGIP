# 🎓 Smart Attendance System using Face Recognition

This is a **Face Recognition-based Attendance System** built with Python and OpenCV. The system automatically detects, recognizes students' faces via webcam, and logs their daily attendance into an organized digital record (CSV).

This project demonstrates practical applications of Computer Vision & Image Processing (CVIP) by automating manual roll calls, reducing time waste, and preventing proxy attendance.

## ✨ Features
- **Centralized CLI Dashboard**: A unified `main.py` entry point to manage everything.
- **Student Registration**: Auto-capture 60 training images from various head poses directly via webcam.
- **Model Training**: Employs Haar Cascade for face detection and OpenCV's LBPH (Local Binary Patterns Histograms) Face Recognizer for training.
- **Real-Time Recognition Multi-tracker**: Effectively scans multiple faces concurrently, requiring continuous frame consistency to reduce false positives.
- **Automated Logging**: Logs recognized students directly into an `attendance.csv` file automatically (and restricts it to once per day per student).
- **Diagnostics Tools**: Built-in scripts like `test_webcam.py` and `debug_recognition.py` for troubleshooting.

## ⚙️ Tech Stack
- **Python** (Core Language)
- **OpenCV (`opencv-python`, `opencv-contrib-python`)** (Face Detection, Image Processing, real-time bounding boxes)
- **Numpy** (Matrix / Array manipulations for image data)

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/YOUR-USERNAME/Smart_Attendance_Monitoring_CGIP.git
cd Smart_Attendance_Monitoring_CGIP
pip install -r requirements.txt
```

### 2. Validating System (Optional)
Run the diagnostic script to ensure your camera and system structure are accessible.
```bash
python quick_start.py
```

### 3. Start the Application
Boot up the main control panel:
```bash
python main.py
```

From the Main Menu, run through the workflow linearly:
1. **Register New Student (Option 1):** Follow prompts to type ID and Name. Look at the camera to collect dataset faces. 
2. **Train Model (Option 2):** Compiles the registered datasets into `trainer.yml` for the AI to benchmark against.
3. **Start Attendance (Option 3):** Launch real-time detection. When a face matches the model, it draws a green box, marks attendance, and pushes the record safely to the CSV! 

Press *'Q'* at any time to close an active camera window.

## 📂 Project Structure
```text
Smart_Attendance_Monitoring_CGIP/
│
├── main.py                  # Centralized CLI application logic
├── register_face.py         # Capture face images & assigns labels
├── train_model.py           # Preprocesses images & trains LBPH logic
├── recognize_attendance.py  # Performs real-time recognition loop
├── debug_recognition.py     # Diagnostics tool to see live confidence figures
│
├── requirements.txt         # Required Python packages
├── dataset/                 # Stores raw .jpg imagery per student ID
├── trainer/                 # Contains compiled .yml learning model
└── attendance/              # Output directory for daily attendance.csv
```

## 🧠 How it Works Under the Hood
This project uses **Haar Cascades** for fast initial bounding-box detection in frames. Captured regions of interest (ROI) are converted to grayscale and normalized. 

During the recognition loop, the **LBPH Recognizer** attempts to match real-time ROI histograms against the trained `trainer.yml` weights. If the confidence distance figure falls safely below the target threshold for consecutive frames, the subject identity is confirmed!

## 📝 License
Feel free to use for educational and structural inspiration!
