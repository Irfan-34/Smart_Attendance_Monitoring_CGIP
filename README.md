# Smart Attendance System (Production Release)

This system has been upgraded from a basic OpenCV Haar Cascades script to a robust, scalable microservices architecture utilizing Deep Learning, Vector Databases, and WebRTC.

## 🚀 Architecture Highlights

1. **Backend API (FastAPI)**: Handles all core logic, database transactions, and model inferences natively on port `8000`.
2. **Deep Learning Engine (DeepFace)**: Uses `RetinaFace` for ultra-accurate multi-face detection and `ArcFace` to generate 512-dimensional facial embeddings.
3. **Storage (SQLite & ChromaDB)**: Relational metadata and attendance logs are securely stored in SQLite, while facial signatures are indexed in ChromaDB for instantaneous Cosine Similarity retrieval.
4. **Anti-Spoofing**: Lightweight heuristic liveness check ensures that flat 2D representations (photos) are blocked.
5. **Frontend Dashboard (Streamlit)**: A modern, real-time web interface running on port `8501`. Features live webcam streaming, dual-status dynamic attendance cards, and admin dashboard rendering.

---

## 🛠️ How to Run in Production

You need to run two separate processes to start the system: the **Backend API** and the **Frontend Dashboard**.

### 1. Install Dependencies
Make sure your environment is activated and install the updated requirements:
```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Backend
The backend engine must be running for the frontend to work.
```bash
# Run this in Terminal 1
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```
*The API Swagger Docs will be available at: http://localhost:8000/docs*

### 3. Start the Streamlit Frontend
Once the backend is live, launch the user interface.
```bash
# Run this in Terminal 2
python -m streamlit run app.py
```
*The Web App will be available at: http://localhost:8501*

---

## 📝 Features & Usage
- **Live View**: Stand in front of the camera. The system will detect multiple faces simultaneously, verify liveness, compute geographic vectors, and explicitly render individual status cards (✅ Recognized, ⚠️ Already Present, ❌ Not Recognized).
- **Registration**: Navigate to the *"📝 Registration"* tab, enter a Student ID and Name, snap a clear photo, and their ArcFace embedding will be instantly stored in ChromaDB.
- **Admin Logs**: Navigate to the *"📊 Admin Logs"* tab to view and filter the SQLite attendance records for any specific day.
