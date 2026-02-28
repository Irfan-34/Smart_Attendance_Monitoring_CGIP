"""
Smart Attendance System - Streamlit Web Dashboard
Provides a modern UI for Registration, Live Attendance, and Admin Logs.
Communicates with the FastAPI backend.
"""

import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io

# API Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Smart Attendance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a beautiful modern design
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .success { background-color: rgba(0, 255, 0, 0.1); border: 1px solid #00ff00; }
    .error { background-color: rgba(255, 0, 0, 0.1); border: 1px solid #ff0000; }
    .warning { background-color: rgba(255, 165, 0, 0.1); border: 1px solid #ffa500; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------------
st.sidebar.title("🎓 Smart Attendance")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["📸 Live View", "📝 Registration", "📊 Admin Logs"])
st.sidebar.markdown("---")

# Test backend connection
try:
    requests.get(f"{API_URL}/health", timeout=2)
    st.sidebar.success("Backend: ONLINE", icon="🟢")
except Exception:
    st.sidebar.error("Backend: OFFLINE", icon="🔴")
    st.sidebar.warning("Please start `python server.py`")

# -------------------------------------------------------------------
# Page 1: Live View (Attendance)
# -------------------------------------------------------------------
if page == "📸 Live View":
    st.title("Live Attendance Tracker")
    st.markdown("Look at the camera and snap a photo to mark your attendance.")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Streamlit Native Camera Input
        camera_image = st.camera_input("Take a picture")

    with col2:
        st.subheader("Attendance Status")
        status_placeholder = st.empty()
        
    if camera_image is not None:
        with col2:
            with st.spinner("Analyzing face..."):
                try:
                    # Convert Streamlit image to bytes for the API
                    files = {"file": ("image.jpg", camera_image.getvalue(), "image/jpeg")}
                    
                    response = requests.post(f"{API_URL}/recognize", files=files)
                    data = response.json()
                    
                    if response.status_code == 200:
                        if data.get("status") in ["error", "spoof", "unknown"]:
                            st.markdown(f"""
                            <div class="status-box error" style="margin-top: 10px;">
                                <h3>❌ {data.get("status").upper()}</h3>
                                <p>{data.get('message', 'Failed to process request.')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            if "faces_detected" in data:
                                st.info(f"Detected {data['faces_detected']} face(s) in frame.")
                            
                        # Loop through all processed faces
                        for face_res in data.get("results", []):
                            if face_res["status"] == "success":
                                st.markdown(f"""
                                <div class="status-box success" style="margin-top: 10px;">
                                    <h3>✅ Recognized!</h3>
                                    <h2>{face_res['student']['name']}</h2>
                                    <p>ID: {face_res['student']['id']}</p>
                                    <p>Confidence Distance: {face_res['distance']:.3f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            elif face_res["status"] == "duplicate":
                                st.markdown(f"""
                                <div class="status-box warning" style="margin-top: 10px;">
                                    <h3>⚠️ Already Present</h3>
                                    <h2>{face_res['student']['name']}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            else: # unknown
                                st.markdown(f"""
                                <div class="status-box error" style="margin-top: 10px;">
                                    <h3>❌ Not Recognized</h3>
                                    <p>{face_res['message']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        error_msg = data.get("detail", "Unknown error")
                        st.error(f"Error: {error_msg}")
                        
                except Exception as e:
                    status_placeholder.error(f"Failed to connect to backend: {e}")

# -------------------------------------------------------------------
# Page 2: Registration
# -------------------------------------------------------------------
elif page == "📝 Registration":
    st.title("Register New Student")
    st.markdown("Fill out the details below and take a clear photo.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        student_id = st.text_input("Student ID (e.g., 1001)")
        student_name = st.text_input("Full Name")
        
    with col2:
        reg_image = st.camera_input("Capture Reference Photo")
        
    if st.button("Register Student", use_container_width=True):
        if not student_id or not student_name:
            st.error("Please fill in both ID and Name.")
        elif reg_image is None:
            st.error("Please take a reference photo.")
        else:
            with st.spinner("Extracting facial embeddings..."):
                try:
                    files = {"file": ("image.jpg", reg_image.getvalue(), "image/jpeg")}
                    data = {
                        "student_id": student_id,
                        "name": student_name
                    }
                    response = requests.post(f"{API_URL}/register", data=data, files=files)
                    
                    if response.status_code == 200:
                        st.success(f"✅ Successfully registered {student_name}!")
                        st.balloons()
                    else:
                        st.error(f"❌ Error: {response.json().get('detail', 'Registration failed')}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

# -------------------------------------------------------------------
# Page 3: Admin Logs
# -------------------------------------------------------------------
elif page == "📊 Admin Logs":
    st.title("Attendance Records")
    
    cols = st.columns([1, 1, 2])
    with cols[0]:
        filter_date = st.date_input("Filter by Date")
    with cols[1]:
        if st.button("🔄 Refresh Data"):
            st.rerun()
            
    try:
        url = f"{API_URL}/attendance"
        if filter_date:
            url += f"?date={filter_date.strftime('%Y-%m-%d')}"
            
        response = requests.get(url)
        
        if response.status_code == 200:
            logs = response.json()["logs"]
            if not logs:
                st.info("No attendance records found.")
            else:
                df = pd.DataFrame(logs)
                df = df.rename(columns={
                    "student_id": "ID",
                    "name": "Name",
                    "date": "Date",
                    "time": "Time",
                    "distance": "AI Confidence (Distance)"
                })
                # Reorder columns
                df = df[["Date", "Time", "ID", "Name", "AI Confidence (Distance)"]]
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Simple analytics
                st.markdown("### 📈 Daily Statistics")
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric("Total Present Today", len(df["ID"].unique()))
                metric_col2.metric("Total Scans", len(df))
                
    except Exception as e:
        st.error(f"Failed to fetch logs: {e}")
