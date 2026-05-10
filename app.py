import streamlit as st
import cv2
import time
import tempfile
from ultralytics import YOLO
import os
from datetime import datetime
import easyocr
import re

# --- Page Config ---
st.set_page_config(page_title="Radar Detection System", page_icon="🚦", layout="wide")

st.title("🚦 Real-Time Radar & Traffic Detection")
st.markdown("Detects **Cars**, **Driver**, and **Car Plates** using our fine-tuned YOLO model.")

# --- Sidebar Configuration ---
st.sidebar.header("Settings")

# Input Source
source_option = st.sidebar.radio("Select Input Source:", ("Webcam", "Video File"))

# Confidence Threshold Slider
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# --- Model Loading ---
@st.cache_resource
def load_model():
    # Update this path if your model saved differently!
    model_path = 'runs/detect/radar_yolo_model/weights/best.pt'
    if not os.path.exists(model_path):
        st.warning(f"Custom weights not found at `{model_path}`. Falling back to default YOLOv8s.")
        return YOLO('yolov8s.pt')
    return YOLO(model_path)

model = load_model()

@st.cache_resource
def load_ocr():
    with st.spinner("Loading OCR Model (this takes a moment on first run)..."):
        return easyocr.Reader(['en'], gpu=True)

reader = load_ocr()

# --- Main App Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    stframe = st.empty()
    stop_button = st.button("🛑 Stop Video Process", key="stop_btn")

with col2:
    st.markdown("### 📸 Unique Plate Logs")
    log_placeholder = st.empty()

# Keep track of logged plates to avoid spamming
if 'logged_plates' not in st.session_state:
    st.session_state.logged_plates = []

def process_video(cap):
    prev_time = 0
    
    while cap.isOpened():
        # Short-circuit if stop is pressed (triggers a script rerun)
        if stop_button:
            break
            
        ret, frame = cap.read()
        if not ret:
            st.info("Video ended or cannot be read.")
            break
            
        # 1. Run YOLO Inference
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # 2. Extract plotted image (bounding boxes + labels + confidence)
        annotated_frame = results[0].plot()
        
        # 3. Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        # Overlay FPS on the top-left corner
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (15, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        
        # --- BONUS: Domain Specific Action (Automated Toll Logging & OCR) ---
        boxes = results[0].boxes
        current_time_str = datetime.now().strftime("%H:%M:%S")
        
        for box in boxes:
            if int(box.cls[0]) == 1: # Class 1 is car_plate
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    plate_crop = frame[y1:y2, x1:x2]
                    ocr_results = reader.readtext(plate_crop)
                    
                    for (bbox, text, prob) in ocr_results:
                        clean_text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
                        
                        if len(clean_text) >= 3 and not any(log.get('text') == clean_text for log in st.session_state.logged_plates):
                            st.session_state.logged_plates.append({'time': time.time(), 'stamp': current_time_str, 'text': clean_text})
                
        # Display the logs dynamically in the side column
        with log_placeholder.container():
            for log in reversed(st.session_state.logged_plates[-10:]):
                st.info(f"📟 **{log['text']}** \n\n📍 *Detected at {log['stamp']}*")

        # 4. Display Frame in Streamlit
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)

    cap.release()


# --- Execution Logic ---
if source_option == "Webcam":
    st.info("Click 'Start Webcam' to begin.")
    if st.button("Start Webcam"):
        # 0 is usually the default laptop webcam
        cap = cv2.VideoCapture(0)
        process_video(cap)

elif source_option == "Video File":
    uploaded_file = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'avi', 'mov'])
    if uploaded_file is not None:
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        if st.button("Process Video"):
            cap = cv2.VideoCapture(tfile.name)
            process_video(cap)
