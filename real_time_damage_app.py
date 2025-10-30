import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
from car_pipeline import damage_cost_map, estimate_cost

st.set_page_config(
    page_title="Real-Time Car Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚗 Real-Time Vehicle Damage Detection & Cost Estimation")
st.write("This app uses your webcam to analyze car damages in real-time and provides AED estimates using AI.")

run_detection = st.button("Start Webcam Detection")
stop_detection = st.button("Stop Detection")
status_placeholder = st.empty()

model = YOLO('best (6).pt')  # Match your custom weights
cap = None

if run_detection:
    cap = cv2.VideoCapture(0)
    detection_active = True
    status_placeholder.success("Webcam started.")
    fps_time = time.time()

    # Create 2 columns: small left for camera, wide right for results
    col1, col2 = st.columns([1,2], gap="large")
    with col1:
        webcam_placeholder = st.empty()
    with col2:
        results_placeholder = st.empty()

    while detection_active:
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("Failed to get frame from webcam.")
            break

        # Resize webcam frame small to fit left box (e.g., quarter size)
        frame_resized = cv2.resize(frame, (320, 240))
        # Run YOLO prediction
        results = model(frame_resized)
        detections = results[0].boxes
        damage_info = []
        for box in detections:
            class_id = int(box.cls.cpu().numpy())
            confidence = float(box.conf.cpu().numpy())
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].cpu().numpy())
            label = f"{damage_cost_map[class_id][0].replace('_',' ').title()} ({confidence*100:.1f}%)"
            color = (255, 0, 0)
            cv2.rectangle(frame_resized, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame_resized, label, (x_min, max(y_min - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            damage_info.append({
                'class_id': class_id,
                'confidence': confidence
            })
        # Estimate cost for the frame
        total_cost, cost_breakdown = estimate_cost(damage_info)
        cost_lines = [f"<li><strong>{c['type'].replace('_',' ').title()}</strong>: AED {c['estimated_cost']:.2f}</li>" for c in cost_breakdown]
        # Convert BGR to RGB for display
        st_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        # Display in left column
        webcam_placeholder.image(st_image, channels="RGB", width=320, caption="Webcam Feed")
        # Display detection & cost in right column
        results_html = f"""
        <h4>Frame Repair Estimate</h4>
        <p style='font-size:1.15em'><strong>Total (AED):</strong> <span style='color:#0d6efd;font-weight:bold;'>{total_cost:.2f}</span></p>
        <ul style='font-size:1em'>{''.join(cost_lines) if cost_lines else '<li>No damage detected.</li>'}</ul>
        """
        results_placeholder.markdown(results_html, unsafe_allow_html=True)
        # Throttle to 10 FPS for responsiveness
        time_elapsed = time.time() - fps_time
        if time_elapsed < 0.1:
            time.sleep(0.1 - time_elapsed)
        fps_time = time.time()
        if stop_detection:
            detection_active = False
            break
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    status_placeholder.info("Detection stopped.")
