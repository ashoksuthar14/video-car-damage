import streamlit as st
import cv2
from ultralytics import YOLO
import time
from car_pipeline import damage_cost_map, estimate_cost

# CSS for card UI
st.markdown("""
    <style>
    .site-header {text-align: center;font-size: 1.85em;font-weight: 700;background: #2246c5ef;
        color: #fff;padding: 25px 0 13px 0;box-shadow: 0 2px 18px #8da2ce26;border-radius: 0 0 28px 28px;}
    .container {display: flex;justify-content: center;align-items: flex-start;gap: 55px;margin: 20px 0 24px 0;}
    .card {background: #fff;border-radius: 20px;
        box-shadow: 0 6px 32px #6c789966, 0 1.5px 8px #bfcadf12;padding: 28px 18px 28px 18px;
        display: flex;flex-direction: column;align-items: center;}
    .video-box {background: #f7f8fa;border-radius: 17px; width: 382px; height: 290px;
        box-shadow: 0 0.7px 9px #bbb5;position: relative;margin-bottom: 8px;}
    .vid-label {position: absolute; left: 17px; bottom: 16px; background: #1949dddf;
        color: #fff; border-radius: 7px; font-size: 1.08em; padding: 4px 19px;
        letter-spacing: .03em; font-weight: 460; box-shadow: 0 2px 6px #0002;}
    .footer {text-align:center; padding:16px 0 6px 0; color: #8ab2e2; font-size: 1em; letter-spacing:.12em;margin-bottom:0;}
    .right-top {position: absolute; right: 16px; top: 16px; z-index: 5;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="site-header">AI Car Damage Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="footer">Powered by <b>YOLOv8</b> + <b>Streamlit</b> &mdash; 2024</div>', unsafe_allow_html=True)

# Session state for clean control
if 'running' not in st.session_state:
    st.session_state['running'] = False
status_placeholder = st.empty()
col1, col2 = st.columns([1, 1], gap="large")

if not st.session_state['running']:
    # Only render the start button when not running
    if st.button("Start Webcam Detection", key="start_button"):
        st.session_state['running'] = True
        st.rerun()
else:
    stop_btn = st.button("Stop Detection", key="stop_button", help="Stops webcam detection")
    if stop_btn:
        st.session_state['running'] = False
        st.rerun()
    status_placeholder.success("Webcam started.")
    cap = cv2.VideoCapture(0)
    detection_active = True
    while detection_active and st.session_state['running']:
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("Failed to get frame from webcam.")
            break
        frame_resized = cv2.resize(frame, (320, 240))
        results = YOLO('best (6).pt')(frame_resized)
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
            damage_info.append({'class_id': class_id, 'confidence': confidence})
        total_cost, cost_breakdown = estimate_cost(damage_info)
        cost_lines = [f"<li><strong>{c['type'].replace('_',' ').title()}</strong>: AED {c['estimated_cost']:.2f}</li>" for c in cost_breakdown]
        st_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        with col1:
            st.markdown('<div class="card"><div class="video-box">', unsafe_allow_html=True)
            st.image(st_image, channels="RGB", width=380, caption="Webcam Feed")
            st.markdown('<div class="vid-label">Live Detection</div></div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card"><div class="video-box">', unsafe_allow_html=True)
            st.markdown(
                f"""
                <h4>Frame Repair Estimate</h4>
                <p style='font-size:1.15em'><strong>Total (AED):</strong> <span style='color:#0d6efd;font-weight:bold;'>{total_cost:.2f}</span></p>
                <ul style='font-size:1em'>{''.join(cost_lines) if cost_lines else '<li>No damage detected.</li>'}</ul>
                """, unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
        # Throttle to ~10 fps
        time.sleep(0.10)
        # If stop button pressed, break early
        if not st.session_state['running']:
            detection_active = False
            break
    cap.release()
    cv2.destroyAllWindows()
    status_placeholder.info("Detection stopped.")
