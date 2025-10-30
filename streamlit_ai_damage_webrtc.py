import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
from ultralytics import YOLO
import av
from car_pipeline import damage_cost_map, estimate_cost

# Card-style UI header
st.markdown("""
    <style>
    .site-header {text-align: center;font-size: 1.85em;font-weight: 700;background: #2246c5ef;
        color: #fff;padding: 25px 0 13px 0;box-shadow: 0 2px 18px #8da2ce26;border-radius: 0 0 28px 28px;}
    .footer {text-align:center; padding:16px 0 6px 0; color: #8ab2e2; font-size: 1em; letter-spacing:.12em;margin-bottom:0;}
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Live Damage Detection", page_icon="🚗")
st.markdown('<div class="site-header">AI Car Damage Detection — Live Camera</div>', unsafe_allow_html=True)
st.markdown('<div class="footer">Powered by <b>YOLOv8</b> + <b>Streamlit/webrtc</b> — 2024</div>', unsafe_allow_html=True)

model = YOLO('best (6).pt')

class DamageProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (320, 240))
        results = model(img_resized)
        detections = results[0].boxes
        damage_info = []
        for box in detections:
            class_id = int(box.cls.cpu().numpy())
            confidence = float(box.conf.cpu().numpy())
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].cpu().numpy())
            label = f"{damage_cost_map[class_id][0].replace('_',' ').title()} ({confidence*100:.1f}%)"
            color = (255, 0, 0)
            cv2.rectangle(img_resized, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(img_resized, label, (x_min, max(y_min - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            damage_info.append({'class_id': class_id, 'confidence': confidence})
        total_cost, _ = estimate_cost(damage_info)
        cv2.rectangle(img_resized, (0, 0), (200, 30), (245,245,245), -1)
        cv2.putText(img_resized, f"Repair: AED {total_cost:.2f}", (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,60,230), 2)
        return av.VideoFrame.from_ndarray(img_resized, format="bgr24")

webrtc_streamer(key="ai-damage", video_processor_factory=DamageProcessor)
