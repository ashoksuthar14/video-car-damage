import os
import time
import cv2
import torch
import av
import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
from car_pipeline import damage_cost_map, estimate_cost

st.set_page_config(page_title="Car Damage - WebRTC (Streamlit)", page_icon="🚗", layout="wide")
st.markdown("""
    <style>
    .site-header {text-align: center;font-size: 1.75em;font-weight: 700;background: #2246c5ef;
        color: #fff;padding: 20px 0 7px 0;box-shadow: 0 2px 18px #8da2ce26;border-radius: 0 0 24px 24px;}
    .footer {text-align:center; padding:12px 0 4px 0; color: #8ab2e2; font-size: 1em;}
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="site-header">AI Car Damage Detection — Live Camera</div>', unsafe_allow_html=True)
st.markdown('<div class="footer">Powered by <b>YOLOv8</b> + <b>Streamlit/webrtc</b> — 2024</div>', unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best (6).pt')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)

rtc_configuration = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
})

with st.sidebar:
    st.header("Settings")
    conf_thr = st.slider("Confidence threshold", 0.05, 0.75, 0.25, 0.05)
    imgsz = st.select_slider("Inference size", options=[320, 480, 640], value=320)
    mirror = st.checkbox("Mirror video", value=True)
    async_mode = st.selectbox("Async Processing", [True, False], index=0, help="Try both for best FPS on your machine.")

class DamageTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_cost = 0.0
        self._last_time = time.time()
        self._frame_counter = 0
        self.fps = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if mirror:
            img = cv2.flip(img, 1)
        work = np.ascontiguousarray(img)
        try:
            results = model.predict(work, conf=conf_thr, imgsz=imgsz, device=DEVICE, verbose=False)
        except TypeError:
            results = model.predict(work, conf=conf_thr, imgsz=imgsz, verbose=False)
        r0 = results[0] if len(results) else None
        annotated = r0.plot() if r0 is not None else work.copy()
        damage_info = []
        if r0 is not None and r0.boxes is not None:
            for box in r0.boxes:
                cls = int(box.cls.cpu().numpy())
                conf = float(box.conf.cpu().numpy())
                if cls in damage_cost_map:
                    damage_info.append({"class_id": cls, "confidence": conf})
        total_cost, _ = estimate_cost(damage_info)
        self.last_cost = float(total_cost)
        # FPS calculation
        self._frame_counter += 1
        now = time.time()
        if now - self._last_time >= 1.0:
            self.fps = self._frame_counter / (now - self._last_time)
            self._last_time = now
            self._frame_counter = 0
        # Overlay info
        textbg = (245,245,245)
        cv2.rectangle(annotated, (0, 0), (430, 46), textbg, -1)
        banner = f"Repair: AED {total_cost:.2f}" if total_cost > 0 else "AI: No Damage Detected"
        cv2.putText(annotated, banner, (10,32), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (0,60,230), 2)
        cv2.putText(annotated, f"FPS: {self.fps:.1f}", (300, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100,50,240), 2)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

col1, col2 = st.columns([4,2])
with col1:
    ctx = webrtc_streamer(
        key="damage-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=DamageTransformer,
        async_processing=async_mode,
    )
with col2:
    info = st.empty()
    st.caption("Tip: For smoothest experience, use default settings or set Detection Quality to 'Fast'.")
    while True:
        if ctx and ctx.video_transformer:
            info.metric("Estimated Repair Cost (AED)", f"{ctx.video_transformer.last_cost:.2f}")
        time.sleep(0.3)

