# AI Video Call with Real-Time Car Damage Detection

This project is a prototype video calling system (like Google Meet/Zoom) where every video frame that passes between users is analyzed for car damage by a YOLOv8 deep learning model, with live annotation!

**Features:**
- Real-time video calls in browser (WebRTC)
- Live damage detection overlay on all frames using YOLOv8 (Ultralytics)
- Backend in Python (Flask, Socket.IO, aiortc, AV, OpenCV, Ultralytics)
- Frontend in pure HTML/JS (getUserMedia + WebRTC + socket.io.js)

## How it Works
- User A and User B both open the client in the browser.
- They click "Start Call", which connects to the backend using WebRTC, exchanging SDP/ICE via Socket.IO for signaling.
- All local video sent to the server is processed live by the YOLOv8 model and annotated before being bounced back as remote video.

## Dependencies
```
pip install -r requirements.txt
```

## Running
- Start backend: `python app.py` (or as instructed)
- Open `http://localhost:5000/` in two browser tabs or two computers.
- Click "Start Call" and see the detected damages appear in real time!

## Customization
- Replace `best (6).pt` with your own YOLOv8 weights for your car damage dataset.

_This is a research/developer prototype!_

## Alternative: Streamlit WebRTC (recommended if Flask/aiortc detects nothing)
- Install deps: `pip install -r requirements.txt`
- Run: `streamlit run webrtc_streamlit_app.py`
- In the sidebar, adjust confidence and inference size if needed. You should see boxes and a banner overlay.
