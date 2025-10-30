import asyncio
import logging
import os
from flask import Blueprint, render_template, send_from_directory, request
from flask_socketio import emit, SocketIO
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from aiortc.sdp import candidate_from_sdp
import av
from ultralytics import YOLO
import cv2
import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-damage-backend")

webrtc_blueprint = Blueprint('webrtc', __name__)
relay = MediaRelay()

# Model and device setup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best (6).pt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH)

# Connection management
pcs = set()
peer_map = {}

class YoloVideoTrack(VideoStreamTrack):
    """Real-time YOLO video processing track"""
    kind = "video"
    
    def __init__(self, track):
        super().__init__()
        self.track = track
        
    async def recv(self):
        # Get frame from incoming track
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        
        # Run YOLO inference - simplified for debugging
        try:
            # Use direct call with minimal parameters
            results = model(img, verbose=False)
            
            # Use YOLO's built-in plotting for reliability
            if results and len(results) > 0:
                annotated_img = results[0].plot()
                logger.info(f"YOLO detected {len(results[0].boxes) if results[0].boxes else 0} objects")
            else:
                annotated_img = img.copy()
                logger.info("No YOLO results")
                
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            annotated_img = img.copy()
        
        # Always add a processing indicator
        cv2.putText(annotated_img, "AI Processing", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Convert back to av.VideoFrame
        new_frame = av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        
        return new_frame

@webrtc_blueprint.route('/client.js')
def client_js():
    return send_from_directory('static', 'client.js')

@webrtc_blueprint.route('/')
def index():
    return render_template('index.html')

def create_webrtc_app(socketio):
    """Create WebRTC app with socket handlers"""
    
    @socketio.on('offer', namespace='/signal')
    def on_offer(message):
        sid = request.sid
        logger.info(f"Received offer from {sid}")
        
        offer = RTCSessionDescription(sdp=message['sdp'], type=message['type'])
        pc = RTCPeerConnection()
        pcs.add(pc)
        peer_map[sid] = pc
        
        @pc.on("track")
        def on_track(track):
            logger.info(f"Track received: {track.kind}")
            if track.kind == "video":
                # Add YOLO processing track
                yolo_track = YoloVideoTrack(relay.subscribe(track))
                pc.addTrack(yolo_track)
                logger.info("Added YOLO processing track")
        
        async def process_offer():
            try:
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                
                emit('answer', {
                    'sdp': pc.localDescription.sdp,
                    'type': pc.localDescription.type
                }, namespace='/signal')
                logger.info("Answer sent")
                
            except Exception as e:
                logger.error(f"Error processing offer: {e}")
        
        asyncio.ensure_future(process_offer())
    
    @socketio.on('candidate', namespace='/signal')
    def on_candidate(message):
        pc = peer_map.get(request.sid)
        if not pc:
            return
            
        sdp = message.get('candidate')
        if not sdp:
            return
            
        try:
            cand = candidate_from_sdp(sdp)
            cand.sdpMid = message.get('sdpMid')
            cand.sdpMLineIndex = message.get('sdpMLineIndex')
            asyncio.ensure_future(pc.addIceCandidate(cand))
        except Exception as e:
            logger.error(f"Error adding ICE candidate: {e}")
    
    @socketio.on('bye', namespace='/signal')
    def on_bye():
        sid = request.sid
        logger.info(f"Bye from {sid}")
        cleanup_peer(sid)
    
    @socketio.on('disconnect', namespace='/signal')
    def on_disconnect():
        sid = request.sid
        logger.info(f"Disconnect from {sid}")
        cleanup_peer(sid)
    
    def cleanup_peer(sid):
        """Clean up peer connection"""
        pc = peer_map.pop(sid, None)
        if pc:
            asyncio.ensure_future(pc.close())
            pcs.discard(pc)
    
    return webrtc_blueprint
