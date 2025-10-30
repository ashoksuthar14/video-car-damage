import eventlet
import eventlet.wsgi
from flask import Flask, render_template
from flask_socketio import SocketIO
from server import create_webrtc_app

app = Flask(__name__, template_folder="templates", static_folder="static")
socketio = SocketIO(app, cors_allowed_origins='*')
# Register aiortc/video call Blueprint
webrtc_blueprint = create_webrtc_app(socketio)
app.register_blueprint(webrtc_blueprint, url_prefix='/signal')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
