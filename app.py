import os
import time
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
app = Flask(__name__)

# --- DATABASE & SECURITY CONFIGURATION ---
app.config['SECRET_KEY'] = 'sky_ai_super_secret_key_2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sky_ai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- DATABASE MODELS ---
# --- DATABASE MODELS ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    scans = db.relationship('Scan', backref='owner', lazy=True)

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(300), nullable=True) # Optional for now
    acne_count = db.Column(db.Integer, default=0)
    clear_skin_score = db.Column(db.Float, nullable=True) 
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Initialize database tables
with app.app_context():
    db.create_all()

print("🧠 Loading Sky-AI V2 Brain...")
model = YOLO("model/v2_brain.pt") 

# Ensure output directories exist
os.makedirs("extracted_output/acne", exist_ok=True)
os.makedirs("extracted_output/clear", exist_ok=True)
os.makedirs("extracted_output/snapshots", exist_ok=True)

# Shared frame state
latest_frame = None 
latest_acne_count = 0

def generate_frames():
    global latest_frame, latest_acne_count
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  
    acne_counter = 0
    clear_counter = 0
    last_save_time = time.time()
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
                
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # 1. Run YOLO inference
            results = model(frame, conf=0.50)
            has_acne_in_frame = False
            current_frame_acne_count = 0
            
            # 2. Process detections
            for r in results:
                boxes = r.boxes
                current_frame_acne_count = len(boxes)
                if current_frame_acne_count > 0:
                    has_acne_in_frame = True
                    
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = int(box.conf[0] * 100)
                    class_name = model.names[int(box.cls[0])]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 150), 2)
                    label = f"{class_name} {confidence}%"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150), 2)

                    # Auto-save acne crops
                    if time.time() - last_save_time > 2.0 and acne_counter < 100:
                        crop_y1, crop_y2 = max(0, y1), min(h, y2)
                        crop_x1, crop_x2 = max(0, x1), min(w, x2)
                        acne_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        if acne_crop.size > 0:
                            acne_counter += 1
                            cv2.imwrite(f"extracted_output/acne/acne_{acne_counter}.png", acne_crop)
                            last_save_time = time.time()

            # Auto-save clear crops
            if not has_acne_in_frame and time.time() - last_save_time > 3.0 and clear_counter < 50:
                cx, cy = w // 2, h // 2
                clear_crop = frame[cy-50:cy+50, cx-50:cx+50]
                clear_counter += 1
                cv2.imwrite(f"extracted_output/clear/clear_{clear_counter}.png", clear_crop)
                last_save_time = time.time()

            # Cache latest state
            latest_frame = frame.copy()
            latest_acne_count = current_frame_acne_count

            # Encode and yield
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

# --- WEB ROUTES ---
@app.route('/')
@login_required
def index():
    # Fetch the logged-in user's scans, newest first
    user_scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).all()
    return render_template('index.html', scans=user_scans)
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            return "Email already exists! Please go to login."

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            return "Invalid email or password. Please try again."
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, port=5000)

@app.route('/take_snapshot', methods=['POST'])
@login_required
def take_snapshot():
    global latest_frame
    
    # 1. Make sure the camera is actually running
    if latest_frame is None:
        return jsonify({"status": "error", "message": "Camera not ready! Please wait."})
    
    # 2. Run the YOLO AI on that specific frozen frame
    results = model(latest_frame, conf=0.15)
    
    # 3. Count how many blemishes the AI found
    # results[0].boxes contains all the detections in the frame
    acne_count = len(results[0].boxes)
    
    # 4. Save this data to the logged-in user's database profile!
    new_scan = Scan(user_id=current_user.id, acne_count=acne_count)
    db.session.add(new_scan)
    db.session.commit()
    
    # 5. Send a success message back to the website
    return jsonify({
        "status": "success", 
        "message": f"Scan Saved! AI detected {acne_count} blemishes."
    })