# Sky-AI: Real-Time Edge Dermatological Clinical Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows&logoColor=white)

**Sky-AI** is an edge-native, real-time computer vision platform for non-invasive dermatological screening and blemish tracking. Built with Flask, OpenCV, YOLO, and SQLite, it processes live video streams locally to evaluate skin conditions in real time, keeping all inference on-device to protect biometric privacy.

> **Medical Disclaimer:** Sky-AI is a screening and tracking aid only. It is not a certified medical device and does not provide diagnoses. Consult a qualified dermatologist for any medical concerns.

---

## Table of Contents

- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running the Application](#running-the-application)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

| Feature | Description |
|---|---|
| **Edge Computer Vision** | Real-time blemish and dark-spot detection using a custom YOLO model (`v2_brain.pt`) with sub-100 ms inference latency (~11 FPS). |
| **100% Biometric Privacy** | All image processing happens on local hardware. No facial or medical imagery is uploaded to third-party cloud servers. |
| **Clinical Dashboard** | A clinical-grade dark-mode UI with live MJPEG streaming, scan capture controls, and session state tracking. |
| **Persistent Telemetry** | A relational SQLite database tracks user registration, authentication sessions, and historical snapshots with automated lesion count indexing. |
| **Hardware-Agnostic Camera Routing** | Works with built-in laptop webcams and high-resolution mobile virtual cameras (e.g., Iriun Webcam) via DirectShow or standard capture backends. |

---

## Tech Stack

| Domain | Technology |
|---|---|
| **Backend** | Python 3.10+, Flask, Flask-Login, Flask-SQLAlchemy |
| **Computer Vision** | OpenCV (`cv2`), Ultralytics YOLO (`v2_brain.pt` / PyTorch) |
| **Database** | SQLite, SQLAlchemy ORM |
| **Frontend** | HTML5, CSS3 (Clinical Dark Theme), JavaScript (Fetch API) |
| **Camera Interface** | Windows DirectShow (`CAP_DSHOW`), Iriun Virtual Webcam |

---

## Project Structure

```text
Skincare_AI/
│
├── instance/
│   └── sky_ai.db               # SQLite database (Users & Scans)
├── static/                     # Static styling, CSS, and UI assets
├── templates/
│   ├── index.html              # Main clinical dashboard & live feed
│   ├── login.html              # User authentication interface
│   └── register.html           # Account registration view
├── extracted_output/
│   └── snapshots/              # Stored capture frames
├── v2_brain.pt                 # Trained YOLO weights for blemish detection
├── app.py                      # Core Flask server & stream generator
├── find_cam.py                 # Diagnostic script to probe camera indices
├── test_iriun.py               # Direct OpenCV feed diagnostic script
├── requirements.txt            # Project dependencies
└── README.md                   # System documentation
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- *(Optional)* [Iriun Webcam](https://iriun.com/) installed on both PC and mobile device for high-resolution capture

### Installation

**1. Clone the repository**

```powershell
git clone https://github.com/your-username/Skincare_AI.git
cd Skincare_AI
```

**2. Create and activate a virtual environment** (Windows PowerShell)

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```powershell
pip install -r requirements.txt
```

Core dependencies include:

```text
flask
flask-login
flask-sqlalchemy
opencv-python
ultralytics
torch
torchvision
```

---

## Running the Application

### Step 1: Detect your camera index

Before launching the server, find the index assigned to your target camera:

```powershell
python find_cam.py
```

### Step 2: Configure the capture source

Open `app.py` and set your capture target. Use index `0` for the default webcam, or index `1` with `cv2.CAP_DSHOW` for virtual cameras:

```python
cap = cv2.VideoCapture(0)
```

### Step 3: Start the server

```powershell
python app.py
```

Then open your browser at:

```text
http://127.0.0.1:5000/
```

---

## Configuration

| Setting | Location | Default | Description |
|---|---|---|---|
| Camera source | `app.py` | `cv2.VideoCapture(0)` | Camera index and optional backend (`cv2.CAP_DSHOW`). |
| Confidence threshold | `app.py` | `conf=0.45` | Minimum confidence for a detection to be shown. Raise it to reduce false positives. |
| Model weights | `v2_brain.pt` | n/a | Trained YOLO weights used for blemish detection. |

Example of adjusting the inference threshold:

```python
results = model(frame, conf=0.45)
```

---

## Troubleshooting

<details>
<summary><b>Camera backend conflict (<code>VIDEOIO(DSHOW)</code> warning)</b></summary>

If Windows throws warnings when accessing capture indices, omit `cv2.CAP_DSHOW` and use standard index mapping:

```python
cap = cv2.VideoCapture(0)
```
</details>

<details>
<summary><b>Using the Iriun mobile webcam</b></summary>

1. Launch Iriun on both the PC and the mobile device **before** starting the server.
2. If Windows maps the integrated webcam to both indices `0` and `1`, open **Device Manager → Cameras** and temporarily disable the internal webcam.
3. Set `cap = cv2.VideoCapture(0)` in `app.py`.
</details>

<details>
<summary><b>False positives in the background</b></summary>

If walls or background noise trigger detections, increase the inference threshold in `app.py`:

```python
results = model(frame, conf=0.45)
```
</details>

---

## Roadmap

- [x] User authentication (register, login, session management)
- [x] Real-time inference integration with Flask streaming
- [x] Snapshot capture and database schema integration
- [ ] Face-cropping preprocessing pipeline (MediaPipe / Haar Cascades) to isolate facial skin from room backgrounds
- [ ] Historical telemetry charts (blemish reduction trends over time)
- [ ] Exportable clinical PDF summary reports

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

---

## Author

**Krishna Sharma** · [GitHub](https://github.com/Krishna2006-babu) · [LinkedIn](https://www.linkedin.com/in/krishna-sharma-80a10b21a/)