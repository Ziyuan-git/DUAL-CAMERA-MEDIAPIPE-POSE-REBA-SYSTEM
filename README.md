# Dual-Camera REBA Ergonomic Assessment System

A real-time ergonomic assessment tool using two cameras and MediaPipe pose detection to automatically compute **Rapid Entire Body Assessment (REBA)** scores for workplace risk analysis.

---

## Features

- **Dual-camera capture** — synchronized recording from two cameras for improved landmark coverage
- **MediaPipe pose detection** — 33-point body landmark tracking with visibility filtering
- **Camera fusion** — projects low-visibility landmarks from Camera 1 into Camera 0 space to correct occluded joints
- **EMA smoothing** — Exponential Moving Average (alpha = 0.7) applied to landmark coordinates to reduce jitter
- **Stability tracking** — 10-frame rolling window per landmark to detect unstable detections
- **Full REBA scoring** — computes Section A (neck, trunk, legs) and Section B (upper arm, lower arm, wrist) scores, Table A/B/C lookups, and final REBA score with action level
- **Natural angle correction** — subtracts calibrated neutral posture baselines from raw angles
- **Post-recording adjustment form** — Tkinter GUI to input manual adjustments (neck/trunk twist, force/load, coupling, activity score)
- **Automatic video re-encoding** — uses FFmpeg to correct FPS in saved recordings; falls back to rename if FFmpeg is unavailable
- **Comprehensive CSV export** — per-frame angles, REBA scores, landmark visibility, stability metrics, and session summary statistics

---

## Project Structure

```
.
├── reba_main.py          # Main application — all camera, pose, and REBA logic
├── requirements.txt      # Python dependencies
├── USM_logo.png          # Logo displayed in the adjustment form (optional)
├── camera0_<timestamp>.mp4          # Output — Camera 0 recording
├── camera1_<timestamp>.mp4          # Output — Camera 1 recording
├── camera_<timestamp>_cam0_frames.csv   # Output — Camera 0 per-frame data
└── camera_<timestamp>_cam1_frames.csv   # Output — Camera 1 per-frame data
```

---

## Setup & Installation

### Requirements

- Python 3.8 or higher
- Two USB cameras
- FFmpeg (optional, recommended for accurate video FPS)

### Install FFmpeg (recommended)

**Windows:** Download from [ffmpeg.org](https://ffmpeg.org) and add to PATH.

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

The required packages are:

```
mediapipe
numpy
opencv-python
pillow
```

---

## How to Run

### 1. Connect both cameras

Plug in Camera 0 (side view — subject's right side) and Camera 1 (angled view) before running.

### 2. Run the script

In the terminal: 

```bash
py -3.11 reba_main_FormatONly.py
```

By default, the script uses:
```python
DualCameraCapture(cam0_idx=1, cam1_idx=0, fps=30)
```

If your cameras are not detected, edit the `cam0_idx` and `cam1_idx` values at the bottom of `reba_main.py` to match your system's camera indices (typically `0` and `1`).

### 3. Calibration

On launch, the system automatically runs a **5-second calibration phase**. During this time:
- Stand in a **natural upright posture** (Camera 0 should capture your right sagittal plane)
- The system records baseline angles for trunk, neck, knees, arms, and wrists
- Camera FPS is measured and synchronised

### 4. Recording

After calibration, press **Enter** to begin recording. The live view window shows:
- Pose landmarks overlaid on both camera feeds
- Real-time angle display (neck, trunk, knees, arms, wrists)
- Corrected landmarks highlighted in red where camera fusion is active

Press **`Q`** to stop recording.

### 5. Post-recording adjustments

A form will appear to collect manual inputs that cannot be detected automatically:

| Field | Description |
|---|---|
| Neck Twisted / Side Bent | Manual observation |
| Trunk Twisted / Side Bent | Manual observation |
| Weight held (kg) | Object weight for force/load score |
| Shock loading | Sudden or rapid force buildup |
| Shoulder Raised / Arm Abducted | Upper arm adjustments |
| Arm Supported / Leaning | Reduces upper arm score |
| Wrist Bent / Twisted | Per-hand wrist adjustments |
| Coupling | Handle quality (Good / Fair / Poor / Unacceptable) |
| Activity | Static posture, repeated actions, rapid changes |

Click **Submit** when done.

### 6. Outputs

After the session, the following files are saved in the working directory:

| File | Description |
|---|---|
| `camera0_<timestamp>.mp4` | Camera 0 recording at true FPS |
| `camera1_<timestamp>.mp4` | Camera 1 recording at true FPS |
| `camera_<timestamp>_cam0_frames.csv` | Per-frame REBA data from Camera 0 |
| `camera_<timestamp>_cam1_frames.csv` | Per-frame REBA data from Camera 1 |

The CSV files include per-frame: angles, REBA scores (raw, corrected, natural), action levels, landmark visibility, stability scores, and a session summary appended at the bottom.

---

## Troubleshooting

**Camera not detected**
- Ensure both cameras are connected before running
- Close any other applications using the cameras (e.g. video conferencing apps)
- Try swapping `cam0_idx` and `cam1_idx` values in `reba_main.py`

**`import cv2` cannot be resolved**
- Make sure you are using the correct Python interpreter where packages are installed
- In VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter*
- Run `python -c "import cv2; print(cv2.__version__)"` to verify installation

**FFmpeg not found**
- Videos will still be saved but FPS accuracy may be lower
- Install FFmpeg from [ffmpeg.org](https://ffmpeg.org) for best results

**High frame drop rate**
- Reduce `fps` value in `DualCameraCapture(fps=30)` to e.g. `fps=15`
- Close background applications to free up CPU for MediaPipe processing
