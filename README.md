# Cognitive Load Detector

> Non-intrusive, real-time cognitive load estimation using webcam and keyboard interaction cues.  
> No wearable sensors. No custom-trained models. Runs on Raspberry Pi 5.

---

## Overview

This project estimates a user's cognitive load (Low / Medium / High) from behavioral cues captured through a standard webcam — or a smartphone via DroidCam. It uses Google's pre-trained **MediaPipe Face Mesh** to extract 468 facial landmarks per frame and applies a **rule-based classifier** over a sliding temporal window.

No EEG. No GSR band. No GPU. No labeled dataset required.

---

## How It Works

```
Webcam (DroidCam)
      ↓
MediaPipe Face Mesh  →  Visual Features (EAR, Blink Rate, Gaze σ, Head ΔH)
                                          ↓
Keyboard (optional)  →  IKI Features  →  Rule-Based Fusion  →  LOW / MEDIUM / HIGH
```

### Features Extracted

| Feature | Symbol | Description |
|---|---|---|
| Eye Aspect Ratio | EAR | Detects blinks via 6 eye landmarks |
| Blink Rate | f_blink | Blinks per second over window W |
| Gaze Stability | σ_gaze | Std deviation of iris position |
| Head Pose Variation | ΔH | Frame-to-frame nose-tip displacement |
| Inter-Keystroke Interval | IKI | Typing pauses as cognitive effort signal |

### Classification Logic

```
Blink rate elevated?
├── NO  → Fixation long & gaze stable?
│         ├── YES → LOW LOAD
│         └── NO  → MEDIUM LOAD
└── YES → Head motion ↑ or typing pauses ↑?
          ├── YES → HIGH LOAD
          └── NO  → MEDIUM LOAD
```

---

## Project Structure

```
cognitive-load-detector/
├── src/
│   ├── blink_detector.py       # EAR computation + blink detection
│   ├── gaze_tracker.py         # Iris tracking + gaze stability
│   ├── head_pose.py            # Nose-tip displacement tracking
│   ├── keystroke_logger.py     # IKI computation via pynput
│   ├── feature_aggregator.py   # Sliding window aggregation
│   ├── classifier.py           # Rule-based fusion logic
│   └── main.py                 # Entry point — full pipeline
├── docs/
│   ├── architecture.png        # System pipeline diagram
│   └── flowchart.png           # Classification logic flowchart
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements

- Python 3.10+
- Webcam or smartphone with [DroidCam](https://www.dev47apps.com/) (USB or WiFi)
- Raspberry Pi 5 (8GB recommended) or any Linux/Windows machine

### Install

```bash
git clone https://github.com/KrishSingh2302/cognitive-load-detector.git
cd cognitive-load-detector
pip install -r requirements.txt
```

### Run

```bash
python src/main.py
```

To use DroidCam, open DroidCam on your phone, connect via USB or WiFi, then run the script. The webcam index will be `1` or `2` — edit `CAMERA_INDEX` in `main.py` if needed.

---

## Technical Details

### EAR Formula

The Eye Aspect Ratio (EAR), introduced by Soukupá & Čech (CVWW 2016), is computed from six eye landmarks P1–P6:

```
EAR = ( ||P2 - P6|| + ||P3 - P5|| ) / ( 2 * ||P1 - P4|| )
```

A blink is registered when `EAR < 0.21` for 3 consecutive frames.

### Sliding Window

All features are aggregated over a rolling window of `W = 30 seconds`. The classifier evaluates the aggregated feature vector at the end of each window and resets.

### Hardware Target

| Spec | Value |
|---|---|
| Platform | Raspberry Pi 5 — 8 GB |
| OS | Raspberry Pi OS Lite 64-bit |
| Target FPS | ≥ 15 fps |
| MediaPipe | CPU-only inference |

---

## Status

| Module | Status |
|---|---|
| Blink detection (EAR) | ✅ Implemented |
| Gaze stability | 🔄 In progress |
| Head pose variation | 🔄 In progress |
| Keystroke IKI | 🔄 In progress |
| Rule-based classifier | 🔄 In progress |
| Full pipeline | 🔄 In progress |
| Pi 5 deployment | ⏳ Planned |

---

## References

1. Sweller, J. (1988). Cognitive load during problem solving. *Cognitive Science*, 12(2), 257–285.
2. Soukupá, T. & Čech, J. (2016). Real-time eye blink detection using facial landmarks. *CVWW 2016*.
3. Lugaresi, C. et al. (2019). MediaPipe: A framework for building perception pipelines. *arXiv:1906.08172*.
4. Epp, C., Lippold, M., & Mandryk, R. L. (2011). Identifying emotional states using keystroke dynamics. *CHI 2011*.
