# Gesture & ASL Prototype (Landmark-Based Hand Recognition)

## Overview
This repository contains a **camera-based hand gesture and American Sign Language (ASL) prototype** built to validate a landmark-based recognition pipeline.

The goal of this project is **not** to provide a full-scale sign language system, but to:
- verify the feasibility of hand-landmark representations,
- understand real-time behavior vs offline evaluation,
- and identify clear next steps for more advanced modeling.

The project currently supports **two scoped modes**:
1. **Basic gesture recognition** (fist, palm, point)
2. **ASL static fingerspelling prototype** (limited subset, static signs only)

---

## Key Features
- Real-time webcam input using OpenCV  
- Hand landmark extraction via MediaPipe  
- Wrist-relative landmark normalization  
- Left/right hand mirroring for consistency  
- Confidence-based rejection (`Unknown`) for unseen gestures  
- Temporal smoothing for stable live predictions  
- Separate pipelines and models for **gestures** and **ASL**

---

## Supported Modes

### 1️⃣ Basic Gesture Mode
Recognizes a small set of static hand gestures:
- `fist`
- `palm`
- `point`

This mode is intended to validate the base pipeline and rejection logic.

---

### 2️⃣ ASL Static Fingerspelling Mode
A **scoped ASL prototype** supporting a limited subset of **static fingerspelling signs** (e.g., A–E).

⚠️ This is **not full ASL recognition**.  
Dynamic signs, sentence-level recognition, and grammar are **out of scope** for this prototype.

---

## Folder Structure (Clean View)

gesture-recognition/
├── data/
│ ├── basic_gestures/
│ │ ├── fist/
│ │ ├── palm/
│ │ └── point/
│ │
│ └── asl_static/
│ ├── A/
│ ├── B/
│ ├── C/
│ ├── D/
│ └── E/
│
├── models/
│ ├── basic_gesture_model.pkl
│ └── asl_static_model.pkl
│
├── collect_data.py
├── train_model.py
├── predict_live.py
├── README.md
└── requirements.txt


Virtual environments (`env/`, `venv/`) and cache files are intentionally excluded.

---

## How It Works (High Level)

1. Webcam captures live frames  
2. MediaPipe extracts 21 hand landmarks  
3. Landmarks are normalized relative to the wrist  
4. Left-hand landmarks are mirrored to match right-hand geometry  
5. A Random Forest classifier predicts the class  
6. Confidence threshold + temporal smoothing determine final output  
7. Low-confidence inputs are labeled as `Unknown`

---

## Running the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
2️⃣ Collect Data
python collect_data.py
Choose mode (gesture or asl) and provide the label when prompted.

3️⃣ Train Model
python train_model.py
Select training mode (gesture or asl).
Models are saved separately under models/.

4️⃣ Run Live Prediction
python predict_live.py
Select mode (gesture or asl) to run real-time inference.

Design Decisions & Limitations
Static signs only: temporal and continuous gestures are intentionally excluded.

Small, controlled label set: avoids overclaiming and keeps evaluation meaningful.

Random Forest classifier: chosen for stability on small landmark datasets.

Confidence calibration: accuracy alone is insufficient for live systems; rejection logic is required.

These limitations are deliberate and define clear directions for future work.

Future Work
Potential extensions include:

Temporal sequence modeling (LSTM / Transformer)

Continuous ASL sign recognition

Multi-gesture and sentence-level modeling

Cross-user generalization and robustness studies

Angle-based or graph-based landmark representations

Disclaimer
This project is a prototype and research exploration, not a production-ready or complete sign language recognition system.
Its purpose is to validate assumptions, surface challenges, and provide a foundation for further work.

