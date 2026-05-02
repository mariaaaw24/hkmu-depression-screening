# 🧠 AI-Powered Multimodal Depression Detection
### Harnessing Visual Cues for Early, Privacy-Preserving Screening
> **Project Code:** 2025-KevinHung-1 | **Institution:** Hong Kong Metropolitan University  
> 🏆 **Best Paper Award** – ISCAIE 2026 (Penang, Malaysia)

---

## 📖 Overview
This project develops a privacy-preserving AI system for early depression screening using **quaternion-encoded visual behavioral cues** (head pose, eye gaze, and blink dynamics) extracted from the DAIC-WOZ dataset. By fusing 3D geometric features into 4D hypercomplex representations, the model captures spatial and rotational relationships that traditional 3D concatenation misses. 

The system achieves an **F1-score of 0.80 (+27% vs. 3D baseline)** and **doubles recall (33% → 67%)** for depressed class detection, while requiring **zero raw video/audio storage**. A minimal Flask prototype demonstrates local, real-time deployment feasibility.

---

## ✨ Key Features
| Feature | Description |
|---------|-------------|
| 🔒 **Privacy-First Architecture** | Uses only pre-extracted geometric features. No raw media processing or storage. |
| 📐 **Quaternion Fusion** | Novel 4D encoding preserves rotational/spatial dependencies between visual cues. |
| ⚖️ **Robust Imbalance Handling** | SMOTE resampling + class-weighted models address the 1:2.7 class ratio. |
| 📊 **Rigorous Evaluation** | 7 classical ML models, 5-fold Stratified CV, Cohen’s Kappa, ROC-AUC, and official DAIC-WOZ splits. |
| 🌐 **Minimal Web Prototype** | Flask-based demonstrator for local, ethical risk assessment (`localhost:5000`). |

---

## 📁 Project Structure
HKMUResearch/
├── data/               # Pre-extracted feature matrices (.xls) & label files
├── models/             # Serialized models (.pkl) & scalers
├── Minimal_prototype/  # Flask web demonstrator
│   ├── app.py
│   ├── templates/      # HTML views
│   └── static/         # HKMU-logo, styles, scripts
├── python files/       # Training & evaluation pipelines
│   ├── H2.py, E2.py, BH2.py, BE2.py   # Official DAIC-WOZ protocol scripts
│   └── H.py, E.py, BH.py, BE.py       # Exploratory/internal validation scripts
├── requirements.txt    # Project dependencies
└── README.md           # This file

---

## 🛠️ Setup & Installation
This project is designed to run using a global Python installation.

### 1. Install Dependencies

**Windows**
Open your terminal/command prompt and run:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**mac0S**
Open your terminal and run:
```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 2. Verify Installation
Run this command to ensure all libraries are loaded:
```bash
python -c "import pandas, sklearn, xgboost, catboost, flask; print('✅ All dependencies loaded successfully.')"
```

## Usage
### Reproducing Training Results
Run the official DAIC-WOZ protocol scripts located in the 'python files' directory:
(Make sure you are in this directory: 'C:\Users\...\HKMUResearch>' before you execute the following)
```bash
python "python files/H2.py"      # Head Pose Quaternion
python "python files/E2.py"      # Eye Gaze Quaternion
```

### Running the Minimal Prototype
(Make sure you are in this directory: 'C:\Users\...\HKMUResearch>' before you execute the following)
```bash
python "Minimal_prototype/app.py"
```

Open your browser to: http://localhost:5000

⚠️ Troubleshooting: Port Conflicts
On macOS, Port 5000 is often used by AirPlay. If you get an "Address already in use" error:

Run via: flask run --port 5001

Or, change the port in app.py: app.run(debug=True, port=5001)

## Model Performance
| Feature Set | Encoding | F1-Score | Recall |
|-------------|----------|----------|--------|
|Head Pose| Quaternion (4D)| 0.80 | 0.67 |
|Head Pose | Standard (3D) | 0.63 | 0.33 |

## Ethical Considerations & Data
- Dataset: DAIC-WOZ (Distress Analysis Interview Corpus).
- Privacy: Only geometric feature vectors are used; no raw PII is processed.
- Disclaimer: This is a pre-clinical screening demonstrator only. It is not a diagnostic tool and should never replace professional medical evaluation.

## 📧 Contact
Maria Dharshini School of Science and Technology, Hong Kong Metropolitan University
📧 s1312639@live.hkmu.edu.hk
Last updated: May 2026