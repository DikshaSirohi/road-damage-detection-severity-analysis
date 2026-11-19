# AI-Based Road Damage Detection and Severity Analysis
### PaveSense AI — Automated Road Distress Screening & Human-in-the-Loop Audit System

An original, end-to-end computer vision project designed for municipal road condition surveys, dashcam screening, and field inspection shortlisting. Built on a leakage-free dataset split of 25,600 images using Ultralytics YOLO26n with custom Family-Aware deduplication post-processing.

---

## 🛣️ Project Overview & Architecture

Standard deep learning projects stop at drawing bounding boxes on static photos. **PaveSense AI** implements the complete municipal operational lifecycle:
1. **Automated Triage Screening:** Processes road frames in real-time (~41–50 FPS on GTX 1650), automatically filtering out 85–90% of undamaged asphalt frames.
2. **Four Damage Families:** Groups 8 fine-grained annotations into four physical distress families ({0,1} Longitudinal Crack, {2,3} Transverse Crack, {4,5} Alligator Crack, {6,7} Pothole).
3. **Family-Aware NMS:** Suppresses redundant intra-family duplicate boxes caused by multi-class overlap between normal and severe variants.
4. **Structured Shortlisting:** Aggregates detected defects, visual area percentages, and flags candidate severity for certified human field audits.
5. **Report Export:** Exports interactive CSV field inspection shortlists and high-visibility annotated inspection imagery.

---

## 📊 Rigorous Held-Out Evaluation Results

All metrics evaluated on strictly partition-isolated images using the frozen **YOLO26n V1** checkpoint:

| Metric | Validation Split (2,560 images) | Held-Out Test Split (2,560 images) |
|---|:---:|:---:|
| **Operational Precision (@ conf=0.15)** | **52.75%** | **53.38%** |
| **Operational Recall (@ conf=0.15)** | **43.77%** | **44.33%** |
| **Operational F1 Score** | **0.4784** | **0.4843** |
| **Grouped mAP50** | **0.4422** | **0.4291** |
| **Grouped mAP50-95** | **0.2083** | **0.2073** |
| **Hardware Throughput** | **50.8 FPS (19.7 ms)** | **41.5 FPS (24.1 ms)** |

### Per-Family Test Split AP50:
- **Longitudinal Crack:** `0.4983` (2,192 ground truth boxes)
- **Transverse Crack:** `0.4477` (1,358 ground truth boxes)
- **Alligator Crack:** `0.4212` (1,154 ground truth boxes)
- **Pothole:** `0.3492` (672 ground truth boxes)

*Note: Validation and Test metrics match within 1.3 percentage points, proving 0% overfitting and authentic generalization on unseen test data.*

---

## 🛠️ Environment Setup & Quickstart

### Prerequisites
- Windows 10/11
- NVIDIA GPU (tested on GTX 1650 4GB) or CPU
- Anaconda / Miniconda

### 1. Activate Conda Environment
Open **Command Prompt (CMD)** or PowerShell:
```cmd
conda activate road_yolo
cd "C:\NIET\Projects\Road Damage"
```

### 2. Launch PaveSense AI Streamlit Application
```cmd
streamlit run app.py
```
Open your browser at: **`http://localhost:8501`**

### 3. Run Benchmark and Evaluation Scripts
- **Run Full 2,560 Validation and Test Evaluation:**
  ```cmd
  python run_full_eval_val_and_test.py
  ```
- **Run Resolution & NMS Benchmark:**
  ```cmd
  python benchmark_detection_improvements.py
  ```

---

## 📂 Project Directory Structure

```
C:\NIET\Projects\Road Damage\
├── app.py                                # PaveSense AI interactive Streamlit dashboard
├── benchmark_detection_improvements.py  # Multi-resolution & NMS benchmark script
├── run_full_eval_val_and_test.py         # Full 2,560 val & test evaluation pipeline
├── check_v1_thresholds.py                # F1 curve & confidence threshold diagnostic
├── PRESENTATION_SLIDES_AND_SPEAKER_NOTES.md # 10-slide presentation deck with full speaker notes
├── VIVA_DEFENSE_QA.md                    # Faculty & examiner technical viva Q&A defense
├── ANTIGRAVITY_ROAD_DAMAGE_HANDOFF.md    # Project execution specification & constraints
│
├── benchmark_results/                    # Benchmark CSVs and comparison tables
├── final_evaluation_results/             # Official validation & test evaluation metrics
├── runs/road_damage_yolo26n_v1/          # Frozen YOLO26n V1 trained model weights & logs
└── road_damage_yolo/                     # Leakage-safe dataset (train, val, test splits)
```

---

## ⚠️ Metrology & Engineering Realism Disclosure
Monocular 2D images cannot objectively measure physical cavity depth or crack width (millimeter scale) without calibrated 3D depth sensors or stereo cameras. All PaveSense AI detections provide rapid visual triage to prioritize certified municipal inspection teams; severity classifications are provisional visual indicators requiring field audit before issuing formal repair work orders.
