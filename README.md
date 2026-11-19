
# AI-Based Road Damage Detection and Severity Analysis

### PaveSense AI — Automated Road Distress Screening & Human-in-the-Loop Audit System

An end-to-end computer vision project designed for municipal road condition surveys, dashcam screening, automated road damage detection, and field inspection shortlisting.

Built using **Ultralytics YOLO26n**, a leakage-aware dataset preparation pipeline of **25,600 road images**, custom **Family-Aware Non-Maximum Suppression (NMS)**, provisional visual severity screening, and an interactive **Streamlit dashboard**.

---

## 🛣️ Project Overview & Architecture

**PaveSense AI** extends conventional object detection by integrating road damage identification with an inspection-oriented workflow.

### Key Features

1. **Automated Triage Screening:** Processes road images using YOLO26n, with reported evaluation throughput of approximately 41–50 FPS on NVIDIA GTX 1650 hardware.

2. **Four Damage Families:** Groups eight fine-grained damage classes into four physical distress families:
   - `{0,1}` Longitudinal Crack
   - `{2,3}` Transverse Crack
   - `{4,5}` Alligator Crack
   - `{6,7}` Pothole

3. **Family-Aware NMS:** Suppresses redundant detections within the same damage family, reducing duplicate bounding boxes caused by overlapping normal and severe subclass predictions.

4. **Provisional Severity Screening:** Uses visual subclass evidence to identify potential severity indicators requiring human verification.

5. **Structured Inspection Shortlisting:** Aggregates detected defects and visual damage information to support field inspection prioritization.

6. **Interactive Dashboard:** Provides image-based detection, annotated visualizations, and configurable detection parameters.

7. **Report Export:** Generates CSV-based inspection records and annotated road damage images.

---

## 🧠 System Architecture

The overall workflow follows:

```text
           Road Image Input
                  |
                  v
          Image Preprocessing
                  |
                  v
         YOLO26n Object Detection
                  |
                  v
      8 Fine-Grained Damage Classes
                  |
                  v
         4 Damage Family Mapping
                  |
                  v
       Family-Aware Deduplication
                  |
                  v
       Provisional Severity Analysis
                  |
                  v
         PaveSense AI Dashboard
                  |
           +------+------+
           |             |
           v             v
     Annotated Image   CSV Report
           |             |
           +------+------+
                  |
                  v
       Human Field Verification
```

The system is designed to assist inspection teams rather than independently make final engineering decisions.

---

## 🔍 Road Damage Classification

The model recognizes eight fine-grained road damage categories.

| Class ID | Damage Type | Damage Family |
|---|---|---|
| 0 | Longitudinal Crack | Longitudinal Crack |
| 1 | Longitudinal Crack Wide | Longitudinal Crack |
| 2 | Transverse Crack | Transverse Crack |
| 3 | Transverse Crack Wide | Transverse Crack |
| 4 | Alligator Crack | Alligator Crack |
| 5 | Alligator Crack Sunken | Alligator Crack |
| 6 | Pothole | Pothole |
| 7 | Pothole Deep | Pothole |

The four-family mapping helps consolidate predictions referring to the same physical road defect.

---

## 📊 Dataset Preparation

The project uses a dataset of **25,600 road images**, divided into training, validation, and testing subsets.

### Dataset Distribution

| Dataset Split | Images | Percentage |
|---|---:|---:|
| Training | 20,480 | 80% |
| Validation | 2,560 | 10% |
| Testing | 2,560 | 10% |
| **Total** | **25,600** | **100%** |

### Dataset Validation Pipeline

The dataset preparation workflow includes:

- Image and annotation validation
- Empty annotation detection
- Image resolution verification
- Exact duplicate identification
- Near-duplicate image investigation
- Perceptual hash analysis
- Leakage-group construction
- Dataset partitioning
- YOLO annotation preparation

### Leakage-Aware Splitting

Near-duplicate images appearing across training and testing partitions can produce misleading model evaluation results.

The project uses duplicate investigation and leakage-group construction to reduce cross-partition contamination.

Relevant scripts:

```text
check_exact_duplicates.py
check_near_duplicates.py
build_leakage_groups.py
create_dataset_split.py
prepare_yolo_dataset.py
validate_labels.py
```

Dataset split information is preserved in:

```text
dataset_split.csv
leakage_groups.csv
```

---

## 🚀 Model Training

### Training Configuration

| Parameter | Configuration |
|---|---|
| Detection Model | YOLO26n |
| Image Resolution | 640 × 640 |
| Epoch Limit | 20 |
| Batch Size | 4 |
| Early Stopping Patience | 5 |
| Optimizer | Automatic |
| Pretrained Weights | Enabled |
| Mixed Precision | Enabled |
| Random Seed | 42 |
| Training Hardware | NVIDIA GTX 1650 (4 GB VRAM) |

### Training Script

```cmd
python train_road_damage.py
```

The repository also contains:

```text
finetune_road_damage.py
```

for model fine-tuning experiments.

Training requires the prepared dataset and appropriate model initialization files.

Generated model weights and training results are stored locally and excluded from GitHub.

---

## 📈 Held-Out Evaluation Results

Evaluation was performed using the frozen **YOLO26n V1 checkpoint**, with separate validation and held-out test partitions.

### Overall Detection Performance

| Metric | Validation (2,560 Images) | Held-Out Test (2,560 Images) |
|---|:---:|:---:|
| **Operational Precision (@ conf=0.15)** | **52.75%** | **53.38%** |
| **Operational Recall (@ conf=0.15)** | **43.77%** | **44.33%** |
| **Operational F1 Score** | **0.4784** | **0.4843** |
| **Grouped mAP@50** | **0.4422** | **0.4291** |
| **Grouped mAP@50:95** | **0.2083** | **0.2073** |
| **Hardware Throughput** | **50.8 FPS** | **41.5 FPS** |
| **Latency** | **19.7 ms** | **24.1 ms** |

### Per-Family Test Split AP@50

| Damage Family | AP@50 | Ground Truth Boxes |
|---|:---:|---:|
| Longitudinal Crack | 0.4983 | 2,192 |
| Transverse Crack | 0.4477 | 1,358 |
| Alligator Crack | 0.4212 | 1,154 |
| Pothole | 0.3492 | 672 |

### Evaluation Interpretation

Validation and test performance show relatively small differences across the reported metrics.

This suggests that the model achieves comparable performance on these evaluation partitions.

However, similar validation and test scores do not prove the complete absence of overfitting or guarantee performance on road environments outside the dataset.

Further evaluation on geographically diverse datasets and real-world conditions is necessary to assess broader generalization.

### Evaluation Scripts

Run the complete validation and test evaluation:

```cmd
python run_full_eval_val_and_test.py
```

Run four-family evaluation:

```cmd
python evaluate_v1_four_families.py
```

Run detection benchmarking:

```cmd
python benchmark_detection_improvements.py
```

Evaluation requires the original dataset and trained model checkpoint.

---

## ⚙️ Family-Aware Non-Maximum Suppression

A common problem in object detection is the generation of overlapping bounding boxes for the same physical object.

In road damage detection, a single defect can receive predictions from multiple related classes.

For example:

```text
Pothole
Pothole Deep
```

Both predictions may refer to the same physical pothole.

### Family-Aware Deduplication Workflow

1. Obtain raw YOLO predictions.
2. Map fine-grained classes into their corresponding damage families.
3. Group predictions by damage family.
4. Sort predictions using confidence scores.
5. Compare overlapping bounding boxes.
6. Suppress redundant intra-family predictions.
7. Preserve relevant subclass evidence.
8. Generate consolidated damage detections.

**Family-Aware NMS IoU Threshold: `0.60`**

This post-processing approach reduces redundant damage counts while retaining information for provisional severity screening.

---

## ⚠️ Provisional Severity Analysis

PaveSense AI uses visual evidence from fine-grained damage subclasses to support preliminary severity screening.

Examples include:

- Wide longitudinal cracks
- Wide transverse cracks
- Sunken alligator cracks
- Deep-looking potholes

The system distinguishes standard distress observations from visual cues suggesting potentially more severe damage.

### Important Limitation

Visual damage severity is not equivalent to measured physical severity.

A single RGB image cannot reliably determine:

- Crack width in millimeters
- Actual pothole depth
- Pavement structural condition
- Subsurface deterioration
- Required repair thickness

Severity indications are therefore **provisional visual classifications**, not certified engineering measurements.

Final severity assessment requires professional field verification.

---

## 💻 PaveSense AI — Streamlit Dashboard

The project includes an interactive Streamlit application implemented in:

```text
app.py
```

### Dashboard Features

- Road image upload
- AI-based road damage detection
- Annotated bounding box visualization
- Adjustable confidence thresholds
- Family-aware detection processing
- Damage family summaries
- Provisional severity indicators
- Inspection record generation
- CSV report export
- Human verification support

The dashboard provides a user-friendly interface for reviewing model predictions and supporting road inspection workflows.

---

## 🛠️ Environment Setup & Quickstart

### Prerequisites

- Windows 10/11
- Python
- Anaconda or Miniconda
- Git
- Compatible CPU or NVIDIA GPU
- Trained YOLO26n model checkpoint for inference

The project was developed using an NVIDIA GTX 1650 GPU with 4 GB VRAM.

### 1. Clone the GitHub Repository

Open Command Prompt (CMD):

```cmd
git clone https://github.com/DikshaSirohi/road-damage-detection-severity-analysis.git
```

Move into the project directory:

```cmd
cd road-damage-detection-severity-analysis
```

### 2. Create a Conda Environment

```cmd
conda create -n road_yolo python=3.11 -y
```

Activate the environment:

```cmd
conda activate road_yolo
```

### 3. Install Dependencies

```cmd
pip install -r requirements.txt
```

If Streamlit or Pandas is not installed:

```cmd
pip install streamlit pandas
```

**Note:** GPU-specific PyTorch installations may require a compatible CUDA environment and package index. The listed setup commands are a starting point and have not been validated on a fresh installation.

### 4. Configure Model Weights

The application expects the trained checkpoint at:

```text
runs/road_damage_yolo26n_v1/weights/best.pt
```

The trained checkpoint must be supplied separately.

Model weights are excluded from GitHub to avoid uploading large binary files.

### 5. Launch the Streamlit Application

```cmd
streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

The application requires the trained checkpoint and compatible dependencies.

---

## 📂 GitHub Repository Structure

The repository contains source code, dataset validation utilities, training scripts, and evaluation tools.

```text
road-damage-detection-severity-analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── app.py
│
├── train_road_damage.py
├── finetune_road_damage.py
├── prepare_yolo_dataset.py
├── create_dataset_split.py
├── validate_labels.py
│
├── run_full_eval_val_and_test.py
├── evaluate_v1_four_families.py
├── finish_four_family_metrics.py
├── benchmark_detection_improvements.py
├── check_v1_thresholds.py
│
├── analyze_groups.py
├── build_leakage_groups.py
├── check_exact_duplicates.py
├── check_near_duplicates.py
├── check_empty_labels.py
├── check_images.py
├── check_mirror_annotations.py
├── check_resolutions.py
├── compare_orb_candidates.py
├── finalize_duplicate_pairs.py
│
├── diagnose_detection_pipeline.py
├── investigate_detection_issue.py
├── inspect_filenames.py
├── inspect_phash_zero.py
├── inspect_severity_classes.py
├── inspect_severity_crops.py
├── inspect_source_ids.py
│
├── make_distance2_contact_sheets.py
├── test_provisional_severity.py
├── test_verification_run.py
│
├── dataset_split.csv
└── leakage_groups.csv
```

### Files Excluded from GitHub

Large datasets, model weights, temporary outputs, and generated analysis results are maintained locally.

Examples include:

```text
combined_annotatedv2/
combined_annotatedv2_working/
road_damage_yolo/
runs/
benchmark_results/
final_evaluation_results/
detection_investigation_results/
four_family_v1_results/
severity_audit/
severity_audit_crops/
```

These files are not required to browse the source code but may be necessary for model training, evaluation, or application execution.

---

## 🔬 Experimental Workflow

The project follows an end-to-end experimental process:

```text
Raw Road Damage Dataset
          |
          v
Image and Annotation Validation
          |
          v
Duplicate and Leakage Investigation
          |
          v
Leakage-Aware Dataset Splitting
          |
          v
YOLO Dataset Preparation
          |
          v
YOLO26n Model Training
          |
          v
Validation and Held-Out Testing
          |
          v
Four-Family Detection Aggregation
          |
          v
Family-Aware NMS
          |
          v
Provisional Severity Screening
          |
          v
Streamlit Inspection Dashboard
          |
          v
CSV Reports and Human Verification
```

---

## 🎯 Applications

Potential applications of PaveSense AI include:

- Municipal road condition surveys
- Automated road defect screening
- Dashcam-based road monitoring
- Preliminary maintenance inspection
- Road infrastructure documentation
- Inspection prioritization
- AI-assisted transportation infrastructure monitoring

The system is designed as an inspection-support prototype rather than a replacement for certified pavement assessment.

---

## 🚧 Limitations

### Detection Accuracy

The model may generate false positives or miss certain road defects, particularly under challenging visual conditions.

### Environmental Conditions

Detection accuracy may vary with lighting, road texture, camera viewpoint, weather, and image quality.

### Generalization

Evaluation on the project's held-out test split does not guarantee equivalent accuracy across different road networks or geographic regions.

### Severity Estimation

Severity indicators are based on visual subclass evidence and do not measure physical damage dimensions.

### Dataset Availability

Large image datasets are not included in the GitHub repository.

### Model Checkpoint Availability

The trained model checkpoint is stored locally and is not distributed through this repository.

Full reproduction of model evaluation requires the original dataset and trained checkpoint.

---

## 🔮 Future Improvements

Potential extensions include:

1. Improved detection of small cracks and potholes.
2. Evaluation on geographically diverse road datasets.
3. Video-based road damage detection.
4. GPS-assisted inspection mapping.
5. Improved performance under low-light and adverse weather conditions.
6. More efficient edge-device deployment.
7. Depth estimation using calibrated stereo cameras or 3D sensors.
8. Advanced inspection report generation.
9. Improved uncertainty estimation.
10. Real-world field validation with road inspection teams.

---

## ⚖️ Metrology & Engineering Realism Disclosure

Monocular 2D images cannot objectively measure physical pothole depth, crack width in millimeters, or structural pavement deterioration without suitable calibration and additional measurement methods.

All PaveSense AI detections provide **visual screening and inspection-support information**.

Provisional severity indicators require human field verification before being used for formal damage grading, maintenance decisions, or engineering repair work orders.

**PaveSense AI is a research and inspection-support prototype, not a certified pavement measurement system.**

---

## 📌 Project Information

**Project Name:** AI-Based Road Damage Detection and Severity Analysis

**Application Name:** PaveSense AI

**Core Model:** Ultralytics YOLO26n

**Primary Language:** Python

**Application Framework:** Streamlit

**Repository:**

https://github.com/DikshaSirohi/road-damage-detection-severity-analysis

---

## ✅ Conclusion

PaveSense AI demonstrates an end-to-end deep learning workflow for automated road damage detection and preliminary severity screening.

The project combines dataset quality analysis, leakage-aware partitioning, YOLO26n model training, held-out evaluation, family-aware detection post-processing, and an interactive inspection dashboard.

By integrating automated visual analysis with human verification, the system provides a practical foundation for AI-assisted road infrastructure inspection and future research into intelligent pavement monitoring.
