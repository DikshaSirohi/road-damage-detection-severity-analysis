"""PaveSense AI - Road Damage Detection and Severity Screening System
Official Project: AI-Based Road Damage Detection and Severity Analysis
Uses frozen YOLO26n V1 model with four-family aggregation, competing subclass evidence preservation,
provisional visual severity analysis, and human-in-the-loop audit workflow.
"""

from pathlib import Path
import io
import time
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------
# Page Configuration & Dark-Theme High-Contrast Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="PaveSense AI — Road Damage Detection & Screening",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    .project-subtitle {
        font-size: 1.05rem;
        font-weight: 500;
        color: #94A3B8;
        margin-bottom: 0.8rem;
    }
    .project-meta-pill {
        display: inline-block;
        background-color: #1E293B;
        color: #CBD5E1;
        border: 1px solid #334155;
        border-radius: 9999px;
        padding: 4px 14px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 1.2rem;
    }
    .dark-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .disclaimer-banner {
        background-color: #2D1F03;
        border-left: 4px solid #F59E0B;
        border-radius: 6px;
        padding: 12px 18px;
        margin-bottom: 1.2rem;
        color: #FDE68A;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    [data-testid="stMetricValue"] {
        color: #38BDF8 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="metric-container"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    .badge-unverified {
        display: inline-block;
        background-color: #78350F;
        color: #FDE68A;
        border: 1px solid #D97706;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-clear {
        display: inline-block;
        background-color: #064E3B;
        color: #A7F3D0;
        border: 1px solid #059669;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model Mapping & Class Definitions
# ---------------------------------------------------------
MODEL_PATH = Path("runs/road_damage_yolo26n_v1/weights/best.pt")

CLASS_NAMES_8 = {
    0: "longitudinal crack",
    1: "longitudinal crack wide",
    2: "transverse crack",
    3: "transverse crack wide",
    4: "alligator crack",
    5: "alligator crack sunken",
    6: "pothole",
    7: "pothole deep"
}

# (Family Name, Visual Variant, Is Severe Subclass)
FAMILY_MAPPING = {
    0: ("Longitudinal Crack", "Standard", False),
    1: ("Longitudinal Crack", "Wide (Severe)", True),
    2: ("Transverse Crack", "Standard", False),
    3: ("Transverse Crack", "Wide (Severe)", True),
    4: ("Alligator Crack", "Standard", False),
    5: ("Alligator Crack", "Sunken (Severe)", True),
    6: ("Pothole", "Standard", False),
    7: ("Pothole", "Deep (Severe)", True),
}

FAMILY_COLORS = {
    "Longitudinal Crack": "#3B82F6",
    "Transverse Crack": "#10B981",
    "Alligator Crack": "#F59E0B",
    "Pothole": "#EF4444"
}

# ---------------------------------------------------------
# Cached Model Loader with GPU Warm-up
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading and warming up YOLO26n V1 Model on GPU...")
def load_v1_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None, "None"
    
    device_str = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    model = YOLO(str(MODEL_PATH))
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy_img, device=device_str, verbose=False)
    
    return model, device_name

# ---------------------------------------------------------
# Box IoU & Evidence-Preserving Deduplication
# ---------------------------------------------------------
def box_iou_numpy(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
    b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = b1_area + b2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def process_and_deduplicate_with_evidence(raw_detections, iou_thresh=0.60, enable_dedup=True):
    """
    Suppresses redundant duplicate bounding boxes within the SAME family,
    but crucially PRESERVES all competing subclass evidence (e.g. standard vs severe variants).
    Suppressed duplicate boxes DO NOT increment the distinct defect count.
    """
    if not enable_dedup or len(raw_detections) <= 1:
        for d in raw_detections:
            d['competing_evidence'] = []
        return raw_detections
    
    sorted_dets = sorted(raw_detections, key=lambda d: d['confidence'], reverse=True)
    kept = []
    
    for d in sorted_dets:
        matched = False
        for k in kept:
            if d['family'] == k['family']:
                iou = box_iou_numpy(d['box_xyxy'], k['box_xyxy'])
                if iou >= iou_thresh:
                    # Record suppressed duplicate as competing subclass evidence
                    k['competing_evidence'].append({
                        'label': d['source_label'],
                        'conf': d['confidence'],
                        'variant': d['variant'],
                        'is_severe': d['is_severe'],
                        'iou': iou
                    })
                    matched = True
                    break
        if not matched:
            d['competing_evidence'] = []
            kept.append(d)
            
    return kept

def evaluate_provisional_severity(primary_variant, primary_conf, is_severe, competing_list):
    """
    Determines evidence-backed provisional severity indicator.
    Does NOT mechanically map standard to Low or severe to High.
    Accurately reports when model subclasses conflict or provide visual cues.
    """
    has_severe_competing = any(c['is_severe'] for c in competing_list)
    has_standard_competing = any(not c['is_severe'] for c in competing_list)
    
    if is_severe and has_standard_competing:
        comp_str = ", ".join([f"{c['label']} ({c['conf']:.0%})" for c in competing_list])
        return f"Ambiguous Subclasses (Severe {primary_conf:.0%} vs Standard {comp_str})"
    elif (not is_severe) and has_severe_competing:
        comp_str = ", ".join([f"{c['label']} ({c['conf']:.0%})" for c in competing_list])
        return f"Ambiguous Subclasses (Standard {primary_conf:.0%} vs Severe {comp_str})"
    elif is_severe:
        return f"Visual Severe Cue ({primary_variant}) — Gauge Verification Required"
    else:
        return "Standard Surface Distress (Visual Cue Only)"

def annotate_image(image: Image.Image, detections: list):
    """Annotates PIL Image with high-visibility color-coded boxes and labels."""
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    w, h = image.size
    
    try:
        font = ImageFont.truetype("arial.ttf", size=max(13, int(min(w, h) * 0.024)))
    except Exception:
        font = ImageFont.load_default()
        
    for idx, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det['box_xyxy']
        family = det['family']
        color = FAMILY_COLORS.get(family, "#EF4444")
        
        for offset in range(3):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)
            
        label_text = f"#{idx} {family} ({det['confidence']:.0%})"
        
        try:
            bbox = draw.textbbox((x1, max(0, y1 - 22)), label_text, font=font)
        except Exception:
            bbox = (x1, max(0, y1 - 20), x1 + len(label_text) * 8, y1)
            
        draw.rectangle(bbox, fill=color)
        draw.text((x1 + 4, max(0, y1 - 22) + 2), label_text, fill="#FFFFFF", font=font)
        
    return draw_img

# ---------------------------------------------------------
# Main Application
# ---------------------------------------------------------
def main():
    st.markdown('<div class="brand-title">PaveSense AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="project-subtitle">AI-Based Road Damage Detection and Severity Analysis</div>', unsafe_allow_html=True)
    
    # Rigorous Metrology Realism Disclosure
    st.markdown("""
    <div class="disclaimer-banner">
        <strong>⚠️ Metrology & Physical Severity Disclosure:</strong> 
        Monocular 2D images cannot measure physical pothole cavity depth or crack width (millimeter scale) without calibrated 3D depth sensors. 
        PaveSense AI provides automated screening to prioritize human engineer field inspections. 
        Visual variants (wide, sunken, deep) are provisional model cues and are explicitly marked as <em>Physical Severity: Unverified</em> pending field inspection.
    </div>
    """, unsafe_allow_html=True)
    
    # -----------------------------------------------------
    # Sidebar: Inspection & Model Controls
    # -----------------------------------------------------
    st.sidebar.markdown("### ⚙️ Inspection Configuration")
    
    road_id = st.sidebar.text_input(
        "Road Section / Chainage ID", 
        value="NH-24, KM 14.8", 
        help="Asset identifier for road section reporting"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Detection Parameters")
    
    conf_thresh = st.sidebar.slider(
        "Confidence Operating Threshold", 
        min_value=0.05, 
        max_value=0.70, 
        value=0.15, 
        step=0.05,
        help="Empirically verified on 2,560 validation images: conf=0.15 achieves peak model F1 (0.48) balancing detection of subtle cracks/potholes against false alarms."
    )
    
    iou_thresh = st.sidebar.slider(
        "NMS Overlap Threshold", 
        min_value=0.30, 
        max_value=0.80, 
        value=0.60, 
        step=0.05,
        help="IoU threshold for non-maximum suppression (default 0.60)."
    )
    
    enable_family_nms = st.sidebar.checkbox(
        "Family-Aware Duplicate Suppression", 
        value=True,
        help="Suppresses duplicate overlapping detections from the same family while preserving competing subclass evidence."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Input Selection")
    
    input_source = st.sidebar.radio(
        "Select Road Image Source:",
        ("Upload Custom Road Photo", "Load Demo Validation Sample")
    )
    
    image_to_process = None
    sample_name = ""
    
    if input_source == "Upload Custom Road Photo":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Road Image (JPG, PNG, JPEG):", 
            type=["jpg", "jpeg", "png"],
            help="Upload field photo or dashcam snapshot"
        )
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
            sample_name = uploaded_file.name
    else:
        sample_options = {
            "Czech_000086.jpg (Conflicting Subclasses: Standard & Sunken Alligator)": "road_damage_yolo/images/val/Czech_000086.jpg",
            "Czech_000809.jpg (Multi-Defect: Alligator, Cracks & Deep Pothole)": "road_damage_yolo/images/val/Czech_000809.jpg",
            "Czech_000070.jpg (Longitudinal & Transverse Cracks)": "road_damage_yolo/images/val/Czech_000070.jpg",
            "Czech_000014.jpg (Clean Road - Background Negative)": "road_damage_yolo/images/val/Czech_000014.jpg",
        }
        selected_sample = st.sidebar.selectbox("Choose Sample (from 2,560 Validation Split):", list(sample_options.keys()))
        sample_path = Path(sample_options[selected_sample])
        if sample_path.exists():
            image_to_process = Image.open(sample_path)
            sample_name = sample_path.name
            st.sidebar.info(f"Loaded: `{sample_name}`\n*(Source: Held-out Validation Split)*")

    # Load Model with Warm-up
    model, device_name = load_v1_model()
    if model is None:
        st.stop()
        
    device_param = 0 if torch.cuda.is_available() else "cpu"

    # -----------------------------------------------------
    # Inference & Evidence Extraction
    # -----------------------------------------------------
    if image_to_process is not None:
        t0 = time.time()
        
        results = model.predict(
            source=image_to_process,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=640,
            device=device_param,
            verbose=False
        )
        inference_time_ms = (time.time() - t0) * 1000
        
        raw_boxes = results[0].boxes
        raw_detections = []
        img_w, img_h = image_to_process.size
        img_area = img_w * img_h
        
        if raw_boxes is not None and len(raw_boxes) > 0:
            for b in raw_boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                cls_id = int(b.cls[0].item())
                confidence = float(b.conf[0].item())
                
                source_label = CLASS_NAMES_8.get(cls_id, f"Class {cls_id}")
                family_name, variant, is_sev = FAMILY_MAPPING.get(cls_id, ("Unknown", "Standard", False))
                
                box_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                area_pct = (box_area / img_area) * 100.0 if img_area > 0 else 0.0
                
                raw_detections.append({
                    "box_xyxy": xyxy,
                    "class_id": cls_id,
                    "source_label": source_label,
                    "family": family_name,
                    "variant": variant,
                    "is_severe": is_sev,
                    "confidence": confidence,
                    "area_pct": area_pct
                })
                
        # Deduplicate within family while preserving competing subclass evidence
        detections = process_and_deduplicate_with_evidence(
            raw_detections, 
            iou_thresh=iou_thresh, 
            enable_dedup=enable_family_nms
        )
            
        annotated_img = annotate_image(image_to_process, detections)
        
        # -----------------------------------------------------
        # Visual Display & Inspection Metrics
        # -----------------------------------------------------
        col_img, col_metrics = st.columns([1.35, 1.0])
        
        with col_img:
            st.markdown("#### 📸 Inspection Image Feed")
            st.image(
                annotated_img, 
                use_container_width=True, 
                caption=f"PaveSense Analyzed: {sample_name} | {len(detections)} defect(s) detected in {inference_time_ms:.1f} ms"
            )
            
            buf = io.BytesIO()
            annotated_img.save(buf, format="JPEG", quality=95)
            st.download_button(
                label="📥 Download Annotated Inspection Image",
                data=buf.getvalue(),
                file_name=f"pavesense_{sample_name}",
                mime="image/jpeg"
            )
            
        with col_metrics:
            st.markdown("#### 📊 Screening Status & Family Counts")
            
            family_counts = {
                "Pothole": sum(1 for d in detections if d['family'] == "Pothole"),
                "Alligator Crack": sum(1 for d in detections if d['family'] == "Alligator Crack"),
                "Longitudinal Crack": sum(1 for d in detections if d['family'] == "Longitudinal Crack"),
                "Transverse Crack": sum(1 for d in detections if d['family'] == "Transverse Crack"),
            }
            
            m1, m2 = st.columns(2)
            m1.metric("Defects Detected", len(detections))
            
            if len(detections) == 0:
                m2.markdown('<div class="badge-clear">🟢 No Visual Defects Above Threshold</div>', unsafe_allow_html=True)
            else:
                m2.markdown('<div class="badge-unverified">⚠️ Physical Severity Unverified</div>', unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            f1, f2 = st.columns(2)
            f1.metric("🕳️ Potholes", family_counts["Pothole"])
            f2.metric("🐊 Alligator Cracks", family_counts["Alligator Crack"])
            
            f3, f4 = st.columns(2)
            f3.metric("📏 Longitudinal Cracks", family_counts["Longitudinal Crack"])
            f4.metric("📐 Transverse Cracks", family_counts["Transverse Crack"])
            
            st.markdown(f"""
            <div class="dark-card" style="margin-top: 14px;">
                <strong>Asset ID:</strong> <code>{road_id}</code><br>
                <strong>Evaluation Backbone:</strong> Frozen YOLO26n V1<br>
                <strong>Hardware Accelerator:</strong> {device_name}<br>
                <strong>Inference Latency:</strong> {inference_time_ms:.1f} ms (Warm)<br>
                <strong>Operating Settings:</strong> Conf ≥ {conf_thresh:.2f} | NMS IoU ≤ {iou_thresh:.2f}
            </div>
            """, unsafe_allow_html=True)
            
        # -----------------------------------------------------
        # Detailed Defect Shortlist Table & Human-in-the-Loop Audit
        # -----------------------------------------------------
        st.markdown("---")
        st.markdown("#### 📋 Field Shortlist, Competing Subclass Evidence & Human Review")
        
        if len(detections) > 0:
            table_data = []
            for i, d in enumerate(detections, 1):
                comp_evidence = d.get('competing_evidence', [])
                if comp_evidence:
                    comp_str = "; ".join([f"{c['label']} ({c['conf']:.1%})" for c in comp_evidence])
                else:
                    comp_str = "None (Unanimous)"
                    
                prov_severity = evaluate_provisional_severity(
                    d["variant"], 
                    d["confidence"], 
                    d["is_severe"], 
                    comp_evidence
                )
                
                table_data.append({
                    "Defect #": f"#{i}",
                    "Damage Family": d["family"],
                    "Annotated Visual Variant": f"{d['source_label']} ({d['confidence']:.1%})",
                    "Competing Subclass Evidence": comp_str,
                    "Provisional Severity Indicator": prov_severity,
                    "Physical Severity": "Unverified (2D photo)",
                    "Auditor Assigned Severity (Human)": "Pending / Not Rated",
                    "Review Notes (Engineer)": "Inspection required",
                    "Audit Decision": "Pending Field Audit"
                })
                
            df_defects = pd.DataFrame(table_data)
            
            edited_df = st.data_editor(
                df_defects,
                column_config={
                    "Auditor Assigned Severity (Human)": st.column_config.SelectboxColumn(
                        "Auditor Assigned Severity (Human)",
                        options=[
                            "Pending / Not Rated",
                            "Low Distress",
                            "Medium Distress",
                            "High Distress",
                            "Uncertain / Unclear"
                        ],
                        required=True,
                        help="Certified engineer assigns physical severity after field measurement."
                    ),
                    "Review Notes (Engineer)": st.column_config.TextColumn(
                        "Review Notes (Engineer)",
                        required=False,
                        help="Field audit notes, depth gauge readings, or site context."
                    ),
                    "Audit Decision": st.column_config.SelectboxColumn(
                        "Review Action",
                        options=[
                            "Pending Field Audit", 
                            "Confirmed True Defect", 
                            "False Alarm / Shadow / Patch", 
                            "Escalate to PWD Work-Order"
                        ],
                        required=True
                    )
                },
                disabled=[
                    "Defect #", "Damage Family", "Annotated Visual Variant", 
                    "Competing Subclass Evidence", "Provisional Severity Indicator", "Physical Severity"
                ],
                use_container_width=True,
                hide_index=True
            )
            
            csv_export = edited_df.to_csv(index=False)
            st.download_button(
                label="📄 Export Shortlist Report with Severity Audit (CSV)",
                data=csv_export,
                file_name=f"pavesense_report_{sample_name.replace('.jpg', '')}.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No visual road defects detected above current operating threshold. Road section appears in serviceable visual condition.")
            
        # -----------------------------------------------------
        # Presentation Defense Section
        # -----------------------------------------------------
        st.markdown("---")
        with st.expander("💡 Practical Significance and Operational Value (Engineering Workflow)"):
            st.markdown("""
            ### The Core Practical Challenge
            A standalone deep learning model that merely draws bounding boxes does not repair a road or generate municipal action.
            
            ### PaveSense AI Operational Value Pipeline:
            ```
            [1. Video / Mobile Survey] → Continuous capture of road network frames
                         ↓
            [2. PaveSense AI Triage]   → Automatically filters out sound pavement (~90% of frames)
                         ↓
            [3. Shortlist Generation]  → Aggregates candidate damage locations with competing subclass evidence
                         ↓
            [4. Engineer Field Audit]  → Human assigns Low/Medium/High/Uncertain with calibrated depth probes
                         ↓
            [5. Work-Order Dispatch]   → Generates formal PWD repair work-orders for confirmed distress
            ```
            
            ### Technical Ground Truth on Model Accuracy
            - **Honest Metrics:** On 2,560 fixed validation images and 2,560 held-out test images, the model achieves **53.4% Precision, 44.3% Recall, and 42.9% mAP50** on the four damage families.
            - **No Fabricated Claims:** We do not claim 90%+ accuracy. The system is designed as a high-throughput **screening and shortlisting assistant**, eliminating hundreds of hours of manual video scanning.
            """)
    else:
        st.info("👆 Please select a demo validation sample or upload a road photo from the sidebar to inspect.")

if __name__ == "__main__":
    main()
