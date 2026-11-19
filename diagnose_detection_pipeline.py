"""Detection Pipeline Investigation Script
Tests Czech_000809.jpg across confidence thresholds (0.10, 0.15, 0.20, 0.25),
evaluates raw 8-class vs 4-family post-processing, and generates before/after images.
"""
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'runs' / 'road_damage_yolo26n_v1' / 'weights' / 'best.pt'
IMG_PATH = ROOT / 'road_damage_yolo' / 'images' / 'val' / 'Czech_000809.jpg'

CLASS_NAMES = {
    0: 'longitudinal crack',
    1: 'longitudinal crack wide',
    2: 'transverse crack',
    3: 'transverse crack wide',
    4: 'alligator crack',
    5: 'alligator crack sunken',
    6: 'pothole',
    7: 'pothole deep'
}

FAMILY_NAMES = {
    0: 'Longitudinal Crack',
    1: 'Longitudinal Crack',
    2: 'Transverse Crack',
    3: 'Transverse Crack',
    4: 'Alligator Crack',
    5: 'Alligator Crack',
    6: 'Pothole',
    7: 'Pothole'
}

COLORS = {
    'Longitudinal Crack': (37, 99, 235),     # Blue
    'Transverse Crack': (5, 150, 105),       # Green
    'Alligator Crack': (217, 119, 6),        # Amber
    'Pothole': (220, 38, 38)                 # Red
}

def box_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

def family_nms(boxes, confs, clss, iou_thresh=0.60):
    """Suppress within-family duplicates only when overlap is high (e.g. severe+normal duplicate)."""
    if len(boxes) <= 1:
        return boxes, confs, clss
    order = np.argsort(-confs)
    keep = []
    for i in order:
        fam_i = FAMILY_NAMES[clss[i]]
        discard = False
        for k in keep:
            fam_k = FAMILY_NAMES[clss[k]]
            if fam_i == fam_k:
                if box_iou(boxes[i], boxes[k]) >= iou_thresh:
                    discard = True
                    break
        if not discard:
            keep.append(i)
    return [boxes[i] for i in keep], [confs[i] for i in keep], [clss[i] for i in keep]

def draw_detections(img_pil, boxes, confs, clss, title):
    draw_img = img_pil.copy()
    draw = ImageDraw.Draw(draw_img)
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()
        
    for i, (b, c, cls) in enumerate(zip(boxes, confs, clss), 1):
        fam = FAMILY_NAMES[cls]
        raw_name = CLASS_NAMES[cls]
        color = COLORS[fam]
        # RGB color
        draw.rectangle(b, outline=color, width=3)
        label = f"#{i} {fam} ({raw_name} {c:.0%})"
        try:
            tb = draw.textbbox((b[0], max(0, b[1] - 18)), label, font=font)
        except Exception:
            tb = (b[0], max(0, b[1] - 18), b[0] + len(label)*7, b[1])
        draw.rectangle(tb, fill=color)
        draw.text((b[0] + 2, max(0, b[1] - 18) + 1), label, fill=(255, 255, 255), font=font)
    return draw_img

def main():
    model = YOLO(str(WEIGHTS))
    img = Image.open(IMG_PATH)
    out_dir = ROOT / 'detection_investigation_results'
    out_dir.mkdir(exist_ok=True)
    
    thresholds = [0.10, 0.15, 0.20, 0.25]
    
    print(f"Investigating {IMG_PATH.name}...")
    
    for conf in thresholds:
        # Standard raw prediction
        res = model.predict(img, conf=conf, iou=0.5, imgsz=640, verbose=False)[0]
        raw_boxes = [b.xyxy[0].tolist() for b in res.boxes]
        raw_confs = [float(b.conf[0]) for b in res.boxes]
        raw_clss = [int(b.cls[0]) for b in res.boxes]
        
        # Family-aware NMS
        fnms_boxes, fnms_confs, fnms_clss = family_nms(raw_boxes, np.array(raw_confs), raw_clss, iou_thresh=0.60)
        
        print(f"\n--- Confidence Threshold: {conf:.2f} ---")
        print(f"Raw YOLO detections: {len(raw_boxes)}")
        for b, c, cls in zip(raw_boxes, raw_confs, raw_clss):
            print(f"  Raw: {CLASS_NAMES[cls]:<25} ({FAMILY_NAMES[cls]}) conf={c:.3f}")
        print(f"After Family-NMS (IoU=0.60): {len(fnms_boxes)}")
        for b, c, cls in zip(fnms_boxes, fnms_confs, fnms_clss):
            print(f"  F-NMS: {CLASS_NAMES[cls]:<25} ({FAMILY_NAMES[cls]}) conf={c:.3f}")
            
        # Draw and save
        img_raw = draw_detections(img, raw_boxes, raw_confs, raw_clss, f"Raw conf={conf:.2f}")
        img_raw.save(out_dir / f"czech809_raw_conf_{int(conf*100)}.jpg")
        
        img_fnms = draw_detections(img, fnms_boxes, fnms_confs, fnms_clss, f"FNMS conf={conf:.2f}")
        img_fnms.save(out_dir / f"czech809_fnms_conf_{int(conf*100)}.jpg")

    print("\nSaved comparison images to detection_investigation_results/")

if __name__ == '__main__':
    main()
