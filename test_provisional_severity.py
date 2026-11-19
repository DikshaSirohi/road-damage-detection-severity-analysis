from ultralytics import YOLO
from PIL import Image
import numpy as np

m = YOLO('runs/road_damage_yolo26n_v1/weights/best.pt')

CLASS_NAMES_8 = {
    0: 'longitudinal crack', 1: 'longitudinal crack wide',
    2: 'transverse crack', 3: 'transverse crack wide',
    4: 'alligator crack', 5: 'alligator crack sunken',
    6: 'pothole', 7: 'pothole deep'
}
FAMILY_MAPPING = {
    0: ('Longitudinal Crack', 'Standard', False),
    1: ('Longitudinal Crack', 'Wide (Severe)', True),
    2: ('Transverse Crack', 'Standard', False),
    3: ('Transverse Crack', 'Wide (Severe)', True),
    4: ('Alligator Crack', 'Standard', False),
    5: ('Alligator Crack', 'Sunken (Severe)', True),
    6: ('Pothole', 'Standard', False),
    7: ('Pothole', 'Deep (Severe)', True),
}

def box_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

def process_and_deduplicate(raw_dets, iou_thresh=0.60):
    sorted_dets = sorted(raw_dets, key=lambda d: d['confidence'], reverse=True)
    kept = []
    for d in sorted_dets:
        matched = False
        for k in kept:
            if d['family'] == k['family']:
                if box_iou(d['box_xyxy'], k['box_xyxy']) >= iou_thresh:
                    k['competing'].append({
                        'label': d['source_label'],
                        'conf': d['confidence'],
                        'variant': d['variant'],
                        'is_severe': d['is_severe'],
                        'iou': box_iou(d['box_xyxy'], k['box_xyxy'])
                    })
                    matched = True
                    break
        if not matched:
            d['competing'] = []
            kept.append(d)
    return kept

targets = [
    ('Czech_000086.jpg', 'road_damage_yolo/images/val/Czech_000086.jpg'),
    ('Uploaded road image', 'C:/Users/Diksha/Downloads/testtttttttttttttttttt_roaddddddddddd.jpg')
]

for name, path in targets:
    img = Image.open(path)
    res = m.predict(img, conf=0.15, iou=0.60, device=0, verbose=False)[0]
    raw = []
    for b in res.boxes:
        cls_id = int(b.cls[0])
        fam, var, is_sev = FAMILY_MAPPING[cls_id]
        raw.append({
            'box_xyxy': b.xyxy[0].tolist(),
            'class_id': cls_id,
            'source_label': CLASS_NAMES_8[cls_id],
            'family': fam,
            'variant': var,
            'is_severe': is_sev,
            'confidence': float(b.conf[0])
        })
    kept = process_and_deduplicate(raw)
    print(f'\n=== {name} ===')
    print(f'Raw boxes: {len(raw)} | Deduplicated defects: {len(kept)}')
    for i, d in enumerate(kept, 1):
        comp_str = ', '.join([f"{c['label']} ({c['conf']:.1%})" for c in d['competing']]) if d['competing'] else 'None (Unanimous)'
        print(f"  #{i} {d['family']} | Primary: {d['source_label']} ({d['confidence']:.1%}) | Competing: {comp_str}")
