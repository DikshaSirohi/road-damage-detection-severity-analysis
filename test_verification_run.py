from app import load_v1_model, suppress_family_duplicates
from PIL import Image
import time, torch

model, dev = load_v1_model()
print('Device initialized:', dev)

targets = [
    ('Czech_000809.jpg', 'road_damage_yolo/images/val/Czech_000809.jpg'),
    ('Uploaded road image (Downloads)', 'C:/Users/Diksha/Downloads/testtttttttttttttttttt_roaddddddddddd.jpg')
]

for name, path in targets:
    img = Image.open(path)
    t0 = time.time()
    res = model.predict(img, conf=0.15, iou=0.60, device=0 if torch.cuda.is_available() else 'cpu', verbose=False)[0]
    dt = (time.time() - t0) * 1000
    
    raw_boxes = []
    for b in res.boxes:
        c_id = int(b.cls[0])
        conf_val = float(b.conf[0])
        fam = 'Alligator Crack' if c_id in [4,5] else ('Longitudinal Crack' if c_id in [0,1] else ('Transverse Crack' if c_id in [2,3] else 'Pothole'))
        raw_boxes.append({
            'box_xyxy': b.xyxy[0].tolist(),
            'confidence': conf_val,
            'class_id': c_id,
            'family': fam
        })
        
    fnms = suppress_family_duplicates(raw_boxes, iou_thresh=0.60)
    
    print(f'\n=== {name} ===')
    print(f'Inference time: {dt:.1f} ms')
    print(f'Raw 8-class detections (conf >= 0.15): {len(raw_boxes)}')
    for b in raw_boxes:
        print(f"  Class {b['class_id']} ({model.names[b['class_id']]}) conf={b['confidence']:.3f}")
    print(f'Displayed after Family NMS (IoU=0.60): {len(fnms)}')
    for b in fnms:
        print(f"  Family: {b['family']}, conf={b['confidence']:.3f}, box={[round(x,1) for x in b['box_xyxy']]}")
