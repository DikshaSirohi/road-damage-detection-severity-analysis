"""Independent Detection Accuracy Benchmark for PaveSense AI.
Evaluates frozen YOLO26n V1 across:
  - Resolutions: imgsz 640 vs 800 vs 960
  - Post-processing: Standard NMS vs Family-Aware NMS
  - Tiled / Patch-based inference vs Full-Image
  - Metrics: Precision, Recall, F1 (@ conf=0.15), mAP50, mAP50-95, Latency (ms)
"""

from pathlib import Path
import time
import csv
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'runs' / 'road_damage_yolo26n_v1' / 'weights' / 'best.pt'
VAL_IMG_DIR = ROOT / 'road_damage_yolo' / 'images' / 'val'
VAL_LBL_DIR = ROOT / 'road_damage_yolo' / 'labels' / 'val'
OUT_DIR = ROOT / 'benchmark_results'
OUT_DIR.mkdir(exist_ok=True)

# 8-class to 4-family mapping
# 0,1 -> Longitudinal (0)
# 2,3 -> Transverse (1)
# 4,5 -> Alligator (2)
# 6,7 -> Pothole (3)
MAPPING = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
FAMILIES = {0: 'Longitudinal', 1: 'Transverse', 2: 'Alligator', 3: 'Pothole'}
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)

def load_ground_truth(label_path, w, h):
    classes, boxes = [], []
    if label_path.exists():
        for line in label_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            x, y, bw, bh = map(float, parts[1:])
            classes.append(int(MAPPING[cls]))
            boxes.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    if classes:
        return np.array(classes, dtype=np.int64), torch.tensor(boxes, dtype=torch.float32)
    return np.array([], dtype=np.int64), torch.zeros((0, 4), dtype=torch.float32)

def suppress_family_duplicates_np(boxes, confs, clss, iou_thresh=0.60):
    if len(boxes) <= 1:
        return boxes, confs, clss, 0
    order = np.argsort(-confs)
    keep = []
    suppressed_count = 0
    boxes_np = np.asarray(boxes)
    for i in order:
        fam_i = clss[i]
        discard = False
        for k in keep:
            if fam_i == clss[k]:
                # Calculate IoU
                b1, b2 = boxes_np[i], boxes_np[k]
                x1 = max(b1[0], b2[0])
                y1 = max(b1[1], b2[1])
                x2 = min(b1[2], b2[2])
                y2 = min(b1[3], b2[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                union = a1 + a2 - inter
                iou = inter / union if union > 0 else 0
                if iou >= iou_thresh:
                    discard = True
                    suppressed_count += 1
                    break
        if not discard:
            keep.append(i)
    return [boxes[i] for i in keep], [confs[i] for i in keep], [clss[i] for i in keep], suppressed_count

def match_detections(pred_boxes, pred_classes, gt_boxes, gt_classes):
    n_pred, n_gt = len(pred_classes), len(gt_classes)
    correct = np.zeros((n_pred, 10), dtype=bool)
    if not n_pred or not n_gt:
        return correct
    iou = box_iou(gt_boxes, torch.tensor(pred_boxes, dtype=torch.float32)).cpu().numpy()
    same_cls = (gt_classes[:, None] == pred_classes[None, :])
    iou[~same_cls] = 0
    for k, threshold in enumerate(IOU_THRESHOLDS):
        rows, cols = np.where(iou >= threshold)
        if rows.size == 0:
            continue
        matches = np.column_stack([rows, cols, iou[rows, cols]])
        matches = matches[np.argsort(-matches[:, 2])]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.argsort(-matches[:, 2])]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), k] = True
    return correct

def run_tiled_inference(model, img, imgsz=640, conf=0.001, iou=0.5):
    """2x2 overlapping grid tiled inference merged with full image."""
    w, h = img.size
    full_res = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, device=0, verbose=False)[0]
    boxes = [b.xyxy[0].tolist() for b in full_res.boxes]
    confs = [float(b.conf[0]) for b in full_res.boxes]
    clss = [int(b.cls[0]) for b in full_res.boxes]
    
    # 2x2 tiles with 20% overlap
    tiles = [
        (0, 0, int(w * 0.6), int(h * 0.6)),
        (int(w * 0.4), 0, w, int(h * 0.6)),
        (0, int(h * 0.4), int(w * 0.6), h),
        (int(w * 0.4), int(h * 0.4), w, h),
    ]
    for x1, y1, x2, y2 in tiles:
        tile_crop = img.crop((x1, y1, x2, y2))
        tile_res = model.predict(tile_crop, imgsz=imgsz, conf=conf, iou=iou, device=0, verbose=False)[0]
        for b in tile_res.boxes:
            tb = b.xyxy[0].tolist()
            # Unscale to full image
            boxes.append([tb[0] + x1, tb[1] + y1, tb[2] + x1, tb[3] + y1])
            confs.append(float(b.conf[0]))
            clss.append(int(b.cls[0]))
            
    return boxes, confs, clss

def evaluate_configuration(model, img_paths, imgsz=640, use_fnms=False, use_tiling=False, fnms_iou=0.60):
    all_tp = []
    all_conf = []
    all_pred = []
    all_gt = []
    total_suppressed = 0
    latencies = []
    
    device_id = 0 if torch.cuda.is_available() else 'cpu'
    
    for p in img_paths:
        lbl_p = VAL_LBL_DIR / (p.stem + '.txt')
        img = Image.open(p)
        w, h = img.size
        gt_cls, gt_boxes = load_ground_truth(lbl_p, w, h)
        all_gt.append(gt_cls)
        
        t0 = time.time()
        if use_tiling:
            pred_boxes, pred_confs, pred_ids = run_tiled_inference(model, img, imgsz=imgsz, conf=0.001, iou=0.50)
        else:
            res = model.predict(img, imgsz=imgsz, conf=0.001, iou=0.50, device=device_id, verbose=False)[0]
            pred_boxes = [b.xyxy[0].tolist() for b in res.boxes]
            pred_confs = [float(b.conf[0]) for b in res.boxes]
            pred_ids = [int(b.cls[0]) for b in res.boxes]
        latencies.append((time.time() - t0) * 1000)
        
        # Map 8 classes to 4 families
        pred_fam = [int(MAPPING[c]) for c in pred_ids]
        
        # Apply Family-Aware NMS if enabled
        if use_fnms and len(pred_boxes) > 0:
            pred_boxes, pred_confs, pred_fam, supp = suppress_family_duplicates_np(
                pred_boxes, np.array(pred_confs), pred_fam, iou_thresh=fnms_iou
            )
            total_suppressed += supp
            
        tp = match_detections(pred_boxes, np.array(pred_fam, dtype=np.int64), gt_boxes, gt_cls)
        all_tp.append(tp)
        all_conf.append(np.array(pred_confs))
        all_pred.append(np.array(pred_fam, dtype=np.int64))
        
    tps = np.concatenate(all_tp) if len(all_tp) and any(len(x) for x in all_tp) else np.zeros((0, 10), dtype=bool)
    confs = np.concatenate(all_conf) if len(all_conf) and any(len(x) for x in all_conf) else np.zeros(0)
    preds = np.concatenate(all_pred) if len(all_pred) and any(len(x) for x in all_pred) else np.zeros(0, dtype=np.int64)
    gts = np.concatenate(all_gt) if len(all_gt) and any(len(x) for x in all_gt) else np.zeros(0, dtype=np.int64)
    
    # mAP calculation using ap_per_class
    if len(tps) > 0:
        ap_res = ap_per_class(tps, confs, preds, gts, names=FAMILIES)
        precisions, recalls, aps = ap_res[2], ap_res[3], ap_res[5]
        map50 = float(np.mean(aps[:, 0]))
        map5095 = float(np.mean(aps))
    else:
        map50, map5095 = 0.0, 0.0
        
    # Precision, Recall, F1 at operational threshold conf=0.15
    mask15 = confs >= 0.15
    n_pred15 = int(np.sum(mask15))
    if n_pred15 > 0 and len(gts) > 0:
        tp15 = int(np.sum(tps[mask15, 0]))
        p15 = tp15 / n_pred15
        r15 = tp15 / len(gts)
        f115 = (2 * p15 * r15 / (p15 + r15)) if (p15 + r15) > 0 else 0.0
    else:
        p15, r15, f115 = 0.0, 0.0, 0.0
        
    avg_latency = float(np.mean(latencies[5:])) if len(latencies) > 5 else float(np.mean(latencies))
    
    return {
        'precision_015': p15,
        'recall_015': r15,
        'f1_015': f115,
        'map50': map50,
        'map5095': map5095,
        'suppressed_duplicates': total_suppressed,
        'avg_latency_ms': avg_latency
    }

def main():
    print("=" * 70)
    print("PAVESENSE AI — INDEPENDENT ACCURACY & RESOLUTION BENCHMARK")
    print("=" * 70)
    
    model = YOLO(str(WEIGHTS))
    
    # Warm up GPU
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, device=0 if torch.cuda.is_available() else 'cpu', verbose=False)
    
    # Fixed representative subset: 300 validation images (every 8th image out of 2,560)
    all_val_imgs = sorted(list(VAL_IMG_DIR.glob('*.jpg')))
    subset_imgs = all_val_imgs[::8][:300]
    print(f"Using fixed representative validation subset: {len(subset_imgs)} images from {len(all_val_imgs)}")
    
    configs = [
        ("imgsz 640 | Standard NMS", 640, False, False),
        ("imgsz 640 | Family-Aware NMS", 640, True, False),
        ("imgsz 800 | Standard NMS", 800, False, False),
        ("imgsz 800 | Family-Aware NMS", 800, True, False),
        ("imgsz 960 | Standard NMS", 960, False, False),
        ("imgsz 960 | Family-Aware NMS", 960, True, False),
        ("imgsz 640 | Tiled + Family NMS", 640, True, True),
    ]
    
    results = []
    
    print(f"\n{'Configuration':<32} | {'P@0.15':<7} | {'R@0.15':<7} | {'F1@0.15':<7} | {'mAP50':<7} | {'mAP50-95':<8} | {'Suppr':<6} | {'Latency':<8}")
    print("-" * 96)
    
    for name, sz, fnms, tiled in configs:
        t_start = time.time()
        res = evaluate_configuration(model, subset_imgs, imgsz=sz, use_fnms=fnms, use_tiling=tiled)
        res['name'] = name
        results.append(res)
        
        print(f"{name:<32} | {res['precision_015']:7.4f} | {res['recall_015']:7.4f} | {res['f1_015']:7.4f} | {res['map50']:7.4f} | {res['map5095']:8.4f} | {res['suppressed_duplicates']:6d} | {res['avg_latency_ms']:6.1f} ms")
        
    # Save CSV
    csv_path = OUT_DIR / 'benchmark_comparison.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Configuration', 'Precision@0.15', 'Recall@0.15', 'F1@0.15', 'mAP50', 'mAP50_95', 'Suppressed_Duplicates', 'Latency_ms'])
        for r in results:
            writer.writerow([r['name'], f"{r['precision_015']:.4f}", f"{r['recall_015']:.4f}", f"{r['f1_015']:.4f}", f"{r['map50']:.4f}", f"{r['map5095']:.4f}", r['suppressed_duplicates'], f"{r['avg_latency_ms']:.1f}"])
            
    print(f"\nSaved benchmark results to: {csv_path}")

if __name__ == '__main__':
    main()
