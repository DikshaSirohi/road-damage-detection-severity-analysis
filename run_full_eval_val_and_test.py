"""Full 2,560-Image Validation and Test Evaluation for PaveSense AI
Evaluates frozen YOLO26n V1 with benchmark-selected settings:
  - imgsz = 640
  - conf = 0.15 (operational point) / conf = 0.001 (AP curve)
  - Family-Aware NMS IoU = 0.60
Calculates overall and per-family metrics on held-out validation and test sets.
"""

from pathlib import Path
import time
import csv
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'runs' / 'road_damage_yolo26n_v1' / 'weights' / 'best.pt'
OUT_DIR = ROOT / 'final_evaluation_results'
OUT_DIR.mkdir(exist_ok=True)

MAPPING = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
FAMILIES = {0: 'Longitudinal Crack', 1: 'Transverse Crack', 2: 'Alligator Crack', 3: 'Pothole'}
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
        return boxes, confs, clss
    order = np.argsort(-confs)
    keep = []
    boxes_np = np.asarray(boxes)
    for i in order:
        fam_i = clss[i]
        discard = False
        for k in keep:
            if fam_i == clss[k]:
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
                    break
        if not discard:
            keep.append(i)
    return [boxes[i] for i in keep], [confs[i] for i in keep], [clss[i] for i in keep]

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

def evaluate_split(model, split_name, img_dir, lbl_dir):
    print(f"\n=======================================================")
    print(f"EVALUATING SPLIT: {split_name.upper()} (2,560 images)")
    print(f"=======================================================")
    
    img_paths = sorted(list(img_dir.glob('*.jpg')))
    if len(img_paths) != 2560:
        raise ValueError(f"Expected 2560 images in {split_name}, found {len(img_paths)}")
        
    all_tp = []
    all_conf = []
    all_pred = []
    all_gt = []
    
    t0 = time.time()
    
    # Run batch inference with stream=True for speed
    stream = model.predict(
        source=str(img_dir),
        imgsz=640,
        conf=0.001,
        iou=0.50,
        device=0,
        batch=16,
        stream=True,
        verbose=False
    )
    
    processed = 0
    for result in stream:
        p = Path(result.path)
        lbl_p = lbl_dir / (p.stem + '.txt')
        h, w = result.orig_shape
        gt_cls, gt_boxes = load_ground_truth(lbl_p, w, h)
        all_gt.append(gt_cls)
        
        pred_boxes = result.boxes.xyxy.cpu().numpy().tolist()
        pred_confs = result.boxes.conf.cpu().numpy().tolist()
        pred_ids = result.boxes.cls.cpu().numpy().astype(np.int64).tolist()
        
        pred_fam = [int(MAPPING[c]) for c in pred_ids]
        
        # Apply Family-Aware NMS
        if len(pred_boxes) > 0:
            pred_boxes, pred_confs, pred_fam = suppress_family_duplicates_np(
                pred_boxes, np.array(pred_confs), pred_fam, iou_thresh=0.60
            )
            
        tp = match_detections(pred_boxes, np.array(pred_fam, dtype=np.int64), gt_boxes, gt_cls)
        all_tp.append(tp)
        all_conf.append(np.array(pred_confs))
        all_pred.append(np.array(pred_fam, dtype=np.int64))
        
        processed += 1
        if processed % 640 == 0:
            print(f"[{split_name.upper()}] Processed {processed}/2560 images ({time.time() - t0:.1f} s)...")
            
    total_time = time.time() - t0
    fps = processed / total_time
    
    tps = np.concatenate(all_tp)
    confs = np.concatenate(all_conf)
    preds = np.concatenate(all_pred)
    gts = np.concatenate(all_gt)
    
    # Calculate AP per family
    ap_res = ap_per_class(tps, confs, preds, gts, names=FAMILIES)
    precisions, recalls, f1s, aps, unique_classes = ap_res[2], ap_res[3], ap_res[4], ap_res[5], ap_res[6]
    
    overall_map50 = float(np.mean(aps[:, 0]))
    overall_map5095 = float(np.mean(aps))
    
    # Operational metrics at conf=0.15
    mask15 = confs >= 0.15
    n_pred15 = int(np.sum(mask15))
    tp15 = int(np.sum(tps[mask15, 0]))
    p15 = tp15 / n_pred15 if n_pred15 > 0 else 0.0
    r15 = tp15 / len(gts) if len(gts) > 0 else 0.0
    f115 = (2 * p15 * r15 / (p15 + r15)) if (p15 + r15) > 0 else 0.0
    
    # Per family breakdown
    family_rows = []
    for cls_id, fam_name in FAMILIES.items():
        matches = np.where(unique_classes == cls_id)[0]
        if len(matches) > 0:
            idx = int(matches[0])
            fam_p = float(precisions[idx])
            fam_r = float(recalls[idx])
            fam_f1 = float(f1s[idx]) if len(f1s) > idx else 0.0
            fam_ap50 = float(aps[idx, 0])
            fam_ap5095 = float(np.mean(aps[idx]))
        else:
            fam_p = fam_r = fam_f1 = fam_ap50 = fam_ap5095 = 0.0
            
        gt_count = int(np.sum(gts == cls_id))
        pred_count_15 = int(np.sum((preds == cls_id) & mask15))
        
        family_rows.append({
            'family': fam_name,
            'gt_boxes': gt_count,
            'pred_boxes_015': pred_count_15,
            'precision': fam_p,
            'recall': fam_r,
            'f1': fam_f1,
            'ap50': fam_ap50,
            'ap50_95': fam_ap5095
        })
        
    # Write CSV
    csv_file = OUT_DIR / f"{split_name}_family_metrics.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Damage Family', 'GT Boxes', 'Pred Boxes (@0.15)', 'Precision', 'Recall', 'F1', 'AP50', 'AP50_95'])
        for r in family_rows:
            writer.writerow([r['family'], r['gt_boxes'], r['pred_boxes_015'], f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}", f"{r['ap50']:.4f}", f"{r['ap50_95']:.4f}"])
            
    summary_text = (
        f"--- {split_name.upper()} RESULTS (2,560 images) ---\n"
        f"Inference Time: {total_time:.1f} s ({fps:.1f} FPS, {1000/fps:.1f} ms/image)\n"
        f"Total Ground Truth Boxes: {len(gts)}\n"
        f"Operational Point (conf=0.15): Precision={p15:.4f}, Recall={r15:.4f}, F1={f115:.4f}\n"
        f"Overall Grouped mAP50: {overall_map50:.4f}\n"
        f"Overall Grouped mAP50-95: {overall_map5095:.4f}\n\n"
        f"{'Family':<20} | {'GT':<6} | {'P':<7} | {'R':<7} | {'AP50':<7} | {'AP50-95':<8}\n"
        + "-" * 62 + "\n"
    )
    for r in family_rows:
        summary_text += f"{r['family']:<20} | {r['gt_boxes']:<6} | {r['precision']:<7.4f} | {r['recall']:<7.4f} | {r['ap50']:<7.4f} | {r['ap50_95']:<8.4f}\n"
        
    print("\n" + summary_text)
    (OUT_DIR / f"{split_name}_summary.txt").write_text(summary_text, encoding='utf-8')
    
    return {
        'split': split_name,
        'p15': p15,
        'r15': r15,
        'f115': f115,
        'map50': overall_map50,
        'map5095': overall_map5095,
        'fps': fps,
        'family_rows': family_rows
    }

def main():
    model = YOLO(str(WEIGHTS))
    
    # 1. Evaluate full Validation split (2,560 images)
    val_res = evaluate_split(
        model, 
        'validation', 
        ROOT / 'road_damage_yolo' / 'images' / 'val',
        ROOT / 'road_damage_yolo' / 'labels' / 'val'
    )
    
    # 2. Evaluate held-out Test split (2,560 images)
    test_res = evaluate_split(
        model, 
        'test', 
        ROOT / 'road_damage_yolo' / 'images' / 'test',
        ROOT / 'road_damage_yolo' / 'labels' / 'test'
    )
    
    # 3. Save side-by-side comparison
    comp_text = (
        "PAVESENSE AI — FROZEN V1 FINAL EVALUATION REPORT\n"
        "Benchmark Configuration: imgsz=640, conf=0.15, Family-Aware NMS IoU=0.60\n"
        "Model Checkpoint: runs/road_damage_yolo26n_v1/weights/best.pt\n\n"
        f"{'Metric':<25} | {'Validation (2,560 imgs)':<22} | {'Test (2,560 imgs)':<20}\n"
        + "=" * 74 + "\n"
        f"{'Operational Precision (@0.15)':<25} | {val_res['p15']:<22.4f} | {test_res['p15']:<20.4f}\n"
        f"{'Operational Recall (@0.15)':<25} | {val_res['r15']:<22.4f} | {test_res['r15']:<20.4f}\n"
        f"{'Operational F1 Score':<25} | {val_res['f115']:<22.4f} | {test_res['f115']:<20.4f}\n"
        f"{'Grouped mAP50':<25} | {val_res['map50']:<22.4f} | {test_res['map50']:<20.4f}\n"
        f"{'Grouped mAP50-95':<25} | {val_res['map5095']:<22.4f} | {test_res['map5095']:<20.4f}\n"
        f"{'Throughput (FPS)':<25} | {val_res['fps']:<22.1f} | {test_res['fps']:<20.1f}\n\n"
        "PER-FAMILY AP50 COMPARISON:\n"
    )
    for vf, tf in zip(val_res['family_rows'], test_res['family_rows']):
        comp_text += f"  {vf['family']:<20}: Val AP50 = {vf['ap50']:.4f} | Test AP50 = {tf['ap50']:.4f}\n"
        
    (OUT_DIR / "final_val_vs_test_comparison.txt").write_text(comp_text, encoding='utf-8')
    print("\n" + comp_text)
    print(f"Results successfully saved to {OUT_DIR}/")

if __name__ == '__main__':
    main()
