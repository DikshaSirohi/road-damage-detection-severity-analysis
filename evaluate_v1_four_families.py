"""Read-only four-family diagnostic for frozen eight-class YOLO26n V1.

Keeps the original model, annotations and split unchanged. Output is a separate
summary CSV/TXT; this is an easier grouped-class task, not eight-class mAP.
"""
from pathlib import Path
import csv
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'runs' / 'road_damage_yolo26n_v1' / 'weights' / 'best.pt'
IMAGES = ROOT / 'road_damage_yolo' / 'images' / 'val'
LABELS = ROOT / 'road_damage_yolo' / 'labels' / 'val'
OUT = ROOT / 'four_family_v1_results'
FAMILIES = {0:'Longitudinal crack',1:'Transverse crack',2:'Alligator crack',3:'Pothole'}
MAPPING = np.array([0,0,1,1,2,2,3,3], dtype=np.int64)
IOU_THRESHOLDS = np.linspace(.5,.95,10)


def ground_truth(label_path, w, h):
    classes, boxes = [], []
    for line in label_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f'Bad annotation in {label_path}: {line}')
        cls = int(parts[0])
        if not 0 <= cls < 8:
            raise ValueError(f'Invalid class {cls} in {label_path}')
        x, y, bw, bh = map(float, parts[1:])
        classes.append(int(MAPPING[cls]))
        boxes.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
    return (np.array(classes, dtype=np.int64),
            torch.tensor(boxes, dtype=torch.float32).reshape(-1,4))


def match_detections(pred_boxes, pred_classes, gt_boxes, gt_classes):
    """One-to-one matches per IoU; same-family predictions only."""
    n_pred, n_gt = len(pred_classes), len(gt_classes)
    correct = np.zeros((n_pred,10), dtype=bool)
    if not n_pred or not n_gt:
        return correct
    iou = box_iou(gt_boxes, pred_boxes).cpu().numpy()
    same_cls = (gt_classes[:,None] == pred_classes[None,:])
    iou[~same_cls] = 0
    for k, threshold in enumerate(IOU_THRESHOLDS):
        rows, cols = np.where(iou >= threshold)
        if rows.size == 0:
            continue
        matches = np.column_stack([rows,cols,iou[rows,cols]])
        matches = matches[np.argsort(-matches[:,2])]
        matches = matches[np.unique(matches[:,1], return_index=True)[1]]
        matches = matches[np.argsort(-matches[:,2])]
        matches = matches[np.unique(matches[:,0], return_index=True)[1]]
        correct[matches[:,1].astype(int),k] = True
    return correct


def main():
    for p in (WEIGHTS, IMAGES, LABELS):
        if not p.exists():
            raise FileNotFoundError(p)
    image_count = len(list(IMAGES.glob('*.jpg')))
    if image_count != 2560:
        raise RuntimeError(f'Expected fixed validation split of 2560 JPG images, found {image_count}')
    model = YOLO(str(WEIGHTS))
    if len(model.names) != 8:
        raise RuntimeError('Expected V1 with eight source classes')
    all_tp, all_conf, all_pred, all_gt = [], [], [], []
    processed = 0
    stream = model.predict(source=str(IMAGES), stream=True, imgsz=640,
                           conf=0.001, iou=0.7, max_det=300,
                           device=0, batch=4, verbose=False, save=False)
    print('Evaluating frozen V1 on 2560 validation images...')
    for result in stream:
        label_path = LABELS / (Path(result.path).stem + '.txt')
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        h,w = result.orig_shape
        gt_cls, gt_boxes = ground_truth(label_path,w,h)
        pred_boxes = result.boxes.xyxy.cpu().float()
        pred_conf = result.boxes.conf.cpu().numpy()
        pred_ids = result.boxes.cls.cpu().numpy().astype(np.int64)
        if np.any((pred_ids < 0) | (pred_ids > 7)):
            raise RuntimeError('Unexpected prediction class ID')
        pred_cls = MAPPING[pred_ids]
        tp = match_detections(pred_boxes, pred_cls, gt_boxes, gt_cls)
        all_tp.append(tp)
        all_conf.append(pred_conf)
        all_pred.append(pred_cls)
        all_gt.append(gt_cls)
        processed += 1
        if processed % 320 == 0:
            print(f'Processed {processed}/{image_count} images')
    if processed != image_count:
        raise RuntimeError(f'Processed {processed}, expected {image_count}')
    tps = np.concatenate(all_tp)
    conf = np.concatenate(all_conf)
    pred = np.concatenate(all_pred)
    gt = np.concatenate(all_gt)
    # Calculate grouped AP directly; compatible with Ultralytics 8.4.x.
    # Save compact stats first so metrics can be recalculated without inference.
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / 'four_family_stats.npz', tp=tps, conf=conf,
                        pred_cls=pred, target_cls=gt)
    ap_result = ap_per_class(tps, conf, pred, gt, names=FAMILIES)
    precisions, recalls, aps, present_classes = ap_result[2:6]
    p = float(np.mean(precisions))
    r = float(np.mean(recalls))
    map50 = float(np.mean(aps[:, 0]))
    map5095 = float(np.mean(aps))
    OUT.mkdir(exist_ok=True)
    rows = []
    for cls,name in FAMILIES.items():
        found = np.where(np.asarray(present_classes)==cls)[0]
        if found.size:
            i = int(found[0])
            cp, cr = float(precisions[i]), float(recalls[i])
            ca50, ca5095 = float(aps[i, 0]), float(np.mean(aps[i]))
        else:
            cp=cr=ca50=ca5095=0.0
        rows.append([name,int((gt==cls).sum()),int((pred==cls).sum()),cp,cr,ca50,ca5095])
    with (OUT/'family_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['family','ground_truth_boxes','predictions','precision','recall','mAP50','mAP50_95'])
        writer.writerows(rows)
    summary = (f'V1 FOUR-FAMILY VALIDATION (derived metric, no retraining)\n'
               f'Images: {processed} | GT boxes: {len(gt)} | Predictions: {len(pred)}\n'
               f'Precision: {p:.4f}\nRecall: {r:.4f}\n'
               f'mAP50: {map50:.4f}\nmAP50-95: {map5095:.4f}\n\n'
               'Family | GT | P | R | AP50 | AP50-95\n')
    for name,n,_,cp,cr,ca50,ca5095 in rows:
        summary += f'{name} | {n} | {cp:.3f} | {cr:.3f} | {ca50:.3f} | {ca5095:.3f}\n'
    summary += ('\nNote: Four families combine severity variants; scores are NOT eight-class severity results. '
                'Postprocessing: conf=.001, IoU=.7, max_det=300; matching uses grouped labels '
                'with greedy one-to-one IoU matches. Different evaluator settings may shift scores.\n')
    (OUT/'summary.txt').write_text(summary,encoding='utf-8')
    print('\n'+summary)
    print('Saved:',OUT/'summary.txt')
    print('Saved:',OUT/'family_metrics.csv')

if __name__ == '__main__':
    main()
