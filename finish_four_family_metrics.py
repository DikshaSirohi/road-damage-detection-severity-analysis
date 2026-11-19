"""Calculate frozen V1 four-family validation metrics from cached predictions.
No inference, training, or dataset modifications. Expects four_family_stats.npz.
"""
from pathlib import Path
import csv
import numpy as np
from ultralytics.utils.metrics import ap_per_class

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'four_family_v1_results'
CACHE = OUT / 'four_family_stats.npz'
NAMES = {0:'Longitudinal crack', 1:'Transverse crack', 2:'Alligator crack', 3:'Pothole'}

if not CACHE.is_file():
    raise FileNotFoundError(f'Missing stats cache: {CACHE}. Please report this; do not rerun inference yet.')

with np.load(CACHE, allow_pickle=False) as data:
    tp = data['tp']
    conf = data['conf']
    pred = data['pred_cls']
    gt = data['target_cls']

if tp.ndim != 2 or tp.shape[1] != 10:
    raise ValueError(f'Expected tp shape (predictions, 10), got {tp.shape}')
if not (len(tp) == len(conf) == len(pred)):
    raise ValueError('Prediction statistics have inconsistent lengths')

# Ultralytics ap_per_class returns: tp, fp, p, r, f1, ap, unique_classes, ...
result = ap_per_class(tp, conf, pred, gt, names=NAMES)
p, r, ap, unique_classes = map(np.asarray, (result[2], result[3], result[5], result[6]))
if ap.ndim != 2 or ap.shape[1] != 10:
    raise ValueError(f'Expected AP matrix shape (classes, 10), got {ap.shape}')
if not (len(p) == len(r) == len(unique_classes) == len(ap)):
    raise ValueError('Returned per-class metrics have inconsistent lengths')

rows = []
for cls, name in NAMES.items():
    matches = np.where(unique_classes == cls)[0]
    if len(matches):
        i = int(matches[0])
        cp, cr, ap50, ap5095 = float(p[i]), float(r[i]), float(ap[i, 0]), float(ap[i].mean())
    else:
        cp = cr = ap50 = ap5095 = 0.0
    rows.append([name, int((gt == cls).sum()), int((pred == cls).sum()), cp, cr, ap50, ap5095])

# Macro-averages over four families (all four occur in fixed validation labels).
summary = (
    'V1 FOUR-FAMILY VALIDATION (grouped diagnostic; no retraining)\n'
    f'Fixed validation images: 2560 | GT boxes: {len(gt)} | Predictions: {len(pred)}\n'
    f'Precision: {np.mean([x[3] for x in rows]):.4f}\n'
    f'Recall: {np.mean([x[4] for x in rows]):.4f}\n'
    f'mAP50: {np.mean([x[5] for x in rows]):.4f}\n'
    f'mAP50-95: {np.mean([x[6] for x in rows]):.4f}\n\n'
    'Family | GT boxes | Precision | Recall | AP50 | AP50-95\n'
)
for name, n, _, cp, cr, a50, a95 in rows:
    summary += f'{name} | {n} | {cp:.3f} | {cr:.3f} | {a50:.3f} | {a95:.3f}\n'
summary += ('\nThis is a grouped four-class diagnostic, not an eight-class severity score. '
            'Uses cached predictions from conf=0.001, NMS IoU=0.7, max_det=300; '
            'AP after class-family mapping and one-to-one IoU matching. '
            'Not directly comparable to an eight-class validation score from a different evaluator.\n')
OUT.mkdir(exist_ok=True)
with (OUT / 'family_metrics.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['family', 'ground_truth_boxes', 'predictions', 'precision', 'recall', 'AP50', 'AP50_95'])
    writer.writerows(rows)
(OUT / 'summary.txt').write_text(summary, encoding='utf-8')
print('\n' + summary)
print('Saved:', OUT / 'summary.txt')
print('Saved:', OUT / 'family_metrics.csv')
