"""Inspect precision, recall, and F1 across confidence thresholds using cached stats.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'four_family_v1_results' / 'four_family_stats.npz'

if not CACHE.is_file():
    print("Cache not found.")
    exit(1)

data = np.load(CACHE)
tp = data['tp'][:, 0]  # IoU 0.5
conf = data['conf']
pred_cls = data['pred_cls']
gt_cls = data['target_cls']

total_gt = len(gt_cls)
print(f"Total GT boxes: {total_gt}, Total raw predictions: {len(conf)}")

thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
print(f"{'Conf':>6} | {'Preds':>7} | {'TP':>6} | {'FP':>6} | {'Precision':>10} | {'Recall':>8} | {'F1':>6}")
print("-" * 62)

for th in thresholds:
    mask = conf >= th
    n_pred = int(np.sum(mask))
    if n_pred == 0:
        continue
    n_tp = int(np.sum(tp[mask]))
    n_fp = n_pred - n_tp
    p = n_tp / n_pred if n_pred > 0 else 0.0
    r = n_tp / total_gt if total_gt > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    print(f"{th:6.2f} | {n_pred:7d} | {n_tp:6d} | {n_fp:6d} | {p:10.4f} | {r:8.4f} | {f1:6.4f}")
