"""Investigate detection discrepancy on Czech_000809.jpg
Checks hash across directories, inference settings, device (CPU vs CUDA),
cold vs warm timing, IoU thresholds, and PIL buffer vs file path.
"""
from pathlib import Path
import hashlib
import io
import time
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'runs' / 'road_damage_yolo26n_v1' / 'weights' / 'best.pt'

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# 1. Hashes of Czech_000809.jpg
print("\n--- HASH CHECK FOR Czech_000809.jpg ---")
for p in ROOT.rglob("Czech_000809.jpg"):
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    print(f"{p}: sha256={h[:16]}..., size={p.stat().st_size} bytes")

val_img_path = ROOT / 'road_damage_yolo' / 'images' / 'val' / 'Czech_000809.jpg'

# 2. Timing: Cold vs Warm on CUDA vs CPU
print("\n--- INFERENCE TIMING & DEVICE CHECK ---")
model = YOLO(str(WEIGHTS))

# Cold run on whatever default device YOLO picks
t0 = time.time()
r_cold = model.predict(str(val_img_path), conf=0.15, verbose=False)[0]
t_cold = (time.time() - t0) * 1000
print(f"Cold run (device default={r_cold.speed}): total wall time={t_cold:.1f} ms")

# Warm run
t0 = time.time()
r_warm = model.predict(str(val_img_path), conf=0.15, verbose=False)[0]
t_warm = (time.time() - t0) * 1000
print(f"Warm run 1: total wall time={t_warm:.1f} ms, speed dict={r_warm.speed}")

t0 = time.time()
r_warm2 = model.predict(str(val_img_path), conf=0.15, verbose=False)[0]
t_warm2 = (time.time() - t0) * 1000
print(f"Warm run 2: total wall time={t_warm2:.1f} ms, speed dict={r_warm2.speed}")

# Explicit CUDA
if torch.cuda.is_available():
    t0 = time.time()
    r_cuda = model.predict(str(val_img_path), conf=0.15, device=0, verbose=False)[0]
    t_cuda = (time.time() - t0) * 1000
    print(f"Explicit device=0: total wall time={t_cuda:.1f} ms, speed dict={r_cuda.speed}")

# Explicit CPU
t0 = time.time()
r_cpu = model.predict(str(val_img_path), conf=0.15, device='cpu', verbose=False)[0]
t_cpu = (time.time() - t0) * 1000
print(f"Explicit device='cpu': total wall time={t_cpu:.1f} ms, speed dict={r_cpu.speed}")

# 3. Check predictions between PIL Image, BytesIO buffer, and File Path
print("\n--- PREDICTION DIFFERENCES ACROSS INPUT TYPES AT CONF 0.15 ---")
# From path
print(f"1. File path (conf=0.15, iou=0.50): {len(r_cold.boxes)} boxes")
for b in r_cold.boxes:
    print(f"   cls={int(b.cls[0])} ({model.names[int(b.cls[0])]}), conf={float(b.conf[0]):.3f}")

# From PIL Image
img_pil = Image.open(val_img_path)
r_pil = model.predict(img_pil, conf=0.15, iou=0.50, verbose=False)[0]
print(f"2. PIL Image.open() (conf=0.15, iou=0.50): {len(r_pil.boxes)} boxes")
for b in r_pil.boxes:
    print(f"   cls={int(b.cls[0])} ({model.names[int(b.cls[0])]}), conf={float(b.conf[0]):.3f}")

# From BytesIO (like Streamlit file_uploader)
buf = io.BytesIO(val_img_path.read_bytes())
img_buf = Image.open(buf)
r_buf = model.predict(img_buf, conf=0.15, iou=0.50, verbose=False)[0]
print(f"3. BytesIO uploaded buffer (conf=0.15, iou=0.50): {len(r_buf.boxes)} boxes")
for b in r_buf.boxes:
    print(f"   cls={int(b.cls[0])} ({model.names[int(b.cls[0])]}), conf={float(b.conf[0]):.3f}")

# 4. Check what happens if IoU=0.45, 0.50, 0.60 or with Family NMS
print("\n--- CHECKING NMS AND DEDUPLICATION AT CONF 0.15 ---")
for iou_val in [0.45, 0.50, 0.60, 0.70]:
    r = model.predict(img_pil, conf=0.15, iou=iou_val, verbose=False)[0]
    print(f"YOLO predict with iou={iou_val:.2f}: {len(r.boxes)} boxes")
    for b in r.boxes:
        print(f"    cls={int(b.cls[0])} ({model.names[int(b.cls[0])]}), conf={float(b.conf[0]):.3f}, xyxy={[round(x,1) for x in b.xyxy[0].tolist()]}")
