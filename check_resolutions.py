from pathlib import Path
from collections import Counter
from PIL import Image

DATASET_DIR = Path(r"C:\NIET\Projects\Road Damage\combined_annotatedv2_working")

resolution_counts = Counter()
total_images = 0
min_width = None
max_width = None
min_height = None
max_height = None

for image_path in DATASET_DIR.glob("*.jpg"):
    with Image.open(image_path) as img:
        width, height = img.size

    total_images += 1
    resolution_counts[(width, height)] += 1

    min_width = width if min_width is None else min(min_width, width)
    max_width = width if max_width is None else max(max_width, width)
    min_height = height if min_height is None else min(min_height, height)
    max_height = height if max_height is None else max(max_height, height)

print("\n=== IMAGE RESOLUTION SUMMARY ===")
print(f"Images checked        : {total_images}")
print(f"Unique resolutions    : {len(resolution_counts)}")
print(f"Minimum width         : {min_width}")
print(f"Maximum width         : {max_width}")
print(f"Minimum height        : {min_height}")
print(f"Maximum height        : {max_height}")

print("\n=== MOST COMMON RESOLUTIONS ===")
for (width, height), count in resolution_counts.most_common(20):
    percentage = (count / total_images) * 100
    print(f"{width} x {height} : {count} images ({percentage:.2f}%)")