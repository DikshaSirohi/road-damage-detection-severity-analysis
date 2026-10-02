from pathlib import Path
from PIL import Image

DATASET_DIR = Path(r"C:\NIET\Projects\Road Damage\combined_annotatedv2_working")

total_images = 0
corrupt_images = []

for image_path in DATASET_DIR.glob("*.jpg"):
    total_images += 1

    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        corrupt_images.append((image_path.name, str(e)))

print("\n=== JPG INTEGRITY CHECK ===")
print(f"Images checked : {total_images}")
print(f"Corrupt/unreadable images : {len(corrupt_images)}")

if corrupt_images:
    print("\n=== PROBLEM IMAGES ===")
    for filename, error in corrupt_images:
        print(f"{filename} | {error}")
else:
    print("\nNo corrupt or unreadable JPG files found.")