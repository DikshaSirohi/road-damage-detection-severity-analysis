from pathlib import Path

DATASET_DIR = Path(r"C:\NIET\Projects\Road Damage\combined_annotatedv2_working")

total_labels = 0
empty_labels = 0
missing_images = []

for label_path in DATASET_DIR.glob("*.txt"):
    if label_path.name.lower() == "classes.txt":
        continue

    total_labels += 1

    if label_path.stat().st_size == 0:
        empty_labels += 1

        image_path = label_path.with_suffix(".jpg")
        if not image_path.exists():
            missing_images.append(label_path.name)

background_percentage = (empty_labels / total_labels) * 100

print("\n=== EMPTY LABEL / BACKGROUND CHECK ===")
print(f"Label files checked       : {total_labels}")
print(f"Empty label files         : {empty_labels}")
print(f"Background percentage     : {background_percentage:.2f}%")
print(f"Empty labels missing JPGs : {len(missing_images)}")

if missing_images:
    print("\n=== EMPTY LABELS WITHOUT MATCHING JPG ===")
    for filename in missing_images:
        print(filename)
else:
    print("\nEvery empty label has a matching JPG image.")