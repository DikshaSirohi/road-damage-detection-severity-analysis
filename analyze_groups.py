from pathlib import Path
from collections import Counter

DATASET_DIR = Path(
    r"C:\NIET\Projects\Road Damage\combined_annotatedv2_working"
)

CLASS_NAMES = {
    0: "longitudinal crack",
    1: "longitudinal crack wide",
    2: "transverse crack",
    3: "transverse crack wide",
    4: "alligator crack",
    5: "alligator crack sunken",
    6: "pothole",
    7: "pothole deep",
}

total_groups = 0
background_groups = 0

groups_per_class = Counter()
boxes_per_class = Counter()

for image_path in DATASET_DIR.glob("*.jpg"):

    # Each original + xmirror pair is one leakage-safe group.
    # We only inspect the original here because the mirror contains
    # the same damage classes.
    if image_path.name.startswith("xmirror_"):
        continue

    total_groups += 1

    label_path = image_path.with_suffix(".txt")

    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not lines:
        background_groups += 1
        continue

    classes_in_this_group = set()

    for line in lines:
        parts = line.split()
        class_id = int(parts[0])

        boxes_per_class[class_id] += 1
        classes_in_this_group.add(class_id)

    # Count a class only once per image group, even if that image
    # contains several boxes of the same class.
    for class_id in classes_in_this_group:
        groups_per_class[class_id] += 1


damage_groups = total_groups - background_groups

print("\n=== GROUP-LEVEL DATASET SUMMARY ===")
print(f"Total original/mirror groups : {total_groups}")
print(f"Damage groups                : {damage_groups}")
print(f"Background groups            : {background_groups}")
print(
    f"Background group percentage  : "
    f"{(background_groups / total_groups) * 100:.2f}%"
)

print("\n=== GROUPS CONTAINING EACH CLASS ===")

for class_id, class_name in CLASS_NAMES.items():
    group_count = groups_per_class[class_id]
    percentage = (group_count / total_groups) * 100

    print(
        f"{class_id} {class_name:<28} "
        f": {group_count:5d} groups ({percentage:6.2f}%)"
    )

print("\n=== BOX COUNTS IN ORIGINAL IMAGES ONLY ===")

for class_id, class_name in CLASS_NAMES.items():
    print(
        f"{class_id} {class_name:<28} "
        f": {boxes_per_class[class_id]:5d} boxes"
    )