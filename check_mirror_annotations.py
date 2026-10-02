from pathlib import Path

DATASET_DIR = Path(r"C:\NIET\Projects\Road Damage\combined_annotatedv2_working")

TOLERANCE = 1e-5

pairs_checked = 0
matching_pairs = 0
mismatched_pairs = []


def read_labels(path):
    boxes = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line:
            continue

        class_id, x, y, w, h = line.split()

        boxes.append(
            (
                int(class_id),
                float(x),
                float(y),
                float(w),
                float(h),
            )
        )

    return boxes


def boxes_match(original_boxes, mirror_boxes):
    if len(original_boxes) != len(mirror_boxes):
        return False

    expected_mirrors = []

    for class_id, x, y, w, h in original_boxes:
        expected_mirrors.append(
            (
                class_id,
                1.0 - x,
                y,
                w,
                h,
            )
        )

    used = [False] * len(mirror_boxes)

    for expected in expected_mirrors:
        found = False

        for i, actual in enumerate(mirror_boxes):
            if used[i]:
                continue

            same_class = expected[0] == actual[0]

            same_values = all(
                abs(expected[j] - actual[j]) <= TOLERANCE
                for j in range(1, 5)
            )

            if same_class and same_values:
                used[i] = True
                found = True
                break

        if not found:
            return False

    return True


for image_path in DATASET_DIR.glob("*.jpg"):

    if image_path.name.startswith("xmirror_"):
        continue

    mirror_image = DATASET_DIR / ("xmirror_" + image_path.name)

    original_label = image_path.with_suffix(".txt")
    mirror_label = mirror_image.with_suffix(".txt")

    pairs_checked += 1

    original_boxes = read_labels(original_label)
    mirror_boxes = read_labels(mirror_label)

    if boxes_match(original_boxes, mirror_boxes):
        matching_pairs += 1
    else:
        mismatched_pairs.append(image_path.name)


print("\n=== MIRROR ANNOTATION CHECK ===")
print(f"Image pairs checked       : {pairs_checked}")
print(f"Matching annotation pairs : {matching_pairs}")
print(f"Mismatched pairs          : {len(mismatched_pairs)}")

if mismatched_pairs:
    print("\n=== FIRST 50 MISMATCHED PAIRS ===")

    for name in mismatched_pairs[:50]:
        print(name)
else:
    print("\nAll mirrored annotations are geometrically consistent.")