
from pathlib import Path
from collections import defaultdict, Counter
import csv
import random

DATASET_DIR = Path("combined_annotatedv2_working")
GROUP_FILE = Path("leakage_groups.csv")
DUPLICATE_FILE = Path("confirmed_duplicate_pairs.txt")
OUTPUT_FILE = Path("dataset_split.csv")

SEED = 42
random.seed(SEED)

SPLIT_RATIOS = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10
}

CLASS_NAMES = {
    0: "longitudinal crack",
    1: "longitudinal crack wide",
    2: "transverse crack",
    3: "transverse crack wide",
    4: "alligator crack",
    5: "alligator crack sunken",
    6: "pothole",
    7: "pothole deep"
}

# --------------------------------------------------
# 1. Load leakage groups
# --------------------------------------------------

groups = defaultdict(list)
image_to_group = {}

with GROUP_FILE.open("r", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        filename = row["filename"]
        group_id = row["group_id"]

        assert filename not in image_to_group

        groups[group_id].append(filename)
        image_to_group[filename] = group_id

assert len(groups) == 12750
assert len(image_to_group) == 25600

actual_images = {
    p.name for p in DATASET_DIR.glob("*.jpg")
}

assert set(image_to_group) == actual_images

# --------------------------------------------------
# 2. Collect group-level class and source statistics
# --------------------------------------------------

group_features = {}
group_original_counts = {}
group_box_counts = {}

total_features = Counter()
total_boxes = Counter()
total_originals = 0

for group_id, members in groups.items():

    features = Counter()
    boxes = Counter()

    originals = [
        name for name in members
        if not name.startswith("xmirror_")
    ]

    assert len(members) == 2 * len(originals)

    for filename in originals:

        mirror = "xmirror_" + filename

        assert mirror in members

        source = Path(filename).stem.rsplit("_", 1)[0]
        features[f"source:{source}"] += 1

        label_path = DATASET_DIR / (
            Path(filename).stem + ".txt"
        )

        assert label_path.is_file(), label_path

        classes = set()

        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():

            if not line.strip():
                continue

            parts = line.split()
            assert len(parts) == 5

            class_id = int(parts[0])
            assert class_id in CLASS_NAMES

            classes.add(class_id)
            boxes[class_id] += 1

        if classes:
            features["damage"] += 1
        else:
            features["background"] += 1

        for class_id in classes:
            features[f"class:{class_id}"] += 1

    group_features[group_id] = features
    group_original_counts[group_id] = len(originals)
    group_box_counts[group_id] = boxes

    total_features.update(features)
    total_boxes.update(boxes)
    total_originals += len(originals)

assert total_originals == 12800
assert sum(total_boxes.values()) == 17955

# --------------------------------------------------
# 3. Define target counts
# --------------------------------------------------

targets = {
    split: {
        feature: count * ratio
        for feature, count in total_features.items()
    }
    for split, ratio in SPLIT_RATIOS.items()
}

image_targets = {
    split: total_originals * ratio
    for split, ratio in SPLIT_RATIOS.items()
}

group_targets = {
    split: len(groups) * ratio
    for split, ratio in SPLIT_RATIOS.items()
}

split_features = {
    split: Counter()
    for split in SPLIT_RATIOS
}

split_originals = Counter()
split_groups = Counter()

assignments = {}

# --------------------------------------------------
# 4. Process rare-class groups first
# --------------------------------------------------

group_order = list(groups)
random.shuffle(group_order)

def rarity_score(group_id):

    features = group_features[group_id]

    score = 0.0

    for feature, count in features.items():

        if feature.startswith("class:"):
            score += count / total_features[feature]

    return score

group_order.sort(
    key=rarity_score,
    reverse=True
)

# --------------------------------------------------
# 5. Assign complete groups to splits
# --------------------------------------------------

def feature_weight(feature):

    if feature.startswith("class:"):
        return 3.0

    if feature.startswith("source:"):
        return 1.5

    return 1.0


def assignment_cost(group_id, split):

    features = group_features[group_id]
    size = group_original_counts[group_id]

    cost = 0.0

    # Original-image balance
    target = image_targets[split]
    current = split_originals[split]

    before = ((current - target) / target) ** 2
    after = ((current + size - target) / target) ** 2

    cost += 8.0 * (after - before)

    # Leakage-group count balance
    target = group_targets[split]
    current = split_groups[split]

    before = ((current - target) / target) ** 2
    after = ((current + 1 - target) / target) ** 2

    cost += 2.0 * (after - before)

    # Class, source and background balance
    for feature, count in features.items():

        target = max(targets[split][feature], 1.0)
        current = split_features[split][feature]

        before = ((current - target) / target) ** 2
        after = (
            (current + count - target) / target
        ) ** 2

        cost += feature_weight(feature) * (
            after - before
        )

    return cost


for group_id in group_order:

    size = group_original_counts[group_id]

    available_splits = [
        split for split in SPLIT_RATIOS
        if split_originals[split] + size
        <= image_targets[split]
    ]

    if not available_splits:
        available_splits = list(SPLIT_RATIOS)

    best_split = min(
        available_splits,
        key=lambda split: assignment_cost(
            group_id, split
        )
    )

    assignments[group_id] = best_split

    split_originals[best_split] += size
    split_groups[best_split] += 1

    split_features[best_split].update(
        group_features[group_id]
    )

# --------------------------------------------------
# 6. Create image-level split mapping
# --------------------------------------------------

image_to_split = {}

for group_id, members in groups.items():

    split = assignments[group_id]

    for filename in members:
        image_to_split[filename] = split

assert len(assignments) == 12750
assert len(image_to_split) == 25600

# --------------------------------------------------
# 7. Verify leakage prevention
# --------------------------------------------------

for group_id, members in groups.items():

    assigned_splits = {
        image_to_split[name]
        for name in members
    }

    assert len(assigned_splits) == 1, (
        f"Leakage detected: {group_id}"
    )

for filename in image_to_split:

    if filename.startswith("xmirror_"):

        original = filename.removeprefix("xmirror_")

        assert (
            image_to_split[filename]
            == image_to_split[original]
        )

duplicate_count = 0

with DUPLICATE_FILE.open(
    "r", encoding="utf-8-sig"
) as f:

    for line in f:

        if not line.strip():
            continue

        image1, image2 = [
            part.strip()
            for part in line.strip().split("|")
        ]

        assert (
            image_to_split[image1]
            == image_to_split[image2]
        )

        duplicate_count += 1

assert duplicate_count == 53

# --------------------------------------------------
# 8. Calculate split statistics
# --------------------------------------------------

split_images = Counter(image_to_split.values())

split_boxes = {
    split: Counter()
    for split in SPLIT_RATIOS
}

for group_id, split in assignments.items():

    split_boxes[split].update(
        group_box_counts[group_id]
    )

assert sum(split_images.values()) == 25600

# Require all eight classes in every split
for split in SPLIT_RATIOS:

    for class_id in CLASS_NAMES:

        assert (
            split_features[split][f"class:{class_id}"] > 0
        ), f"Missing class {class_id} in {split}"

# Require all source countries in every split
source_features = sorted(
    feature for feature in total_features
    if feature.startswith("source:")
)

for split in SPLIT_RATIOS:

    for feature in source_features:

        assert split_features[split][feature] > 0

# --------------------------------------------------
# 9. Save split mapping
# --------------------------------------------------

with OUTPUT_FILE.open(
    "w", newline="", encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "group_id",
        "split"
    ])

    for filename in sorted(image_to_split):

        writer.writerow([
            filename,
            image_to_group[filename],
            image_to_split[filename]
        ])

# --------------------------------------------------
# 10. Print summary
# --------------------------------------------------

print("=== DATASET SPLIT SUMMARY ===")

print(
    f"{'Split':<10}"
    f"{'Groups':>10}"
    f"{'Originals':>12}"
    f"{'Images':>10}"
    f"{'Percent':>10}"
)

for split in SPLIT_RATIOS:

    percentage = split_images[split] / 25600 * 100

    print(
        f"{split:<10}"
        f"{split_groups[split]:>10}"
        f"{split_originals[split]:>12}"
        f"{split_images[split]:>10}"
        f"{percentage:>9.2f}%"
    )

print("\n=== CLASS DISTRIBUTION ===")
print("Original images containing each class:")

print(
    f"{'Class':<28}"
    f"{'Train':>8}"
    f"{'Val':>8}"
    f"{'Test':>8}"
    f"{'Total':>8}"
)

for class_id, class_name in CLASS_NAMES.items():

    feature = f"class:{class_id}"

    print(
        f"{class_name:<28}"
        f"{split_features['train'][feature]:>8}"
        f"{split_features['val'][feature]:>8}"
        f"{split_features['test'][feature]:>8}"
        f"{total_features[feature]:>8}"
    )

print("\n=== BOUNDING BOX COUNTS ===")
print("Includes original and mirrored images:")

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"{class_name:<28}"
        f"{split_boxes['train'][class_id] * 2:>8}"
        f"{split_boxes['val'][class_id] * 2:>8}"
        f"{split_boxes['test'][class_id] * 2:>8}"
        f"{total_boxes[class_id] * 2:>8}"
    )

print("\n=== SOURCE DISTRIBUTION ===")
print("Original images:")

for feature in source_features:

    source = feature.split(":", 1)[1]

    print(
        f"{source:<28}"
        f"{split_features['train'][feature]:>8}"
        f"{split_features['val'][feature]:>8}"
        f"{split_features['test'][feature]:>8}"
        f"{total_features[feature]:>8}"
    )

print("\n=== BACKGROUND DISTRIBUTION ===")

for split in SPLIT_RATIOS:

    background = split_features[split]["background"]

    percentage = (
        background / split_originals[split] * 100
    )

    print(
        f"{split:<10}: "
        f"{background} originals "
        f"({percentage:.2f}%)"
    )

print("\n=== VERIFICATION ===")
print("Complete image coverage       : PASSED")
print("Leakage groups preserved      : PASSED")
print("Original/mirror pairs         : PASSED")
print("Confirmed same-scene pairs    : PASSED")
print("All classes in every split    : PASSED")
print("All sources in every split    : PASSED")

print(f"\nMapping saved to: {OUTPUT_FILE}")
print("No dataset images or annotations modified.")
