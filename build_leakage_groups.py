
from pathlib import Path
from collections import defaultdict, Counter
import csv

DATASET_DIR = Path("combined_annotatedv2_working")
DUPLICATE_FILE = Path("confirmed_duplicate_pairs.txt")
OUTPUT_FILE = Path("leakage_groups.csv")


# Union-Find structure
class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_b] = root_a
            return True

        return False


# --------------------------------------------------
# 1. Load dataset image filenames
# --------------------------------------------------

images = sorted(
    path.name for path in DATASET_DIR.glob("*.jpg")
)

image_set = set(images)

originals = [
    name for name in images
    if not name.startswith("xmirror_")
]

mirrors = [
    name for name in images
    if name.startswith("xmirror_")
]

assert len(images) == 25600, "Unexpected total image count."
assert len(originals) == 12800, "Unexpected original count."
assert len(mirrors) == 12800, "Unexpected mirror count."

print("=== DATASET SUMMARY ===")
print(f"Total images    : {len(images)}")
print(f"Original images : {len(originals)}")
print(f"Mirror images   : {len(mirrors)}")


# --------------------------------------------------
# 2. Merge original/mirror pairs
# --------------------------------------------------

uf = UnionFind(images)

mirror_pairs = 0

for original in originals:
    mirror = "xmirror_" + original

    if mirror not in image_set:
        raise FileNotFoundError(
            f"Missing mirror: {mirror}"
        )

    uf.union(original, mirror)
    mirror_pairs += 1

for mirror in mirrors:
    original = mirror.removeprefix("xmirror_")

    if original not in image_set:
        raise FileNotFoundError(
            f"Missing original: {original}"
        )

print("\n=== MIRROR GROUPING ===")
print(f"Original/mirror pairs merged : {mirror_pairs}")


# --------------------------------------------------
# 3. Load confirmed same-scene relationships
# --------------------------------------------------

duplicate_pairs = []

with DUPLICATE_FILE.open("r", encoding="utf-8-sig") as f:

    for line_number, line in enumerate(f, start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split("|")

        if len(parts) != 2:
            raise ValueError(
                f"Malformed duplicate pair at line {line_number}"
            )

        image1, image2 = [
            part.strip() for part in parts
        ]

        if image1 not in image_set or image2 not in image_set:
            raise ValueError(
                f"Unknown image in pair: {line}"
            )

        if image1 == image2:
            raise ValueError(f"Self-pair detected: {line}")

        if (
            image1.startswith("xmirror_")
            or image2.startswith("xmirror_")
        ):
            raise ValueError(
                f"Unexpected mirror in duplicate list: {line}"
            )

        duplicate_pairs.append((image1, image2))

assert len(duplicate_pairs) == 53, (
    "Expected exactly 53 confirmed relationships."
)

unique_pairs = {
    frozenset(pair) for pair in duplicate_pairs
}

assert len(unique_pairs) == 53, (
    "Duplicate relationships detected."
)

# Verify previously rejected relationships are absent
rejected_pairs = [
    ("Czech_000850.jpg", "Czech_001628.jpg"),
    ("Czech_000971.jpg", "Czech_001039.jpg"),
    ("Czech_001005.jpg", "Czech_001669.jpg"),
    ("Czech_001908.jpg", "Czech_002056.jpg"),
]

for pair in rejected_pairs:
    assert frozenset(pair) not in unique_pairs, (
        f"Rejected relationship found: {pair}"
    )


# --------------------------------------------------
# 4. Merge confirmed same-scene groups
# --------------------------------------------------

successful_merges = 0
already_connected = 0

for image1, image2 in duplicate_pairs:

    if uf.union(image1, image2):
        successful_merges += 1
    else:
        already_connected += 1

print("\n=== SAME-SCENE GROUPING ===")
print(f"Confirmed relationships : {len(duplicate_pairs)}")
print(f"Successful group merges : {successful_merges}")
print(f"Already connected pairs : {already_connected}")


# --------------------------------------------------
# 5. Collect final leakage groups
# --------------------------------------------------

groups = defaultdict(list)

for image in images:
    root = uf.find(image)
    groups[root].append(image)

# Assign stable and readable group IDs
sorted_groups = sorted(
    groups.values(),
    key=lambda members: min(members)
)

image_to_group = {}

for index, members in enumerate(sorted_groups, start=1):

    group_id = f"G{index:05d}"

    for image in members:
        image_to_group[image] = group_id


# --------------------------------------------------
# 6. Verify final groups
# --------------------------------------------------

assert len(image_to_group) == len(images)

# Verify every original and mirror share a group
for original in originals:
    mirror = "xmirror_" + original

    assert image_to_group[original] == image_to_group[mirror], (
        f"Mirror grouping failed: {original}"
    )

# Verify every confirmed same-scene pair shares a group
for image1, image2 in duplicate_pairs:

    assert image_to_group[image1] == image_to_group[image2], (
        f"Duplicate grouping failed: {image1}, {image2}"
    )

# Every group should contain complete original/mirror pairs
for members in sorted_groups:

    assert len(members) % 2 == 0, (
        "Unexpected odd-sized leakage group."
    )

expected_groups = len(originals) - successful_merges

assert len(sorted_groups) == expected_groups, (
    "Unexpected final leakage group count."
)


# --------------------------------------------------
# 7. Save group mapping
# --------------------------------------------------

with OUTPUT_FILE.open(
    "w", newline="", encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "group_id"
    ])

    for image in images:
        writer.writerow([
            image,
            image_to_group[image]
        ])


# --------------------------------------------------
# 8. Print final summary
# --------------------------------------------------

size_distribution = Counter(
    len(members) for members in sorted_groups
)

multi_scene_groups = sum(
    len(members) > 2
    for members in sorted_groups
)

print("\n=== FINAL LEAKAGE GROUP SUMMARY ===")
print(f"Total images assigned    : {len(image_to_group)}")
print(f"Initial mirror groups    : {len(originals)}")
print(f"Final leakage groups     : {len(sorted_groups)}")
print(f"Multi-scene groups       : {multi_scene_groups}")
print(f"Largest group size       : {max(size_distribution)}")

print("\nGroup size distribution:")

for size, count in sorted(size_distribution.items()):
    print(f"  {size} images : {count} groups")

print("\n=== VERIFICATION ===")
print("Original/mirror grouping : PASSED")
print("Same-scene grouping      : PASSED")
print("Complete image coverage  : PASSED")
print("Group count consistency  : PASSED")

print(f"\nMapping saved to: {OUTPUT_FILE}")
print("No dataset images or annotations were modified.")
