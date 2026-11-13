
from pathlib import Path
from collections import Counter, defaultdict
import csv
import shutil

PROJECT_DIR = Path(__file__).resolve().parent

SOURCE_DIR = PROJECT_DIR / "combined_annotatedv2_working"
SPLIT_FILE = PROJECT_DIR / "dataset_split.csv"
OUTPUT_DIR = PROJECT_DIR / "road_damage_yolo"

SPLITS = ("train", "val", "test")

CLASS_NAMES = [
    "longitudinal crack",
    "longitudinal crack wide",
    "transverse crack",
    "transverse crack wide",
    "alligator crack",
    "alligator crack sunken",
    "pothole",
    "pothole deep"
]

EXPECTED_COUNTS = {
    "train": 20480,
    "val": 2560,
    "test": 2560
}

# --------------------------------------------------
# 1. Read and verify split mapping
# --------------------------------------------------

rows = []
seen_images = set()
group_splits = defaultdict(set)

with SPLIT_FILE.open("r", encoding="utf-8-sig") as f:

    for row in csv.DictReader(f):

        filename = row["filename"]
        group_id = row["group_id"]
        split = row["split"]

        assert split in SPLITS
        assert filename not in seen_images

        image_path = SOURCE_DIR / filename
        label_path = SOURCE_DIR / (
            Path(filename).stem + ".txt"
        )

        assert image_path.is_file(), image_path
        assert label_path.is_file(), label_path

        seen_images.add(filename)
        group_splits[group_id].add(split)

        rows.append(row)

assert len(rows) == 25600
assert len(group_splits) == 12750

for group_id, splits in group_splits.items():

    assert len(splits) == 1, (
        f"Leakage detected: {group_id}"
    )

split_counts = Counter(
    row["split"] for row in rows
)

assert dict(split_counts) == EXPECTED_COUNTS

actual_images = {
    path.name for path in SOURCE_DIR.glob("*.jpg")
}

assert seen_images == actual_images

print("Split mapping verification: PASSED", flush=True)

# --------------------------------------------------
# 2. Check available disk space
# --------------------------------------------------

required_bytes = sum(
    (SOURCE_DIR / row["filename"]).stat().st_size
    +
    (
        SOURCE_DIR /
        (Path(row["filename"]).stem + ".txt")
    ).stat().st_size

    for row in rows
)

free_bytes = shutil.disk_usage(PROJECT_DIR).free

print(
    f"Dataset size : "
    f"{required_bytes / 1024**3:.2f} GB"
)

print(
    f"Free space   : "
    f"{free_bytes / 1024**3:.2f} GB"
)

assert free_bytes > required_bytes + 1024**3, (
    "Insufficient disk space."
)

# --------------------------------------------------
# 3. Create YOLO directory structure
# --------------------------------------------------

if OUTPUT_DIR.exists():

    raise FileExistsError(
        f"Output directory already exists: {OUTPUT_DIR}\n"
        "Do not overwrite it automatically."
    )

for split in SPLITS:

    (OUTPUT_DIR / "images" / split).mkdir(
        parents=True, exist_ok=True
    )

    (OUTPUT_DIR / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )

print("\nYOLO directories created.", flush=True)

# --------------------------------------------------
# 4. Copy dataset files
# --------------------------------------------------

copied_images = Counter()
copied_labels = Counter()

total = len(rows)

print("\nCopying dataset files...", flush=True)

for index, row in enumerate(rows, start=1):

    filename = row["filename"]
    split = row["split"]

    label_name = Path(filename).stem + ".txt"

    source_image = SOURCE_DIR / filename
    source_label = SOURCE_DIR / label_name

    target_image = (
        OUTPUT_DIR / "images" / split / filename
    )

    target_label = (
        OUTPUT_DIR / "labels" / split / label_name
    )

    shutil.copy2(source_image, target_image)
    shutil.copy2(source_label, target_label)

    copied_images[split] += 1
    copied_labels[split] += 1

    if index % 1000 == 0 or index == total:

        print(
            f"Copied {index}/{total} image-label pairs "
            f"({index / total * 100:.1f}%)",
            flush=True
        )

# --------------------------------------------------
# 5. Create data.yaml
# --------------------------------------------------

yaml_content = (
    f'path: "{OUTPUT_DIR.as_posix()}"\n'
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "\n"
    "nc: 8\n"
    "\n"
    "names:\n"
)

for class_id, class_name in enumerate(CLASS_NAMES):

    yaml_content += (
        f"  {class_id}: {class_name}\n"
    )

yaml_path = OUTPUT_DIR / "data.yaml"

yaml_path.write_text(
    yaml_content,
    encoding="utf-8"
)

# --------------------------------------------------
# 6. Verify copied dataset
# --------------------------------------------------

all_split_images = {}
total_annotations = 0
class_counts = Counter()
total_backgrounds = 0

print("\n=== YOLO DATASET VERIFICATION ===")

for split in SPLITS:

    images_dir = OUTPUT_DIR / "images" / split
    labels_dir = OUTPUT_DIR / "labels" / split

    image_files = {
        path.stem: path
        for path in images_dir.glob("*.jpg")
    }

    label_files = {
        path.stem: path
        for path in labels_dir.glob("*.txt")
    }

    assert set(image_files) == set(label_files), (
        f"Image-label mismatch in {split}"
    )

    assert len(image_files) == EXPECTED_COUNTS[split]
    assert len(label_files) == EXPECTED_COUNTS[split]

    all_split_images[split] = set(image_files)

    annotations = 0
    backgrounds = 0

    for stem, label_path in label_files.items():

        source_label = SOURCE_DIR / label_path.name
        source_image = SOURCE_DIR / image_files[stem].name

        assert (
            label_path.read_bytes()
            == source_label.read_bytes()
        )

        assert (
            image_files[stem].stat().st_size
            == source_image.stat().st_size
        )

        lines = [
            line.strip()
            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if not lines:
            backgrounds += 1

        for line in lines:

            parts = line.split()

            assert len(parts) == 5

            class_id = int(parts[0])

            assert 0 <= class_id < 8

            class_counts[class_id] += 1
            annotations += 1

    total_annotations += annotations
    total_backgrounds += backgrounds

    print(f"\n{split.upper()}")
    print(f"Images      : {len(image_files)}")
    print(f"Labels      : {len(label_files)}")
    print(f"Annotations : {annotations}")
    print(f"Backgrounds : {backgrounds}")

# --------------------------------------------------
# 7. Verify no overlap between splits
# --------------------------------------------------

assert not (
    all_split_images["train"]
    & all_split_images["val"]
)

assert not (
    all_split_images["train"]
    & all_split_images["test"]
)

assert not (
    all_split_images["val"]
    & all_split_images["test"]
)

assert total_annotations == 35910
assert total_backgrounds == 8318

expected_class_counts = {
    0: 12810,
    1: 2340,
    2: 7456,
    3: 708,
    4: 6918,
    5: 906,
    6: 4224,
    7: 548
}

assert dict(class_counts) == expected_class_counts

# --------------------------------------------------
# 8. Final summary
# --------------------------------------------------

print("\n=== FINAL VERIFICATION ===")

print("Total images       : 25600")
print("Total labels       : 25600")
print(f"Total annotations  : {total_annotations}")
print(f"Background images  : {total_backgrounds}")

print("\nImage-label pairing   : PASSED")
print("Split image counts    : PASSED")
print("Split overlap check   : PASSED")
print("Annotation counts     : PASSED")
print("Class counts          : PASSED")
print("Background counts     : PASSED")

print(f"\nDataset folder: {OUTPUT_DIR}")
print(f"YAML file     : {yaml_path}")

print("\nSUCCESS: YOLO dataset is ready.")
print("Original and working datasets remain untouched.")
