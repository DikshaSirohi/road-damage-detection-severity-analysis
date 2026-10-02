from pathlib import Path
from collections import defaultdict
import re

DATASET_DIR = Path(
    r"C:\NIET\Projects\Road Damage\combined_annotatedv2_working"
)

pattern = re.compile(r"^(.+)_([0-9]+)\.jpg$", re.IGNORECASE)

ids_by_source = defaultdict(list)
unrecognized = []

for image_path in DATASET_DIR.glob("*.jpg"):
    if image_path.name.startswith("xmirror_"):
        continue

    match = pattern.match(image_path.name)

    if not match:
        unrecognized.append(image_path.name)
        continue

    source = match.group(1)
    image_id = int(match.group(2))

    ids_by_source[source].append(image_id)

print("\n=== SOURCE ID ANALYSIS ===")

for source, ids in sorted(ids_by_source.items()):
    ids = sorted(ids)

    min_id = ids[0]
    max_id = ids[-1]

    expected_count = max_id - min_id + 1
    actual_count = len(ids)

    missing_ids = sorted(set(range(min_id, max_id + 1)) - set(ids))

    print(f"\nSource: {source}")
    print(f"Images         : {actual_count}")
    print(f"Minimum ID     : {min_id}")
    print(f"Maximum ID     : {max_id}")
    print(f"Expected range : {expected_count}")
    print(f"Missing IDs    : {len(missing_ids)}")

    if missing_ids:
        print("First missing IDs:")
        print(missing_ids[:50])

print(f"\nUnrecognized filenames: {len(unrecognized)}")