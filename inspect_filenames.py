from pathlib import Path
from collections import Counter
import re

DATASET_DIR = Path(
    r"C:\NIET\Projects\Road Damage\combined_annotatedv2_working"
)

source_counts = Counter()
nonmatching_names = []
original_count = 0

# Expected example:
# Czech_001003.jpg
# India_000123.jpg
# United_States_000456.jpg
pattern = re.compile(r"^(.+)_([0-9]+)\.jpg$", re.IGNORECASE)

for image_path in DATASET_DIR.glob("*.jpg"):

    # Only inspect source/original images here
    if image_path.name.startswith("xmirror_"):
        continue

    original_count += 1

    match = pattern.match(image_path.name)

    if not match:
        nonmatching_names.append(image_path.name)
        continue

    source_name = match.group(1)
    source_counts[source_name] += 1


print("\n=== ORIGINAL FILENAME STRUCTURE ===")
print(f"Original images checked : {original_count}")
print(f"Recognized filenames     : {sum(source_counts.values())}")
print(f"Unrecognized filenames   : {len(nonmatching_names)}")

print("\n=== SOURCE COUNTS ===")
for source, count in source_counts.most_common():
    print(f"{source}: {count}")

if nonmatching_names:
    print("\n=== UNRECOGNIZED FILENAMES ===")
    for name in sorted(nonmatching_names)[:100]:
        print(name)
else:
    print("\nAll original filenames follow the expected <source>_<number>.jpg pattern.")