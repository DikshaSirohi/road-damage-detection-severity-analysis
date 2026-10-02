from pathlib import Path
from collections import defaultdict
import hashlib

DATASET_DIR = Path(
    r"C:\NIET\Projects\Road Damage\combined_annotatedv2_working"
)

hash_to_files = defaultdict(list)
total_images = 0

for image_path in DATASET_DIR.glob("*.jpg"):
    total_images += 1

    sha256 = hashlib.sha256()

    with image_path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    file_hash = sha256.hexdigest()
    hash_to_files[file_hash].append(image_path.name)

duplicate_groups = [
    files
    for files in hash_to_files.values()
    if len(files) > 1
]

duplicate_file_count = sum(len(files) for files in duplicate_groups)

print("\n=== EXACT DUPLICATE IMAGE CHECK ===")
print(f"Images checked             : {total_images}")
print(f"Unique file hashes         : {len(hash_to_files)}")
print(f"Duplicate hash groups      : {len(duplicate_groups)}")
print(f"Files inside duplicate sets: {duplicate_file_count}")

if duplicate_groups:
    print("\n=== DUPLICATE GROUPS ===")

    for i, files in enumerate(duplicate_groups, start=1):
        print(f"\nGroup {i}:")
        for filename in sorted(files):
            print(f"  {filename}")
else:
    print("\nNo byte-identical duplicate JPG files found.")