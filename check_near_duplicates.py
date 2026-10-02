from pathlib import Path
from PIL import Image
import imagehash

DATASET_DIR = Path(
    r"C:\NIET\Projects\Road Damage\combined_annotatedv2_working"
)

# Smaller distance = more visually similar.
# <= 4 is deliberately strict so we only flag strong candidates.
MAX_DISTANCE = 2


class BKNode:
    def __init__(self, hash_value, filename):
        self.hash_value = hash_value
        self.filenames = [filename]
        self.children = {}


def hamming_distance(a, b):
    return (a ^ b).bit_count()


def insert(root, hash_value, filename):
    node = root

    while True:
        distance = hamming_distance(hash_value, node.hash_value)

        if distance == 0:
            node.filenames.append(filename)
            return

        if distance not in node.children:
            node.children[distance] = BKNode(hash_value, filename)
            return

        node = node.children[distance]


def search(node, target_hash, max_distance, results):
    distance = hamming_distance(target_hash, node.hash_value)

    if distance <= max_distance:
        results.append((distance, node))

    low = distance - max_distance
    high = distance + max_distance

    for child_distance, child in node.children.items():
        if low <= child_distance <= high:
            search(child, target_hash, max_distance, results)


image_hashes = []
processed = 0

print("Calculating perceptual hashes...")

for image_path in DATASET_DIR.glob("*.jpg"):

    if image_path.name.startswith("xmirror_"):
        continue

    with Image.open(image_path) as img:
        phash = imagehash.phash(img)

    hash_int = int(str(phash), 16)

    image_hashes.append((image_path.name, hash_int))

    processed += 1

    if processed % 1000 == 0:
        print(f"Hashed {processed} images...")


print("\nSearching for near-duplicate candidates...")

root = None
candidate_pairs = set()

for filename, hash_value in image_hashes:

    if root is None:
        root = BKNode(hash_value, filename)
        continue

    matches = []
    search(root, hash_value, MAX_DISTANCE, matches)

    for distance, node in matches:
        for other_filename in node.filenames:

            pair = tuple(sorted((filename, other_filename)))

            candidate_pairs.add(
                (pair[0], pair[1], distance)
            )

    insert(root, hash_value, filename)


candidate_pairs = sorted(
    candidate_pairs,
    key=lambda x: (x[2], x[0], x[1])
)


print("\n=== NEAR-DUPLICATE CHECK ===")
print(f"Original images checked : {len(image_hashes)}")
print(f"Maximum hash distance   : {MAX_DISTANCE}")
print(f"Candidate pairs found   : {len(candidate_pairs)}")

if candidate_pairs:
    print("\n=== CANDIDATE PAIRS ===")

    for filename1, filename2, distance in candidate_pairs[:200]:
        print(
            f"distance={distance} | "
            f"{filename1} | {filename2}"
        )

    if len(candidate_pairs) > 200:
        print(
            f"\nOnly first 200 shown. "
            f"Total candidates: {len(candidate_pairs)}"
        )
else:
    print("\nNo strong near-duplicate candidates found.")

OUTPUT_FILE = Path(
    r"C:\NIET\Projects\Road Damage\near_duplicate_candidates.txt"
)

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for filename1, filename2, distance in candidate_pairs:
        f.write(f"{distance}|{filename1}|{filename2}\n")

print(f"\nCandidate list saved to:")
print(OUTPUT_FILE)