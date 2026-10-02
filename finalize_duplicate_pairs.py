from pathlib import Path

INPUT_FILE = Path("near_duplicate_candidates.txt")
OUTPUT_FILE = Path("confirmed_duplicate_pairs.txt")

# These numbers refer ONLY to the distance-2 subset.
EXCLUDED_DISTANCE2 = {14, 19, 20, 36}

confirmed_pairs = []
distance2_number = 0

with INPUT_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        parts = line.split("|")

        if len(parts) != 3:
            print(f"Skipping malformed line: {line}")
            continue

        distance, file1, file2 = parts

        try:
            distance = int(distance)
        except ValueError:
            print(f"Skipping invalid distance: {line}")
            continue

        if distance == 0:
            # All 8 distance-0 pairs are confirmed.
            confirmed_pairs.append(f"{file1}|{file2}")

        elif distance == 2:
            # Number ONLY the distance-2 pairs from 1 to 49.
            distance2_number += 1

            if distance2_number not in EXCLUDED_DISTANCE2:
                confirmed_pairs.append(f"{file1}|{file2}")

OUTPUT_FILE.write_text(
    "\n".join(confirmed_pairs) + "\n",
    encoding="utf-8"
)

print("=== CONFIRMED DUPLICATE PAIR UPDATE ===")
print(f"Distance-2 candidates processed : {distance2_number}")
print(f"Excluded distance-2 pairs       : {sorted(EXCLUDED_DISTANCE2)}")
print(f"Confirmed pair relationships    : {len(confirmed_pairs)}")
print(f"Written to                      : {OUTPUT_FILE}")