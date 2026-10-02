from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import math

PROJECT_DIR = Path(r"C:\NIET\Projects\Road Damage")

DATASET_DIR = PROJECT_DIR / "combined_annotatedv2_working"
CANDIDATE_FILE = PROJECT_DIR / "near_duplicate_candidates.txt"
OUTPUT_DIR = PROJECT_DIR / "distance2_contact_sheets"

OUTPUT_DIR.mkdir(exist_ok=True)

THUMB_SIZE = (300, 300)
TEXT_HEIGHT = 35
GAP = 20

PAIRS_PER_SHEET = 10

pairs = []

for line in CANDIDATE_FILE.read_text(encoding="utf-8").splitlines():
    distance, name1, name2 = line.split("|")

    if int(distance) == 2:
        pairs.append((name1, name2))

print(f"Distance-2 pairs found: {len(pairs)}")

sheet_count = math.ceil(len(pairs) / PAIRS_PER_SHEET)

for sheet_index in range(sheet_count):

    start = sheet_index * PAIRS_PER_SHEET
    end = start + PAIRS_PER_SHEET

    sheet_pairs = pairs[start:end]

    row_width = THUMB_SIZE[0] * 2 + GAP
    row_height = THUMB_SIZE[1] + TEXT_HEIGHT + GAP

    sheet = Image.new(
        "RGB",
        (row_width, row_height * len(sheet_pairs)),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for row, (name1, name2) in enumerate(sheet_pairs):

        y = row * row_height

        with Image.open(DATASET_DIR / name1) as img:
            img1 = ImageOps.fit(img.convert("RGB"), THUMB_SIZE)

        with Image.open(DATASET_DIR / name2) as img:
            img2 = ImageOps.fit(img.convert("RGB"), THUMB_SIZE)

        x1 = 0
        x2 = THUMB_SIZE[0] + GAP

        sheet.paste(img1, (x1, y))
        sheet.paste(img2, (x2, y))

        draw.text(
            (x1, y + THUMB_SIZE[1] + 5),
            name1,
            fill="black"
        )

        draw.text(
            (x2, y + THUMB_SIZE[1] + 5),
            name2,
            fill="black"
        )

    output_path = OUTPUT_DIR / f"distance2_sheet_{sheet_index + 1}.jpg"

    sheet.save(output_path, quality=95)

    print(f"Created: {output_path}")

print("\nDone.")