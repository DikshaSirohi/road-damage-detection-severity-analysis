from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

DATASET_DIR = Path(
    r"C:\NIET\Projects\Road Damage\combined_annotatedv2_working"
)

OUTPUT_PATH = Path(
    r"C:\NIET\Projects\Road Damage\phash_zero_candidates.jpg"
)

PAIRS = [
    ("Czech_000618.jpg", "Czech_001050.jpg"),
    ("Czech_000867.jpg", "Czech_000913.jpg"),
    ("Czech_001783.jpg", "Czech_001981.jpg"),
    ("Czech_002030.jpg", "Czech_002124.jpg"),
    ("Czech_002209.jpg", "Czech_003103.jpg"),
    ("Japan_001402.jpg", "Japan_012647.jpg"),
    ("Japan_002321.jpg", "Japan_013130.jpg"),
    ("United_States_001565.jpg", "United_States_001963.jpg"),
]

THUMB_SIZE = (320, 320)
TEXT_HEIGHT = 35
GAP = 20

row_width = THUMB_SIZE[0] * 2 + GAP
row_height = THUMB_SIZE[1] + TEXT_HEIGHT + GAP

sheet = Image.new(
    "RGB",
    (row_width, row_height * len(PAIRS)),
    "white"
)

draw = ImageDraw.Draw(sheet)

for row, (name1, name2) in enumerate(PAIRS):

    y = row * row_height

    img1 = Image.open(DATASET_DIR / name1).convert("RGB")
    img2 = Image.open(DATASET_DIR / name2).convert("RGB")

    img1 = ImageOps.fit(img1, THUMB_SIZE)
    img2 = ImageOps.fit(img2, THUMB_SIZE)

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

sheet.save(OUTPUT_PATH, quality=95)

print("\nContact sheet created:")
print(OUTPUT_PATH)
print(f"Pairs shown: {len(PAIRS)}")