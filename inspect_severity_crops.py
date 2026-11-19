
from pathlib import Path
from PIL import Image, ImageDraw
import random

ROOT = Path(__file__).resolve().parent

IMAGES = ROOT / "road_damage_yolo" / "images" / "train"
LABELS = ROOT / "road_damage_yolo" / "labels" / "train"

OUTPUT = ROOT / "severity_audit_crops"
OUTPUT.mkdir(exist_ok=True)

random.seed(42)

CLASSES = {
    4: "Alligator crack",
    5: "Alligator crack sunken",
    6: "Pothole",
    7: "Pothole deep"
}

samples = {c: [] for c in CLASSES}

for label_file in sorted(LABELS.glob("*.txt")):

    if label_file.name.startswith("xmirror_"):
        continue

    image_file = IMAGES / (label_file.stem + ".jpg")

    if not image_file.exists():
        continue

    seen = set()

    for line in label_file.read_text().splitlines():
        values = line.split()

        if len(values) != 5:
            continue

        class_id = int(values[0])

        if class_id not in CLASSES:
            continue

        if class_id in seen:
            continue

        seen.add(class_id)

        box = list(map(float, values[1:]))

        samples[class_id].append((image_file, box))


print("=== AVAILABLE ORIGINAL TRAINING IMAGES ===")

for class_id, items in samples.items():
    print(f"{CLASSES[class_id]}: {len(items)}")


def create_sheet(normal_id, severe_id, filename):

    tile = 300
    caption = 65
    cols = 3
    rows = 4

    sheet = Image.new(
        "RGB",
        (cols * tile, rows * (tile + caption)),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    selected = []

    for class_id in (normal_id, severe_id):

        chosen = random.sample(samples[class_id], 6)

        for image_file, box in chosen:
            selected.append((class_id, image_file, box))

    for index, (class_id, image_file, box) in enumerate(selected):

        img = Image.open(image_file).convert("RGB")

        w, h = img.size
        xc, yc, bw, bh = box

        # Original bounding-box coordinates
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h

        # Include surrounding road context
        pad_x = max(15, (x2 - x1) * 0.35)
        pad_y = max(15, (y2 - y1) * 0.35)

        left = max(0, int(x1 - pad_x))
        top = max(0, int(y1 - pad_y))
        right = min(w, int(x2 + pad_x))
        bottom = min(h, int(y2 + pad_y))

        crop = img.crop((left, top, right, bottom))

        # Preserve aspect ratio
        scale = min(
            tile / crop.width,
            tile / crop.height
        )

        new_w = max(1, round(crop.width * scale))
        new_h = max(1, round(crop.height * scale))

        crop = crop.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS
        )

        # Draw the original annotation
        crop_draw = ImageDraw.Draw(crop)

        crop_draw.rectangle(
            (
                (x1 - left) * new_w / (right - left),
                (y1 - top) * new_h / (bottom - top),
                (x2 - left) * new_w / (right - left),
                (y2 - top) * new_h / (bottom - top)
            ),
            outline="yellow",
            width=2
        )

        # Center the crop in the contact-sheet tile
        img = Image.new(
            "RGB",
            (tile, tile),
            "black"
        )

        img.paste(
            crop,
            (
                (tile - new_w) // 2,
                (tile - new_h) // 2
            )
        )

        # Calculate contact-sheet position
        col = index % cols
        row = index // cols

        x = col * tile
        y = row * (tile + caption)

        sheet.paste(img, (x, y))

        # Add class name
        draw.text(
            (x + 5, y + tile + 5),
            CLASSES[class_id],
            fill="black"
        )

        # Add image filename
        draw.text(
            (x + 5, y + tile + 23),
            image_file.name,
            fill="black"
        )

    output_file = OUTPUT / filename

    sheet.save(output_file)

    print(f"Saved: {output_file}")


create_sheet(
    6,
    7,
    "pothole_severity.jpg"
)

create_sheet(
    4,
    5,
    "alligator_severity.jpg"
)

print("\nAudit sheets created successfully.")
print("No dataset images or annotations modified.")
