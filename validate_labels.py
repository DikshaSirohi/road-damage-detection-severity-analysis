from pathlib import Path

DATASET_DIR = Path(r"C:\NIET\Projects\Road Damage\combined_annotatedv2_working")

VALID_CLASS_IDS = set(range(8))

total_label_files = 0
empty_label_files = 0
total_annotations = 0

malformed_files = set()
invalid_lines = []

for label_path in DATASET_DIR.glob("*.txt"):
    if label_path.name.lower() == "classes.txt":
        continue

    total_label_files += 1

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        malformed_files.add(label_path.name)
        invalid_lines.append((label_path.name, 0, f"Could not read file: {e}"))
        continue

    non_empty_lines = [line.strip() for line in lines if line.strip()]

    if not non_empty_lines:
        empty_label_files += 1
        continue

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            malformed_files.add(label_path.name)
            invalid_lines.append(
                (label_path.name, line_number, f"Expected 5 values, found {len(parts)}")
            )
            continue

        class_text, x_text, y_text, w_text, h_text = parts

        try:
            class_id = int(class_text)
        except ValueError:
            malformed_files.add(label_path.name)
            invalid_lines.append(
                (label_path.name, line_number, f"Invalid class ID: {class_text}")
            )
            continue

        if class_id not in VALID_CLASS_IDS:
            malformed_files.add(label_path.name)
            invalid_lines.append(
                (label_path.name, line_number, f"Class ID out of range: {class_id}")
            )
            continue

        try:
            x, y, w, h = map(float, (x_text, y_text, w_text, h_text))
        except ValueError:
            malformed_files.add(label_path.name)
            invalid_lines.append(
                (label_path.name, line_number, "Bounding-box values are not numeric")
            )
            continue

        if not (0 <= x <= 1):
            malformed_files.add(label_path.name)
            invalid_lines.append(
                (label_path.name, line_number, f"x_center out of range: {x}")
            )

        if not (0 <= y <= 1):
            malformed_files.add(label_path.name)
            invalid_lines.append(
                (label_path.name, line_number, f"y_center out of range: {y}")
            )

        if not (0 < w <= 1):
            malformed_files.add(label_path.name)
            invalid_lines.append(
                (label_path.name, line_number, f"width invalid: {w}")
            )

        if not (0 < h <= 1):
            malformed_files.add(label_path.name)
            invalid_lines.append(
                (label_path.name, line_number, f"height invalid: {h}")
            )

        total_annotations += 1

print("\n=== YOLO ANNOTATION VALIDATION SUMMARY ===")
print(f"Label files checked : {total_label_files}")
print(f"Empty label files   : {empty_label_files}")
print(f"Annotations checked : {total_annotations}")
print(f"Malformed files     : {len(malformed_files)}")
print(f"Invalid lines found : {len(invalid_lines)}")

if invalid_lines:
    print("\n=== INVALID ANNOTATIONS ===")
    for filename, line_number, problem in invalid_lines:
        print(f"{filename} | line {line_number} | {problem}")
else:
    print("\nNo invalid annotations found.")