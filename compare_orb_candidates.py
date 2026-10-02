from pathlib import Path
import cv2


PROJECT_DIR = Path(r"C:\NIET\Projects\Road Damage")
DATASET_DIR = PROJECT_DIR / "combined_annotatedv2_working"
CANDIDATE_FILE = PROJECT_DIR / "near_duplicate_candidates.txt"
OUTPUT_FILE = PROJECT_DIR / "orb_candidate_report.txt"


def find_image(filename):
    """
    Find an image inside the working dataset.
    near_duplicate_candidates.txt is expected to contain filenames only.
    """
    path = DATASET_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    return path


def load_candidate_pairs():
    pairs = []

    total_lines = 0
    distance_0_count = 0
    distance_2_count = 0

    with CANDIDATE_FILE.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            total_lines += 1

            # Expected format:
            # phash_distance|image1.jpg|image2.jpg
            parts = [part.strip() for part in line.split("|")]

            if len(parts) != 3:
                print(
                    f"Skipping line {line_number}: "
                    f"unexpected format -> {line}"
                )
                continue

            distance_text, filename1, filename2 = parts

            try:
                phash_distance = int(distance_text)
            except ValueError:
                print(
                    f"Skipping line {line_number}: "
                    f"invalid pHash distance -> {distance_text}"
                )
                continue

            if phash_distance == 0:
                distance_0_count += 1
                continue

            if phash_distance == 2:
                distance_2_count += 1
                pairs.append((filename1, filename2))
                continue

            print(
                f"Skipping line {line_number}: "
                f"unexpected pHash distance {phash_distance}"
            )

    print(f"Candidate file entries : {total_lines}")
    print(f"Distance-0 pairs skipped: {distance_0_count}")
    print(f"Distance-2 pairs loaded : {distance_2_count}")

    return pairs


def compare_with_orb(image_path1, image_path2):
    img1 = cv2.imread(str(image_path1), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(image_path2), cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        raise ValueError(f"Could not read image: {image_path1}")

    if img2 is None:
        raise ValueError(f"Could not read image: {image_path2}")

    # ORB detects local keypoints and calculates binary descriptors.
    orb = cv2.ORB_create(
        nfeatures=3000,
        scaleFactor=1.2,
        nlevels=8
    )

    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    kp1_count = len(keypoints1)
    kp2_count = len(keypoints2)

    if descriptors1 is None or descriptors2 is None:
        return {
            "kp1": kp1_count,
            "kp2": kp2_count,
            "good_matches": 0,
            "ratio": 0.0,
            "median_distance": None,
        }

    # Hamming distance is appropriate for ORB's binary descriptors.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # For each feature in image 1, obtain its two closest candidates
    # from image 2.
    knn_matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    # Lowe-style ratio test:
    # A match is more convincing when the best candidate is clearly
    # better than the second-best candidate.
    for pair in knn_matches:
        if len(pair) < 2:
            continue

        best, second_best = pair

        if best.distance < 0.75 * second_best.distance:
            good_matches.append(best)

    good_count = len(good_matches)

    # Normalize against the smaller detected-feature count.
    denominator = min(kp1_count, kp2_count)

    if denominator > 0:
        match_ratio = good_count / denominator
    else:
        match_ratio = 0.0

    if good_matches:
        distances = sorted(match.distance for match in good_matches)
        middle = len(distances) // 2

        if len(distances) % 2 == 1:
            median_distance = distances[middle]
        else:
            median_distance = (
                distances[middle - 1] + distances[middle]
            ) / 2
    else:
        median_distance = None

    return {
        "kp1": kp1_count,
        "kp2": kp2_count,
        "good_matches": good_count,
        "ratio": match_ratio,
        "median_distance": median_distance,
    }


def main():
    print("Loading candidate pairs...")

    pairs = load_candidate_pairs()

    print(f"Candidate pairs loaded: {len(pairs)}")
    print()
    print("Running ORB comparison...")
    print()

    results = []

    for index, (filename1, filename2) in enumerate(pairs, start=1):
        path1 = find_image(filename1)
        path2 = find_image(filename2)

        result = compare_with_orb(path1, path2)

        result["pair_number"] = index
        result["file1"] = filename1
        result["file2"] = filename2

        results.append(result)

        median_text = (
            f"{result['median_distance']:.1f}"
            if result["median_distance"] is not None
            else "N/A"
        )

        print(
            f"{index:02d}. "
            f"{filename1} | {filename2} | "
            f"KP={result['kp1']}/{result['kp2']} | "
            f"Good={result['good_matches']} | "
            f"Ratio={result['ratio']:.4f} | "
            f"MedianDist={median_text}"
        )

    # Sort only for the ranking section.
    strongest = sorted(
        results,
        key=lambda x: (
            x["ratio"],
            x["good_matches"]
        ),
        reverse=True
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        f.write("ORB NEAR-DUPLICATE CANDIDATE REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            "Important: ORB scores are supporting evidence only. "
            "They must NOT be used as an automatic duplicate label.\n\n"
        )

        f.write("ORIGINAL CANDIDATE ORDER\n")
        f.write("-" * 80 + "\n")

        for result in results:
            median_text = (
                f"{result['median_distance']:.1f}"
                if result["median_distance"] is not None
                else "N/A"
            )

            f.write(
                f"{result['pair_number']:02d}. "
                f"{result['file1']} | {result['file2']}\n"
            )
            f.write(
                f"    Keypoints: "
                f"{result['kp1']} / {result['kp2']}\n"
            )
            f.write(
                f"    Good matches: "
                f"{result['good_matches']}\n"
            )
            f.write(
                f"    Match ratio: "
                f"{result['ratio']:.4f}\n"
            )
            f.write(
                f"    Median good-match distance: "
                f"{median_text}\n\n"
            )

        f.write("\n")
        f.write("RANKED BY MATCH RATIO — STRONGEST FIRST\n")
        f.write("-" * 80 + "\n")

        for rank, result in enumerate(strongest, start=1):
            f.write(
                f"{rank:02d}. "
                f"Pair {result['pair_number']:02d} | "
                f"{result['file1']} | {result['file2']} | "
                f"Good={result['good_matches']} | "
                f"Ratio={result['ratio']:.4f}\n"
            )

    print()
    print("=== ORB COMPARISON COMPLETE ===")
    print(f"Pairs processed : {len(results)}")
    print(f"Report written  : {OUTPUT_FILE}")
    print()
    print("No dataset files were modified.")


if __name__ == "__main__":
    main()