"""
Dataset Validation Script for Capsule Vision Inspection.

Validates a YOLO-format detection dataset before training.

Checks:
    - Total images
    - Images per class
    - Annotations per class
    - Missing labels
    - Empty labels
    - Invalid coordinates
    - Duplicate images
    - Image dimensions
    - Class imbalance

Usage:
    python scripts/validate_dataset.py --dataset /path/to/dataset
"""

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

CLASS_NAMES = [
    "Good",
    "Crack",
    "Scratch",
    "Faulty Imprint",
    "Poke",
    "Squeeze",
    "Contamination",
]


def file_hash(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(dataset_path: Path) -> bool:
    """Validate a YOLO detection dataset. Returns True if valid."""
    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    issues = []
    stats = {
        "total_images": 0,
        "splits": {},
        "annotations_per_class": Counter(),
        "missing_labels": [],
        "empty_labels": [],
        "invalid_coords": [],
        "duplicate_images": [],
        "image_dims": [],
    }

    for split in ["train", "val", "test"]:
        img_dir = dataset_path / "images" / split
        lbl_dir = dataset_path / "labels" / split

        if not img_dir.exists():
            print(f"⚠️  {img_dir} does not exist")
            continue

        images = sorted(img_dir.glob("*.*"))
        stats["splits"][split] = len(images)
        stats["total_images"] += len(images)
        print(f"\n{split}: {len(images)} images")

        seen_hashes = {}

        for img_path in images:
            # Check for duplicates
            h = file_hash(img_path)
            if h in seen_hashes:
                stats["duplicate_images"].append((str(img_path), seen_hashes[h]))
            else:
                seen_hashes[h] = str(img_path)

            # Check label file
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                stats["missing_labels"].append(str(img_path))
                continue

            # Read label
            with open(lbl_path) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            if not lines:
                stats["empty_labels"].append(str(img_path))
                continue

            # Parse annotations
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    stats["invalid_coords"].append((str(img_path), line))
                    continue

                try:
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = [float(p) for p in parts[1:]]
                except ValueError:
                    stats["invalid_coords"].append((str(img_path), line))
                    continue

                # Validate coordinates
                if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    stats["invalid_coords"].append((str(img_path), line))
                    continue

                if cls_id >= len(CLASS_NAMES):
                    stats["invalid_coords"].append((str(img_path), f"Invalid class: {cls_id}"))
                    continue

                stats["annotations_per_class"][cls_id] += 1

            # Image dimensions
            try:
                with Image.open(img_path) as img:
                    stats["image_dims"].append(img.size)
            except Exception:
                pass

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images: {stats['total_images']}")
    print(f"Splits: {stats['splits']}")

    print("\nAnnotations per class:")
    for cls_id in range(len(CLASS_NAMES)):
        count = stats["annotations_per_class"].get(cls_id, 0)
        print(f"  {CLASS_NAMES[cls_id]}: {count}")

    # Check issues
    if stats["missing_labels"]:
        issues.append(f"Missing labels: {len(stats['missing_labels'])}")
        for p in stats["missing_labels"][:5]:
            print(f"  ❌ Missing label: {p}")

    if stats["empty_labels"]:
        issues.append(f"Empty labels: {len(stats['empty_labels'])}")
        for p in stats["empty_labels"][:5]:
            print(f"  ❌ Empty label: {p}")

    if stats["invalid_coords"]:
        issues.append(f"Invalid coordinates: {len(stats['invalid_coords'])}")
        for p, line in stats["invalid_coords"][:5]:
            print(f"  ❌ Invalid: {p}: {line}")

    if stats["duplicate_images"]:
        issues.append(f"Duplicate images: {len(stats['duplicate_images'])}")
        for p1, p2 in stats["duplicate_images"][:5]:
            print(f"  ❌ Duplicate: {p1} == {p2}")

    # Class imbalance check
    total_anns = sum(stats["annotations_per_class"].values())
    if total_anns > 0:
        max_count = max(stats["annotations_per_class"].values())
        min_count = min(stats["annotations_per_class"].values())
        if min_count > 0:
            ratio = max_count / min_count
            print(f"\nMax/Min class ratio: {ratio:.1f}")
            if ratio > 10:
                issues.append(f"Severe class imbalance (ratio: {ratio:.1f})")
            elif ratio > 5:
                issues.append(f"Moderate class imbalance (ratio: {ratio:.1f})")

    # Image dimensions
    if stats["image_dims"]:
        widths = [w for w, h in stats["image_dims"]]
        heights = [h for w, h in stats["image_dims"]]
        print(f"\nImage dimensions: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")

    # Final verdict
    print("\n" + "=" * 60)
    if issues:
        print("❌ VALIDATION FAILED")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before training.")
        return False
    else:
        print("✅ VALIDATION PASSED")
        print("Dataset is ready for training.")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate a YOLO detection dataset")
    parser.add_argument("--dataset", required=True, help="Path to the dataset directory")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    valid = validate_dataset(dataset_path)
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()