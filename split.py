"""Split an existing YOLO dataset into train/valid/test with an 80/10/10 ratio."""

import argparse
import collections
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SRC = PROJECT_DIR / "Defect in NEU.v2i.yolov8"
DEFAULT_DST = PROJECT_DIR / "outputs" / "5_split" / "dataset_801010"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_primary_class(image_path):
    """Return the most frequent class in the image for approximate stratification."""
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if not label_path.is_file():
        return "__background__"

    classes = []
    with label_path.open("r", encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"Label harus memiliki 5 kolom: {label_path}:{line_number}")
            try:
                classes.append(int(fields[0]))
                for value in fields[1:]:
                    float(value)
            except ValueError as exc:
                raise ValueError(f"Format label tidak valid: {label_path}:{line_number}") from exc

    if not classes:
        return "__background__"
    return collections.Counter(classes).most_common(1)[0][0]


def split_items(items, labels, test_size, description):
    """Stratify by the most frequent class, falling back when strata are too small."""
    try:
        return train_test_split(
            items, labels, test_size=test_size, stratify=labels, random_state=42
        )
    except ValueError as exc:
        print(f"Peringatan: stratifikasi {description} tidak dapat dilakukan ({exc}).")
        print(f"Membagi {description} secara acak sebagai fallback.")
        return train_test_split(
            items, labels, test_size=test_size, random_state=42
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC,
                        help="Folder dataset sumber (default: Defect in NEU.v2i.yolov8)")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST,
                        help="Folder keluaran (default: outputs/5_split/dataset_801010)")
    args = parser.parse_args()

    source_dir = args.src.resolve()
    destination_dir = args.dst.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Folder sumber tidak ditemukan: {source_dir}")
    if source_dir == destination_dir or source_dir in destination_dir.parents:
        raise ValueError("Folder keluaran harus berada di luar folder dataset sumber.")
    if destination_dir.exists() and any(destination_dir.iterdir()):
        raise FileExistsError(
            f"Folder keluaran sudah ada dan tidak kosong: {destination_dir}. "
            "Pilih folder lain dengan --dst."
        )

    images = []
    for split in ("train", "valid", "test"):
        images_dir = source_dir / split / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Folder gambar tidak ditemukan: {images_dir}")
        images.extend(
            path for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    images.sort()
    if len(images) < 3:
        raise ValueError(f"Minimal 3 gambar diperlukan untuk split; ditemukan {len(images)}.")
    image_names = [image.name for image in images]
    if len(image_names) != len(set(image_names)):
        raise ValueError("Nama file gambar duplikat ditemukan di beberapa split sumber.")

    labels = [image_primary_class(image) for image in images]
    train_images, remainder_images, train_labels, remainder_labels = split_items(
        images, labels, test_size=0.2, description="train dan sisa data"
    )
    valid_images, test_images, _, _ = split_items(
        remainder_images, remainder_labels, test_size=0.5,
        description="valid dan test"
    )

    for split_name, split_images in (
        ("train", train_images),
        ("valid", valid_images),
        ("test", test_images),
    ):
        images_dir = destination_dir / split_name / "images"
        labels_dir = destination_dir / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for image in split_images:
            shutil.copy2(image, images_dir / image.name)
            label = image.parent.parent / "labels" / f"{image.stem}.txt"
            if label.is_file():
                shutil.copy2(label, labels_dir / label.name)

        print(f"{split_name}: {len(split_images)} gambar")

    source_yaml = source_dir / "data.yaml"
    if source_yaml.is_file():
        shutil.copy2(source_yaml, destination_dir / "data.yaml")

    print(f"\nDataset split tersimpan di: {destination_dir}")
    print("Rasio target: 80% train, 10% valid, 10% test.")
    print("Stratifikasi berdasarkan kelas yang paling sering muncul per gambar.")


if __name__ == "__main__":
    main()
