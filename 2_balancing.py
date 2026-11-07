"""
==========================================================================
[2] DATA BALANCING (TRAIN SET ONLY) - NEU-DET STEEL DEFECT
==========================================================================
Tujuan: membuat versi "balanced" dari train set untuk eksperimen
        balanced vs unbalanced (Bab 2.5 & 2.10 di paper).

PENTING (sesuai arahan dosen di catatan kamu):
  - Balancing HANYA diterapkan ke train set.
  - valid/ dan test/ TIDAK disentuh sama sekali, supaya evaluasi tetap adil
    dan mencerminkan kondisi data asli.

Strategi balancing yang dipakai: OVERSAMPLING berbasis gambar.
  1. Hitung jumlah bounding box per kelas di train set (hasil EDA).
  2. Cari kelas dengan jumlah box terbanyak -> jadi target jumlah.
  3. Untuk kelas yang kurang, duplikasi gambar yang mengandung kelas
     tersebut (dengan augmentasi ringan tiap duplikat, bukan copy identik)
     sampai jumlah box-nya mendekati kelas mayoritas.

Catatan: oversampling berbasis gambar dipakai (bukan SMOTE) karena ini
tugas object detection, bukan klasifikasi -> SMOTE tidak berlaku di sini
(perlu disebutkan di Bab 2.6 paper).

Output:
  outputs/2_balancing/train_balanced/images/*.jpg
  outputs/2_balancing/train_balanced/labels/*.txt
  outputs/2_balancing/balancing_report.csv -> untuk Tabel 3
  outputs/2_balancing/data_balanced.yaml

CARA PAKAI:
  python 2_balancing.py
  python 2_balancing.py --data_dir ./path/to/dataset --out_dir ./path/to/train_balanced
==========================================================================
"""

import os
import argparse
import shutil
import random
import yaml
import pandas as pd
import cv2
import albumentations as A
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "outputs" / "5_split" / "dataset_801010"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "outputs" / "2_balancing" / "train_balanced"

def load_class_names(data_yaml_path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["names"]


def get_image_classes(labels_dir):
    """
    Mengembalikan dict: {nama_gambar: set(class_id yang muncul di gambar itu)}
    """
    image_classes = {}
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue
        classes_in_image = set()
        with open(os.path.join(labels_dir, label_file), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    classes_in_image.add(int(parts[0]))
        img_name = os.path.splitext(label_file)[0]
        image_classes[img_name] = classes_in_image
    return image_classes


def count_boxes_per_class(labels_dir, num_classes):
    counts = defaultdict(int)
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue
        with open(os.path.join(labels_dir, label_file), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    counts[int(parts[0])] += 1
    return counts


def get_augmentation_pipeline():
    """
    Augmentasi RINGAN untuk oversampling, supaya duplikat tidak identik
    pixel-per-pixel (membantu generalisasi, bukan cuma duplikasi murni).
    bbox_params memastikan koordinat box ikut ditransformasi dengan benar.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.4),
            A.Rotate(limit=10, p=0.4, border_mode=cv2.BORDER_CONSTANT),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def read_yolo_label(label_path):
    boxes, labels = [], []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            labels.append(int(parts[0]))
            boxes.append([float(x) for x in parts[1:5]])
    return boxes, labels


def write_yolo_label(label_path, boxes, labels):
    with open(label_path, "w") as f:
        for box, label in zip(boxes, labels):
            f.write(f"{label} {' '.join(f'{v:.6f}' for v in box)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR),
                         help="Folder output train yang sudah diseimbangkan")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    data_yaml = data_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Konfigurasi dataset tidak ditemukan: {data_yaml}")
    class_names = load_class_names(data_yaml)
    num_classes = len(class_names)

    train_images_dir = data_dir / "train" / "images"
    train_labels_dir = data_dir / "train" / "labels"
    if not train_images_dir.is_dir():
        raise FileNotFoundError(f"Folder gambar train tidak ditemukan: {train_images_dir}")
    if not train_labels_dir.is_dir():
        raise FileNotFoundError(f"Folder label train tidak ditemukan: {train_labels_dir}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_images_dir = os.path.join(out_dir, "images")
    out_labels_dir = os.path.join(out_dir, "labels")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    # --- Step 1: salin dulu seluruh train asli ke folder balanced ---
    print("Menyalin train set asli ke folder balanced...")
    for img_file in os.listdir(train_images_dir):
        shutil.copy(os.path.join(train_images_dir, img_file),
                    os.path.join(out_images_dir, img_file))
    for label_file in os.listdir(train_labels_dir):
        shutil.copy(os.path.join(train_labels_dir, label_file),
                    os.path.join(out_labels_dir, label_file))

    # --- Step 2: hitung distribusi box SEBELUM balancing ---
    before_counts = count_boxes_per_class(train_labels_dir, num_classes)
    print("\nJumlah box per kelas SEBELUM balancing:")
    for cid, cname in enumerate(class_names):
        print(f"  {cname}: {before_counts.get(cid, 0)}")

    target_count = max(before_counts.values())
    print(f"\nTarget jumlah box per kelas (mengikuti kelas mayoritas): {target_count}")

    # --- Step 3: cari gambar per kelas untuk oversampling ---
    image_classes = get_image_classes(train_labels_dir)
    images_by_class = defaultdict(list)
    for img_name, classes in image_classes.items():
        for cid in classes:
            images_by_class[cid].append(img_name)

    transform = get_augmentation_pipeline()
    after_counts = dict(before_counts)

    # --- Step 4: oversampling untuk kelas yang kurang dari target ---
    for cid, cname in enumerate(class_names):
        current = before_counts.get(cid, 0)
        if current >= target_count or current == 0:
            continue

        deficit = target_count - current
        candidate_images = images_by_class.get(cid, [])
        if not candidate_images:
            print(f"  Peringatan: tidak ada gambar untuk kelas {cname}, dilewati.")
            continue

        # perkiraan rata-rata box kelas ini per gambar kandidat
        avg_box_per_image = max(current / len(candidate_images), 1e-6)
        n_duplicates_needed = max(1, int(deficit / avg_box_per_image))

        print(f"Oversampling kelas '{cname}': butuh ~{n_duplicates_needed} duplikat gambar")

        for i in range(n_duplicates_needed):
            src_img_name = random.choice(candidate_images)

            # cari ekstensi file gambar aslinya
            matches = [f for f in os.listdir(train_images_dir)
                       if os.path.splitext(f)[0] == src_img_name]
            if not matches:
                continue
            src_img_file = matches[0]
            ext = os.path.splitext(src_img_file)[1]

            img_path = os.path.join(train_images_dir, src_img_file)
            label_path = os.path.join(train_labels_dir, src_img_name + ".txt")

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes, labels = read_yolo_label(label_path)

            try:
                augmented = transform(image=image, bboxes=boxes, class_labels=labels)
            except Exception as e:
                print(f"  Lewati {src_img_name} karena error augmentasi: {e}")
                continue

            new_name = f"{src_img_name}_aug{i}"
            new_img_path = os.path.join(out_images_dir, new_name + ext)
            new_label_path = os.path.join(out_labels_dir, new_name + ".txt")

            aug_image_bgr = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(new_img_path, aug_image_bgr)
            write_yolo_label(new_label_path, augmented["bboxes"], augmented["class_labels"])

            for l in augmented["class_labels"]:
                after_counts[l] = after_counts.get(l, 0) + 1

    # --- Step 5: laporan before vs after (Tabel 3) ---
    report_rows = []
    for cid, cname in enumerate(class_names):
        report_rows.append({
            "class_name": cname,
            "before": before_counts.get(cid, 0),
            "after": after_counts.get(cid, 0),
        })
    report_df = pd.DataFrame(report_rows)
    report_path = out_dir.parent / "balancing_report.csv"
    report_df.to_csv(report_path, index=False)

    print("\nTabel 3 - Before vs After Balancing:")
    print(report_df)
    print(f"\nDataset balanced tersimpan di: {out_dir}")
    print(f"Laporan balancing tersimpan di: {report_path}")

    # --- Step 6: buat data.yaml khusus untuk versi balanced ---
    with open(data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    balanced_yaml = dict(data_cfg)
    balanced_yaml["path"] = str(data_dir)
    balanced_yaml["train"] = os.path.relpath(out_images_dir, data_dir).replace(os.sep, "/")
    balanced_yaml["val"] = "valid/images"
    balanced_yaml["test"] = "test/images"
    balanced_yaml_path = out_dir.parent / "data_balanced.yaml"
    with open(balanced_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(balanced_yaml, f)
    print(f"data.yaml untuk training balanced tersimpan di: {balanced_yaml_path}")


if __name__ == "__main__":
    main()
