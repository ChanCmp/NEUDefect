"""
==========================================================================
[1] EXPLORATORY DATA ANALYSIS (EDA) - NEU-DET STEEL DEFECT (Roboflow YOLO)
==========================================================================
Tujuan: menghasilkan semua tabel & chart untuk Bab 2.3 (EDA) di paper.

Asumsi struktur folder dataset (format export YOLOv8 dari Roboflow):

dataset/
  data.yaml
  train/
    images/*.jpg
    labels/*.txt
  valid/
    images/*.jpg
    labels/*.txt
  test/
    images/*.jpg
    labels/*.txt

Setiap file label .txt berisi baris: class_id x_center y_center width height
(semua dinormalisasi 0-1, format standar YOLO).

Output yang dihasilkan (semua masuk ke folder outputs/1_eda/):
  - class_distribution.csv      -> Tabel 2 (images/boxes per class per split)
  - class_distribution.png      -> Gambar 3 (bar chart distribusi kelas)
  - box_size_distribution.png   -> statistik ukuran bounding box
  - boxes_per_image.png         -> histogram jumlah box per gambar
  - summary.txt                 -> ringkasan temuan (balanced/tidak)

CARA PAKAI:
  python 1_eda.py
  python 1_eda.py --data_dir ./path/to/dataset --out_dir ./path/to/output
==========================================================================
"""

import os
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "outputs" / "5_split" / "dataset_801010"
DEFAULT_OUT_DIR = PROJECT_DIR / "outputs" / "1_eda"


def load_class_names(data_yaml_path):
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data["names"]


def parse_split(split_dir, class_names):
    """
    Membaca semua file label di satu split (train/valid/test),
    mengembalikan DataFrame dengan satu baris per bounding box.
    """
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")

    rows = []
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        # Ambil dimensi gambar asli (untuk hitung ukuran box dalam piksel)
        img_path = os.path.join(images_dir, img_file)
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            img_w, img_h = None, None

        if not os.path.exists(label_path):
            # Gambar tanpa anotasi (jarang di NEU-DET, tapi dijaga)
            rows.append({
                "image": img_file, "class_id": None, "class_name": None,
                "box_w_norm": None, "box_h_norm": None,
                "box_w_px": None, "box_h_px": None,
            })
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            rows.append({
                "image": img_file, "class_id": None, "class_name": None,
                "box_w_norm": None, "box_h_norm": None,
                "box_w_px": None, "box_h_px": None,
            })

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            _, _, w_norm, h_norm = map(float, parts[1:5])

            row = {
                "image": img_file,
                "class_id": class_id,
                "class_name": class_names[class_id],
                "box_w_norm": w_norm,
                "box_h_norm": h_norm,
            }
            if img_w and img_h:
                row["box_w_px"] = w_norm * img_w
                row["box_h_px"] = h_norm * img_h
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def categorize_box_size(area_px, small_thresh=32**2, medium_thresh=96**2):
    """Kategori ukuran box ala COCO: small/medium/large (dalam piksel^2)."""
    if pd.isna(area_px):
        return None
    if area_px < small_thresh:
        return "small"
    elif area_px < medium_thresh:
        return "medium"
    else:
        return "large"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR),
                         help="Folder root dataset (berisi data.yaml, train/, valid/, test/)")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = data_dir / "data.yaml"
    class_names = load_class_names(data_yaml)
    print(f"Kelas terdeteksi: {class_names}")

    splits = ["train", "valid", "test"]
    all_dfs = {}

    for split in splits:
        split_dir = data_dir / split
        if not os.path.exists(split_dir):
            print(f"Peringatan: folder {split_dir} tidak ditemukan, dilewati.")
            continue
        df = parse_split(split_dir, class_names)
        df["split"] = split
        all_dfs[split] = df
        print(f"{split}: {df['image'].nunique()} gambar, "
              f"{df['class_id'].notna().sum()} bounding box")

    full_df = pd.concat(all_dfs.values(), ignore_index=True)

    # ---------------------------------------------------------------
    # Rasio split (persentase gambar per split) - supaya langsung
    # kelihatan dataset ini 70/20/10, 80/10/10, atau rasio lain
    # ---------------------------------------------------------------
    split_counts = {split: df["image"].nunique() for split, df in all_dfs.items()}
    total_images = sum(split_counts.values())

    print("\n" + "=" * 50)
    print("RASIO SPLIT DATASET")
    print("=" * 50)
    for split, count in split_counts.items():
        pct = (count / total_images * 100) if total_images else 0
        print(f"{split:6s}: {count:5d} gambar ({pct:.1f}%)")
    print(f"{'total':6s}: {total_images:5d} gambar")
    print("=" * 50)

    # ---------------------------------------------------------------
    # Tabel 2: jumlah gambar & box per kelas per split
    # ---------------------------------------------------------------
    box_counts = (
        full_df.dropna(subset=["class_name"])
        .groupby(["split", "class_name"])
        .size()
        .unstack(fill_value=0)
    )
    box_counts = box_counts.reindex(columns=class_names, fill_value=0)
    box_counts.to_csv(out_dir / "class_distribution.csv")
    print("\nTabel 2 - Jumlah bounding box per kelas per split:")
    print(box_counts)

    # ---------------------------------------------------------------
    # Gambar 3: Bar chart distribusi kelas (train vs valid vs test)
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    box_counts.T.plot(kind="bar", ax=ax)
    ax.set_title("Distribusi Jumlah Bounding Box per Kelas per Split")
    ax.set_xlabel("Kelas Defect")
    ax.set_ylabel("Jumlah Bounding Box")
    ax.legend(title="Split")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close()

    # ---------------------------------------------------------------
    # Statistik ukuran bounding box (small/medium/large)
    # ---------------------------------------------------------------
    box_df = full_df.dropna(subset=["box_w_px", "box_h_px"]).copy()
    box_df["area_px"] = box_df["box_w_px"] * box_df["box_h_px"]
    box_df["size_category"] = box_df["area_px"].apply(categorize_box_size)
    box_df["aspect_ratio"] = box_df["box_w_px"] / box_df["box_h_px"]

    size_dist = box_df.groupby(["split", "size_category"]).size().unstack(fill_value=0)
    size_dist.to_csv(out_dir / "box_size_distribution.csv")
    print("\nDistribusi ukuran bounding box (small/medium/large):")
    print(size_dist)

    fig, ax = plt.subplots(figsize=(8, 5))
    size_dist.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Distribusi Ukuran Bounding Box per Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Jumlah Box")
    plt.tight_layout()
    plt.savefig(out_dir / "box_size_distribution.png", dpi=150)
    plt.close()

    # ---------------------------------------------------------------
    # Histogram jumlah box per gambar
    # ---------------------------------------------------------------
    boxes_per_image = full_df.dropna(subset=["class_name"]).groupby(
        ["split", "image"]
    ).size().reset_index(name="box_count")

    fig, ax = plt.subplots(figsize=(8, 5))
    for split in boxes_per_image["split"].unique():
        subset = boxes_per_image[boxes_per_image["split"] == split]
        ax.hist(subset["box_count"], bins=range(1, subset["box_count"].max() + 2),
                alpha=0.5, label=split)
    ax.set_title("Distribusi Jumlah Bounding Box per Gambar")
    ax.set_xlabel("Jumlah Box dalam Satu Gambar")
    ax.set_ylabel("Frekuensi")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "boxes_per_image.png", dpi=150)
    plt.close()

    # ---------------------------------------------------------------
    # Ringkasan temuan (balanced/tidak) -> untuk teks Bab 2.3 / 3.1
    # ---------------------------------------------------------------
    train_counts = box_counts.loc["train"] if "train" in box_counts.index else None
    summary_lines = ["RINGKASAN EDA", "=" * 50]

    summary_lines.append("Rasio split dataset:")
    for split, count in split_counts.items():
        pct = (count / total_images * 100) if total_images else 0
        summary_lines.append(f"  {split}: {count} gambar ({pct:.1f}%)")
    summary_lines.append(f"  total: {total_images} gambar")
    summary_lines.append("")

    if train_counts is not None:
        max_class = train_counts.idxmax()
        min_class = train_counts.idxmin()
        imbalance_ratio = train_counts.max() / max(train_counts.min(), 1)

        summary_lines.append(
            f"Kelas dengan box terbanyak di train: {max_class} ({train_counts.max()} box)"
        )
        summary_lines.append(
            f"Kelas dengan box tersedikit di train: {min_class} ({train_counts.min()} box)"
        )
        summary_lines.append(f"Rasio imbalance (max/min): {imbalance_ratio:.2f}x")
        summary_lines.append(
            "Kesimpulan: " + (
                "Dataset TIDAK seimbang secara jumlah box per kelas, "
                "meskipun jumlah GAMBAR per kelas relatif seimbang."
                if imbalance_ratio > 1.3 else
                "Dataset relatif seimbang baik dari sisi gambar maupun box."
            )
        )

    summary_text = "\n".join(summary_lines)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n" + summary_text)
    print(f"\nSemua output EDA tersimpan di: {out_dir}")


if __name__ == "__main__":
    main()