"""
==========================================================================
[4] EVALUASI DETAIL & VISUALISASI HASIL PREDIKSI
==========================================================================
Tujuan: menghasilkan output tambahan untuk Bab 3 yang tidak otomatis
        lengkap dari model.val(), yaitu:
  - Tabel per-kelas P/R/mAP50/mAP50-95 (Tabel 5 & 6) dalam format CSV rapi
  - Grid visualisasi prediksi pada beberapa citra test, termasuk
    contoh BENAR dan contoh GAGAL (Gambar 7)

CARA PAKAI:
  python 4_evaluate_and_visualize.py \
      --weights outputs/3_train/balanced_yolov8n/weights/best.pt \
      --split test

Output default: outputs/4_evaluate/<run>_<split>/
==========================================================================
"""

import argparse
import math
import random
from pathlib import Path

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "outputs" / "5_split" / "dataset_801010"
DEFAULT_OUT_DIR = PROJECT_DIR / "outputs" / "4_evaluate"


def per_class_metrics_table(metrics, class_names, out_csv):
    """
    Mengubah hasil metrics.box dari Ultralytics menjadi DataFrame
    per kelas (format Tabel 5/6 di paper: P, R, mAP50, mAP50-95).
    """
    rows = []
    # metrics.box.ap_class_index -> index kelas yang punya hasil evaluasi
    # metrics.box.p, .r, .ap50, .ap -> array per kelas
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        rows.append({
            "class_name": class_names[int(cls_idx)],
            "precision": metrics.box.p[i],
            "recall": metrics.box.r[i],
            "mAP50": metrics.box.ap50[i],
            "mAP50-95": metrics.box.ap[i],
        })

    df = pd.DataFrame(rows)

    # baris rata-rata keseluruhan (mengikuti format Tabel 1 Paskasius)
    avg_row = {
        "class_name": "Rata-rata (all classes)",
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    df = df.round(3)
    df.to_csv(out_csv, index=False)
    print(f"\nTabel per-kelas tersimpan di: {out_csv}")
    print(df.to_string(index=False))
    return df


def visualize_predictions(model, images_dir, class_names, out_path,
                           n_images=9, conf_threshold=0.25):
    """
    Membuat grid 3x3 visualisasi prediksi pada citra test secara acak,
    untuk Gambar 7 (grid prediksi, termasuk contoh gagal jika ada).
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Folder gambar tidak ditemukan: {images_dir}")
    if n_images < 1:
        raise ValueError("n_images harus lebih besar dari 0")
    image_files = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_files:
        raise ValueError(f"Tidak ada gambar yang dapat divisualisasikan di: {images_dir}")
    image_files = random.Random(42).sample(
        image_files, min(n_images, len(image_files))
    )

    grid_size = math.ceil(math.sqrt(len(image_files)))
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(15, 15), squeeze=False
    )
    axes = axes.flatten()

    for idx, img_path in enumerate(image_files):
        results = model.predict(str(img_path), conf=conf_threshold, verbose=False)
        result = results[0]

        img = result.orig_img[:, :, ::-1]  # BGR -> RGB
        ax = axes[idx]
        ax.imshow(img)

        n_detections = len(result.boxes)
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, max(y1 - 5, 0),
                     f"{class_names[cls_id]} {conf:.2f}",
                     color="white", fontsize=8,
                     bbox=dict(facecolor="green", alpha=0.6, pad=1))

        title = img_path.name if n_detections > 0 else f"{img_path.name} (TIDAK TERDETEKSI)"
        ax.set_title(title, fontsize=9,
                     color="black" if n_detections > 0 else "red")
        ax.axis("off")

    # matikan axis kosong jika gambar kurang dari grid
    for idx in range(len(image_files), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Grid visualisasi prediksi tersimpan di: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path ke best.pt")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n_vis", type=int, default=9)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    data_yaml = data_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Konfigurasi dataset tidak ditemukan: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8") as f:
        class_names = yaml.safe_load(f).get("names")
    if isinstance(class_names, dict):
        class_names = [class_names[key] for key in sorted(class_names, key=int)]
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"Daftar nama kelas tidak valid di: {data_yaml}")
    if args.n_vis < 1:
        parser.error("--n_vis harus lebih besar dari 0")

    weights_path = Path(args.weights).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Bobot model tidak ditemukan: {weights_path}")
    run_name = f"{weights_path.parent.parent.name}_{args.split}"
    out_dir = args.out_dir.resolve() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))

    # --- Evaluasi metrik per kelas ---
    # Ultralytics membaca kunci 'val' di data.yaml, bukan 'valid'
    ul_split = "val" if args.split == "valid" else args.split
    metrics = model.val(
        data=str(data_yaml),
        split=ul_split,
        save_json=True,
        project=str(out_dir),
        name="ultralytics",
        exist_ok=True,
    )
    out_csv = out_dir / f"metrics_{args.split}.csv"
    per_class_metrics_table(metrics, class_names, out_csv)

    # --- Visualisasi prediksi ---
    images_dir = data_dir / args.split / "images"
    out_vis_path = out_dir / f"prediction_grid_{args.split}.png"
    visualize_predictions(model, images_dir, class_names, out_vis_path,
                           n_images=args.n_vis)

    print(f"\nSemua output evaluasi tersimpan di: {out_dir}")
    print(f"Output tambahan Ultralytics tersimpan di: {out_dir / 'ultralytics'}")


if __name__ == "__main__":
    main()