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
  Gunakan --data_dir ./path/to/dataset jika dataset berada di lokasi lain.
==========================================================================
"""

import os
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "outputs" / "5_split" / "dataset_801010"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "outputs" / "4_evaluate"


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
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_files = image_files[:n_images]

    if n_images < 1:
        raise ValueError("n_images harus lebih besar dari 0")
    grid_size = int(n_images ** 0.5)
    if grid_size * grid_size < n_images:
        grid_size += 1
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
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

        title = img_file if n_detections > 0 else f"{img_file} (TIDAK TERDETEKSI)"
        ax.set_title(title, fontsize=9,
                     color="black" if n_detections > 0 else "red")
        ax.axis("off")

    # matikan axis kosong jika gambar kurang dari grid
    for idx in range(len(image_files), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Grid visualisasi prediksi tersimpan di: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path ke best.pt")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--n_vis", type=int, default=9)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_yaml = data_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Konfigurasi dataset tidak ditemukan: {data_yaml}")
    with open(data_yaml, "r", encoding="utf-8") as f:
        class_names = yaml.safe_load(f)["names"]

    weights_path = Path(args.weights).resolve()
    run_name = f"{weights_path.parent.parent.name}_{args.split}"
    out_dir = Path(args.out_dir).resolve() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    # --- Evaluasi metrik per kelas ---
    metrics = model.val(
        data=data_yaml,
        split=args.split,
        save_json=True,
        project=str(out_dir.parent),
        name=run_name,
        exist_ok=True,
    )
    out_csv = out_dir / f"metrics_{args.split}.csv"
    per_class_metrics_table(metrics, class_names, out_csv)

    # --- Visualisasi prediksi ---
    images_dir = data_dir / args.split / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Folder gambar split '{args.split}' tidak ditemukan: {images_dir}")
    out_vis_path = out_dir / f"prediction_grid_{args.split}.png"
    visualize_predictions(model, images_dir, class_names, out_vis_path,
                           n_images=args.n_vis)

    print(f"\nSemua output evaluasi tersimpan di: {out_dir}")
    print("File confusion_matrix.png, PR_curve.png, F1_curve.png, dll. "
          "tersimpan di folder tersebut oleh Ultralytics.")


if __name__ == "__main__":
    main()
