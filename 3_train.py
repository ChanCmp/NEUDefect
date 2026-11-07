"""
==========================================================================
[3] TRAINING YOLOv8 - SKENARIO UNBALANCED vs BALANCED
==========================================================================
Tujuan: melatih YOLOv8 dua kali (data asli vs data balanced) dengan
        semua konfigurasi lain SAMA, supaya perbandingan adil
        (Bab 2.10 Skenario Eksperimen).

Prasyarat:
  pip install ultralytics
  Sudah menjalankan 1_eda.py dan 2_balancing.py terlebih dahulu.

Struktur yang dibutuhkan:
  dataset/data.yaml           -> untuk skenario UNBALANCED (train asli)
  outputs/2_balancing/data_balanced.yaml -> untuk skenario BALANCED

CARA PAKAI:
  python 3_train.py --model yolov8n.pt --epochs 100
  python 3_train.py --data_dir ./path/to/dataset --model yolov8n.pt --epochs 100

Semua hasil tersimpan otomatis oleh Ultralytics di:
  outputs/3_train/unbalanced_yolov8n/
  outputs/3_train/balanced_yolov8n/
Termasuk: results.png, results.csv, confusion_matrix.png, F1_curve.png,
          PR_curve.png, labels.jpg, args.yaml, weights/best.pt, dst.
(lihat checklist di catatan paper kamu - semua ini otomatis dibuat)
==========================================================================
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "outputs" / "5_split" / "dataset_801010"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "3_train"
DEFAULT_BALANCED_YAML = Path(__file__).resolve().parent / "outputs" / "2_balancing" / "data_balanced.yaml"


def train_scenario(data_yaml_path, model_name, run_name, output_dir, epochs, imgsz, batch, patience):
    """
    Melatih satu skenario (unbalanced atau balanced) dengan konfigurasi
    yang identik, kecuali data.yaml yang dipakai.
    """
    print(f"\n{'=' * 60}")
    print(f"MEMULAI TRAINING: {run_name}")
    print(f"Dataset config : {data_yaml_path}")
    print(f"{'=' * 60}\n")

    model = YOLO(model_name)  # pretrained COCO weights (transfer learning)

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,          # early stopping
        optimizer="auto",           # Ultralytics otomatis pilih SGD/AdamW
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Augmentasi bawaan Ultralytics (dicatat juga di Bab 2.6 paper):
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        close_mosaic=10,  # matikan mosaic di 10 epoch terakhir, stabilkan training
        project=str(output_dir),
        name=run_name,
        exist_ok=True,
        seed=42,
        verbose=True,
    )
    return model, results


def evaluate_on_split(model, data_yaml_path, split, run_name, output_dir):
    """
    Menjalankan model.val() pada split tertentu (valid/test) dan
    mencetak metrik per kelas - untuk Tabel 5 & 6 di paper.
    """
    print(f"\nEvaluasi pada split '{split}' untuk {run_name}...")
    metrics = model.val(
        data=data_yaml_path,
        split=split,
        save_json=True,
        project=str(output_dir / run_name),
        name=f"{split}_evaluation",
        exist_ok=True,
    )

    print(f"mAP50     : {metrics.box.map50:.3f}")
    print(f"mAP50-95  : {metrics.box.map:.3f}")
    print(f"Precision : {metrics.box.mp:.3f}")
    print(f"Recall    : {metrics.box.mr:.3f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUTPUT_DIR),
                         help="Folder utama untuk semua output training")
    parser.add_argument("--balanced_data", default=str(DEFAULT_BALANCED_YAML),
                         help="Path YAML dataset balanced dari 2_balancing.py")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="yolov8n.pt / yolov8s.pt / yolov8m.pt untuk benchmark ukuran model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    model_tag = os.path.splitext(args.model)[0]  # contoh: "yolov8n"

    data_dir = Path(args.data_dir).resolve()
    unbalanced_yaml = data_dir / "data.yaml"
    balanced_yaml = Path(args.balanced_data).resolve()
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not unbalanced_yaml.is_file():
        raise FileNotFoundError(f"Konfigurasi dataset tidak ditemukan: {unbalanced_yaml}")

    scenarios = [
        ("unbalanced", unbalanced_yaml),
        ("balanced", balanced_yaml),
    ]

    summary = []

    for label, yaml_path in scenarios:
        if not yaml_path.is_file():
            print(f"Lewati skenario '{label}': {yaml_path} tidak ditemukan.")
            continue

        run_name = f"{label}_{model_tag}"
        model, train_results = train_scenario(
            yaml_path, args.model, run_name, output_dir,
            args.epochs, args.imgsz, args.batch, args.patience
        )

        # Evaluasi valid dan test selalu memakai split asli yang sama.
        best_weights = output_dir / run_name / "weights" / "best.pt"
        eval_model = YOLO(best_weights)

        val_metrics = evaluate_on_split(eval_model, yaml_path, "valid", run_name, output_dir)
        test_metrics = evaluate_on_split(eval_model, yaml_path, "test", run_name, output_dir)
        # Kedua YAML memakai valid/test yang sama; balancing hanya mengubah train.

        summary.append({
            "scenario": label,
            "model": args.model,
            "val_mAP50": val_metrics.box.map50,
            "val_mAP50-95": val_metrics.box.map,
            "test_mAP50": test_metrics.box.map50,
            "test_mAP50-95": test_metrics.box.map,
            "test_precision": test_metrics.box.mp,
            "test_recall": test_metrics.box.mr,
        })

    # --- Tabel 7: Perbandingan Eksperimen (balanced vs unbalanced) ---
    import pandas as pd
    summary_df = pd.DataFrame(summary)
    summary_path = output_dir / "experiment_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n" + "=" * 60)
    print("RINGKASAN PERBANDINGAN EKSPERIMEN (Tabel 7)")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\nTersimpan di: {summary_path}")


if __name__ == "__main__":
    main()
