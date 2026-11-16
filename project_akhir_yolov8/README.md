# Pipeline YOLOv8 - Steel Surface Defect Detection (NEU-DET)

## Menjalankan pipeline

1. Pasang dependensi:

   ```bash
   pip install ultralytics albumentations opencv-python pandas matplotlib pyyaml pillow scikit-learn
   ```

2. Buka `steel_defect_pipeline.ipynb`, sesuaikan konfigurasi di sel pertama, lalu jalankan semua sel secara berurutan. Notebook berisi kode untuk split dataset, EDA, balancing, training dua skenario, evaluasi, dan visualisasi prediksi.

Dataset sumber YOLOv8 berada di `Defect in NEU.v2i.yolov8/`. Notebook membuat split 80/10/10 di `outputs/5_split/dataset_801010/` jika belum tersedia.

## Konfigurasi

Atur nilai berikut di sel pertama notebook:

- `MODEL`: model YOLO yang digunakan
- `EPOCHS`: jumlah epoch
- `IMGSZ`: ukuran gambar
- `BATCH`: ukuran batch
- `PATIENCE`: early stopping
- `WEIGHTS_PATH`: otomatis mengarah ke bobot model unbalanced yang dipilih

## Folder output

```
outputs/
  1_eda/                         # tabel dan grafik EDA
  2_balancing_notebook/          # data balanced, YAML, laporan
  3_train/                       # bobot, metrik, dan hasil training
  4_evaluate/                    # hasil evaluasi dan grid prediksi
  5_split/dataset_801010/        # dataset split 80/10/10
```

## Hasil untuk laporan

- `outputs/1_eda/class_distribution.csv` — distribusi kelas
- `outputs/2_balancing_notebook/balancing_report.csv` — laporan balancing
- `outputs/3_train/experiment_comparison.csv` — perbandingan training
- `outputs/4_evaluate/<model>_test/metrics_test.csv` — metrik per kelas
- `outputs/4_evaluate/<model>_test/prediction_grid_test.png` — visualisasi prediksi
