# Pipeline YOLOv8 - Steel Surface Defect Detection (NEU-DET)

Urutan menjalankan script, sesuai alur di paper (Bab 2.1 Flowchart):

```
Data Collection (download dari Roboflow)
        |
        v
1_eda.py                -> EDA (Bab 2.3)
        |
        v
2_balancing.py          -> Balancing train set saja (Bab 2.5)
        |
        v
3_train.py              -> Training 2 skenario: unbalanced vs balanced (Bab 2.10)
        |
        v
4_evaluate_and_visualize.py -> Tabel per-kelas + grid visualisasi (Bab 3.3-3.9)
```

## 0. Persiapan

```bash
pip install ultralytics albumentations opencv-python pandas matplotlib pyyaml pillow scikit-learn
```

Dataset sumber YOLOv8 berada di folder `Defect in NEU.v2i.yolov8/`.
Pipeline memakai hasil split 80/10/10 di
`outputs/5_split/dataset_801010/` sebagai dataset utama:

```
outputs/5_split/dataset_801010/
  data.yaml
  train/images/*.jpg   train/labels/*.txt
  valid/images/*.jpg   valid/labels/*.txt
  test/images/*.jpg    test/labels/*.txt
```

Jalankan `split.py` satu kali untuk membuatnya dari dataset sumber.
Semua skrip pipeline otomatis menggunakan `dataset_801010`. Gunakan
`--data_dir` jika ingin memakai folder dataset lain.

### Membuat split dataset

`split.py` membaca dataset sumber `Defect in NEU.v2i.yolov8`, menggabungkan
gambar dari split train/valid/test, lalu
membuat split baru 80/10/10 di folder `outputs/5_split/dataset_801010/`. Jalankan:

```bash
python split.py
```

Script tidak menimpa folder keluaran yang sudah berisi file; gunakan
`--dst` untuk menentukan folder keluaran lain.

## Folder output

Semua hasil setiap tahap disimpan di folder `outputs/` di root proyek:

```
outputs/
  1_eda/                         # tabel, grafik, ringkasan EDA
  2_balancing/                   # dataset balanced, YAML, laporan
  3_train/                       # run YOLO, bobot, metrik, perbandingan
  4_evaluate/                    # hasil evaluasi dan grid prediksi
  5_split/dataset_801010/        # dataset utama hasil split 80/10/10
```

Folder output bisa diubah melalui `--out_dir`; pada `split.py`, gunakan `--dst`.

## 1. EDA

```bash
python 1_eda.py
```

Menghasilkan: `class_distribution.csv/png`, `box_size_distribution.png`,
`boxes_per_image.png`, `summary.txt` — semua untuk Bab 2.3 dan 3.1.

## 2. Balancing (train set saja)

```bash
python 2_balancing.py
```

Menghasilkan folder balanced, file
`outputs/2_balancing/data_balanced.yaml`, dan
`outputs/2_balancing/balancing_report.csv` (Tabel 3).
Data gambar/label hasil balancing berada di `outputs/2_balancing/train_balanced/`.
Jika folder output sudah berisi hasil run sebelumnya, gunakan
`python 2_balancing.py` akan membuat ulang output default secara otomatis.
Untuk folder custom yang sudah berisi hasil, gunakan `--overwrite`.

## 3. Training (2 skenario otomatis: unbalanced & balanced)

```bash
python 3_train.py --model yolov8n.pt --epochs 100
```

Ganti `--model` ke `yolov8s.pt` atau `yolov8m.pt` kalau mau sekalian
benchmark ukuran model (opsional, bisa jadi eksperimen tambahan).

Hasil otomatis tersimpan di:
```
outputs/3_train/unbalanced_yolov8n/   (results.png, confusion_matrix.png,
outputs/3_train/balanced_yolov8n/      PR_curve.png, labels.jpg, args.yaml,
                                      weights/best.pt, evaluasi valid/test)
```
Plus `outputs/3_train/experiment_comparison.csv` (Tabel 7).

## 4. Evaluasi detail & visualisasi prediksi

Jalankan untuk masing-masing skenario (ganti path weights):

```bash
python 4_evaluate_and_visualize.py \
    --weights outputs/3_train/unbalanced_yolov8n/weights/best.pt \
    --split test

python 4_evaluate_and_visualize.py \
    --weights outputs/3_train/balanced_yolov8n/weights/best.pt \
    --split test
```

CSV metrik, grid prediksi, dan output Ultralytics disimpan di
`outputs/4_evaluate/<run>_<split>/`.
Jalankan juga dengan `--split valid` untuk mengisi Tabel 5 (validation).

## Checklist file yang perlu diambil untuk paper

Setelah training selesai, folder `outputs/3_train/<nama_run>/` otomatis berisi:
- `results.png` + `results.csv` -> Gambar 4 (training curves)
- `confusion_matrix.png` / `confusion_matrix_normalized.png` -> Gambar 6
- `F1_curve.png`, `PR_curve.png`, `P_curve.png`, `R_curve.png` -> Gambar 5
- `labels.jpg` -> Gambar 3 (bisa dibandingkan dengan hasil EDA manual)
- `args.yaml` -> Tabel 4 (hyperparameter)
- `weights/best.pt` -> dipakai di Tahap 4

Dari Tahap 1: `outputs/1_eda/class_distribution.csv` (Tabel 2)
Dari Tahap 2: `outputs/2_balancing/balancing_report.csv` (Tabel 3)
Dari Tahap 3: `outputs/3_train/experiment_comparison.csv` (Tabel 7)
Dari Tahap 4: `outputs/4_evaluate/<model>_<split>/metrics_<split>.csv`
              (Tabel 5 & 6), `prediction_grid_<split>.png` (Gambar 7)

## Catatan penting

- Semua hyperparameter di `3_train.py` sengaja ditulis eksplisit
  (bukan default tersembunyi) supaya bisa langsung disalin ke Tabel 4.
- valid/test selalu mengarah ke folder split asli yang sama pada kedua YAML;
  balancing hanya mengubah path train.
- Kalau GPU Colab timeout, `--epochs` bisa diturunkan dulu (misal 50)
  untuk uji pipeline end-to-end, baru dinaikkan untuk run final.
