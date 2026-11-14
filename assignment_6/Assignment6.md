# Planning Assignment 6

## Ruang lingkup

Assignment 6 dikerjakan terpisah dari project akhir YOLOv8. Fokus tugas ini adalah klasifikasi supervised learning menggunakan K-Nearest Neighbors (KNN) sebagai model utama dan minimal satu model pembanding (SVM).

**Ketentuan utama dari assignment:**

- Dataset harus berlabel.
- Pembagian data: 80% training/validation dan 20% testing.
- Periksa missing values, duplikasi, dan kebutuhan normalisasi/standardisasi.
- Gunakan stratified split jika memungkinkan.
- KNN harus diuji dengan `K = 1, 3, 5, 7, 9`, dan hasil setiap K dicatat.
- Evaluasi menggunakan 5-fold cross-validation.
- Laporkan accuracy, precision, recall, F1-score, dan confusion matrix.
- Kelas harus seimbang atau ketidakseimbangannya perlu ditangani/dijelaskan.
- Pengumpulan: notebook `.ipynb` dan laporan PDF berisi screenshot evaluasi serta pipeline pengerjaan.

## Status pengerjaan saat ini

| Komponen | Kondisi saat ini |
|---|---|
| Dataset | 1.799 gambar usable dan 4.186 bounding box; tidak ditemukan label hilang, label kosong, atau duplikat gambar identik |
| Split | 1.439 gambar development (80%) dan 360 test (20%), stratified berdasarkan kelas dominan gambar |
| Representasi fitur | Crop grayscale 32 x 32 diratakan menjadi vektor piksel (konsep Modul 7), kemudian PCA 150 komponen dengan `whiten=True` (konsep Modul 8) |
| KNN | 5-fold grouped CV untuk K = 1, 3, 5, 7, 9; K terbaik berdasarkan macro-F1 CV adalah 1 |
| SVM | 5-fold grouped CV untuk 3 konfigurasi; konfigurasi terbaik adalah RBF, C=10, gamma=scale |
| Evaluasi test | Accuracy, macro precision/recall/F1, laporan per kelas, dan confusion matrix untuk kedua model |
| Notebook | `Assignment6.ipynb` dibuat dan berhasil dijalankan end-to-end |
| Laporan | `Laporan_Assignment6.pdf` dibuat dari hasil aktual, termasuk tabel dan confusion matrix |

Hasil test dengan fitur piksel-PCA: KNN macro-F1 **0,329** (accuracy **0,402**); SVM macro-F1 **0,768** (accuracy **0,779**). Hasil CV per fold dan evaluasi test tersimpan pada folder `outputs/assignment6_run/`.

Progress implementasi: **sekitar 90%**. Sisa pekerjaan adalah review manual laporan/notebook dan memastikan lingkungan pengumpulan (misalnya Colab) menerima file helper Python bersama notebook.

`KNN.py` dan `SVM.py` adalah prototipe lama dan tidak dipanggil oleh notebook. Jalankan eksperimen Assignment 6 melalui `Assignment6.ipynb`, yang memakai `assignment6_pipeline.py`.

## Rencana pengerjaan

### 1. Tetapkan dan audit dataset

- [x] Tentukan dataset berlabel yang digunakan untuk Assignment 6.
- [x] Catat jumlah gambar dan jumlah sampel/objek per kelas.
- [x] Periksa gambar atau label yang hilang, rusak, kosong, atau formatnya tidak valid.
- [x] Periksa duplikasi gambar dan label.
- [x] Tinjau distribusi kelas dan putuskan penanganan imbalance jika diperlukan.
- [x] Simpan hasil Assignment 6 di folder assignment; data sumber tetap dipakai tanpa mengubah output project YOLOv8.

**Hasil:** ringkasan dataset, kualitas data, dan distribusi kelas.

### 2. Buat pembagian data yang bebas kebocoran

- [x] Bagi data berdasarkan gambar menjadi 80% development (training/validation) dan 20% test.
- [x] Gunakan stratified split berdasarkan kelas dominan gambar.
- [x] Pastikan semua crop dari satu gambar hanya masuk ke satu bagian split.
- [x] Simpan daftar file/hash setiap split agar pembagian dapat direproduksi.
- [x] Jangan gunakan test set untuk memilih K atau parameter model.

**Hasil:** split 80/20 khusus Assignment 6 yang dapat direproduksi.

### 3. Siapkan preprocessing dan fitur

- [x] Validasi bounding box dan label sebelum membuat crop.
- [x] Ekstrak crop gambar berdasarkan bounding box untuk klasifikasi.
- [x] Terapkan grayscale, resize 32 x 32, dan flatten piksel menjadi vektor seperti konsep klasifikasi gambar Modul 7.
- [x] Fit PCA 150 komponen dengan `whiten=True` hanya pada training fold/development set, mengikuti konsep preprocessing PCA Modul 8 dan mencegah data leakage.

**Hasil:** fitur dan label yang siap digunakan model, dengan preprocessing yang konsisten.

### 4. Evaluasi KNN dengan 5-fold CV

- [x] Gunakan 5-fold cross-validation berkelompok berdasarkan gambar pada development set.
- [x] Uji `K = 1, 3, 5, 7, 9`.
- [x] Catat skor tiap fold serta rata-rata dan standar deviasi untuk tiap K.
- [x] Pilih K terbaik berdasarkan macro-F1 CV sebelum evaluasi test.

**Hasil:** tabel perbandingan hasil CV untuk seluruh nilai K yang diwajibkan.

### 5. Evaluasi model pembanding (SVM)

- [x] Gunakan 5-fold cross-validation berkelompok berdasarkan gambar pada development set.
- [x] Lakukan pemilihan parameter hanya di development set.
- [x] Catat konfigurasi dan hasil CV SVM dengan metrik yang konsisten.

**Hasil:** konfigurasi SVM terpilih dan hasil CV untuk dibandingkan dengan KNN.

### 6. Evaluasi akhir pada test set

- [x] Latih ulang KNN terbaik dan SVM terbaik menggunakan seluruh development set.
- [x] Evaluasi masing-masing model pada test set 20% yang sama.
- [x] Laporkan accuracy, precision, recall, dan F1-score.
- [x] Tampilkan confusion matrix dan metrik per kelas.
- [x] Sertakan macro average agar performa pada setiap kelas tercermin.

**Hasil:** tabel evaluasi test final KNN vs SVM dan confusion matrix.

### 7. Buat notebook dan laporan

- [x] Buat notebook khusus Assignment 6 dengan urutan pengerjaan yang jelas.
- [x] Jalankan notebook dari awal hingga akhir.
- [x] Simpan tabel dan visualisasi hasil yang relevan.
- [x] Buat laporan PDF yang menjelaskan pipeline dan hasil.
- [x] Sertakan confusion matrix aktual pada laporan PDF.
- [ ] Review akhir format laporan dan siapkan metadata kelompok/anggota bila diminta dosen.

**Hasil akhir:** notebook `.ipynb` dan laporan PDF.

## Urutan prioritas

1. Tetapkan dataset Assignment 6 dan audit kualitas/distribusi kelas.
2. Buat split khusus 80/20 berdasarkan gambar.
3. Rapikan preprocessing agar tidak terjadi data leakage.
4. Jalankan KNN dengan 5-fold CV untuk K = 1, 3, 5, 7, 9.
5. Jalankan SVM dengan 5-fold CV.
6. Evaluasi model terpilih pada test set.
7. Susun notebook dan laporan PDF.

## Progress saat ini

| Bagian | Status |
|---|---:|
| Dataset dan preprocessing | 100% |
| KNN | 100% |
| SVM | 100% |
| Notebook final | 90% |
| Laporan PDF | 90% |
| **Keseluruhan** | **sekitar 90%** |

Progress belum 100% sampai notebook/laporan direview manual dan berkas siap dikumpulkan dikonfirmasi.
