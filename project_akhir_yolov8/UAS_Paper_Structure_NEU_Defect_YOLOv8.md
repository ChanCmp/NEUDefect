# UAS Paper Structure: Steel Surface Defect Detection with YOLOv8

**Working title options**

- *Implementasi YOLOv8 untuk Deteksi Cacat Permukaan Baja pada Dataset NEU-DET*
- *Steel Surface Defect Detection Using YOLOv8 on the NEU Surface Defect Dataset*
- *Analisis Pengaruh Penyeimbangan Data terhadap Performa YOLOv8 dalam Deteksi Cacat Permukaan Baja* (use this one if you run the balanced-vs-unbalanced experiment)

**Format reference:** the three previous UAS papers follow a journal-article layout, not a thesis layout:
Title → Abstrak/Abstract → Pendahuluan → Metode Penelitian → Hasil dan Pembahasan → Kesimpulan → Ucapan Terima Kasih → Kontribusi Penulis → Daftar Pustaka.
The "Bab" numbering below follows that order. Target is about 10–16 pages, like the references.

**Dataset facts (verify on the Roboflow page / `data.yaml` before writing):**

- Source: Roboflow Universe, project "defect-in-NEU" (workspace `defect-detection-0r2ml`)
- Split: Train 1259 (70%) / Valid 360 (20%) / Test 180 (10%), total 1799 images
- Likely based on NEU-DET: 6 classes (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches), about 300 images per class, 200×200 grayscale
- Task: **object detection** (bounding boxes), not classification

---

## Bagian Awal: Judul, Penulis, Abstrak

### Judul + Identitas Penulis
- Title, all author names, *Mahasiswa Teknik Komputer, Fakultas Teknik & Informatika, Universitas Multimedia Nusantara*, corresponding author email.

### Abstrak (Bahasa Indonesia) + Abstract (English)
Write this **last**. It's one paragraph of about 150–250 words each, covering:

1. Problem: manual steel surface inspection is slow, subjective and error-prone
2. Method: YOLOv8 on NEU-DET (Roboflow), 6 defect classes, 70/20/10 split
3. Process: EDA → preprocessing → augmentation/balancing → training
4. **Key numbers:** Precision, Recall, mAP50, mAP50-95 on the **test set**
5. Best and worst class, plus a one-line conclusion

**Kata kunci / Keywords:** 4–5 terms, e.g. *Deteksi Objek, YOLOv8, Cacat Permukaan Baja, NEU-DET, Inspeksi Visual Otomatis*

> Highlight: Paskasius's abstract is the best template, since it's the same task type. Copy its structure, not its words.

---

## Bab 1: Pendahuluan

### 1.1 Latar Belakang
- Why steel surface quality matters (manufacturing, automotive, construction) and how defects affect product strength and cost.
- The problem with **manual inspection**: inspector fatigue, inconsistency, and too slow for production lines.
- Why computer vision / deep learning is the solution, moving from classical image processing (LBP, texture features) to CNN-based detectors.
- Tie it back to the course: this is image processing applied to industrial inspection.

### 1.2 Penelitian Terkait (Related Work)
Write 2–4 paragraphs, each summarising one prior study with **method + dataset + result**, as Paskasius and Alvin did.

Suggested sources to search for (verify each one before citing):

- Song & Yan (2013), *Applied Surface Science*: the original NEU surface defect database (classification, LBP-based)
- He, Song, Meng & Yan (2020), *IEEE Transactions on Instrumentation and Measurement*: NEU-DET with bounding boxes (detection version)
- Recent YOLO-based steel defect papers (YOLOv5/v7/v8 on NEU-DET); search Google Scholar for "YOLOv8 NEU-DET"
- Redmon et al. (2016): the original YOLO paper, for explaining the one-stage detection concept

> Highlight: end this subsection with the **gap**. For example, prior work used older YOLO versions, didn't analyse class balance, or didn't compare model sizes. That gap justifies your study.

### 1.3 Rumusan Masalah & Tujuan Penelitian (optional but recommended)
- Research questions such as "How well does YOLOv8 detect the 6 defect types?" and "Which defect class is hardest to detect, and why?"
- Objectives that mirror each question.
- The references fold this into the last paragraph of Pendahuluan. You can do the same instead of making a separate heading.

### 1.4 Kontribusi (optional)
- What's new in your work, e.g. benchmarking YOLOv8n/s/m, or testing whether class balancing improves mAP.

---

## Bab 2: Metode Penelitian

### 2.1 Alur Penelitian (Flowchart)
- **Gambar 1: Flowchart.** Follow the class notes order:
  `Start → Data Collection → EDA → Preprocessing → Data Balancing/Augmentation → Training (YOLOv8) → Evaluation → Result good? (No → tune hyperparameters, loop back) → Finish`
- Paskasius's flowchart, with its "Result is good?" decision loop, is a good model. Draw yours in draw.io or Figma.
- Add a short paragraph that walks through the flowchart.

### 2.2 Dataset
- Source (Roboflow Universe link + citation), number of images, image size, format (YOLO txt labels).
- **Table: class list** with a short description of each defect (what crazing, inclusion, patches etc. look like).
- **Figure: 1 sample image per class** with bounding boxes drawn (a 2×3 grid works well).

### 2.3 Exploratory Data Analysis (EDA)
This step is in the class notes but was **weak or missing in all three reference papers**, so it's a chance to score extra.

- **Table: images and bounding boxes per class, per split** (train / valid / test).
- **Bar chart:** class distribution (box count) in train vs valid vs test.
- **Bounding-box statistics:** box sizes (small/medium/large), boxes per image, aspect ratios. Ultralytics auto-generates `labels.jpg`, which you can use.
- Image properties: resolution and grayscale vs RGB. Optionally add a pixel intensity histogram, as Alvin's paper did.
- Finding: say whether the data is balanced. NEU-DET is usually balanced per image but **not per box**.

### 2.4 Preprocessing
- What was applied and **why**: auto-orient, resize to 640×640 (the YOLOv8 default `imgsz`), any normalisation.
- Note what Roboflow already applied versus what was done in code.
- Optional: a before/after image comparison, like Alvin's normalisation table.

### 2.5 Data Splitting & Penyeimbangan Data (Balancing)
- Explain the 70/20/10 split and the purpose of each set: **train** to learn, **valid** to tune and pick the best epoch / prevent overfitting, **test** for the final unseen evaluation.
- Explain the balancing decision (lecturer's advice):
  - Balancing / augmentation is applied **to the train set only**
  - Valid and test stay **untouched, real images** so the evaluation is fair
  - If a split was skewed, mention a stratified re-split (Fathan's paper merged and re-split)
- **Table: data count before vs after balancing.**

### 2.6 Augmentasi Data
- List each augmentation with its parameter value: `hsv_h`, `hsv_s`, `hsv_v`, `fliplr`, `flipud`, `translate`, `scale`, `mosaic`, `close_mosaic`.
- Explain briefly why each helps. Paskasius's section 4 is the template to follow.
- Note: SMOTE (used by Fathan) is for classification and doesn't apply to detection. Say that augmentation was used instead.

### 2.7 Arsitektur Model YOLOv8
- What YOLO is: one-stage, real-time detection that predicts boxes and classes in a single pass.
- YOLOv8 parts at a high level: **Backbone** (feature extraction), **Neck** (multi-scale feature fusion), **Head** (anchor-free detection).
- Which variant(s) you use (n/s/m) and their parameter counts. The table from the Ultralytics docs can go here.
- Optional figure: a simplified YOLOv8 architecture diagram (redraw it yourself; don't copy).

### 2.8 Konfigurasi Training (Hyperparameter)
- **Table:** epochs, batch size, imgsz, optimizer (SGD/AdamW/auto), lr0, lrf, momentum, weight_decay, warmup_epochs, patience (early stopping), pretrained weights (COCO), workers.
- Environment: Google Colab, GPU type (T4 etc.), Ultralytics version, Python version.
- Give a one-line reason for each important choice.

### 2.9 Metrik Evaluasi
Define each metric with its **formula**. None of the references showed formulas, so including them adds rigour.

- **IoU** = area of overlap / area of union
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **AP** = area under the precision-recall curve for one class
- **mAP50**: mean AP across classes at IoU 0.5
- **mAP50-95**: mean AP averaged over IoU thresholds 0.5 to 0.95 (stricter on box accuracy)
- Optional: F1-score and inference speed (ms/image)

### 2.10 Skenario Eksperimen (if you benchmark)
- What is being compared, e.g. YOLOv8n vs s vs m, or balanced vs unbalanced train set.
- Keep all other settings identical so the comparison is fair.

---

## Bab 3: Hasil dan Pembahasan

General rule from the references: **every table or figure needs a paragraph that interprets it.** Don't just show it.

### 3.1 Hasil EDA
- Present the EDA tables and charts if you didn't put them in the Method section, and interpret them (which class dominates, which has small boxes).

### 3.2 Hasil Training (Kurva Pelatihan)
- **Figure: `results.png`** from Ultralytics (train/val box_loss, cls_loss, dfl_loss, precision, recall, mAP50, mAP50-95).
- Explain each graph, as Paskasius did in "Gambar 3.1" plus the numbered explanations.
- Discuss **overfitting or underfitting**: does val loss track train loss? Is there a gap?

### 3.3 Evaluasi pada Validation Set
- **Table:** per-class P, R, mAP50, mAP50-95 + average (Paskasius's Tabel 1 format).

### 3.4 Evaluasi pada Test Set
- Run `model.val(split='test')` with `best.pt`.
- **Table:** same format as 3.3.
- Compare valid vs test: similar numbers mean the model generalises well.

### 3.5 Analisis Kurva Evaluasi
- F1-Confidence, Precision-Recall, Precision-Confidence and Recall-Confidence curves.
- State the **optimal confidence threshold** from the F1 curve.

### 3.6 Confusion Matrix
- Use the normalised confusion matrix (it's easier to read).
- Which classes get confused with each other? Crazing vs rolled-in scale and inclusion vs pitted surface are common candidates, but check your own results.
- Look at **missed detections** (the background row/column).

### 3.7 Analisis per Kelas
- Best and worst class, and **why**: visual similarity, small or thin defects (scratches), low contrast, fuzzy box boundaries (crazing), or fewer annotations.
- Link this back to the EDA findings. This is where the paper becomes analysis rather than a report.

### 3.8 Perbandingan Eksperimen (if you benchmark)
- **Table:** model variant / setup vs mAP50, mAP50-95, params, inference time.
- Discuss the trade-off between accuracy and speed and model size.

### 3.9 Visualisasi Hasil Prediksi
- **Figure: grid of test images** with predicted boxes and confidence. Include both correct predictions and **failure cases**.
- Fathan's Figure 7 and Alvin's Gambar 7 are good formats.

### 3.10 Perbandingan dengan Penelitian Sebelumnya (optional)
- A small table comparing your mAP against the studies from Related Work, if they used the same dataset.

---

## Bab 4: Kesimpulan (dan Saran)

- 1–2 paragraphs that answer the research questions directly with the **key test numbers**.
- Restate the best and worst class and the main reason.
- Note what contributed most (augmentation, balancing, model size etc.).
- **Saran / future work:** more data, higher resolution for small defects, newer YOLO versions (v9–v11), attention modules, deployment on edge devices for production lines.
- Don't introduce new results here.

---

## Bagian Akhir

### Ucapan Terima Kasih
- University + lecturer name (Alvin's paper is the template).

### Kontribusi Penulis
- Who did what: coding/training, EDA, writing, analysis.

### Daftar Pustaka
- Pick **one** citation style and stick to it. The references used both numbered [1] (IEEE-like) and APA, so check with the lecturer.
- Aim for 8–15 references, mostly journal or conference papers from the last 5 years.
- Must include: the dataset (Roboflow page + original NEU paper), Ultralytics YOLOv8 docs, the YOLO origin paper, and the related-work papers.
- Use Google Scholar → "Cite" or Zotero/Mendeley to avoid formatting errors.

---

## Checklist: What to Get from Your Teammate (Coding Side)

Ultralytics saves most of this automatically in `runs/detect/train*/`:

- [ ] `results.png` (training curves) + `results.csv` (raw numbers)
- [ ] `confusion_matrix.png` + `confusion_matrix_normalized.png`
- [ ] `F1_curve.png`, `PR_curve.png`, `P_curve.png`, `R_curve.png`
- [ ] `labels.jpg` (class distribution + box statistics, useful for EDA)
- [ ] `val_batch*_pred.jpg` (prediction visualisations)
- [ ] `args.yaml` (all hyperparameters used, which fills table 2.8 directly)
- [ ] Per-class metrics printed by `model.val()` for **valid AND test**
- [ ] Per-class image and box counts for each split (EDA script)
- [ ] GPU type, training time, and model parameter count
- [ ] For benchmarking: the same set of outputs for each experiment

## Figure & Table Plan (quick count)

| # | Item | Section |
|---|---|---|
| Gambar 1 | Research flowchart | 2.1 |
| Gambar 2 | Sample image per class | 2.2 |
| Tabel 1 | Class list + description | 2.2 |
| Tabel 2 | Images/boxes per class per split | 2.3 |
| Gambar 3 | Class distribution chart / labels.jpg | 2.3 |
| Tabel 3 | Before vs after balancing | 2.5 |
| Tabel 4 | Hyperparameters | 2.8 |
| Gambar 4 | Training curves | 3.2 |
| Tabel 5 | Validation results per class | 3.3 |
| Tabel 6 | Test results per class | 3.4 |
| Gambar 5 | Evaluation curves | 3.5 |
| Gambar 6 | Confusion matrix | 3.6 |
| Tabel 7 | Experiment comparison (optional) | 3.8 |
| Gambar 7 | Prediction samples | 3.9 |

## Writing Tips

- Number and caption every figure and table, and **refer to them in the text** ("seperti ditunjukkan pada Gambar 4…").
- Put captions **below** figures and **above** tables (Paskasius's format).
- Italicise foreign terms in Indonesian text: *bounding box*, *epoch*, *overfitting*, *learning rate*.
- Keep class names consistent everywhere (pick either "rolled-in scale" or "rolled-in_scale").
- Report numbers to 3 decimals consistently (0.866, not 86.6% in one place and 0.87 in another).
- Write the Methods section while training runs, Results after the numbers come in, and the Abstract and Conclusion last.
