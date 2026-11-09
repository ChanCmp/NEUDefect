import cv2, glob, numpy as np
from pathlib import Path
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

DATASET_DIR = Path(__file__).resolve().parent / "outputs" / "5_split" / "dataset_801010"

def load_crops(split_dir, size=64):
    X, y = [], []
    split_dir = Path(split_dir)
    for img_path in glob.glob(str(split_dir / "images" / "*")):
        img_path = Path(img_path)
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        lbl = split_dir / "labels" / f"{img_path.stem}.txt"
        if not lbl.exists(): continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Gambar tidak dapat dibaca: {img_path}")
        h, w = img.shape
        with lbl.open("r", encoding="utf-8") as label_file:
            for line_number, line in enumerate(label_file, start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) < 5:
                    raise ValueError(f"Format label tidak valid: {lbl}:{line_number}")
                c, xc, yc, bw, bh = fields[:5]
                x1 = int((float(xc) - float(bw)/2) * w); x2 = int((float(xc) + float(bw)/2) * w)
                y1 = int((float(yc) - float(bh)/2) * h); y2 = int((float(yc) + float(bh)/2) * h)
                crop = img[max(y1,0):min(y2,h), max(x1,0):min(x2,w)]
                if crop.size == 0: continue
                crop = cv2.resize(crop, (size, size))
                X.append(hog(crop, pixels_per_cell=(8, 8), cells_per_block=(2, 2)))
                y.append(int(c))
    if not X:
        raise ValueError(f"Tidak ada crop berlabel yang ditemukan di: {split_dir}")
    return np.array(X), np.array(y)

Xtr, ytr = load_crops(DATASET_DIR / "train")
Xva, yva = load_crops(DATASET_DIR / "valid")
Xte, yte = load_crops(DATASET_DIR / "test")

scaler = StandardScaler().fit(Xtr)
Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

best_k = max(range(1, min(16, len(ytr) + 1), 2),
             key=lambda k: KNeighborsClassifier(n_neighbors=k, weights="distance")
                           .fit(Xtr, ytr).score(Xva, yva))
knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance").fit(Xtr, ytr)
pred = knn.predict(Xte)
print(f"k terbaik = {best_k}")
print(classification_report(yte, pred))
print(confusion_matrix(yte, pred))