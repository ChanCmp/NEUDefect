import cv2, glob, numpy as np
from pathlib import Path
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
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
        label_path = split_dir / "labels" / f"{img_path.stem}.txt"
        if not label_path.is_file():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Gambar tidak dapat dibaca: {img_path}")
        h, w = img.shape
        with label_path.open("r", encoding="utf-8") as label_file:
            for line_number, line in enumerate(label_file, start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) < 5:
                    raise ValueError(f"Format label tidak valid: {label_path}:{line_number}")
                c, xc, yc, bw, bh = fields[:5]
                x1 = int((float(xc) - float(bw) / 2) * w)
                x2 = int((float(xc) + float(bw) / 2) * w)
                y1 = int((float(yc) - float(bh) / 2) * h)
                y2 = int((float(yc) + float(bh) / 2) * h)
                crop = img[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
                if crop.size == 0:
                    continue
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

# Gabungkan train+valid, tandai mana yang valid (-1 = train, 0 = valid)
X_tv = np.vstack([Xtr, Xva])
y_tv = np.concatenate([ytr, yva])
split = PredefinedSplit([-1] * len(Xtr) + [0] * len(Xva))

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 1e-3, 1e-2],
    "kernel": ["rbf", "linear"],
    "class_weight": [None, "balanced"],
}

grid = GridSearchCV(SVC(), param_grid, cv=split, scoring="f1_macro", n_jobs=-1)
grid.fit(X_tv, y_tv)
print("Parameter terbaik:", grid.best_params_)

pred = grid.best_estimator_.predict(Xte)
print(classification_report(yte, pred))
print(confusion_matrix(yte, pred))
