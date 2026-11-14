"""Reproducible KNN/SVM classification pipeline for Assignment 6."""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = (
    PROJECT_DIR
    / "project_akhir_yolov8"
    / "Defect in NEU.v2i.yolov8"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "assignment6_run"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
CROP_SIZE = 32
PCA_COMPONENTS = 150
K_VALUES = (1, 3, 5, 7, 9)
SVM_CONFIGS = (
    {"kernel": "linear", "C": 1.0},
    {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
    {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
)


def load_dataset(data_dir):
    data_dir = Path(data_dir).resolve()
    yaml_path = data_dir / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML tidak ditemukan: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as file:
        names = yaml.safe_load(file)["names"]
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]

    records = []
    audit_rows = []
    seen_hashes = {}
    counts = Counter()

    for split_name in ("train", "valid", "test"):
        images_dir = data_dir / split_name / "images"
        labels_dir = data_dir / split_name / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Folder images/labels untuk split '{split_name}' tidak lengkap: "
                f"{images_dir}, {labels_dir}"
            )

        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            counts["images_found"] += 1
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                counts["images_missing_label"] += 1
                audit_rows.append({
                    "image": str(image_path),
                    "source_split": split_name,
                    "status": "missing_label",
                    "sha256": "",
                    "n_boxes": 0,
                    "classes": "",
                })
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Gambar tidak dapat dibaca: {image_path}")

            boxes = []
            with label_path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    fields = line.split()
                    if not fields:
                        continue
                    if len(fields) != 5:
                        raise ValueError(
                            f"Format label harus 5 kolom: {label_path}:{line_number}"
                        )
                    try:
                        class_id = int(fields[0])
                        x_center, y_center, width, height = map(float, fields[1:])
                    except ValueError as exc:
                        raise ValueError(
                            f"Label tidak valid: {label_path}:{line_number}"
                        ) from exc
                    if not 0 <= class_id < len(names):
                        raise ValueError(
                            f"Class ID di luar rentang: {label_path}:{line_number}"
                        )
                    if (
                        not all(np.isfinite(v) for v in (x_center, y_center, width, height))
                        or width <= 0
                        or height <= 0
                    ):
                        raise ValueError(
                            f"Bounding box tidak valid: {label_path}:{line_number}"
                        )
                    boxes.append((class_id, x_center, y_center, width, height))

            if not boxes:
                counts["images_empty_label"] += 1
                audit_rows.append({
                    "image": str(image_path),
                    "source_split": split_name,
                    "status": "empty_label",
                    "sha256": "",
                    "n_boxes": 0,
                    "classes": "",
                })
                continue

            image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            class_ids = sorted({box[0] for box in boxes})
            previous = seen_hashes.get(image_hash)
            if previous is not None:
                if previous["class_ids"] != class_ids:
                    raise ValueError(
                        "Gambar duplikat memiliki label kelas yang berbeda: "
                        f"{previous['image_path']} dan {image_path}"
                    )
                counts["duplicate_images"] += 1
                audit_rows.append({
                    "image": str(image_path),
                    "source_split": split_name,
                    "status": "exact_duplicate_skipped",
                    "sha256": image_hash,
                    "n_boxes": len(boxes),
                    "classes": ",".join(names[class_id] for class_id in class_ids),
                })
                continue

            per_class = Counter(box[0] for box in boxes)
            dominant_class = min(
                class_id
                for class_id, class_count in per_class.items()
                if class_count == max(per_class.values())
            )
            record = {
                "image_path": image_path,
                "label_path": label_path,
                "source_split": split_name,
                "image_hash": image_hash,
                "boxes": boxes,
                "class_ids": class_ids,
                "dominant_class": dominant_class,
            }
            records.append(record)
            counts["images_usable"] += 1
            counts["boxes_usable"] += len(boxes)
            audit_rows.append({
                "image": str(image_path),
                "source_split": split_name,
                "status": "usable",
                "sha256": image_hash,
                "n_boxes": len(boxes),
                "classes": ",".join(names[class_id] for class_id in class_ids),
            })

    if not records:
        raise ValueError(f"Tidak ada gambar dengan label valid di {data_dir}")
    return records, names, pd.DataFrame(audit_rows), counts


def split_by_image(records):
    indices = np.arange(len(records))
    strata = np.array([record["dominant_class"] for record in records])
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        stratify=strata,
    )
    train_records = [records[index] for index in train_indices]
    test_records = [records[index] for index in test_indices]
    return train_records, test_records


def extract_features(records):
    features = []
    labels = []
    groups = []
    for record in records:
        image = cv2.imread(str(record["image_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Gambar tidak dapat dibaca: {record['image_path']}")
        image_height, image_width = image.shape
        group_id = record["image_hash"]
        for class_id, x_center, y_center, width, height in record["boxes"]:
            x1 = max(0, int((x_center - width / 2) * image_width))
            y1 = max(0, int((y_center - height / 2) * image_height))
            x2 = min(image_width, int((x_center + width / 2) * image_width))
            y2 = min(image_height, int((y_center + height / 2) * image_height))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                raise ValueError(
                    f"Bounding box menghasilkan crop kosong: {record['label_path']}"
                )
            crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
            feature = crop.astype(np.float32).reshape(-1) / 255.0
            features.append(feature)
            labels.append(class_id)
            groups.append(group_id)
    if not features:
        raise ValueError("Tidak ada bounding box yang dapat diekstrak.")
    return np.asarray(features, dtype=np.float32), np.asarray(labels), np.asarray(groups)


def oversample_indices(y, seed):
    rng = np.random.default_rng(seed)
    classes, class_counts = np.unique(y, return_counts=True)
    target_count = int(class_counts.max())
    selected = []
    for class_id in classes:
        class_indices = np.flatnonzero(y == class_id)
        selected.extend(
            rng.choice(class_indices, size=target_count, replace=True).tolist()
        )
    rng.shuffle(selected)
    return np.asarray(selected, dtype=int)


def score_predictions(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def cross_validate_knn(X, y, groups):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_rows = []
    for fold, (train_idx, validation_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        pca = PCA(
            n_components=PCA_COMPONENTS,
            whiten=True,
            svd_solver="randomized",
            random_state=SEED,
        )
        X_train = pca.fit_transform(X[train_idx])
        X_validation = pca.transform(X[validation_idx])
        sampled_idx = oversample_indices(y[train_idx], SEED + fold)
        for k in K_VALUES:
            model = KNeighborsClassifier(n_neighbors=k, weights="distance")
            model.fit(X_train[sampled_idx], y[train_idx][sampled_idx])
            scores = score_predictions(y[validation_idx], model.predict(X_validation))
            fold_rows.append({
                "model": "KNN",
                "configuration": f"k={k}",
                "fold": fold,
                **scores,
            })
    return pd.DataFrame(fold_rows)


def cross_validate_svm(X, y, groups):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_rows = []
    for fold, (train_idx, validation_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        pca = PCA(
            n_components=PCA_COMPONENTS,
            whiten=True,
            svd_solver="randomized",
            random_state=SEED,
        )
        X_train = pca.fit_transform(X[train_idx])
        X_validation = pca.transform(X[validation_idx])
        for config in SVM_CONFIGS:
            config_name = ",".join(f"{key}={value}" for key, value in config.items())
            model = SVC(class_weight="balanced", **config)
            model.fit(X_train, y[train_idx])
            scores = score_predictions(y[validation_idx], model.predict(X_validation))
            fold_rows.append({
                "model": "SVM",
                "configuration": config_name,
                "fold": fold,
                **scores,
            })
    return pd.DataFrame(fold_rows)


def summarize_cv(fold_results):
    metric_columns = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    summary = (
        fold_results.groupby(["model", "configuration"], as_index=False)[metric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary.sort_values(
        ["model", "f1_macro_mean"],
        ascending=[True, False],
    )


def fit_final_knn(X_train, y_train, k):
    sampled_idx = oversample_indices(y_train, SEED)
    pca = PCA(
        n_components=PCA_COMPONENTS,
        whiten=True,
        svd_solver="randomized",
        random_state=SEED,
    )
    X_reduced = pca.fit_transform(X_train)
    model = KNeighborsClassifier(n_neighbors=k, weights="distance")
    model.fit(X_reduced[sampled_idx], y_train[sampled_idx])
    return pca, model


def fit_final_svm(X_train, y_train, config):
    pca = PCA(
        n_components=PCA_COMPONENTS,
        whiten=True,
        svd_solver="randomized",
        random_state=SEED,
    )
    X_reduced = pca.fit_transform(X_train)
    model = SVC(class_weight="balanced", **config)
    model.fit(X_reduced, y_train)
    return pca, model


def save_confusion_matrix(y_true, y_pred, class_names, model_name, output_dir):
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )
    matrix_df.to_csv(output_dir / f"confusion_matrix_{model_name.lower()}.csv")

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Prediksi",
        ylabel="Label aktual",
        title=f"Confusion Matrix - {model_name}",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_dir / f"confusion_matrix_{model_name.lower()}.png", dpi=160)
    plt.close(figure)


def run(data_dir=DEFAULT_DATA_DIR, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, class_names, audit, counts = load_dataset(data_dir)
    audit.to_csv(output_dir / "dataset_audit.csv", index=False)
    train_records, test_records = split_by_image(records)

    manifest_rows = []
    for split_name, split_records in (
        ("development", train_records),
        ("test", test_records),
    ):
        for record in split_records:
            manifest_rows.append({
                "image": str(record["image_path"]),
                "source_split": record["source_split"],
                "assignment_split": split_name,
                "image_hash": record["image_hash"],
                "dominant_class": class_names[record["dominant_class"]],
                "n_boxes": len(record["boxes"]),
                "classes": ",".join(class_names[class_id] for class_id in record["class_ids"]),
            })
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)

    X_train, y_train, groups_train = extract_features(train_records)
    X_test, y_test, _ = extract_features(test_records)

    class_distribution = pd.DataFrame({
        "class": class_names,
        "development_crops": [
            int(np.sum(y_train == class_id)) for class_id in range(len(class_names))
        ],
        "test_crops": [
            int(np.sum(y_test == class_id)) for class_id in range(len(class_names))
        ],
    })
    class_distribution.to_csv(output_dir / "class_distribution.csv", index=False)

    print(
        f"Dataset: {counts['images_found']} images ditemukan, "
        f"{counts['images_usable']} usable, {counts['boxes_usable']} bounding boxes, "
        f"{counts['duplicate_images']} duplicate dilewati."
    )
    print(
        f"Split gambar: development {len(train_records)} "
        f"({len(train_records) / len(records):.1%}), "
        f"test {len(test_records)} ({len(test_records) / len(records):.1%})."
    )
    print(f"Fitur crop: development={len(y_train)}, test={len(y_test)}.")
    print("\nDistribusi crop sebelum balancing train:")
    print(class_distribution.to_string(index=False))

    cv_folds = pd.concat(
        [
            cross_validate_knn(X_train, y_train, groups_train),
            cross_validate_svm(X_train, y_train, groups_train),
        ],
        ignore_index=True,
    )
    cv_folds.to_csv(output_dir / "cv_fold_results.csv", index=False)
    cv_summary = summarize_cv(cv_folds)
    cv_summary.to_csv(output_dir / "cv_summary.csv", index=False)
    print("\nRingkasan 5-fold CV:")
    print(cv_summary.to_string(index=False))

    knn_summary = cv_summary[cv_summary["model"] == "KNN"]
    best_knn_row = knn_summary.sort_values(
        ["f1_macro_mean", "accuracy_mean"],
        ascending=False,
    ).iloc[0]
    best_k = int(best_knn_row["configuration"].split("=")[1])

    svm_summary = cv_summary[cv_summary["model"] == "SVM"]
    best_svm_row = svm_summary.sort_values(
        ["f1_macro_mean", "accuracy_mean"],
        ascending=False,
    ).iloc[0]
    best_svm_config = next(
        config
        for config in SVM_CONFIGS
        if ",".join(f"{key}={value}" for key, value in config.items())
        == best_svm_row["configuration"]
    )

    final_models = []
    knn_scaler, knn_model = fit_final_knn(X_train, y_train, best_k)
    final_models.append(
        ("KNN", f"k={best_k}", knn_scaler, knn_model)
    )
    svm_scaler, svm_model = fit_final_svm(X_train, y_train, best_svm_config)
    final_models.append(
        ("SVM", best_svm_row["configuration"], svm_scaler, svm_model)
    )

    test_rows = []
    report_rows = []
    for model_name, configuration, pca, model in final_models:
        y_prediction = model.predict(pca.transform(X_test))
        scores = score_predictions(y_test, y_prediction)
        test_rows.append({
            "model": model_name,
            "configuration": configuration,
            **scores,
        })
        report = classification_report(
            y_test,
            y_prediction,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(
            output_dir / f"classification_report_{model_name.lower()}.csv"
        )
        report_rows.append({
            "model": model_name,
            "configuration": configuration,
            **scores,
        })
        save_confusion_matrix(
            y_test,
            y_prediction,
            class_names,
            model_name,
            output_dir,
        )

    test_metrics = pd.DataFrame(test_rows)
    test_metrics.to_csv(output_dir / "test_metrics.csv", index=False)
    print("\nHasil evaluasi test (20% holdout):")
    print(test_metrics.to_string(index=False))

    summary_lines = [
        "Assignment 6 - Ringkasan eksekusi",
        f"Dataset: {Path(data_dir).resolve()}",
        f"Seed split: {SEED}",
        f"Ukuran crop: {CROP_SIZE}x{CROP_SIZE}; fitur: piksel grayscale flatten + PCA {PCA_COMPONENTS} komponen (whiten=True)",
        f"Gambar ditemukan: {counts['images_found']}",
        f"Gambar usable: {counts['images_usable']}",
        f"Gambar tanpa label: {counts['images_missing_label']}",
        f"Gambar dengan label kosong: {counts['images_empty_label']}",
        f"Duplikat exact dilewati: {counts['duplicate_images']}",
        f"Bounding box usable: {counts['boxes_usable']}",
        f"Gambar development/test: {len(train_records)}/{len(test_records)}",
        f"Crop development/test: {len(y_train)}/{len(y_test)}",
        f"KNN terbaik berdasarkan macro-F1 CV: {best_k}",
        f"SVM terbaik berdasarkan macro-F1 CV: {best_svm_row['configuration']}",
        "",
        "Distribusi crop:",
        class_distribution.to_string(index=False),
        "",
        "Metrik test:",
        test_metrics.to_string(index=False),
    ]
    (output_dir / "run_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": output_dir,
        "class_distribution": class_distribution,
        "cv_summary": cv_summary,
        "test_metrics": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
