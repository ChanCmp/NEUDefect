"""Generate the Assignment 6 PDF report from pipeline output files."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "assignment6_run"
DEFAULT_PDF_PATH = Path(__file__).resolve().parent / "Laporan_Assignment6.pdf"


def add_text_page(pdf, title, paragraphs):
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(0.08, 0.94, title, fontsize=18, weight="bold", va="top")
    y = 0.88
    for paragraph in paragraphs:
        figure.text(0.08, y, paragraph, fontsize=11, va="top", wrap=True)
        y -= 0.075 if len(paragraph) < 120 else 0.105
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def add_table_page(pdf, title, dataframe, columns, headers, font_size=8):
    figure, axis = plt.subplots(figsize=(11.69, 8.27))
    axis.axis("off")
    axis.set_title(title, fontsize=16, weight="bold", pad=20)
    table = axis.table(
        cellText=dataframe[columns].values,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.65)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def add_image_page(pdf, title, image_path):
    figure, axis = plt.subplots(figsize=(8.27, 11.69))
    axis.axis("off")
    axis.set_title(title, fontsize=16, weight="bold", pad=20)
    image = plt.imread(image_path)
    axis.imshow(image)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def generate_report(output_dir=DEFAULT_OUTPUT_DIR, pdf_path=DEFAULT_PDF_PATH):
    output_dir = Path(output_dir).resolve()
    pdf_path = Path(pdf_path).resolve()
    required = [
        "run_summary.txt",
        "class_distribution.csv",
        "cv_summary.csv",
        "test_metrics.csv",
        "confusion_matrix_knn.png",
        "confusion_matrix_svm.png",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Output pipeline belum lengkap; file tidak ditemukan: "
            + ", ".join(missing)
        )

    summary_text = (output_dir / "run_summary.txt").read_text(encoding="utf-8")
    summary = {}
    for line in summary_text.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            summary[key] = value
    distribution = pd.read_csv(output_dir / "class_distribution.csv")
    cv_summary = pd.read_csv(output_dir / "cv_summary.csv")
    test_metrics = pd.read_csv(output_dir / "test_metrics.csv")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        add_text_page(
            pdf,
            "Laporan Assignment 6 - Klasifikasi Defect NEU",
            [
                "Tujuan: membandingkan K-Nearest Neighbors (KNN) sebagai model utama "
                "dengan Support Vector Machine (SVM) sebagai model pembanding.",
                "Dataset berisi gambar defect berlabel bounding box. Setiap bounding "
                "box dipotong menjadi satu sampel klasifikasi; crop grayscale di-resize "
                "menjadi 32 x 32 lalu diratakan menjadi vektor piksel, mengikuti alur "
                "representasi gambar pada Modul 7.",
                "PCA dengan 150 komponen dan whiten=True diterapkan seperti contoh "
                "preprocessing eigenfaces pada Modul 8. PCA hanya di-fit pada bagian "
                "training setiap fold atau development set sebelum test.",
                f"Gambar ditemukan: {summary.get('Gambar ditemukan', 'N/A')}; "
                f"gambar usable: {summary.get('Gambar usable', 'N/A')}; "
                f"bounding box usable: {summary.get('Bounding box usable', 'N/A')}.",
                f"Split berdasarkan gambar: development "
                f"{summary.get('Gambar development/test', 'N/A')} (80/20), "
                f"seed {summary.get('Seed split', '42')}. Stratified split memakai "
                "kelas dominan tiap gambar. Gambar yang sama tidak dibagi ke kedua "
                "split; 5-fold CV juga dikelompokkan berdasarkan hash gambar.",
                "Preprocessing dan scaling dipelajari dari data training setiap fold. "
                "KNN dilatih dengan oversampling kelas hanya pada bagian training "
                "tiap fold; SVM memakai class_weight='balanced'. Test set tidak "
                "digunakan untuk memilih parameter.",
                f"Jumlah crop: development {summary.get('Crop development/test', 'N/A')}; "
                "test pada baris yang sama setelah tanda '/'.",
            ],
        )

        distribution_table = distribution.copy()
        distribution_table.columns = [
            "Kelas",
            "Crop development",
            "Crop test",
        ]
        add_table_page(
            pdf,
            "Distribusi kelas per split",
            distribution_table,
            list(distribution_table.columns),
            list(distribution_table.columns),
            font_size=10,
        )

        cv_display = cv_summary.copy()
        cv_display["Akurasi (mean +/- std)"] = cv_display.apply(
            lambda row: f"{row['accuracy_mean']:.3f} +/- {row['accuracy_std']:.3f}",
            axis=1,
        )
        cv_display["Macro-F1 (mean +/- std)"] = cv_display.apply(
            lambda row: f"{row['f1_macro_mean']:.3f} +/- {row['f1_macro_std']:.3f}",
            axis=1,
        )
        add_table_page(
            pdf,
            "Hasil Stratified Group 5-fold Cross-Validation",
            cv_display,
            ["model", "configuration", "Akurasi (mean +/- std)", "Macro-F1 (mean +/- std)"],
            ["Model", "Konfigurasi", "Akurasi (mean +/- std)", "Macro-F1 (mean +/- std)"],
            font_size=8,
        )

        test_display = test_metrics.copy()
        for column in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            test_display[column] = test_display[column].map(lambda value: f"{value:.3f}")
        test_display.columns = [
            "Model",
            "Konfigurasi",
            "Accuracy",
            "Precision macro",
            "Recall macro",
            "F1-score macro",
        ]
        add_table_page(
            pdf,
            "Evaluasi akhir pada test set 20%",
            test_display,
            list(test_display.columns),
            list(test_display.columns),
            font_size=9,
        )
        add_image_page(
            pdf,
            "Confusion Matrix - KNN",
            output_dir / "confusion_matrix_knn.png",
        )
        add_image_page(
            pdf,
            "Confusion Matrix - SVM",
            output_dir / "confusion_matrix_svm.png",
        )

        knn_f1 = float(
            test_metrics.loc[test_metrics["model"] == "KNN", "f1_macro"].iloc[0]
        )
        svm_f1 = float(
            test_metrics.loc[test_metrics["model"] == "SVM", "f1_macro"].iloc[0]
        )
        best_k = summary.get("KNN terbaik berdasarkan macro-F1 CV", "N/A")
        best_svm = summary.get("SVM terbaik berdasarkan macro-F1 CV", "N/A")
        add_text_page(
            pdf,
            "Kesimpulan dan catatan",
            [
                f"Berdasarkan development set, konfigurasi KNN terpilih adalah K={best_k}; "
                f"konfigurasi SVM terpilih adalah {best_svm}.",
                f"Pada test set, macro-F1 KNN = {knn_f1:.3f} dan macro-F1 SVM = {svm_f1:.3f}. "
                "Hasil ini berasal dari satu holdout test set dan tidak boleh digunakan "
                "untuk mengubah konfigurasi model setelah evaluasi.",
                "Distribusi crop antar kelas tidak identik. Penanganan imbalance "
                "diterapkan pada data training saja; distribusi test dipertahankan "
                "apa adanya agar evaluasi mencerminkan data holdout.",
                "Keterbatasan: hasil bergantung pada representasi piksel-PCA dan crop "
                "bounding box yang tersedia. Analisis lanjutan dapat membandingkan fitur atau "
                "parameter lain, tetapi pemilihan tambahan harus tetap dibatasi pada "
                "development set.",
            ],
        )
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pdf_path", type=Path, default=DEFAULT_PDF_PATH)
    args = parser.parse_args()
    path = generate_report(args.output_dir, args.pdf_path)
    print(f"Laporan tersimpan: {path}")


if __name__ == "__main__":
    main()
