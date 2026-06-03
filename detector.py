import cv2
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# =============================================================
# PIPELINE PENGOLAHAN CITRA DIGITAL — DETEKSI PLAT NOMOR
#
# Tahapan:
#   1. Grayscale Conversion
#   2. Histogram (Equalization)
#   3. Filter Spasial (Gaussian Blur) — reduksi noise
#   4. Deteksi Tepi (Sobel + Canny)
#   5. Thresholding Otsu — binarisasi global optimal
#   6. Morfologi (Dilasi horizontal + Closing) — penguatan region plat
#   7. Temukan Kontur kandidat
#   8. Ekstraksi Fitur geometri (5 fitur)
#   9. Klasifikasi KNN (k=5, weight=distance)
# =============================================================


# =============================================
# TAHAP 9 — DATA LATIH & MODEL KNN
# Fitur: [area, aspect_ratio, solidity,
#         extent, perimeter_ratio]
# =============================================

# =============================================
# LOAD DATASET ASLI DARI CSV
# =============================================

DATASET_PATH = "data/dataset.csv"

dataset = pd.read_csv(DATASET_PATH)

required_columns = [
    "area",
    "aspect_ratio",
    "solidity",
    "extent",
    "perimeter_ratio",
    "label"
]

for col in required_columns:
    if col not in dataset.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset.csv")

train_features = dataset[
    [
        "area",
        "aspect_ratio",
        "solidity",
        "extent",
        "perimeter_ratio"
    ]
].values.astype(np.float32)

train_labels = dataset["label"].values.astype(np.int32)

print("=" * 50)
print("DATASET BERHASIL DIMUAT")
print("Jumlah Data :", len(dataset))
print("Jumlah Plat :", (train_labels == 1).sum())
print("Jumlah Non Plat :", (train_labels == 0).sum())
print("=" * 50)

# =============================================
# TRAIN KNN
# =============================================

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance"
)

knn.fit(train_features, train_labels)


# =============================================
# TAHAP 8 — EKSTRAKSI FITUR GEOMETRI
# =============================================

def extract_features(contour):
    """
    Mengekstrak 5 fitur geometri dari satu kontur:

    1. Area          — luas piksel region (px²)
    2. Aspect Ratio  — lebar / tinggi bounding box
    3. Solidity      — area / convex hull area  (kepadatan bentuk)
    4. Extent        — area / bounding rect area (seberapa padat di kotak)
    5. Perimeter Ratio — 4πA / P² (isoperimetric quotient, mendekati 1 = lingkaran)
    """
    area          = cv2.contourArea(contour)
    x, y, w, h    = cv2.boundingRect(contour)
    aspect_ratio  = float(w) / h if h > 0 else 0

    hull          = cv2.convexHull(contour)
    hull_area     = cv2.contourArea(hull)
    solidity      = area / hull_area if hull_area > 0 else 0

    rect_area     = w * h
    extent        = area / rect_area if rect_area > 0 else 0

    perimeter     = cv2.arcLength(contour, True)
    perim_ratio   = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    return np.array([[area, aspect_ratio, solidity, extent, perim_ratio]],
                    dtype=np.float32)


# =============================================
# TAHAP 1–6 — PIPELINE PREPROCESSING & SEGMENTASI
# Murni PCD: histogram, filter spasial, tepi, morfologi
# =============================================

def preprocess_and_segment(image):
    """
    Mengembalikan:
      - all_contours : list semua kontur kandidat dari dua jalur
      - steps        : dict citra hasil tiap tahapan (untuk visualisasi)
    """
    steps = {}

    # ── TAHAP 1: Konversi ke Grayscale ──────────────────────────────
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps['1_grayscale'] = gray

    # ── TAHAP 2: Ekualisasi Histogram (pemerataan kontras) ───────────
    # Menggunakan histogram equalization global (materi dasar PCD)
    equalized = cv2.equalizeHist(gray)
    steps['2_histogram_eq'] = equalized

    # ── TAHAP 3: Filter Spasial — Gaussian Blur ──────────────────────
    # Kernel 5×5, σ=0 (dihitung otomatis dari ukuran kernel)
    # Tujuan: meredam noise frekuensi tinggi sebelum deteksi tepi
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    steps['3_gaussian_blur'] = blurred

    # ── TAHAP 4a: Deteksi Tepi — Sobel ──────────────────────────────
    # Sobel X + Y → magnitude gradien
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel   = cv2.magnitude(sobel_x, sobel_y)
    sobel   = np.uint8(np.clip(sobel, 0, 255))
    steps['4a_sobel'] = sobel

    # ── TAHAP 4b: Deteksi Tepi — Canny ──────────────────────────────
    # Canny menggunakan hysteresis thresholding + non-maximum suppression
    canny = cv2.Canny(blurred, 40, 130)
    steps['4b_canny'] = canny

    # ── TAHAP 5: Thresholding Otsu ───────────────────────────────────
    # Metode Otsu mencari threshold global optimal dari histogram
    _, otsu = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    steps['5_otsu'] = otsu

    # ── TAHAP 6: Morfologi ───────────────────────────────────────────
    # Kernel horizontal (17×3): menyambung karakter plat yang berdampingan
    # Kernel persegi (5×5): menutup celah kecil (Closing = Dilasi + Erosi)

    # Jalur A: dari Canny → Dilasi horizontal → Closing
    k_horiz  = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    k_square = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    dilated_a = cv2.dilate(canny, k_horiz, iterations=1)
    closed_a  = cv2.morphologyEx(dilated_a, cv2.MORPH_CLOSE, k_square)
    steps['6a_morph_canny'] = closed_a

    # Jalur B: dari Otsu → Dilasi horizontal → Closing
    dilated_b = cv2.dilate(otsu, k_horiz, iterations=1)
    closed_b  = cv2.morphologyEx(dilated_b, cv2.MORPH_CLOSE, k_square)
    steps['6b_morph_otsu'] = closed_b

    # ── TAHAP 7: Temukan Kontur ──────────────────────────────────────
    contours_a, _ = cv2.findContours(closed_a, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    contours_b, _ = cv2.findContours(closed_b, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

    all_contours = list(contours_a) + list(contours_b)
    return all_contours, steps


# =============================================
# FILTER KANDIDAT BERDASARKAN ATURAN GEOMETRI PLAT
# =============================================

def filter_candidates(contours, img_h, img_w):
    """
    Aturan geometri plat nomor Indonesia:
      - Aspect ratio   : 1.8 – 7.0  (plat panjang horizontal)
      - Area           : 0.5% – 35% dari luas gambar
      - Solidity       : ≥ 0.50      (bentuk cukup padat)
    """
    img_area   = img_h * img_w
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.005 or area > img_area * 0.35:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        ar = float(w) / h if h > 0 else 0
        if ar < 1.8 or ar > 7.0:
            continue

        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 0
        if solidity < 0.50:
            continue

        candidates.append(cnt)

    return candidates


# =============================================
# FUNGSI UTAMA DETEKSI
# =============================================

def detect_plate(image_path, brightness=0):
    """
    Menjalankan seluruh pipeline PCD dan mengembalikan:
      (result_path, label, features_info, step_paths)
    """

    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        return None, "Gagal membaca gambar", {}, {}

    # Resize ke resolusi standar
    image = cv2.resize(image, (800, 600))

    # Penyesuaian kecerahan: f(x) = α·x + β
    image = cv2.convertScaleAbs(image, alpha=1.0, beta=int(brightness))

    img_h, img_w   = image.shape[:2]
    result_img     = image.copy()

    # ── Jalankan pipeline segmentasi ──
    all_contours, steps = preprocess_and_segment(image)

    # ── Filter kandidat ──
    candidates = filter_candidates(all_contours, img_h, img_w)

    label        = "Plat Tidak Ditemukan"
    color        = (0, 0, 255)
    features_info = {}
    plate_region = None

    # ── Pilih kandidat terbaik via KNN ──
    best_contour = None
    best_score   = -1.0

    for cnt in candidates:
        feats = extract_features(cnt)
        try:
            proba = knn.predict_proba(feats)[0]
            score = float(proba[1])          # P(kelas=plat)
        except Exception:
            pred  = knn.predict(feats)[0]
            score = 1.0 if pred == 1 else 0.0

        if score > best_score:
            best_score   = score
            best_contour = cnt

    # ── Keputusan akhir ──
    if best_contour is not None and best_score >= 0.40:
        feats      = extract_features(best_contour)
        prediction = int(knn.predict(feats)[0])

        area         = float(cv2.contourArea(best_contour))
        x, y, w, h   = cv2.boundingRect(best_contour)
        ar           = float(w) / h if h > 0 else 0.0
        hull         = cv2.convexHull(best_contour)
        hull_area    = float(cv2.contourArea(hull))
        solidity     = area / hull_area if hull_area > 0 else 0.0
        rect_area    = w * h
        extent       = area / rect_area if rect_area > 0 else 0.0
        perimeter    = float(cv2.arcLength(best_contour, True))
        perim_ratio  = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

        features_info = {
            'area'           : int(area),
            'aspect_ratio'   : round(ar, 3),
            'solidity'       : round(solidity, 3),
            'extent'         : round(extent, 3),
            'perimeter_ratio': round(perim_ratio, 4),
            'confidence'     : round(best_score * 100, 1),
            'bounding_box'   : {'x': x, 'y': y, 'w': w, 'h': h},
        }

        if prediction == 1:
            label = "Plat Indonesia Terdeteksi"
            color = (0, 220, 80)
        else:
            label = "Bukan Plat Indonesia"
            color = (0, 60, 220)

        # Bounding box + sudut dekoratif
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 3)
        clen, thick = 18, 4
        for cx, cy in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
            dx = -clen if cx == x else clen
            dy = -clen if cy == y else clen
            cv2.line(result_img, (cx, cy), (cx + dx, cy), color, thick)
            cv2.line(result_img, (cx, cy), (cx, cy + dy), color, thick)

        # Crop region plat
        pad  = 10
        px1  = max(0, x - pad);  py1 = max(0, y - pad)
        px2  = min(img_w, x+w+pad); py2 = min(img_h, y+h+pad)
        plate_region = image[py1:py2, px1:px2]

        # Label dengan background
        label_text = f"{label} ({features_info['confidence']}%)"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
        cv2.rectangle(result_img, (x, y - th - 14), (x + tw + 10, y), color, -1)
        cv2.putText(result_img, label_text, (x + 5, y - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

    # ── Simpan citra hasil ──
    result_folder = "static/results"
    os.makedirs(result_folder, exist_ok=True)
    result_path   = os.path.join(result_folder, "result.jpg")
    cv2.imwrite(result_path, result_img)

    # ── Simpan setiap tahapan preprocessing ──
    step_paths = {}
    step_map = {
        '1_grayscale'    : ('step1_grayscale.jpg',    None),
        '2_histogram_eq' : ('step2_histeq.jpg',       None),
        '3_gaussian_blur': ('step3_gaussian.jpg',     None),
        '4a_sobel'       : ('step4a_sobel.jpg',       cv2.COLORMAP_MAGMA),
        '4b_canny'       : ('step4b_canny.jpg',       None),
        '5_otsu'         : ('step5_otsu.jpg',         None),
        '6a_morph_canny' : ('step6a_morph_canny.jpg', None),
        '6b_morph_otsu'  : ('step6b_morph_otsu.jpg',  None),
    }

    for key, (fname, cmap) in step_map.items():
        if key not in steps:
            continue
        img_step = steps[key]
        if cmap is not None:
            img_step = cv2.applyColorMap(img_step, cmap)
        spath = os.path.join(result_folder, fname)
        cv2.imwrite(spath, img_step)
        step_paths[key] = f"results/{fname}"

    # Crop plat (jika ada)
    if plate_region is not None and plate_region.size > 0:
        ppath = os.path.join(result_folder, "plate_crop.jpg")
        cv2.imwrite(ppath, plate_region)
        step_paths['plate_crop'] = "results/plate_crop.jpg"

    return result_path, label, features_info, step_paths
