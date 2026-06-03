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
#   2. Histogram Equalization
#   3. Filter Spasial (Gaussian Blur) — reduksi noise
#   4a. Sobel Edge Detection — mendeteksi perubahan intensitas piksel
#   4b. Canny Edge Detection — mendeteksi tepi objek secara lebih akurat
#   5.  Otsu Thresholding    — mengubah citra menjadi biner secara otomatis
#   6.  Temukan Kontur kandidat (dari Sobel + Canny + Otsu)
#   7.  Ekstraksi Fitur geometri (5 fitur)
#   8.  Klasifikasi KNN (k=5, weight=distance)
# =============================================================

DATASET_PATH = "data/dataset.csv"

dataset = pd.read_csv(DATASET_PATH)

required_columns = ["area", "aspect_ratio", "solidity", "extent", "perimeter_ratio", "label"]
for col in required_columns:
    if col not in dataset.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset.csv")

train_features = dataset[["area", "aspect_ratio", "solidity", "extent", "perimeter_ratio"]].values.astype(np.float32)
train_labels   = dataset["label"].values.astype(np.int32)

print("=" * 50)
print("DATASET BERHASIL DIMUAT")
print("Jumlah Data    :", len(dataset))
print("Jumlah Plat    :", (train_labels == 1).sum())
print("Jumlah Non Plat:", (train_labels == 0).sum())
print("=" * 50)

knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn.fit(train_features, train_labels)


def extract_features(contour):
    area         = cv2.contourArea(contour)
    x, y, w, h   = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    hull         = cv2.convexHull(contour)
    hull_area    = cv2.contourArea(hull)
    solidity     = area / hull_area if hull_area > 0 else 0
    rect_area    = w * h
    extent       = area / rect_area if rect_area > 0 else 0
    perimeter    = cv2.arcLength(contour, True)
    perim_ratio  = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    return np.array([[area, aspect_ratio, solidity, extent, perim_ratio]], dtype=np.float32)


def preprocess_and_segment(image):
    steps = {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps['1_grayscale'] = gray

    equalized = cv2.equalizeHist(gray)
    steps['2_histogram_eq'] = equalized

    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    steps['3_gaussian_blur'] = blurred

    # Tahap 4a: Sobel — mendeteksi perubahan intensitas piksel
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel   = cv2.magnitude(sobel_x, sobel_y)
    sobel   = np.uint8(np.clip(sobel, 0, 255))
    steps['4a_sobel'] = sobel
    _, sobel_bin = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)

    # Tahap 4b: Canny — mendeteksi tepi objek secara lebih akurat
    canny = cv2.Canny(blurred, 40, 130)
    steps['4b_canny'] = canny

    # Tahap 5: Otsu — mengubah citra menjadi biner secara otomatis
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    steps['5_otsu'] = otsu

    contours_a, _ = cv2.findContours(sobel_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_b, _ = cv2.findContours(canny,     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_c, _ = cv2.findContours(otsu,      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_contours = list(contours_a) + list(contours_b) + list(contours_c)
    return all_contours, steps


def filter_candidates(contours, img_h, img_w):
    """
    Filter geometri plat Indonesia:
      - aspect_ratio : 1.5 – 7.0
      - area         : 0.3% – 40% luas gambar  ← dinaikkan untuk plat putih border tebal
      - solidity     : >= 0.40
      - extent       : >= 0.25
      - tinggi       : 10px – 40% tinggi gambar ← dilonggarkan
    """
    img_area   = img_h * img_w
    candidates = []
    seen       = set()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.003 or area > img_area * 0.40:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h < 10 or h > img_h * 0.40:
            continue

        ar = float(w) / h if h > 0 else 0
        if ar < 1.5 or ar > 7.0:
            continue

        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 0
        if solidity < 0.40:
            continue

        rect_area = w * h
        extent    = area / rect_area if rect_area > 0 else 0
        if extent < 0.25:
            continue

        key = (x // 15, y // 15, w // 15, h // 15)
        if key in seen:
            continue
        seen.add(key)

        candidates.append(cnt)

    return candidates


def detect_plate(image_path, brightness=0):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Gagal membaca gambar", {}, {}

    image = cv2.resize(image, (800, 600))
    image = cv2.convertScaleAbs(image, alpha=1.0, beta=int(brightness))

    img_h, img_w = image.shape[:2]
    result_img   = image.copy()

    all_contours, steps = preprocess_and_segment(image)
    candidates          = filter_candidates(all_contours, img_h, img_w)

    print(f"Kandidat lolos filter: {len(candidates)}")

    label         = "Plat Tidak Ditemukan"
    color         = (0, 0, 255)
    features_info = {}
    plate_region  = None
    best_contour  = None
    best_score    = -1.0

    for cnt in candidates:
        feats = extract_features(cnt)
        try:
            proba = knn.predict_proba(feats)[0]
            score = float(proba[1])
            print(f"  ar={feats[0][1]:.2f} score={score:.2f} proba={proba}")
        except Exception:
            pred  = knn.predict(feats)[0]
            score = 1.0 if pred == 1 else 0.0

        if score > best_score:
            best_score   = score
            best_contour = cnt

    print(f"Best score: {best_score:.3f}")

    if best_contour is not None and best_score >= 0.30:
        feats      = extract_features(best_contour)
        prediction = int(knn.predict(feats)[0])

        area        = float(cv2.contourArea(best_contour))
        x, y, w, h  = cv2.boundingRect(best_contour)
        ar          = float(w) / h if h > 0 else 0.0
        hull        = cv2.convexHull(best_contour)
        hull_area   = float(cv2.contourArea(hull))
        solidity    = area / hull_area if hull_area > 0 else 0.0
        rect_area   = w * h
        extent      = area / rect_area if rect_area > 0 else 0.0
        perimeter   = float(cv2.arcLength(best_contour, True))
        perim_ratio = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

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

        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 3)
        clen, thick = 18, 4
        for cx, cy in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
            dx = -clen if cx == x else clen
            dy = -clen if cy == y else clen
            cv2.line(result_img, (cx, cy), (cx + dx, cy), color, thick)
            cv2.line(result_img, (cx, cy), (cx, cy + dy), color, thick)

        pad  = 10
        px1  = max(0, x - pad);     py1 = max(0, y - pad)
        px2  = min(img_w, x+w+pad); py2 = min(img_h, y+h+pad)
        plate_region = image[py1:py2, px1:px2]

        label_text = f"{label} ({features_info['confidence']}%)"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
        cv2.rectangle(result_img, (x, y - th - 14), (x + tw + 10, y), color, -1)
        cv2.putText(result_img, label_text, (x + 5, y - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

    result_folder = "static/results"
    os.makedirs(result_folder, exist_ok=True)
    result_path   = os.path.join(result_folder, "result.jpg")
    cv2.imwrite(result_path, result_img)

    step_paths = {}
    step_map = {
        '1_grayscale'    : ('step1_grayscale.jpg', None),
        '2_histogram_eq' : ('step2_histeq.jpg',    None),
        '3_gaussian_blur': ('step3_gaussian.jpg',  None),
        '4a_sobel'       : ('step4a_sobel.jpg',    cv2.COLORMAP_MAGMA),
        '4b_canny'       : ('step4b_canny.jpg',    None),
        '5_otsu'         : ('step5_otsu.jpg',      None),
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

    if plate_region is not None and plate_region.size > 0:
        ppath = os.path.join(result_folder, "plate_crop.jpg")
        cv2.imwrite(ppath, plate_region)
        step_paths['plate_crop'] = "results/plate_crop.jpg"

    return result_path, label, features_info, step_paths