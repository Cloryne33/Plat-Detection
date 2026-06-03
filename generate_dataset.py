"""
Generate dataset.csv dari foto plat.

Struktur folder:
  dataset/plat nomor/   ← foto plat Indonesia  (label=1)
  dataset/non plat/     ← foto non-plat        (label=0)

Jalankan: py generate_dataset.py
"""

import os
import cv2
import numpy as np
import pandas as pd

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def extract_features(cnt):
    area        = cv2.contourArea(cnt)
    x, y, w, h  = cv2.boundingRect(cnt)
    ar          = float(w) / h if h > 0 else 0
    hull        = cv2.convexHull(cnt)
    hull_area   = cv2.contourArea(hull)
    solidity    = area / hull_area if hull_area > 0 else 0
    extent      = area / (w * h) if (w * h) > 0 else 0
    perimeter   = cv2.arcLength(cnt, True)
    perim_ratio = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    return area, ar, solidity, extent, perim_ratio


def get_plate_contour(image_path):
    """
    Ambil kontur yang paling mirip plat:
    - aspect ratio 1.5–7.0 (horizontal)
    - solidity >= 0.4
    - extent >= 0.3
    - area 0.5%–25% luas gambar
    Dari kandidat yang lolos, pilih yang skor kemiripannya tertinggi.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    img      = cv2.resize(img, (800, 600))
    img_area = 800 * 600
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq       = cv2.equalizeHist(gray)
    blur     = cv2.GaussianBlur(eq, (5, 5), 0)

    # Tiga sumber kontur
    sx, sy   = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel    = np.uint8(np.clip(cv2.magnitude(sx, sy), 0, 255))
    _, sbin  = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)
    canny    = cv2.Canny(blur, 40, 130)
    _, otsu  = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    all_cnts = []
    for src in [sbin, canny, otsu]:
        c, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_cnts += list(c)

    best, best_score = None, -1

    for cnt in all_cnts:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.005 or area > img_area * 0.25:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h < 10:
            continue

        ar = float(w) / h if h > 0 else 0
        if ar < 1.5 or ar > 7.0:          # harus horizontal
            continue

        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 0
        if solidity < 0.40:
            continue

        extent = area / (w * h) if (w * h) > 0 else 0
        if extent < 0.30:
            continue

        # Skor: makin dekat ar=3.0 + solidity tinggi + extent tinggi
        score = (1 - abs(ar - 3.0) / 5.0) * 0.4 + solidity * 0.4 + extent * 0.2
        if score > best_score:
            best_score, best = score, cnt

    return best


def process_folder(folder, label):
    rows  = []
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)]
    print(f"\nFolder: '{folder}' | {len(files)} foto | label={label}")

    for fname in files:
        fpath = os.path.join(folder, fname)
        cnt   = get_plate_contour(fpath)

        if cnt is None:
            print(f"  [SKIP] {fname} — tidak ada kontur plat")
            continue

        area, ar, sol, ext, pr = extract_features(cnt)
        rows.append([round(area,1), round(ar,6), round(sol,6), round(ext,6), round(pr,6), label])
        print(f"  [OK]   {fname} | ar={ar:.2f} sol={sol:.2f} ext={ext:.2f} area={int(area)}")

    return rows


# ── Main ──
rows  = process_folder("dataset/plat nomor", label=1)
rows += process_folder("dataset/non plat",   label=0)

df = pd.DataFrame(rows, columns=["area","aspect_ratio","solidity","extent","perimeter_ratio","label"])

os.makedirs("data", exist_ok=True)
df.to_csv("data/dataset.csv", index=False)

print("\n" + "=" * 50)
print("DATASET SELESAI DIBUAT")
print(f"Plat (1)      : {(df.label==1).sum()}")
print(f"Non-plat (0)  : {(df.label==0).sum()}")
print(f"Total         : {len(df)}")
print("Disimpan ke   : data/dataset.csv")
print("=" * 50)