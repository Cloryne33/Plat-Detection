import os
import cv2
import numpy as np
import pandas as pd

DATASET_FOLDER = "dataset/plat nomor"

data = []

for filename in os.listdir(DATASET_FOLDER):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(DATASET_FOLDER, filename)

    image = cv2.imread(path)

    if image is None:
        continue

    image = cv2.resize(image, (800, 600))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    equalized = cv2.equalizeHist(gray)

    blurred = cv2.GaussianBlur(equalized, (5,5), 0)

    canny = cv2.Canny(blurred, 40, 130)

    contours, _ = cv2.findContours(
        canny,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        continue

    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)

    x,y,w,h = cv2.boundingRect(largest)

    aspect_ratio = w / h if h > 0 else 0

    hull = cv2.convexHull(largest)

    hull_area = cv2.contourArea(hull)

    solidity = area / hull_area if hull_area > 0 else 0

    rect_area = w * h

    extent = area / rect_area if rect_area > 0 else 0

    perimeter = cv2.arcLength(largest, True)

    perimeter_ratio = (
        (4*np.pi*area)/(perimeter**2)
        if perimeter > 0 else 0
    )

    data.append([
        area,
        aspect_ratio,
        solidity,
        extent,
        perimeter_ratio,
        1
    ])

df = pd.DataFrame(
    data,
    columns=[
        "area",
        "aspect_ratio",
        "solidity",
        "extent",
        "perimeter_ratio",
        "label"
    ]
)

os.makedirs("data", exist_ok=True)

df.to_csv("data/dataset.csv", index=False)

print("Dataset berhasil dibuat")
print("Jumlah data:", len(df))