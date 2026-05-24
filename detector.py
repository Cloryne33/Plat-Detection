import cv2
import numpy as np
import os

from sklearn.neighbors import KNeighborsClassifier

# =====================================
# TRAINING DATA
# =====================================

train_features = np.array([

    # VALID PLAT INDONESIA
    [25000, 3.5, 0.88],
    [30000, 4.0, 0.90],
    [40000, 3.8, 0.87],
    [50000, 4.5, 0.92],
    [20000, 3.2, 0.85],
    [22000, 3.3, 0.86],

    # TIDAK VALID / NOISE
    [1050, 2.50, 0.35],
    [1100, 5.00, 0.30],
    [1200, 0.30, 0.25],
    [4354, 2.83, 0.05],
    [1080, 6.00, 0.20],
    [1150, 0.15, 0.18],

], dtype=np.float32)

train_labels = np.array([

    1, 1, 1, 1, 1, 1,

    0, 0, 0, 0, 0, 0

])

# =====================================
# TRAIN KNN
# =====================================

knn = KNeighborsClassifier(
    n_neighbors=3
)

knn.fit(
    train_features,
    train_labels
)

# =====================================
# FEATURE EXTRACTION
# =====================================

def extract_features(contour):

    area = cv2.contourArea(contour)

    x, y, w, h = cv2.boundingRect(
        contour
    )

    aspect_ratio = (
        float(w) / h
        if h != 0
        else 0
    )

    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(
        hull
    )

    solidity = (
        area / hull_area
        if hull_area != 0
        else 0
    )

    return np.array(

        [[
            area,
            aspect_ratio,
            solidity
        ]],

        dtype=np.float32

    )

# =====================================
# DETECTION FUNCTION
# =====================================

def detect_plate(image_path, brightness):

    # =========================
    # READ IMAGE
    # =========================

    image = cv2.imread(image_path)

    # =========================
    # VALIDATION
    # =========================

    if image is None:

        return None, "Gagal membaca gambar"

    # =========================
    # RESIZE IMAGE
    # =========================

    image = cv2.resize(
        image,
        (800, 600)
    )

    # =========================
    # APPLY BRIGHTNESS
    # =========================

    image = cv2.convertScaleAbs(

        image,

        alpha=1,

        beta=int(brightness)

    )

    # =========================
    # PREPROCESSING
    # =========================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(

        gray,

        (15, 15),

        0

    )

    edged = cv2.Canny(

        blurred,

        50,

        150

    )

    # =========================
    # MORPHOLOGY
    # =========================

    kernel = cv2.getStructuringElement(

        cv2.MORPH_RECT,

        (5, 5)

    )

    closing = cv2.morphologyEx(

        edged,

        cv2.MORPH_CLOSE,

        kernel

    )

    # =========================
    # FIND CONTOURS
    # =========================

    contours, _ = cv2.findContours(

        closing,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )

    label = "Plat Tidak Ditemukan"

    color = (0, 0, 255)

    # =========================
    # DETECT PLATE
    # =========================

    if contours:

        largest = max(

            contours,

            key=cv2.contourArea

        )

        if cv2.contourArea(largest) > 1000:

            features = extract_features(
                largest
            )

            prediction = knn.predict(
                features
            )[0]

            x, y, w, h = cv2.boundingRect(
                largest
            )

            # =========================
            # PREDICTION
            # =========================

            if prediction == 1:

                label = "Plat Indonesia"

                color = (0, 255, 0)

            else:

                label = "Bukan Plat Indonesia"

                color = (0, 0, 255)

            # =========================
            # DRAW RECTANGLE
            # =========================

            cv2.rectangle(

                image,

                (x, y),

                (x + w, y + h),

                color,

                3

            )

            # =========================
            # DRAW LABEL
            # =========================

            cv2.putText(

                image,

                label,

                (x, y - 10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                color,

                2

            )

    # =========================
    # SAVE RESULT
    # =========================

    result_folder = "static/results"

    os.makedirs(

        result_folder,

        exist_ok=True

    )

    result_path = os.path.join(

        result_folder,

        "result.jpg"

    )

    cv2.imwrite(

        result_path,

        image

    )

    # =========================
    # RETURN
    # =========================

    return result_path, label