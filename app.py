from flask import Flask, render_template, request, url_for
from detector import detect_plate

import os
import base64

app = Flask(__name__)

# =========================
# MAX UPLOAD SIZE
# =========================

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# =========================
# UPLOAD FOLDER
# =========================

UPLOAD_FOLDER = 'uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# buat folder uploads otomatis
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# HOME
# =========================

@app.route('/', methods=['GET', 'POST'])
def index():

    result_image = None
    label = None

    if request.method == 'POST':

        brightness = int(
            request.form.get(
                'brightness',
                0
            )
        )

        # =========================
        # UPLOAD IMAGE
        # =========================

        file = request.files.get('image')

        if file and file.filename != '':

            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'],
                file.filename
            )

            file.save(filepath)

            print("UPLOAD:", filepath)

            _, label = detect_plate(
                filepath,
                brightness
            )

        # =========================
        # CAMERA IMAGE
        # =========================

        elif request.form.get('camera_image'):

            image_data = request.form[
                'camera_image'
            ]

            image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(
                image_data
            )

            # FIX: simpan sebagai .jpg (konsisten dengan ekstensi)
            filepath = os.path.join(
                UPLOAD_FOLDER,
                'camera_capture.jpg'
            )

            with open(filepath, 'wb') as f:

                f.write(image_bytes)

            print("CAMERA:", filepath)

            _, label = detect_plate(
                filepath,
                brightness
            )

        # FIX: gunakan url_for agar path gambar benar di template
        if label is not None:
            result_image = url_for(
                'static',
                filename='results/result.jpg'
            )

    return render_template(
        'index.html',
        result_image=result_image,
        label=label
    )

# =========================
# RUN
# =========================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
