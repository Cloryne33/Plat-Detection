from flask import Flask, render_template, request, url_for, jsonify
from detector import detect_plate

import os
import base64
import time
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def numpy_safe(obj):
    """Konversi tipe numpy ke Python native agar bisa di-serialize JSON."""
    if isinstance(obj, dict):
        return {k: numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    result_image = None
    label = None
    features_info = {}
    step_paths = {}
    error = None
    ts = int(time.time())

    brightness = int(request.form.get('brightness', 0))

    try:
        file = request.files.get('image')

        if file and file.filename != '' and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            _, label, features_info, step_paths = detect_plate(filepath, brightness)

        elif request.form.get('camera_image'):
            image_data = request.form['camera_image']
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            filepath = os.path.join(UPLOAD_FOLDER, 'camera_capture.jpg')
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            _, label, features_info, step_paths = detect_plate(filepath, brightness)

        else:
            error = "Tidak ada gambar yang dikirim"

        if label is not None:
            result_image = url_for('static', filename='results/result.jpg') + f'?t={ts}'
            step_urls = {k: url_for('static', filename=v) + f'?t={ts}' for k, v in step_paths.items()}
        else:
            step_urls = {}

    except Exception as e:
        error = f"Error: {str(e)}"
        step_urls = {}

    return jsonify({
        'result_image': result_image,
        'label': label,
        'features': numpy_safe(features_info),
        'steps': step_urls,
        'error': error
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
