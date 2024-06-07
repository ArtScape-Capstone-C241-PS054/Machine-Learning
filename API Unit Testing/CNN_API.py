from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import requests
import threading
import time
import tempfile

# Load model yg sdh dilatih
model = load_model(r'C:\Users\user\Documents\Capstone Analisis Sentimen\genre_classification_84.h5')

# Load label kelasnya
class_indices = {
    0: 'Abstrak',
    1: 'Cubism',
    2: 'Dadaisme',
    3: 'Fauvisme',
    4: 'Impresionisme',
    5: 'Nouveau',
    6: 'Pop',
    7: 'Realisme',
    8: 'Renaissance',
    9: 'Surealisme'
}

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Dimensi gambar
IMG_HEIGHT = 299
IMG_WIDTH = 299

# Fungsi untuk preprocess gambar
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = img_to_array(img)
    x /= 255.0
    x = np.expand_dims(x, axis=0)
    return x

# Rute untuk halaman utama dengan formulir unggah
@app.route('/')
def home():
    return '''
    <h1>Upload gambar untuk prediksi genre</h1>
    <form method="POST" action="/predict" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*">
        <input type="submit" value="Upload">
    </form>
    '''

# Rute untuk menangani prediksi gambar
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return '<p>Tidak ada file dalam permintaan.</p>', 400
        
        file = request.files['file']
        
        if file.filename == '':
            return '<p>Tidak ada file yang dipilih untuk diupload.</p>', 400
        
        if file:
            # Gunakan file sementara untuk menangani unggahan
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                file_path = tmp_file.name
                file.save(file_path)

            try:
                # Preprocess gambar
                image_tensor = preprocess_image(file_path)

                # Prediksi probabilitas kelas
                classes = model.predict(image_tensor)

                # Dapatkan indeks dari 3 kelas teratas dengan probabilitas tertinggi
                top_3_indices = np.argsort(classes[0])[-3:][::-1]

                # Persiapkan hasil prediksi
                predictions = []
                for i in top_3_indices:
                    class_name = class_indices[i]
                    probability = classes[0][i]
                    predictions.append({'class': class_name, 'probability': float(probability)})

                # Hasilkan HTML dengan hasil prediksi
                result_html = '<h1>Hasil Prediksi</h1>'
                for pred in predictions:
                    result_html += f"<p>Kelas: {pred['class']} - Probabilitas: {pred['probability']:.2f}</p>"

                return result_html
            finally:
                # Pastikan file sementara dihapus
                os.remove(file_path)

        return '<p>Jenis file yang diizinkan adalah jpg, jpeg, png.</p>', 400
    except Exception as e:
        return f'<p>Error: {str(e)}</p>', 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)