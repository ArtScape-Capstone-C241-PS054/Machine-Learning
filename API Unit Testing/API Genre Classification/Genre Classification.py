from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import tempfile
import logging

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)

# Memuat model klasifikasi gambar
image_model = load_model(r'C:\Users\user\Documents\Capstone Analisis Sentimen\genre_classification_84.h5')

# Memuat indeks kelas label untuk klasifikasi gambar
class_indices = {
    0: 'Abstract',
    1: 'Cubism',
    2: 'Dadaism',
    3: 'Fauvism',
    4: 'Impressionism',
    5: 'Nouveau',
    6: 'Pop',
    7: 'Realism',
    8: 'Renaissance',
    9: 'Surrealism'
}

# Dimensi gambar
IMG_HEIGHT = 299
IMG_WIDTH = 299

# Fungsi untuk pra-pemrosesan gambar
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = img_to_array(img)
    x /= 255.0
    x = np.expand_dims(x, axis=0)
    return x

# Halaman utama untuk pengujian API
@app.route('/')
def home():
    return '''
    <h1>Upload an image for genre prediction</h1>
    <form method="POST" action="/predict_recommend" enctype="multipart/form-data">
        <label for="file">Upload an image for genre prediction:</label>
        <input type="file" name="file" accept="image/*"><br><br>
        <input type="submit" value="Upload and Predict">
    </form>
    '''

# Rute untuk menangani prediksi genre
@app.route('/predict_recommend', methods=['POST'])
def predict_genre():
    try:
        file = request.files.get('file')
        if file and file.filename != '':
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                file_path = tmp_file.name
                file.save(file_path)

            try:
                # Pra-pemrosesan gambar
                image_tensor = preprocess_image(file_path)

                # Prediksi probabilitas kelas
                classes = image_model.predict(image_tensor)
                top_3_indices = np.argsort(classes[0])[-3:][::-1]

                # Siapkan hasil prediksi gambar
                image_predictions = []
                for i in top_3_indices:
                    class_name = class_indices[i]
                    probability = classes[0][i]
                    image_predictions.append({'class': class_name, 'probability': float(probability)})

                return jsonify({'predictions': image_predictions})

            finally:
                os.remove(file_path)
        else:
            return jsonify({'error': 'No image uploaded for genre prediction.'}), 400

    except Exception as e:
        logging.error(f"Error in genre prediction: {e}")
        return jsonify({'error': 'Error in genre prediction.'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)