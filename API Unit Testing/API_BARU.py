from flask import Flask, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import tempfile
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
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

# Memuat model analisis sentimen dan objek
try:
    sentiment_model = load_model(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\sentiment_analysis_model.h5')
    tf_idf = joblib.load(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\tf_idf_vectorizer.pkl')
    chi2_features = joblib.load(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\chi2_features.pkl')
    key_norm = pd.read_csv(r'C:\Users\user\Documents\Capstone Analisis Sentimen\Dataset\key_norm.csv')
    logging.info("Sentiment analysis model and objects loaded successfully.")
except Exception as e:
    logging.error(f"Error loading sentiment analysis model or objects: {e}")

# Definisikan fungsi pra-pemrosesan teks untuk analisis sentimen
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) 
    text = re.sub(r'#\w+', '', text) 
    text = re.sub(r'https?://\S+', '', text)  
    text = re.sub(r'<[^>]+>', '', text)  
    text = re.sub(r'[^A-Za-z\s]', '', text)  
    return text

def normalize_text(text):
    words = text.split()
    normalized_words = []
    for word in words:
        match = key_norm[key_norm['singkat'] == word]
        normalized_words.append(match['hasil'].values[0] if not match.empty else word)
    return ' '.join(normalized_words).lower()

stopwords_ind = stopwords.words('indonesian')
more_stopwords = ['tsel', 'gb', 'rb']  
stopwords_ind.extend(more_stopwords)

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stopwords_ind])

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stem_text(text):
    return stemmer.stem(text)

def preprocess_text_pipeline(text):
    text = preprocess_text(text)
    text = normalize_text(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text

# Definisikan fungsi prediksi untuk analisis sentimen
def predict_sentiment(input_text):
    try:
        preprocessed_text = preprocess_text_pipeline(input_text)
        tf_idf_vec = tf_idf.transform([preprocessed_text]).toarray()
        kbest_vec = chi2_features.transform(tf_idf_vec)
        prediction = (sentiment_model.predict(kbest_vec) > 0.5).astype("int32")
        return 'positive' if prediction == 1 else 'negative'
    except Exception as e:
        logging.error(f"Error in sentiment analysis preprocessing or prediction: {e}")
        return 'error'

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

# Rute untuk halaman utama dengan formulir unggah
@app.route('/')
def home():
    return '''
    <h1>Upload an image for genre prediction and enter a text for sentiment analysis</h1>
    <form method="POST" action="/predict" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*">
        <input type="text" name="text" placeholder="Enter text for sentiment analysis">
        <input type="submit" value="Upload and Predict">
    </form>
    '''

# Rute untuk menangani prediksi untuk gambar dan teks
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return '<p>No file in request.</p>', 400
        
        file = request.files['file']
        text = request.form['text']

        # Prediksi gambar
        if file.filename != '':
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

                image_result_html = '<h1>Image Prediction Result</h1>'
                for pred in image_predictions:
                    image_result_html += f"<p>Class: {pred['class']} - Probability: {pred['probability']:.2f}</p>"
            finally:
                os.remove(file_path)
        else:
            image_result_html = '<p>No image uploaded.</p>'

        # Prediksi analisis sentimen
        if text.strip() != '':
            sentiment_prediction = predict_sentiment(text)
            text_result_html = f'<h1>Sentiment Analysis Result</h1><p>Sentiment: {sentiment_prediction}</p>'
        else:
            text_result_html = '<p>No text for sentiment analysis</p>'

         # Gabungkan hasil gambar dan teks
        return image_result_html + text_result_html

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return '<p>Error in prediction.</p>', 500
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)