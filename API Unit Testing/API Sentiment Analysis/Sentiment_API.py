from flask import Flask, request, jsonify
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
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
import logging
from sklearn.preprocessing import LabelEncoder

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)

# Memuat model analisis sentimen dan objek
try:
    sentiment_model = load_model(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\sentiment_analysis_model.h5')
    tf_idf = joblib.load(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\tf_idf_vectorizer.pkl')
    chi2_features = joblib.load(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\chi2_features.pkl')
    key_norm = pd.read_csv(r'C:\Users\user\Documents\Capstone Analisis Sentimen\Dataset\key_norm.csv')
    logging.info("Sentiment analysis model and objects loaded successfully.")
except Exception as e:
    logging.error(f"Error loading sentiment analysis model or objects: {e}")

# Fungsi pra-pemrosesan teks untuk analisis sentimen
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

# Fungsi prediksi untuk analisis sentimen
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

# Rute untuk halaman utama dengan formulir unggah
@app.route('/')
def home():
    return '''
    <h1>Sentiment Analysis</h1>
    <form method="POST" action="/predict_recommend" enctype="multipart/form-data">
        <h2>Input Text Please</h2>
        <label for="text">Enter text for sentiment analysis:</label>
        <input type="text" name="text" placeholder="Enter text for sentiment analysis"><br><br>
        <input type="submit" value="Predict">
    </form>
    '''

# Rute untuk menangani prediksi genre, analisis sentimen, dan rekomendasi seni
@app.route('/predict_recommend', methods=['POST'])
def predict_recommend():
    try:
        file = request.files.get('file')
        text = request.form.get('text', '').strip()

        image_result_html = ''
        text_result_html = ''
        recommendations_html = ''

        # Prediksi analisis sentimen
        if text:
            sentiment_prediction = predict_sentiment(text)
            text_result_html = f'<h1>Sentiment Analysis Result</h1><p>Sentiment: {sentiment_prediction}</p>'
        else:
            text_result_html = '<p>No text for sentiment analysis</p>'

        # Gabungkan hasil gambar, teks, dan rekomendasi
        return image_result_html + text_result_html + recommendations_html

    except Exception as e:
        logging.error(f"Error in prediction or recommendation: {e}")
        return '<p>Error in prediction or recommendation.</p>', 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)
