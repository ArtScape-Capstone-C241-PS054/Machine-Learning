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

# Memuat model rekomendasi seni
recommendation_model = load_model(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\RecSys_CBF.h5')
df = pd.read_csv(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\dataset_dummy.csv')

# Encode genre_seni dan user_id
genre_encoder = LabelEncoder()
user_encoder = LabelEncoder()

df['genre_encoded'] = genre_encoder.fit_transform(df['genre_seni'])
df['user_encoded'] = user_encoder.fit_transform(df['user_id'])

# Get embeddings
user_embeddings = recommendation_model.get_layer('user_embedding').get_weights()[0]
genre_embeddings = recommendation_model.get_layer('genre_embedding').get_weights()[0]

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

# Fungsi rekomendasi berdasarkan embedding
def recommend_art_for_new_user(new_user_ratings, num_recommendations=5):
    rated_genres = [rating[0] for rating in new_user_ratings]
    rated_ratings = [rating[1] for rating in new_user_ratings]
    
    rated_genre_indices = genre_encoder.transform(rated_genres)
    new_user_embedding = np.average(
        genre_embeddings[rated_genre_indices], axis=0, weights=rated_ratings)
    
    similarities = np.dot(genre_embeddings, new_user_embedding)
    
    recommended_indices = np.argsort(similarities)[-num_recommendations:][::-1]
    recommended_genres = genre_encoder.inverse_transform(recommended_indices)
    
    return recommended_genres

# Rute untuk halaman utama dengan formulir unggah
@app.route('/')
def home():
    return '''
    <h1>Upload an image for genre prediction, enter text for sentiment analysis, and/or upload an image for art recommendation</h1>
    <form method="POST" action="/predict_recommend" enctype="multipart/form-data">
        <h2>Genre Prediction & Sentiment Analysis</h2>
        <label for="file">Upload an image for genre prediction:</label>
        <input type="file" name="file" accept="image/*"><br><br>
        <label for="text">Enter text for sentiment analysis:</label>
        <input type="text" name="text" placeholder="Enter text for sentiment analysis"><br><br>
        <input type="submit" value="Upload and Predict">
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

        # Prediksi gambar untuk genre dan rekomendasi
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

                image_result_html = '<h1>Image Prediction Result</h1>'
                for pred in image_predictions:
                    image_result_html += f"<p>Class: {pred['class']} - Probability: {pred['probability']:.2f}</p>"

                # Siapkan input untuk rekomendasi
                top_genre_index = np.argmax(classes[0])
                top_genre = class_indices[top_genre_index]

                # Asumsi rating tertinggi untuk genre teratas
                new_user_ratings = [(top_genre, 10)]
                num_recommendations = 5

                # Dapatkan rekomendasi
                recommended_genres = recommend_art_for_new_user(new_user_ratings, num_recommendations)
                
                recommendations_html = '<h1>Recommended Art Genres</h1>'
                for genre in recommended_genres:
                    recommendations_html += f'<p>{genre}</p>'
            finally:
                os.remove(file_path)
        else:
            image_result_html = '<p>No image uploaded for genre prediction or recommendation.</p>'

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
