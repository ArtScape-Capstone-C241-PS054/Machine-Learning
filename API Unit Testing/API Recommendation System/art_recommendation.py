from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)

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
    
    return recommended_genres.tolist()

# Endpoint untuk rekomendasi seni
@app.route('/recommend_art', methods=['POST'])
def recommend_art():
    try:
        data = request.get_json()
        ratings = data.get('ratings', [])
        num_recommendations = data.get('num_recommendations', 5)
        
        if not ratings:
            return jsonify({'error': 'No ratings provided'}), 400
        
        recommended_genres = recommend_art_for_new_user(ratings, num_recommendations)
        return jsonify({'recommendations': recommended_genres})
    except Exception as e:
        logging.error(f"Error in recommendation: {e}")
        return jsonify({'error': 'An error occurred during recommendation'}), 500

# Halaman utama dengan formulir unggah
@app.route('/')
def home():
    return '''
    <h1>Art Recommendation API</h1>
    <form method="POST" action="/recommend_art" enctype="application/json">
        <label for="ratings">Enter your ratings (JSON format):</label><br>
        <textarea name="ratings" rows="4" cols="50" placeholder='[["Abstract", 8], ["Cubism", 5]]'></textarea><br><br>
        <label for="num_recommendations">Number of recommendations:</label><br>
        <input type="text" name="num_recommendations" placeholder="5"><br><br>
        <input type="submit" value="Get Recommendations">
    </form>
    '''

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
