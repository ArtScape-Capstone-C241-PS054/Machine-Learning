from flask import Flask, request, jsonify
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load model and other necessary objects
try:
    model = load_model(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\sentiment_analysis_model.h5')
    tf_idf = joblib.load(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\tf_idf_vectorizer.pkl')
    chi2_features = joblib.load(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\chi2_features.pkl')
    key_norm = pd.read_csv(r'C:\Users\user\Documents\Capstone Analisis Sentimen\Dataset\key_norm.csv')
    logging.info("Model and objects loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or objects: {e}")

# Define text preprocessing functions
def casefolding(text):
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text

def text_normalize(text):
    words = text.split()
    normalized_words = []
    for word in words:
        match = key_norm[key_norm['singkat'] == word]
        normalized_words.append(match['hasil'].values[0] if not match.empty else word)
    text = ' '.join(normalized_words)
    return text.lower()

stopwords_ind = stopwords.words('indonesian')
more_stopwords = ['tsel', 'gb', 'rb']
stopwords_ind.extend(more_stopwords)

def remove_stop_words(text):
    return " ".join([word for word in text.split() if word not in stopwords_ind])

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(text):
    return stemmer.stem(text)

def text_preprocessing_process(text):
    logging.debug(f"Original text: {text}")
    text = casefolding(text)
    logging.debug(f"After casefolding: {text}")
    text = text_normalize(text)
    logging.debug(f"After normalization: {text}")
    text = remove_stop_words(text)
    logging.debug(f"After removing stopwords: {text}")
    text = stemming(text)
    logging.debug(f"After stemming: {text}")
    return text

# Define prediction function
def preprocess_and_predict(input_text):
    try:
        preprocessed_text = text_preprocessing_process(input_text)
        logging.debug(f"Preprocessed text: {preprocessed_text}")
        tf_idf_vec = tf_idf.transform([preprocessed_text]).toarray()
        kbest_vec = chi2_features.transform(tf_idf_vec)
        prediction = (model.predict(kbest_vec) > 0.5).astype("int32")
        logging.debug(f"Prediction result: {prediction}")
        return 'komentar positive' if prediction == 1 else 'komentar negative'
    except Exception as e:
        logging.error(f"Error in preprocessing or prediction: {e}")
        return 'error'

# Define API endpoint
@app.route('/')
def index():
    return 'Welcome to the Sentiment Analysis API!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_text = data['text']
        logging.info(f"Received text for prediction: {input_text}")
        prediction = preprocess_and_predict(input_text)
        return jsonify({'prediction': prediction})
    except Exception as e:
        logging.error(f"Error in /predict endpoint: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

# Run the app
if __name__ == '__main__':
    try:
        app.run(debug=True, port=5001, host='0.0.0.0')
    except Exception as e:
        logging.error(f"Error starting the Flask app: {e}")
