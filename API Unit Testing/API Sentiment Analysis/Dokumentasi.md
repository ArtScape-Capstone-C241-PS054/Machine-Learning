# Flask API Documentation

This documentation provides an overview and detailed information about the Flask API for sentiment analysis. The API accepts text input, preprocesses it, and returns a sentiment prediction (positive or negative).

## Table of Contents

1. [Setup Instructions](#setup-instructions)
2. [API Endpoints](#api-endpoints)
    - [Home](#home)
    - [Predict Sentiment](#predict-sentiment)
3. [Model and Data](#model-and-data)
4. [Text Preprocessing](#text-preprocessing)
5. [Error Handling](#error-handling)
6. [Logging](#logging)

## Setup Instructions

1. **Clone the Repository:**

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install the Required Packages:**

    Make sure you have Python and pip installed. Then, install the required Python packages:

    ```sh
    pip install Flask tensorflow pandas scikit-learn nltk Sastrawi joblib
    ```

3. **Download NLTK Stopwords:**

    ```sh
    python -m nltk.downloader stopwords
    ```

4. **Model and Data Files:**

    Place your model file (`sentiment_analysis_model.h5`), TF-IDF vectorizer (`tf_idf_vectorizer.pkl`), chi-squared feature selector (`chi2_features.pkl`), and normalization CSV (`key_norm.csv`) in the specified directory. Update the paths in the code if necessary:

    ```python
    model = load_model(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\sentiment_analysis_model.h5')
    tf_idf = joblib.load(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\tf_idf_vectorizer.pkl')
    chi2_features = joblib.load(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\chi2_features.pkl')
    key_norm = pd.read_csv(r'C:\Users\user\Documents\Capstone Analisis Sentimen\Dataset\key_norm.csv')
    ```

5. **Run the Application:**

    Run the Flask application with:

    ```sh
    python app.py
    ```

    The application will start on `http://0.0.0.0:5001`.

## API Endpoints

### Home

- **URL:** `/`
- **Method:** `GET`
- **Description:** Returns a welcome message.
- **Response:** Plain text message.

#### Example

```text
Welcome to the Sentiment Analysis API!
```

### Predict Sentiment

- **URL:** `/predict`
- **Method:** `POST`
- **Description:** Accepts text input, preprocesses it, and returns the sentiment prediction.
- **Request:** JSON object with a `text` field.
- **Response:** JSON object with a `prediction` field indicating 'komentar positive' or 'komentar negative'.

#### Example

**Request:**

```json
{
    "text": "Saya sangat senang dengan layanan ini!"
}
```

**Response:**

```json
{
    "prediction": "komentar positive"
}
```

## Model and Data

- **Model:** The model used for prediction is a TensorFlow/Keras model saved in the `h5` format.
- **TF-IDF Vectorizer:** Used for converting text data into numerical format.
- **Chi-squared Feature Selector:** Used for selecting important features from the text data.
- **Normalization CSV:** Contains mappings for text normalization.

## Text Preprocessing

The following preprocessing steps are applied to the input text:

1. **Case Folding:** Convert text to lowercase and remove unnecessary characters.

    ```python
    def casefolding(text):
        text = text.lower()
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        return text
    ```

2. **Text Normalization:** Normalize text based on a predefined dictionary.

    ```python
    def text_normalize(text):
        words = text.split()
        normalized_words = []
        for word in words:
            match = key_norm[key_norm['singkat'] == word]
            normalized_words.append(match['hasil'].values[0] if not match.empty else word)
        text = ' '.join(normalized_words)
        return text.lower()
    ```

3. **Remove Stop Words:** Remove common stop words.

    ```python
    stopwords_ind = stopwords.words('indonesian')
    more_stopwords = ['tsel', 'gb', 'rb']
    stopwords_ind.extend(more_stopwords)

    def remove_stop_words(text):
        return " ".join([word for word in text.split() if word not in stopwords_ind])
    ```

4. **Stemming:** Reduce words to their base form.

    ```python
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemming(text):
        return stemmer.stem(text)
    ```

5. **Complete Preprocessing:**

    ```python
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
    ```

## Error Handling

- **Invalid Input:** If the input JSON does not contain the required `text` field, the API returns a `400` status code with an error message.
- **Prediction Error:** If there is an error during the prediction process, the API returns a `500` status code with an error message.

```python
return jsonify({'error': 'An error occurred during prediction'}), 500
```

## Logging

- The application uses Python's `logging` module to log errors and other informational messages.

```python
logging.basicConfig(level=logging.DEBUG)
```

This setup ensures that any issues encountered during the execution of the API are logged for debugging purposes.

## License

Include a license if applicable.

---

By following the setup instructions and utilizing the provided endpoints, users can interact with the sentiment analysis API to send text and receive sentiment predictions.