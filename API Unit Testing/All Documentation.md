# Flask API Documentation

This repository contains three Flask APIs: an image genre classification API, a text sentiment analysis API, and an art recommendation API. Each API is described in detail below, including setup instructions, endpoints, and other relevant information.

## Table of Contents

1. [Setup Instructions](#setup-instructions)
2. [API Endpoints](#api-endpoints)
    - [Image Genre Classification](#image-genre-classification)
        - [Home](#home-1)
        - [Predict Genre](#predict-genre)
    - [Text Sentiment Analysis](#text-sentiment-analysis)
        - [Home](#home-2)
        - [Predict Sentiment](#predict-sentiment)
    - [Art Recommendation](#art-recommendation)
        - [Home](#home-3)
        - [Recommend Art](#recommend-art)
3. [Model and Data](#model-and-data)
4. [Preprocessing Functions](#preprocessing-functions)
5. [Error Handling](#error-handling)
6. [Logging](#logging)
7. [License](#license)

## Setup Instructions

1. **Install the Required Packages:**

    Make sure you have Python and pip installed. Then, install the required Python packages:

    ```sh
    pip install Flask tensorflow pandas scikit-learn nltk Sastrawi joblib
    ```

2. **Download NLTK Stopwords:**

    ```sh
    python -m nltk.downloader stopwords
    ```

3. **Place Model and Data Files:**

    - **Image Genre Classification:** Place `genre_classification_84.h5` in the specified directory.
    - **Text Sentiment Analysis:** Place `sentiment_analysis_model.h5`, `tf_idf_vectorizer.pkl`, `chi2_features.pkl`, and `key_norm.csv` in the specified directory.
    - **Art Recommendation:** Place `RecSys_CBF.h5` and `dataset_dummy.csv` in the specified directory.

    Update the paths in the code if necessary.

4. **Run the Applications:**

    For each API, navigate to its directory and run the Flask application:

    ```sh
    python app.py
    ```

    Each application will start on `http://0.0.0.0:5001`.

## API Endpoints

### Image Genre Classification

#### Home

- **URL:** `/`
- **Method:** `GET`
- **Description:** Returns an HTML form for uploading an image for genre prediction.
- **Response:** HTML form to upload an image and get genre prediction.

#### Example

```html
<h1>Upload an image for genre prediction</h1>
<form method="POST" action="/predict_recommend" enctype="multipart/form-data">
    <label for="file">Upload an image for genre prediction:</label>
    <input type="file" name="file" accept="image/*"><br><br>
    <input type="submit" value="Upload and Predict">
</form>
```

#### Predict Genre

- **URL:** `/predict_recommend`
- **Method:** `POST`
- **Description:** Accepts an image file and returns the top 3 genre predictions.
- **Request:** Multipart form data with an image file.
- **Response:** JSON object with `predictions` field containing a list of top 3 genre predictions and their probabilities.

#### Example

**Request:**

Upload an image file using the form.

**Response:**

```json
{
    "predictions": [
        {"class": "Surrealism", "probability": 0.85},
        {"class": "Cubism", "probability": 0.10},
        {"class": "Abstract", "probability": 0.05}
    ]
}
```

### Text Sentiment Analysis

#### Home

- **URL:** `/`
- **Method:** `GET`
- **Description:** Returns a welcome message.
- **Response:** Plain text message.

#### Example

```text
Welcome to the Sentiment Analysis API!
```

#### Predict Sentiment

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

### Art Recommendation

#### Home

- **URL:** `/`
- **Method:** `GET`
- **Description:** Returns an HTML form for entering user ratings and the number of recommendations.
- **Response:** HTML form to submit user ratings and get recommendations.

#### Example

```html
<h1>Art Recommendation API</h1>
<form method="POST" action="/recommend_art" enctype="application/json">
    <label for="ratings">Enter your ratings (JSON format):</label><br>
    <textarea name="ratings" rows="4" cols="50" placeholder='[["Abstract", 8], ["Cubism", 5]]'></textarea><br><br>
    <label for="num_recommendations">Number of recommendations:</label><br>
    <input type="text" name="num_recommendations" placeholder="5"><br><br>
    <input type="submit" value="Get Recommendations">
</form>
```

#### Recommend Art

- **URL:** `/recommend_art`
- **Method:** `POST`
- **Description:** Accepts user ratings for art genres and returns genre recommendations.
- **Request:** JSON object with `ratings` field (list of genre and rating pairs) and optional `num_recommendations` field.
- **Response:** JSON object with `recommendations` field containing a list of recommended genres.

#### Example

**Request:**

```json
{
    "ratings": [["Abstract", 8], ["Cubism", 5]],
    "num_recommendations": 5
}
```

**Response:**

```json
{
    "recommendations": ["Impressionism", "Surrealism", "Renaissance", "Pop", "Realism"]
}
```

## Model and Data

- **Image Genre Classification:** TensorFlow/Keras model for predicting art genres from images.
- **Text Sentiment Analysis:** TensorFlow/Keras model for predicting sentiment from text. Uses TF-IDF vectorizer and chi-squared feature selector.
- **Art Recommendation:** TensorFlow/Keras collaborative filtering model for recommending art genres based on user ratings. Uses label encoding for genres and user IDs.

## Preprocessing Functions

### Text Preprocessing for Sentiment Analysis

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

- **Invalid Input:** If the input is not as expected, the APIs return a `400` status code with an error message.
- **Processing Error:** If there is an error during processing, the APIs return a `500` status code with an error message.

```python
return jsonify({'error': 'An error occurred during processing'}), 500
```

## Logging

- The application uses Python's `logging` module to log errors and other informational messages.

```python
logging.basicConfig(level=logging.DEBUG)
```

This setup ensures that any issues encountered during the execution of the API are

 logged for debugging purposes.


By following the setup instructions and utilizing the provided endpoints, users can interact with the image genre classification, text sentiment analysis, and art recommendation APIs to get predictions and recommendations based on their inputs.
