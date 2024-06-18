# Flask API Documentation

This documentation provides an overview and detailed information about the Flask API for art recommendation. The API accepts user ratings for different art genres and returns genre recommendations based on a collaborative filtering model.

## Table of Contents

1. [Setup Instructions](#setup-instructions)
2. [API Endpoints](#api-endpoints)
    - [Home](#home)
    - [Recommend Art](#recommend-art)
3. [Model and Data](#model-and-data)
4. [Recommendation Function](#recommendation-function)
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
    pip install Flask tensorflow pandas scikit-learn
    ```

3. **Model and Data Files:**

    Place your model file (`RecSys_CBF.h5`) and dataset file (`dataset_dummy.csv`) in the specified directory. Update the paths in the code if necessary:

    ```python
    recommendation_model = load_model(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\RecSys_CBF.h5')
    df = pd.read_csv(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\dataset_dummy.csv')
    ```

4. **Run the Application:**

    Run the Flask application with:

    ```sh
    python app.py
    ```

    The application will start on `http://0.0.0.0:5001`.

## API Endpoints

### Home

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

### Recommend Art

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

- **Model:** The model used for recommendations is a TensorFlow/Keras collaborative filtering model saved in the `h5` format.
- **Data:** The dataset contains user ratings for different art genres. The genres and user IDs are encoded using `LabelEncoder`.

```python
df['genre_encoded'] = genre_encoder.fit_transform(df['genre_seni'])
df['user_encoded'] = user_encoder.fit_transform(df['user_id'])
```

## Recommendation Function

The recommendation function computes the average embedding for the rated genres and returns the most similar genres based on cosine similarity.

```python
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
```

## Error Handling

- **No Ratings Provided:** If no ratings are provided in the request, the API returns a `400` status code with an error message.
- **Recommendation Error:** If there is an error during the recommendation process, the API returns a `500` status code with an error message.

```python
return jsonify({'error': 'No ratings provided'}), 400
return jsonify({'error': 'An error occurred during recommendation'}), 500
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

By following the setup instructions and utilizing the provided endpoints, users can interact with the art recommendation API to submit ratings and receive genre recommendations.