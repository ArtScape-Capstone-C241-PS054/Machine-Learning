# Flask API Documentation

This documentation provides an overview and detailed information about the Flask API for image genre prediction. The API is designed to accept image uploads, process them using a pre-trained TensorFlow model, and return the top three predicted genres for the uploaded image.

## Table of Contents

1. [Setup Instructions](#setup-instructions)
2. [API Endpoints](#api-endpoints)
    - [Home](#home)
    - [Predict Genre](#predict-genre)
3. [Model and Data](#model-and-data)
4. [Error Handling](#error-handling)
5. [Logging](#logging)

## Setup Instructions

1. **Clone the Repository:**

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install the Required Packages:**

    Make sure you have Python and pip installed. Then, install the required Python packages:

    ```sh
    pip install Flask tensorflow
    ```

3. **Model File:**

    Place your pre-trained model file (`genre_classification_84.h5`) in the specified directory. Update the `load_model` path if necessary:

    ```python
    image_model = load_model(r'C:\Users\user\Documents\Capstone Analisis Sentimen\genre_classification_84.h5')
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
- **Description:** Returns an HTML form for uploading an image file.
- **Response:** HTML form to upload an image.

#### Example

```html
<h1>Upload an image for genre prediction</h1>
<form method="POST" action="/predict_recommend" enctype="multipart/form-data">
    <label for="file">Upload an image for genre prediction:</label>
    <input type="file" name="file" accept="image/*"><br><br>
    <input type="submit" value="Upload and Predict">
</form>
```

### Predict Genre

- **URL:** `/predict_recommend`
- **Method:** `POST`
- **Description:** Accepts an image file, processes it, and returns the top three predicted genres.
- **Request:** Form data with an image file.
- **Response:** JSON object with top three predicted genres and their probabilities.

#### Example

**Request:**

Upload an image using the form provided in the home endpoint.

**Response:**

```json
{
    "predictions": [
        {"class": "Impressionism", "probability": 0.85},
        {"class": "Realism", "probability": 0.10},
        {"class": "Renaissance", "probability": 0.05}
    ]
}
```

## Model and Data

- **Model:** The model used for prediction is a TensorFlow/Keras model saved in the `h5` format.
- **Image Preprocessing:**
    - The image is resized to `299x299` pixels.
    - The pixel values are normalized by dividing by `255.0`.
    - The image array is expanded to match the model input shape.

```python
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = img_to_array(img)
    x /= 255.0
    x = np.expand_dims(x, axis=0)
    return x
```

## Error Handling

- **No Image Uploaded:** If no image is uploaded, the API returns a `400` status code with an error message.
- **Prediction Error:** If there is an error during the prediction process, the API returns a `500` status code with an error message.

```python
return jsonify({'error': 'No image uploaded for genre prediction.'}), 400
return jsonify({'error': 'Error in genre prediction.'}), 500
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

By following the setup instructions and utilizing the provided endpoints, users can interact with the image genre prediction API to upload images and receive genre predictions.