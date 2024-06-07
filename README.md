# Machine-Learning
Repository for Machine Learning

# 1.Sentiment Analysis

This project is a sentiment analysis pipeline for analyzing comments from Instagram to detect cyberbullying. The project involves data preprocessing, feature extraction, feature selection, model building, evaluation, and saving the model for future use.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Preprocessing](#preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Model Saving](#model-saving)
- [Model Inference](#model-inference)

## Installation

To run this project, you need to have the following libraries installed:

- pandas
- numpy
- re
- seaborn
- matplotlib
- tensorflow
- sklearn
- nltk
- Sastrawi
- joblib

You can install them using pip:

```sh
pip install pandas numpy seaborn matplotlib tensorflow scikit-learn nltk Sastrawi joblib
```

## Usage

1. Clone this repository.
2. Open the `Sentiment_Analisis.ipynb` notebook.
3. Follow the steps in the notebook to run the entire sentiment analysis pipeline.

## Project Structure

```
.
├── code
│   ├── Sentiment_Analisis.ipynb
│   ├── clean_data_baru.csv
│   ├── sentiment_analysis_model.h5
│   └── sentiment_analysis_model.tflite
├── Dataset
│   ├── dataset_komentar_instagram_cyberbullying.csv
│   └── key_norm.csv
```

## Preprocessing

### Import Libraries

The necessary libraries are imported at the beginning of the notebook.

### Load Data

The dataset is loaded and some initial exploratory data analysis is performed:

```python
file_path = "C:/Users/user/Documents/Capstone Analisis Sentimen/Dataset/dataset_komentar_instagram_cyberbullying.csv"
df = pd.read_csv(file_path)
df.rename(columns={'Instagram Comment Text' : 'ArtScape Comment Text'}, inplace=True)
```

### Label Encoding

Sentiment labels are encoded into numerical values:

```python
le = LabelEncoder()
df['Sentiment'] = le.fit_transform(df['Sentiment'].values)
```

### Data Information

Basic information and statistics about the data are displayed:

```python
df.info()
df.columns
```

### Sentiment Distribution

Visualize the distribution of sentiment categories:

```python
height = df['Sentiment'].value_counts()
labels = ('Sentiment positive', 'Sentiment negative')
y_pos = np.arange(len(labels))

plt.figure(figsize=(6, 5), dpi=(80))
plt.ylim(0, 300)
plt.title('Distribusi Kategori Sentiment', fontweight='bold')
plt.xlabel('Kategori', fontweight='bold')
plt.ylabel('Jumlah', fontweight='bold')
plt.bar(y_pos, height, color=['royalblue', 'skyblue'])
plt.xticks(y_pos, labels)
plt.show()
```

## Text Preprocessing

### Case Folding

Convert text to lowercase and remove unwanted characters:

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

### Word Normalization

Normalize text using a key normalization file:

```python
file_path = "C:/Users/user/Documents/Capstone Analisis Sentimen/Dataset/key_norm.csv"
key_norm = pd.read_csv(file_path)

def text_normalize(text):
    text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] if (key_norm['singkat'] == word).any() else word for word in text.split()])
    text = str.lower(text)
    return text
```

### Stopword Removal

Remove stopwords from the text:

```python
stopwords_ind = stopwords.words('indonesian')
more_stopword = ['tsel', 'gb', 'rb']
stopwords_ind = stopwords_ind + more_stopword

def remove_stop_words(text):
    return " ".join([word for word in text.split() if word not in stopwords_ind])
```

### Stemming

Perform stemming on the text using Sastrawi:

```python
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(text):
    return stemmer.stem(text)
```

### Text Preprocessing Pipeline

Apply all preprocessing steps to the text:

```python
def text_preprocessing_process(text):
    text = casefolding(text)
    text = text_normalize(text)
    text = remove_stop_words(text)
    text = stemming(text)
    return text

df['clean_teks'] = df['ArtScape Comment Text'].apply(text_preprocessing_process)
df.to_csv('clean_data_baru.csv')
```

## Feature Engineering

### Feature Extraction (TF-IDF and N-Gram)

Extract features using TF-IDF:

```python
tf_idf = TfidfVectorizer(ngram_range=(1, 1))
x_tf_idf = tf_idf.fit_transform(df['clean_teks']).toarray()
```

### Feature Selection (Chi-Square)

Select important features using Chi-Square:

```python
chi2_features = SelectKBest(chi2, k=1000)
x_kbest_features = chi2_features.fit_transform(x_tf_idf, df['Sentiment'])
```

## Model Building

### Define and Compile Model

Build and compile a neural network model:

```python
model = Sequential()
model.add(Dense(128, input_shape=(x_kbest_features.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Train-Test Split

Split data into training and testing sets:

```python
x_train, x_test, y_train, y_test = train_test_split(x_kbest_features, df['Sentiment'], test_size=0.2, random_state=42)
```

### Train the Model

Train the model:

```python
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
```

## Model Evaluation

Evaluate model performance:

```python
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Akurasi: {accuracy * 100:.2f}%')

y_pred = (model.predict(x_test) > 0.5).astype("int32")

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
```

## Model Saving

Save the trained model:

```python
model.save('sentiment_analysis_model.h5')
```

Convert and save the model to TensorFlow Lite format:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('sentiment_analysis_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model has been converted to TFLite and saved as 'sentiment_analysis_model.tflite'")
```

## Model Inference

Define a function to preprocess input text and make predictions:

```python
def preprocess_and_predict(input_text):
    preprocessed_text = text_preprocessing_process(input_text)
    tf_idf_vec = tf_idf.transform([preprocessed_text]).toarray()
    kbest_vec = chi2_features.transform(tf_idf_vec)
    prediction = (model.predict(kbest_vec) > 0.5).astype("int32")
    return 'komentar positive' if prediction == 1 else 'komentar negative'

input_text = "Lukisan nya bagus banget"
print(f'Input Text: {input_text}')
print(f'Hasil Prediksi: {preprocess_and_predict(input_text)}')
```

This README provides an overview of the sentiment analysis project and explains how to use the provided code to perform sentiment analysis on Instagram comments.

# 2. Artwork Genre Classification
```markdown
Artwork Genre Classification

This repository contains code for classifying artwork images into different genres using TensorFlow. The project involves detailed steps for data pipelining, data splitting, model building, training, evaluation, model saving, and prediction.

## Data Pipelining

The data pipelining process involves downloading and preprocessing artwork images.

- **Downloading Dataset**: The dataset is downloaded from GitHub using the `git clone` command.
- **Setting Dataset Path**: The path to the dataset folder is set.
- **Creating Preprocessed Directory**: A new directory is created to store preprocessed images.
- **Image Preprocessing**: Images are resized and converted to PNG format using the `preprocess_and_save_image` function.

```python
# Downloading dataset and setting dataset path
dataset_dir = 'NEW ART DATASETS/ARTSCAPE'

# Creating preprocessed directory
preprocessed_dir = 'preprocessed_art'
if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

# Image preprocessing
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', '.tiff', '.gif')):
            image_path = os.path.join(root, file)
            relative_path = os.path.relpath(image_path, dataset_dir)
            save_path = os.path.join(preprocessed_dir, os.path.splitext(relative_path)[0] + '.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            preprocess_and_save_image(image_path, save_path)
```

## Data Splitting

The data splitting process involves splitting the dataset into training and validation sets.

- **Creating Train and Validation Directories**: New directories are created to store training and validation images.
- **Splitting Images**: Images are split into training and validation sets using `train_test_split`.
- **Copying Images**: Images are copied to the respective directories based on the split.

```python
# Splitting dataset into train and validation sets
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)

# Splitting images and copying to train and validation directories
for class_name in class_names:
    class_original_dir = os.path.join(original_dataset_dir, class_name)
    class_train_dir = os.path.join(train_dir, class_name)
    os.makedirs(class_train_dir, exist_ok=True)
    class_validation_dir = os.path.join(validation_dir, class_name)
    os.makedirs(class_validation_dir, exist_ok=True)

    image_paths = [os.path.join(class_original_dir, image_name) for image_name in os.listdir(class_original_dir)]
    train_image_paths, validation_image_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

    for image_path in train_image_paths:
        image_name = os.path.basename(image_path)
        target_path = os.path.join(class_train_dir, image_name)
        copyfile(image_path, target_path)

    for image_path in validation_image_paths:
        image_name = os.path.basename(image_path)
        target_path = os.path.join(class_validation_dir, image_name)
        copyfile(image_path, target_path)
```

## Model Building

The model building process involves building a deep learning model using transfer learning (Xception).

- **Loading Base Model**: Xception base model is loaded.
- **Freezing Base Model Layers**: Base model layers are frozen to prevent retraining.
- **Adding Custom Layers**: Custom dense layers are added on top of the base model.
- **Compiling Model**: Model is compiled with optimizer, loss function, and metrics.

```python
# Loading Xception base model
base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling='avg')

# Freezing base model layers
for layer in base_model.layers:
    layer.trainable = False

# Adding custom layers
model = Sequential()
model.add(base_model)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compiling model
model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

## Model Training

The model training process involves training the model on the training dataset.

- **Defining Callbacks**: ModelCheckpoint callback is defined to save the best model during training.
- **Training Model**: Model is trained using the `fit` function.

```python
# Defining ModelCheckpoint callback
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=False, save_freq='epoch')

# Training model with checkpoint callback
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, callbacks=[checkpoint], verbose=1)
```

## Model Evaluation

The model evaluation process involves evaluating the model on the validation dataset.

- **Evaluating Model**: Model is evaluated using the `evaluate` function.
- **Calculating Metrics**: Accuracy and loss are calculated and printed.

```python
# Evaluating model
score = model.evaluate(validation_dataset, verbose=0)

# Printing accuracy and loss
print("Accuracy: {}%, Loss:{}".format(score[1]*100, score[0]))
```

## Model Saving

The model saving process involves saving the trained model for future use.

- **Saving Model**: Trained model is saved to a .h5 file using the `save` function.

```python
# Saving the model to a .h5 file
model.save('genre_classification_84.h5')
```

## Prediction

The prediction process involves predicting genres for new artwork images.

- **Loading Model**: Trained model is loaded from the saved .h
