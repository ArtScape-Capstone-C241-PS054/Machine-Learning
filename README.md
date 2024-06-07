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
# Artwork Genre Classification

This repository contains code for classifying artwork images into different genres using TensorFlow. The project involves detailed steps for data pipelining, data splitting, model building, training, evaluation, model saving, and prediction.

# **DATA PIPELINING**

## Extract

```python
# Mengunduh dataset dari GitHub dengan git clone
# !git clone https://github.com/zidan2808/NEW-ART-DATASETS.git

# Mengatur path ke folder dataset
dataset_dir = 'NEW ART DATASETS/ARTSCAPE'

# Buat direktori baru untuk menyimpan gambar yang telah diproses
preprocessed_dir = 'preprocessed_art'
if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

# Fungsi untuk mendapatkan daftar kelas dan jumlah masing-masing kelas
def get_class_distribution(dataset_dir):
    class_counts = Counter()

    # Loop melalui subdirektori dalam dataset_dir
    for root, dirs, files in os.walk(dataset_dir):
        for dir_name in dirs:
            class_dir = os.path.join(root, dir_name)
            num_images = len([f for f in os.listdir(class_dir)
                             if os.path.isfile(os.path.join(class_dir, f))])
            # Nama kelas diambil dari nama subdirektori terakhir
            class_name = os.path.basename(class_dir)
            class_counts[class_name] += num_images

    return class_counts


# Mendapatkan jumlah masing-masing kelas
class_counts = get_class_distribution(dataset_dir)

# Menghitung total jumlah kelas dan total jumlah file
num_classes = len(class_counts)
total_file_count = sum(class_counts.values())

# Membuat DataFrame dari hasil penghitungan
class_counts_df = pd.DataFrame.from_dict(
    class_counts, orient='index', columns=['Jumlah'])
class_counts_df.index.name = 'Genre'
class_counts_df.reset_index(inplace=True)

# Menampilkan tabel
print(class_counts_df)

# Menampilkan total jumlah kelas dan total jumlah file
print(f"\nTotal Jenis Genre Seni: {num_classes}")
print(f"Total Jumlah FIle: {total_file_count}")

# Fungsi untuk mendapatkan daftar kelas dan jumlah masing-masing kelas
def get_class_distribution(dataset_dir):
    class_counts = Counter()
    extension_counts = Counter()

    # Loop melalui subdirektori dalam dataset_dir
    for root, dirs, files in os.walk(dataset_dir):
        for dir_name in dirs:
            class_dir = os.path.join(root, dir_name)
            num_images = len([f for f in os.listdir(class_dir)
                             if os.path.isfile(os.path.join(class_dir, f))])
            # Nama kelas diambil dari nama subdirektori terakhir
            class_name = os.path.basename(class_dir)
            class_counts[class_name] += num_images

            # Menghitung jumlah file berdasarkan ekstensi
            for file in os.listdir(class_dir):
                if os.path.isfile(os.path.join(class_dir, file)):
                    file_extension = os.path.splitext(file)[1].lower()
                    extension_counts[file_extension] += 1

    return class_counts, extension_counts


# Mendapatkan jumlah masing-masing kelas dan ekstensi file
class_counts, extension_counts = get_class_distribution(dataset_dir)

# Membuat DataFrame dari hasil penghitungan ekstensi
extension_counts_df = pd.DataFrame.from_dict(
    extension_counts, orient='index', columns=['Jumlah'])
extension_counts_df.index.name = 'Ekstensi'
extension_counts_df.reset_index(inplace=True)

# Menampilkan tabel
print(extension_counts_df)


## Data Splitting

The dataset is split into training and validation sets.

```python
# Splitting data into training and validation sets
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)

# Splitting path gambar menjadi bagian training dan validasi
for class_name in class_names:
    # Path ke direktori kelas di dataset asli
    class_original_dir = os.path.join(original_dataset_dir, class_name)

    # Path ke direktori kelas di dataset baru untuk training dan validasi
    class_train_dir = os.path.join(train_dir, class_name)
    os.makedirs(class_train_dir, exist_ok=True)
    class_validation_dir = os.path.join(validation_dir, class_name)
    os.makedirs(class_validation_dir, exist_ok=True)

    # Splitting path gambar menjadi bagian training dan validasi
    image_paths = [os.path.join(class_original_dir, image_name)
                   for image_name in os.listdir(class_original_dir)]
    train_image_paths, validation_image_paths = train_test_split(
        image_paths, test_size=0.2, random_state=42)

    # Menyalin gambar ke direktori training
    for image_path in train_image_paths:
        image_name = os.path.basename(image_path)
        target_path = os.path.join(class_train_dir, image_name)
        copyfile(image_path, target_path)

    # Menyalin gambar ke direktori validation
    for image_path in validation_image_paths:
        image_name = os.path.basename(image_path)
        target_path = os.path.join(class_validation_dir, image_name)
        copyfile(image_path, target_path)

print("Pemisahan dataset selesai.")
```

## Model Building

The model building process involves building a deep learning model using transfer learning (Xception).

### Loading Base Model

Xception base model is loaded.

```python
# Loading Xception base model
base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling='avg')
```

### Freezing Base Model Layers

Base model layers are frozen to prevent retraining.

```python
# Freezing base model layers
for layer in base_model.layers:
    layer.trainable = False
```

### Adding Custom Layers

Custom dense layers are added on top of the base model.

```python
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
```

### Compiling Model

Model is compiled with optimizer, loss function, and metrics.

```python
# Compiling model
model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

## Model Training

The model training process involves training the model on the training dataset.

### Defining Callbacks

ModelCheckpoint callback is defined to save the best model during training.

```python
# Defining ModelCheckpoint callback
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=False, save_freq='epoch')
```

### Training Model

Model is trained using the `fit` function.

```python
# Training model with checkpoint callback
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, callbacks=[checkpoint], verbose=1)
```

## Model Evaluation

The model evaluation process involves evaluating the model on the validation dataset.

### Evaluating Model

Model is evaluated using the `evaluate` function.

```python
# Evaluating model
score = model.evaluate(validation_dataset, verbose=0)
```

### Calculating Metrics

Accuracy and loss are calculated and printed.

```python
# Printing accuracy and loss
print("Accuracy: {}%, Loss:{}".format(score[1]*100, score[0]))
```

## Model Saving

The model saving process involves saving the trained model for future use.

### Saving Model

Trained model is saved to a .h5 file using the `save` function.

```python
# Saving the model to a .h5 file
model.save('genre_classification_84.h5')
```

## Prediction

The prediction process involves predicting genres for new artwork images.

### Loading Model

Trained model is loaded from the saved .h5 file.

```python
# Loading the saved model
loaded_model = load_model('genre_classification_84.h5')
```
```
