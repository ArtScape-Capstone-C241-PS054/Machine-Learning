### 1. Mengimpor Pustaka dan Modul
```python
from flask import Flask, request
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
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
import logging
```
Ini adalah langkah awal di mana kita mengimpor semua pustaka dan modul yang akan kita gunakan dalam aplikasi Flask, termasuk Flask itu sendiri untuk membuat aplikasi web, dan pustaka-pustaka lain untuk memproses teks dan gambar.

### 2. Inisialisasi Aplikasi Flask
```python
app = Flask(__name__)
```
Ini adalah bagian di mana kita membuat instance aplikasi Flask yang akan digunakan untuk menangani rute dan permintaan.

### 3. Konfigurasi Logging
```python
logging.basicConfig(level=logging.DEBUG)
```
Ini mengatur konfigurasi logging pada level DEBUG sehingga kita dapat melihat pesan debug saat menjalankan aplikasi.

### 4. Memuat Model Klasifikasi Gambar
```python
image_model = load_model(r'C:\Users\user\Documents\Capstone Analisis Sentimen\genre_classification_84.h5')
```
Ini memuat model klasifikasi gambar yang akan digunakan untuk memprediksi genre dari gambar yang diunggah.

### 5. Memuat Indeks Kelas Label untuk Klasifikasi Gambar
```python
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
```
Ini adalah kamus yang berisi indeks kelas label untuk klasifikasi gambar. Setiap angka merepresentasikan kelas tertentu.

### 6. Memuat Model Analisis Sentimen dan Objek
```python
try:
    sentiment_model = load_model(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\sentiment_analysis_model.h5')
    tf_idf = joblib.load(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\tf_idf_vectorizer.pkl')
    chi2_features = joblib.load(r'C:\Users\user\Documents\Capstone Analisis Sentimen\code\chi2_features.pkl')
    key_norm = pd.read_csv(r'C:\Users\user\Documents\Capstone Analisis Sentimen\Dataset\key_norm.csv')
    logging.info("Sentiment analysis model and objects loaded successfully.")
except Exception as e:
    logging.error(f"Error loading sentiment analysis model or objects: {e}")
```
Ini memuat model analisis sentimen dan objek-objek yang diperlukan seperti vektorisasi TF-IDF dan fitur-fitur chi-squared yang telah disimpan sebelumnya.

### 7. Fungsi Pra-Pemrosesan Teks untuk Analisis Sentimen
```python
def preprocess_text(text):
    # Implementasi pra-pemrosesan teks
    ...
```
Ini adalah serangkaian fungsi-fungsi yang digunakan untuk pra-pemrosesan teks sebelum meneruskannya ke model analisis sentimen.

### 8. Definisi Fungsi Prediksi untuk Analisis Sentimen
```python
def predict_sentiment(input_text):
    # Implementasi prediksi analisis sentimen
    ...
```
Ini adalah fungsi yang digunakan untuk memprediksi sentimen dari teks yang diberikan.

### 9. Fungsi Pra-Pemrosesan Gambar
```python
def preprocess_image(image_path):
    # Implementasi pra-pemrosesan gambar
    ...
```
Ini adalah fungsi yang digunakan untuk pra-pemrosesan gambar sebelum meneruskannya ke model klasifikasi gambar.

### 10. Rute untuk Halaman Utama
```python
@app.route('/')
def home():
    # Implementasi halaman utama
    ...
```
Ini adalah rute yang digunakan untuk menampilkan halaman utama aplikasi web, yang berisi formulir untuk mengunggah gambar dan teks.

### 11. Rute untuk Prediksi
```python
@app.route('/predict', methods=['POST'])
def predict():
    # Implementasi rute untuk menangani prediksi gambar dan teks
    ...
```
Ini adalah rute yang menangani permintaan prediksi, baik untuk gambar maupun teks.

### 12. Menjalankan Aplikasi Flask
```python
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)
```
Ini adalah bagian terakhir dari kode yang menjalankan aplikasi Flask di localhost pada port 5001. Jika aplikasi dijalankan secara langsung (bukan diimpor sebagai modul), ia akan memulai server Flask untuk menangani permintaan masuk.
