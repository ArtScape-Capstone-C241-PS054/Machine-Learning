# %% [markdown]
# # Preprocessing

# %% [markdown]
# ## Import Library

# %%
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from joblib import dump, load

# %%
file_path = "C:/Users/user/Documents/Capstone Analisis Sentimen/Dataset/dataset_komentar_instagram_cyberbullying.csv"
df = pd.read_csv(file_path)
df.rename(
    columns={'Instagram Comment Text': 'ArtScape Comment Text'}, inplace=True)

# %%
df.head()

# %% [markdown]
# #### Label Encoding

# %% [markdown]
# Membuat dataset yang bersifat kategoris
#
# *0 = negative*
# *1 = positive*

# %%
le = LabelEncoder()
df['Sentiment'] = le.fit_transform(df['Sentiment'].values)
df

# %% [markdown]
# *Menampilkan Informasi tentang DataFrame*

# %%
df.info()

# %% [markdown]
# *Menampilkan daftar nama kolom dalam DataFrame*

# %%
df.columns

# %% [markdown]
# *Mencetak total jumlah komentar dengan pesan yang sesuai dan Mencetak jumlah komentar dengan sentimen positif dan sentimen negatif beserta pesannya*

# %%
print('Total Sentiment Comment ArtScape:', df.shape[0], 'df\n')
print('terdiri dari (Sentiment):')
print('[0] Sentiment positive\t:', df[df.Sentiment == 1].shape[0], 'data\n')
print('[1] Sentiment Negative\t:', df[df.Sentiment == 0].shape[0], 'data\n')

# %% [markdown]
# ### Distributor Kategori Sentiment ###

# %% [markdown]
# *visualisasi distribusi kategori sentimen dalam DataFrame*

# %%
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

# %% [markdown]
# # Text Preprocessing

# %% [markdown]
# ## Case Folding

# %% [markdown]
# *lakukan beberapa langkah pemrosesan teks, termasuk case folding (mengubah semua huruf menjadi huruf kecil) dan penghapusan pola teks tertentu seperti nama pengguna Twitter, hashtag, URL, tag HTML, dan karakter non-alfabet*

# %%


def casefolding(text):
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text

# %% [markdown]
# *ambil sampel data dari kolom 'ArtScape Comment Text' pada DataFrame df, kemudian melakukan case folding menggunakan fungsi casefolding yang telah didefinisikan sebelumnya, dan akhirnya mencetak hasilnya*


# %%
raw_sample = df['ArtScape Comment Text'].iloc[17]
case_folding = casefolding(raw_sample)

print('Raw df\t:', raw_sample)
print('Case folding\t:', case_folding)

# %% [markdown]
# ## Word Normalization

# %% [markdown]
# *membaca file CSV yang berisi data kunci yang telah dinormalisasi, kemudian mencetak beberapa baris pertama dari data tersebut dan mengembalikan dimensi DataFrame*

# %%
file_path = "C:/Users/user/Documents/Capstone Analisis Sentimen/Dataset/key_norm.csv"
key_norm = pd.read_csv(file_path)
print(key_norm.head())
key_norm.shape

# %% [markdown]
# *lakukan normalisasi teks dengan menggunakan data kunci yang telah dibaca sebelumnya*

# %%


def text_normalize(text):
    text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] if (
        key_norm['singkat'] == word).any() else word for word in text.split()])
    text = str.lower(text)
    return text

# %% [markdown]
# ## Filtering (Stopword Removal)

# %% [markdown]
# *mengambil daftar kata-kata stopwords dalam bahasa Indonesia dari modul stopwords dalam NLTK (Natural Language Toolkit), kemudian menambahkan beberapa kata tambahan ke dalam daftar tersebut*


# %%
stopwords_ind = stopwords.words('indonesian')
more_stopword = ['tsel', 'gb', 'rb']
stopwords_ind = stopwords_ind + more_stopword

# %% [markdown]
# *hapus kata-kata stopwords dari teks yang diberikan*

# %%


def remove_stop_words(text):
    return " ".join([word for word in text.split() if word not in stopwords_ind])

# %% [markdown]
# *ambil contoh teks dari kolom 'ArtScape Comment Text' dalam DataFrame df, kemudian melakukan beberapa tahapan pemrosesan teks seperti case folding, normalisasi teks, dan penghapusan kata-kata stopwords, serta mencetak hasilnya*


# %%
raw_sample = df['ArtScape Comment Text'].iloc[17]
case_folding = casefolding(raw_sample)
normalized_text = text_normalize(case_folding)
stopword_removal = remove_stop_words(normalized_text)

print('Raw df\t\t:', raw_sample)
print('Case folding\t:', case_folding)
print('Stopword Removal\t:', stopword_removal)

# %% [markdown]
# ## Steaming

# %% [markdown]
# *gunakan pustaka Sastrawi untuk membuat objek stemmer, yang akan digunakan untuk melakukan stemming pada kata-kata dalam bahasa Indonesia*

# %%
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# %% [markdown]
# *lakukan stemming pada teks yang diberikan menggunakan objek stemmer yang telah dibuat sebelumnya*

# %%


def stemming(text):
    return stemmer.stem(text)

# %% [markdown]
# *lakukan beberapa tahapan pemrosesan teks secara berurutan, termasuk case folding, normalisasi kata, penghapusan kata-kata stopwords, dan stemming*

# %%


def text_preprocessing_process(text):
    text = casefolding(text)
    text = text_normalize(text)
    text = remove_stop_words(text)
    text = stemming(text)
    return text

# %% [markdown]
# *lakukan pra-pemrosesan teks pada satu contoh teks yang diambil dari kolom 'ArtScape Comment Text' dalam DataFrame, meliputi langkah-langkah case folding, normalisasi teks, penghapusan kata-kata stopwords, dan stemming, serta mencetak hasilnya untuk memeriksa perubahan yang terjadi pada teks setelah melalui setiap tahapan.*


# %%
raw_sample = df['ArtScape Comment Text'].iloc[17]
case_folding = casefolding(raw_sample)
normalized_text = text_normalize(case_folding)
stopword_removal = remove_stop_words(normalized_text)
text_stemming = stemming(stopword_removal)

print('Raw df\t\t:', raw_sample)
print('Case folding\t:', case_folding)
print('Stopword Removal\t:', stopword_removal)
print('Stemming\t\t:', text_stemming)

# %% [markdown]
# # Text Preprocessing Pipeline
# #### Pipeline ini bertujuan untuk membersihkan dan menormalkan teks sehingga lebih cocok untuk analisis dan pemrosesan oleh model atau algoritma.

# %% [markdown]
# *mengukur waktu eksekusi dan menerapkan fungsi text_preprocessing_process pada setiap baris teks dalam kolom 'ArtScape Comment Text' dari DataFrame, dengan tujuan untuk melakukan pra-pemrosesan teks pada data tersebut dan menyimpan hasilnya dalam kolom baru 'clean_teks'.*

# %%
# % % time
df['clean_teks'] = df['ArtScape Comment Text'].apply(
    text_preprocessing_process)

# %% [markdown]
# *menampilkan DataFrame*

# %%
df

# %% [markdown]
# *menyimpannya ke dalam file CSV yang disebut 'clean_data.csv'*

# %%
df.to_csv('clean_data_baru.csv')

# %% [markdown]
# # Feature Engineering

# %% [markdown]
# *menyiapkan data untuk proses feature engineering. Variabel x berisi teks yang telah dibersihkan dari DataFrame df, sedangkan variabel y berisi label sentimen yang sesuai dengan teks tersebut*

# %%
x = df['clean_teks']
y = df['Sentiment']

# %% [markdown]
# *menampilkan variabel 'x' yang berisi teks yang telah dibersihkan*

# %%
x

# %% [markdown]
# *menampilkan variabel 'y' yang berisi label sentimen yang sesuai*

# %%
y

# %% [markdown]
# ## Feature Extraction (TF-IDF and N-Gram)
#
# *gunakan metode TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengubah teks yang telah dibersihkan dalam variabel x menjadi representasi matriks TF-IDF*

# %%
tf_idf = TfidfVectorizer(ngram_range=(1, 1))
x_tf_idf = tf_idf.fit_transform(x).toarray()

# %% [markdown]
# *cetak jumlah total fitur (kata-kata) yang telah diekstraksi dari teks menggunakan TF-IDF, serta daftar fitur tersebut*

# %%
print(len(tf_idf.get_feature_names_out()))
print(tf_idf.get_feature_names_out())

# %% [markdown]
# *gunakan objek TF-IDF yang sudah di-fit sebelumnya pada data training, teks dalam variabel x dapat diubah menjadi representasi matriks TF-IDF yang disimpan dalam variabel x_tf_idf untuk digunakan sebagai fitur dalam analisis atau pemodelan selanjutnya.*

# %%
x_tf_idf = tf_idf.transform(x).toarray()
x_tf_idf

# %% [markdown]
# *membuat DataFrame baru yang disebut df_tf_idf menggunakan matriks TF-IDF yang telah dibuat sebelumnya (x_tf_idf). Setiap kolom dalam DataFrame ini akan mewakili satu fitur (kata-kata) dari matriks TF-IDF yang dihasilkan sebelumnya.*

# %%
df_tf_idf = pd.DataFrame(x_tf_idf, columns=tf_idf.get_feature_names_out())
df_tf_idf

# %% [markdown]
# ### Feature Selection (Chi Square)
#
# *menyiapkan data untuk proses seleksi fitur menggunakan metode Chi-Square (Chi-squared). Variabel x berisi matriks TF-IDF yang telah diubah menjadi array numpy dari DataFrame df_tf_idf, sedangkan variabel y berisi label sentimen yang sesuai dengan data tersebut.*

# %%
x = np.array(df_tf_idf)
y = np.array(y)

# %% [markdown]
# *lakukan seleksi fitur menggunakan metode Chi-Square (Chi-squared) lalu mencetak informasi ini dapat memantau efek dari seleksi fitur terhadap jumlah fitur yang dipertahankan dalam model.*

# %%
chi2_features = SelectKBest(chi2, k=1000)
x_kbest_features = chi2_features.fit_transform(x, y)

print('original feature number:', x.shape[1])
print('reduced feature number:', x_kbest_features.shape[1])

# %% [markdown]
# *membuat DataFrame baru df_chi2 yang berisi skor Chi-Square dari setiap fitur setelah proses seleksi fitur dilakukan.*

# %%
df_chi2 = pd.DataFrame(chi2_features.scores_, columns=['nilai'])
df_chi2

# %% [markdown]
# # Building Model
#
# *Model jaringan saraf dibangun dengan tiga lapisan Dense yang masing-masing memiliki dropout 0.5, diikuti oleh lapisan output dengan aktivasi sigmoid, dan kemudian model dikompilasi menggunakan optimizer 'adam', fungsi loss 'binary_crossentropy', dan metrik evaluasi 'accuracy'.*

# %%
model = Sequential()
model.add(Dense(128, input_shape=(
    x_kbest_features.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# %% [markdown]
# *membagi dataset menjadi data latih (train) dan data uji (test), yang merupakan praktek umum dalam pembelajaran mesin (machine learning) untuk mengevaluasi kinerja model.*

# %%
x_train, x_test, y_train, y_test = train_test_split(
    x_kbest_features, y, test_size=0.2, random_state=42)

# %% [markdown]
# *melatih model menggunakan data latih yang telah dibagi sebelumnya (x_train dan y_train)*

# %%
history = model.fit(x_train, y_train, epochs=20,
                    batch_size=32, validation_split=0.2)

# %% [markdown]
# # Evaluasi kinerja model

# %%
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Akurasi: {accuracy * 100:.2f}%')

y_pred = (model.predict(x_test) > 0.5).astype("int32")

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# %% [markdown]
# *menyimpan model yang telah dilatih ke dalam format H5*

# %%
model.save('sentiment_analysis_model.h5')

# %% [markdown]
# *konversi model Keras menjadi format TensorFlow Lite (TFLite) dengan menyimpannya ke file 'sentiment_analysis_model.tflite'.*

# %%
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('sentiment_analysis_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model has been converted to TFLite and saved as 'sentiment_analysis_model.tflite'")

# %% [markdown]
# ### Finalizing

# %% [markdown]
# *Fungsi preprocess_and_predict melakukan pra-pemrosesan pada teks input, kemudian menerapkan model yang telah dilatih untuk menghasilkan prediksi sentimen berdasarkan teks yang telah diproses, dengan keluaran 'komentar positif' jika prediksi lebih dari 0.5 dan 'komentar negatif' jika tidak.*

# %%


def preprocess_and_predict(input_text):
    preprocessed_text = text_preprocessing_process(input_text)
    tf_idf_vec = tf_idf.transform([preprocessed_text]).toarray()
    kbest_vec = chi2_features.transform(tf_idf_vec)
    prediction = (model.predict(kbest_vec) > 0.5).astype("int32")
    return 'komentar positive' if prediction == 1 else 'komentar negative'


input_text = "Lukisan nya bagus banget"
print(f'Input Text: {input_text}')
print(f'Hasil Prediksi: {preprocess_and_predict(input_text)}')
