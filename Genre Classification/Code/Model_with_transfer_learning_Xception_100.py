# %% [markdown]
# # **IMPORT IMPORTANT LIBRARIES**
#
#

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import itertools
import cv2 as cv
import PIL
import PIL.Image

from PIL import Image
from collections import Counter
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from shutil import copyfile
from sklearn.model_selection import train_test_split
# from google.colab import files


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # **SET HYPERPARAMETER**

# %%
BATCH_SIZE = 50
IMG_HEIGHT = 299
IMG_WIDTH = 299
EPOCHS = 100

# %% [markdown]
# # **DATA PIPELINING**

# %% [markdown]
# # **Extract**

# %%
# Mengunduh dataset dari GitHub dengan git clone
# !git clone https://github.com/zidan2808/NEW-ART-DATASETS.git

# Mengatur path ke folder dataset
dataset_dir = '../Dataset/ARTSCAPE'

# Buat direktori baru untuk menyimpan gambar yang telah diproses
preprocessed_dir = '../Dataset/preprocessed_art'
if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

# %%
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

# %%
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

# %% [markdown]
# # **Transform**

# %%
# def preprocess_and_save_image(image_path, save_path):
#     try:
#         with Image.open(image_path) as img:
#             # Jika gambar dalam mode CMYK, konversi ke RGB
#             if img.mode == 'CMYK':
#                 img = img.convert('RGB')

#             # Mengubah ukuran gambar
#             img = img.resize((IMG_HEIGHT, IMG_WIDTH))

#             # Menyimpan gambar sebagai PNG
#             img.save(save_path, 'PNG')
#     except OSError as e:
#         print(f"Error processing image {image_path}: {e}")


# # Menelusuri direktori dataset dan memproses gambar
# for root, dirs, files in os.walk(dataset_dir):
#     for file in files:
#         if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', '.tiff', '.gif')):
#             image_path = os.path.join(root, file)
#             relative_path = os.path.relpath(image_path, dataset_dir)
#             save_path = os.path.join(
#                 preprocessed_dir, os.path.splitext(relative_path)[0] + '.png')

#             # Membuat direktori jika belum ada
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)

#             # Memproses dan menyimpan gambar
#             preprocess_and_save_image(image_path, save_path)

# print("Semua gambar telah diproses dan disimpan sebagai PNG.")

# %%
# Mengurutkan kelas berdasarkan jumlah file
sorted_class_counts = sorted(
    class_counts.items(), key=lambda x: x[1], reverse=False)
sorted_class_names = [item[0] for item in sorted_class_counts]
sorted_file_counts = [item[1] for item in sorted_class_counts]

# Setel palet warna berdasarkan jumlah kelas
palette = sns.color_palette("plasma", len(sorted_class_counts))

# Membuat plot
plt.figure(figsize=(12, 8))
plt.barh(sorted_class_names, sorted_file_counts, color=palette)
plt.xlabel('Number of Image Files')
plt.ylabel('Class Name')
plt.title('Number of Image Files for Each Class in the Dataset')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Menampilkan plot
plt.show()

# %% [markdown]
# # **Load**

# %%
# # Path ke direktori dataset asli
# original_dataset_dir = '../Dataset/preprocessed_art'

# # Path ke direktori baru untuk dataset yang sudah dipisah
# base_dir = '../Dataset/new_dir'
# os.makedirs(base_dir, exist_ok=True)

# # Membuat sub-direktori untuk data training dan validasi
# train_dir = os.path.join(base_dir, 'train')
# os.makedirs(train_dir, exist_ok=True)
# validation_dir = os.path.join(base_dir, 'validation')
# os.makedirs(validation_dir, exist_ok=True)

# # Daftar nama kelas (nama direktori di dalam original_dataset_dir)
# class_names = os.listdir(original_dataset_dir)

# # Memisahkan data menjadi bagian training dan validation
# for class_name in class_names:
#     # Path ke direktori kelas di dataset asli
#     class_original_dir = os.path.join(original_dataset_dir, class_name)

#     # Path ke direktori kelas di dataset baru untuk training dan validasi
#     class_train_dir = os.path.join(train_dir, class_name)
#     os.makedirs(class_train_dir, exist_ok=True)
#     class_validation_dir = os.path.join(validation_dir, class_name)
#     os.makedirs(class_validation_dir, exist_ok=True)

#     # Memisahkan path gambar menjadi bagian training dan validasi
#     image_paths = [os.path.join(class_original_dir, image_name)
#                    for image_name in os.listdir(class_original_dir)]
#     train_image_paths, validation_image_paths = train_test_split(
#         image_paths, test_size=0.2, random_state=42)

#     # Menyalin gambar ke direktori training
#     for image_path in train_image_paths:
#         image_name = os.path.basename(image_path)
#         target_path = os.path.join(class_train_dir, image_name)
#         copyfile(image_path, target_path)

#     # Menyalin gambar ke direktori validation
#     for image_path in validation_image_paths:
#         image_name = os.path.basename(image_path)
#         target_path = os.path.join(class_validation_dir, image_name)
#         copyfile(image_path, target_path)

# print("Pemisahan dataset selesai.")

# %%
# Path ke direktori train dan validasi
train_dir = '../Dataset/new_dir/train'
validation_dir = '../Dataset/new_dir/validation'

# Fungsi untuk menghitung jumlah file dalam direktori


def count_files_in_directory(directory):
    return sum(len(files) for _, _, files in os.walk(directory))


# Menghitung jumlah file dalam direktori train dan validasi
num_train_files = count_files_in_directory(train_dir)
num_validation_files = count_files_in_directory(validation_dir)

print(f"Jumlah file dalam direktori train: {num_train_files}")
print(f"Jumlah file dalam direktori validation: {num_validation_files}")

# %%
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.2],
    channel_shift_range=0.1,
    fill_mode='nearest'
)

train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_dataset = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# %%
class_indices = train_dataset.class_indices
num_classes = len(class_indices)
print(f"Jumlah kelas: {num_classes}")
print("Label kelas:", class_indices)

# %%
class_indices = validation_dataset.class_indices
num_classes = len(class_indices)
print(f"Jumlah kelas: {num_classes}")
print("Label kelas:", class_indices)

# %%
# Menampilkan beberapa contoh gambar


def plot_images(images_arr, labels_arr, class_indices, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(20, 20))
    axes = axes.flatten()
    for img, lbl, ax in zip(images_arr, labels_arr, axes):
        ax.imshow(img)
        label = list(class_indices.keys())[
            list(class_indices.values()).index(np.argmax(lbl))]
        ax.set_title(label)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# %%
# Mendapatkan batch gambar dan label
images, labels = next(train_dataset)

# Menampilkan 5 contoh gambar
plot_images(images[:5], labels[:5], class_indices, num_images=5)

# %%
# Mendapatkan batch gambar dan label
images, labels = next(validation_dataset)

# Menampilkan 5 contoh gambar
plot_images(images[:5], labels[:5], class_indices, num_images=5)

# %% [markdown]
# #### Imagenet Exception

# %%
base_model = tf.keras.applications.Xception(
    include_top=False,
    weights='imagenet',
    input_shape=(299, 299, 3),
    pooling='avg'
)

for layer in base_model.layers:
    layer.trainable = False

model = base_model.output
model = Dense(128, activation='relu')(model)
model = Dropout(.5)(model)
model = Dense(256, activation='relu')(model)
model = Dropout(.5)(model)
model = Dense(512, activation='relu')(model)
model = Dropout(.5)(model)
output_layer = Dense(num_classes, activation='softmax')(model)

model = tf.keras.models.Model(inputs=base_model.input, outputs=output_layer)

# %%
model.compile(optimizer=RMSprop(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
# Buat model checkpoint
checkpoint = ModelCheckpoint(
    '../Code/best_model.h5',              # Nama file untuk menyimpan model
    monitor='val_accuracy',       # Metrik yang dipantau
    # Verbosity mode, 1 = menampilkan pesan saat checkpoint disimpan
    verbose=1,
    save_best_only=True,          # Hanya menyimpan model terbaik
    mode='max',                   # Mode untuk metrik yang dipantau, 'max' untuk akurasi
    # Menyimpan seluruh model (False) atau hanya bobot (True)
    save_weights_only=False,
    # Frekuensi penyimpanan checkpoint, 'epoch' untuk setiap akhir epoch
    save_freq='epoch'
)

# %%
# Train the model with checkpoint callback
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint],
    verbose=1
)

# %%
model.summary()

# %%
# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.show()

# %%
val_loss, val_accuracy = model.evaluate(validation_dataset)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# %%
# Save the model to a .h5 file
model.save('../Code/genre_classification_84.h5')

# %%
# Simpan model ke format .keras
model.save('../Code/genre_classification_84.keras', save_format='keras')

# %%
for class_name in train_dataset.class_indices:
    print(class_name)

# %%
id2label = dict((v, k) for k, v in train_dataset.class_indices.items())
id2label

# %%
model = tf.keras.models.load_model('../Code/genre_classification_84.keras')

# %%
y_actual = []
y_pred = []

for class_name in train_dataset.class_indices:
    path_folder = os.path.join(validation_dir, class_name)

    for filename in os.listdir(path_folder):
        file_ = os.path.join(path_folder, filename)
        img = cv.imread(file_)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (299, 299))
        prediction = model.predict(np.array([img])/255)
        index = np.argmax(prediction)

        y_actual.append(class_name)
        y_pred.append(id2label[index])

print(y_actual)
print(y_pred)

# %%
print(classification_report(y_actual, y_pred))

# %%


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap='terrain_r'):
    if normalize:
        cm = cm.astype('float64') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")

    else:
        print("Confusion matrix")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, weight='bold', fontsize=16)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = ".2f" if normalize else 'd'
    thresh = cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                 fontsize=12, weight='bold', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.xlabel('Predicted label', fontsize=16)


# %%
classes = [class_name for class_name in train_dataset.class_indices]
classes

# %%
cnf_matrix = confusion_matrix(y_actual, y_pred)
np.set_printoptions(precision=2)

plt.figure(figsize=(10, 10))
plt.xticks(rotation=90)
plot_confusion_matrix(cnf_matrix, classes=classes,
                      normalize=True, cmap='terrain_r')
plt.savefig("../Images/confusion-matrix-val.png")

# %% [markdown]
# **PREDIKSI GAMBAR DI VALIDATION IMAGES**

# %%
class_indices = validation_dataset.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}

# %%


def display_images(images, predictions, true_labels, title, class_names, num_images=9):
    plt.figure(figsize=(16, 16))
    for i in range(num_images):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        color = 'green' if predictions[i] == true_labels[i] else 'red'
        plt.title("True: {}\nPred: {}".format(
            class_names[true_labels[i]], class_names[predictions[i]]), color=color)
        plt.axis("off")
    plt.suptitle(title, size=20)
    plt.show()

# %%
# Kumpulkan gambar, prediksi, dan label sebenarnya dari dataset validasi


def collect_validation_data(num_images):
    validation_images = []
    validation_labels = []

    # Reset iterator sebelum loop
    validation_dataset.reset()

    # Collect a manageable number of images and labels
    for i in range(num_images):
        images, labels = validation_dataset.next()
        validation_images.append(images[0])
        validation_labels.append(np.argmax(labels, axis=-1)[0])

    validation_images = np.array(validation_images)
    validation_labels = np.array(validation_labels)

    return validation_images, validation_labels


# Tentukan jumlah gambar yang ingin diplot
num_images = 6  # Ubah nilai ini sesuai kebutuhan

# Kumpulkan data validasi
validation_images, validation_labels = collect_validation_data(num_images)

# Prediksi
probabilities = model.predict(validation_images, batch_size=num_images)
predicted_classes = np.argmax(probabilities, axis=1)

# Tampilkan gambar dengan prediksi dan label sebenarnya
display_images(validation_images, predicted_classes,
               validation_labels, "Validation Images Predictions", idx_to_class, num_images)

# %%
# Load the model (replace 'your_model.h5' with your actual model file)
model = load_model('../Code/genre_classification_84.h5')

# Function to load class labels from the train_generator


def load_class_labels(generator):
    class_indices = generator.class_indices
    return {v: k for k, v in class_indices.items()}


# Assuming 'validation_dataset' is already defined and used to load the class labels
class_labels = load_class_labels(validation_dataset)

# Specify the directory containing images
image_directory = '..\test_images'

# Loop through the files in the directory
for filename in os.listdir(image_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Predicting images
        path = os.path.join(image_directory, filename)
        img = load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        x = img_to_array(img)
        x /= 255.0
        x = np.expand_dims(x, axis=0)

        image_tensor = np.vstack([x])
        classes = model.predict(image_tensor)

        # Get the indices of the top 3 classes with the highest probabilities
        top_3_indices = np.argsort(classes[0])[-3:][::-1]

        print(f"Predictions for {filename}:")
        for i in top_3_indices:
            class_name = class_labels[i]
            probability = classes[0][i]
            print(f"Class: {class_name}, Probability: {probability:.2f}")

# %% [markdown]
# **UNGGAH FILE DI GOOGLE COLAB**

# %%
# import numpy as np
# from google.colab import files
# from tensorflow.keras.utils import load_img, img_to_array

# # Fungsi untuk memuat label kelas dari train_generator


# def load_class_labels(generator):
#     class_indices = generator.class_indices
#     return {v: k for k, v in class_indices.items()}


# class_labels = load_class_labels(validation_dataset)

# # Mengunggah file
# uploaded = files.upload()

# for fn in uploaded.keys():

#     # predicting images
#     path = '/content/' + fn
#     img = load_img(path, target_size=(150, 150))
#     x = img_to_array(img)
#     x /= 255.0
#     x = np.expand_dims(x, axis=0)

#     image_tensor = np.vstack([x])
#     classes = model.predict(image_tensor)

#     # Mendapatkan indeks kelas dengan probabilitas tertinggi
#     top_3_indices = np.argsort(classes[0])[-3:][::-1]

#     print(f"Predictions for {fn}:")
#     for i in top_3_indices:
#         class_name = class_labels[i]
#         probability = classes[0][i]
#         print(f"Class: {class_name}, Probability: {probability:.2f}")

# %%
# # Ekspor daftar paket yang aktif ke file requirements.txt
# !pip freeze > requirements.txt

# # Verifikasi bahwa file telah dibuat dengan benar
# with open('requirements.txt', 'r') as file:
#     print(file.read())
