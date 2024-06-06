# %%
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud
from datetime import datetime
from ast import literal_eval
from typing import Dict, Text
from collections import Counter
import tensorflow_recommenders as tfrs
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
import string
# !pip install - q tensorflow-recommenders

# %%

warnings.filterwarnings('ignore')

# %%
rating_df = pd.read_csv('..\Dataset\dataset_dummy.csv')
rating_df.head(100)

# %%
rating_df.info()

# %%
# Melihat jumlah masing-masing nilai unik dalam kolom "genre_seni"
genre_counts = rating_df['genre_seni'].value_counts()

# Membuat dataframe dari nilai unik dan jumlah kemunculannya
genre_counts_df = pd.DataFrame(genre_counts.items(), columns=[
                               'Genre Seni', 'Jumlah'])

# Mencetak dataframe
print(genre_counts_df)

# %%
# Menghitung jumlah unique user_id
jumlah_unique_user = rating_df['user_id'].nunique()

# Mencetak jumlah unique user
print("Jumlah Unique User:", jumlah_unique_user)

# %%
# Menghitung jumlah masing-masing rating dan mengurutkannya secara ascending
jumlah_rating = rating_df['rating'].value_counts().sort_index()

# Mencetak jumlah masing-masing rating yang sudah diurutkan secara ascending
print("Jumlah Masing-masing Rating:")
print(jumlah_rating)

# %%
genre_seni_df = rating_df[['genre_seni_id', 'genre_seni']]

# %%
unique_genre_ids = genre_seni_df['genre_seni_id'].unique()

# %%
unique_genres = genre_seni_df['genre_seni'].unique()

# %%
genre_df = pd.DataFrame(
    {'genre_seni_id': unique_genre_ids, 'genre_seni': unique_genres})
genre_df = genre_df.sort_values('genre_seni_id')
genre_df.head(10)

# %%
rating_df.head(10)

# %%
rating_df['user_id'] = rating_df['user_id'].astype(str)
genre_df['genre_seni_id'] = rating_df['genre_seni_id'].astype(str)

ratings = tf.data.Dataset.from_tensor_slices(
    dict(rating_df[['user_id', 'genre_seni', 'rating']]))

genres = tf.data.Dataset.from_tensor_slices(
    dict(genre_df[['genre_seni']]))

ratings = ratings.map(lambda x: {
    "genre_seni": x["genre_seni"],
    "user_id": x["user_id"],
    "rating": int(x["rating"])
})

genres = genres.map(lambda x: x["genre_seni"])

# %%
rating_df.info()

# %%
genre_df.info()

# %%
# Set the seed for reproducibility
tf.random.set_seed(16)
# shuffled = ratings.shuffle(16, seed=8, reshuffle_each_iteration=False)

# Calculate the size of the training and testing sets
train_size = int(len(ratings) * 0.8)
test_size = len(ratings) - train_size

# Split the dataset
train = ratings.take(train_size)
test = ratings.skip(train_size).take(test_size)

# Print the sizes of the training and testing sets
print('Training set size:', len(train))
print('Testing set size:', len(test))

# %%
genre_seni = genres.batch(100)
user_ids = ratings.batch(100).map(lambda x: x["user_id"])

unique_genre_titles = np.unique(np.concatenate(list(genre_seni)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

print('Unique Movies: {}'.format(len(unique_genre_titles)))
print('Unique users: {}'.format(len(unique_user_ids)))

# %%


class GenreModels(tfrs.models.Model):

    def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
        # We take the loss weights in the constructor: this allows us to instantiate
        # several model objects with different loss weights.

        super().__init__()

        embedding_dimension = 64

        # User and movie models.
        self.genre_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_genre_titles, mask_token=None),
            tf.keras.layers.Embedding(
                len(unique_genre_titles) + 1, embedding_dimension)
        ])
        self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(
                len(unique_user_ids) + 1, embedding_dimension)
        ])

        # A small model to take in user and movie embeddings and predict ratings.
        # We can make this as complicated as we want as long as we output a scalar
        # as our prediction.
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        # The tasks.
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=genres.batch(128).map(self.genre_model)
            )
        )

        # The loss weights.
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model.
        genre_embeddings = self.genre_model(features["genre_seni"])

        return (
            user_embeddings,
            genre_embeddings,
            # We apply the multi-layered rating model to a concatentation of
            # user and movie embeddings.
            self.rating_model(
                tf.concat([user_embeddings, genre_embeddings], axis=1)
            ),
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        ratings = features.pop("rating")

        user_embeddings, genre_embeddings, rating_predictions = self(features)

        # We compute the loss for each task.
        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating_predictions,
        )
        retrieval_loss = self.retrieval_task(user_embeddings, genre_embeddings)

        # And combine them using the loss weights.
        return (self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss)


# %%
model = GenreModels(rating_weight=1.0, retrieval_weight=1.0)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

cached_train = train.shuffle(16).batch(100).cache()
cached_test = test.batch(100).cache()

model.fit(cached_train, epochs=100)

# %%
metrics = model.evaluate(cached_test, return_dict=True)

print(
    f"\nRetrieval top-5 accuracy: {metrics['factorized_top_k/top_5_categorical_accuracy'] * 100:.2f}%")
print(f"RMSE: {metrics['root_mean_squared_error']:.2f}")

# %%


def predict_genres(user, top_n=5):
    # Create a model that takes in raw query features, and
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

    # Recommend genres out of the entire genres dataset.
    index.index_from_dataset(
        tf.data.Dataset.zip(
            (genres.batch(100), genres.batch(100).map(model.genre_model)))
    )

    # Ensure k doesn't exceed the number of unique genres
    num_genres = len(unique_genre_titles)
    k = min(top_n, num_genres)

    # Get recommendations.
    _, titles = index(tf.constant([str(user)]), k=k)

    print(f'Top {top_n} recommendations genre for user {user}:\n')
    for i, title in enumerate(titles[0, :top_n].numpy()):
        print(f'{i+1}. {title.decode("utf-8")}')


def predict_rating(user, genre):
    trained_genre_embeddings, trained_user_embeddings, predicted_rating = model({
        "user_id": np.array([str(user)]),
        "genre_title": np.array([genre])
    })
    print(f"Predicted rating for {genre}: {predicted_rating.numpy()[0][0]}")


# %%
rating_df[rating_df['user_id'] == '1']

# %%
predict_genres(1, 5)

# %%
model.save_weights('tfrs.h5')

# %%
# Load the model weights
model.load_weights('tfrs.h5')

# Define genres dataset and unique_genre_titles if not already defined
# Example placeholder (you need to replace this with actual data)
unique_genre_titles = np.unique(np.concatenate(list(genre_seni)))

# Create a tf.data.Dataset of genres
genres = tf.data.Dataset.from_tensor_slices(unique_genre_titles)


def predict_genres(user, top_n=5):
    # Create a model that takes in raw query features, and
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

    # Recommend genres out of the entire genres dataset.
    index.index_from_dataset(
        tf.data.Dataset.zip(
            (genres.batch(100), genres.batch(100).map(model.genre_model)))
    )

    # Ensure k doesn't exceed the number of unique genres
    num_genres = len(unique_genre_titles)
    k = min(top_n, num_genres)

    # Get recommendations.
    _, titles = index(tf.constant([str(user)]), k=k)

    print(f'Top {top_n} recommendations genre for user {user}:\n')
    for i, title in enumerate(titles[0, :top_n].numpy()):
        print(f'{i+1}. {title.decode("utf-8")}')


# Example usage:
# Replace 'example_user_id' with the actual user ID you want to predict for.
predict_genres('1', top_n=5)

# %%

# Load the model weights
model.load_weights('tfrs.h5')

# Define genres dataset and unique_genre_titles if not already defined
# Example placeholder (replace this with actual data)
unique_genre_titles = np.unique(np.concatenate(list(genre_seni)))

# Create a tf.data.Dataset of genres
genres = tf.data.Dataset.from_tensor_slices(unique_genre_titles)


def predict_genres(user, top_n=5):
    # Create a model that takes in raw query features, and
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

    # Recommend genres out of the entire genres dataset.
    index.index_from_dataset(
        tf.data.Dataset.zip(
            (genres.batch(100), genres.batch(100).map(model.genre_model)))
    )

    # Ensure k doesn't exceed the number of unique genres
    num_genres = len(unique_genre_titles)
    k = min(top_n, num_genres)

    # Get recommendations.
    _, titles = index(tf.constant([str(user)]), k=k)

    print(f'Top {top_n} recommended genres for user {user}:\n')
    for i, title in enumerate(titles[0, :top_n].numpy()):
        print(f'{i+1}. {title.decode("utf-8")}')


# Interactive loop to get user input
while True:
    user_id = input("Enter user ID (or 'exit' to quit): ")
    if user_id.lower() == 'exit':
        break
    top_n = input("Enter the number of top genres to recommend: ")
    try:
        top_n = int(top_n)
    except ValueError:
        print("Invalid input for number of genres. Please enter an integer.")
        continue

    predict_genres(user_id, top_n)
