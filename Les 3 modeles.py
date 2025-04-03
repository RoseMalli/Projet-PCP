import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from tensorflow import keras
from sklearn.model_selection import train_test_split

# 🔹 Chargement des données à partir du fichier CSV
dataset = pd.read_csv('user_ratings_genres_mov.csv')

# Afficher un aperçu des premières lignes du dataset pour vérifier
print(dataset.head())

# --------- SVD Model ---------
# Création de la matrice utilisateur-film
user_item_matrix = dataset.pivot_table(index='userId', columns='title', values='rating')
user_item_matrix = user_item_matrix.fillna(0)

# Appliquer la décomposition SVD
svd = TruncatedSVD(n_components=2)
svd_matrix = svd.fit_transform(user_item_matrix)

# Reconstitution de la matrice des prédictions
pred_matrix = np.dot(svd_matrix, svd.components_)

# Erreur MSE pour SVD
mse = mean_squared_error(user_item_matrix, pred_matrix)
print(f"Erreur Quadratique Moyenne (SVD): {mse:.2f}")

# --------- KNN Model ---------
# Appliquer KNN
knn = NearestNeighbors(metric='cosine', algorithm='brute')

# Ajuster le nombre de voisins pour correspondre au nombre d'échantillons disponibles
n_neighbors = min(5, len(user_item_matrix))

knn.fit(user_item_matrix)

# Sélection d'un utilisateur pour les recommandations KNN
user_input = 999  # Utilisateur fictif
if user_input in user_item_matrix.index:
    distances, indices = knn.kneighbors([user_item_matrix.loc[user_input]], n_neighbors=n_neighbors)

    # Affichage des recommandations KNN
    recommended_movies = user_item_matrix.iloc[indices[0]].mean().sort_values(ascending=False).index[:5]
    print(f"Films recommandés par KNN pour l'utilisateur {user_input}: {recommended_movies.tolist()}")
else:
    print(f"L'utilisateur {user_input} n'existe pas dans le dataset.")

# --------- NFM Model ---------
# Encodage des utilisateurs et des films
user_ids = dataset["userId"].astype("category").cat.codes.values
movie_ids = dataset["title"].astype("category").cat.codes.values
ratings = dataset["rating"].values

# Séparer les données en train et test
X_train, X_test, y_train, y_test = train_test_split(
    np.vstack((user_ids, movie_ids)).T, ratings, test_size=0.2, random_state=42
)

# Définition du modèle NFM
embedding_size = 10
num_users = len(dataset["userId"].unique())
num_movies = len(dataset["title"].unique())

user_input_keras = keras.layers.Input(shape=(1,))
movie_input_keras = keras.layers.Input(shape=(1,))

user_embedding = keras.layers.Embedding(num_users, embedding_size)(user_input_keras)
movie_embedding = keras.layers.Embedding(num_movies, embedding_size)(movie_input_keras)

dot_product = keras.layers.Dot(axes=1)([user_embedding, movie_embedding])
output = keras.layers.Flatten()(dot_product)

model = keras.Model(inputs=[user_input_keras, movie_input_keras], outputs=output)
model.compile(optimizer="adam", loss="mse")

# Entraînement du modèle NFM
model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=5, verbose=1)

# Prédiction de la note pour un exemple utilisateur-film
user_input = 999  # Utilisateur fictif
movie_input = 'Inception'  # Film fictif
if user_input in dataset["userId"].values and movie_input in dataset["title"].values:
    user_idx = np.where(user_ids == user_input)[0][0]  # Index de l'utilisateur encodé
    movie_idx = np.where(movie_ids == movie_input)[0][0]  # Index du film encodé

    # Prédiction de la note
    pred_rating = model.predict([[user_idx], [movie_idx]])[0][0]
    print(f"Note prédite pour l'utilisateur {user_input} et le film '{movie_input}': {pred_rating:.2f}")
else:
    print(f"L'utilisateur {user_input} ou le film '{movie_input}' n'existe pas dans le dataset.")
