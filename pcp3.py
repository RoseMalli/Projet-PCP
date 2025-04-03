import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from tensorflow import keras

# ======= Chargement des données =======
df = pd.read_csv("user_ratings_genres_mov.csv")
user_item_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# ======= Configuration de la page =======
st.set_page_config(page_title="Recommandation de Films", layout="centered")
st.title("🎬 Recommandation de Films")
st.caption("Choisissez 3 films que vous avez aimés et obtenez des recommandations personnalisées.")

# ======= Formulaire utilisateur =======
films = sorted(df['title'].unique())
with st.form("formulaire"):
    film1 = st.selectbox("🎥 Film 1", films)
    note1 = st.slider("⭐ Note 1", 0.5, 5.0, 3.0, 0.5)

    film2 = st.selectbox("🎥 Film 2", [f for f in films if f != film1])
    note2 = st.slider("⭐ Note 2", 0.5, 5.0, 3.0, 0.5)

    film3 = st.selectbox("🎥 Film 3", [f for f in films if f not in [film1, film2]])
    note3 = st.slider("⭐ Note 3", 0.5, 5.0, 3.0, 0.5)

    submitted = st.form_submit_button("🎯 Obtenir mes recommandations")

# ======= Si le formulaire est validé =======
if submitted:
    new_user_id = 999
    already_rated = [film1, film2, film3]

    # Ajout de l'utilisateur dans le DataFrame
    for film, note in zip(already_rated, [note1, note2, note3]):
        genre = df[df['title'] == film].iloc[0]['genres']
        df = pd.concat([df, pd.DataFrame([{
            'userId': new_user_id, 'title': film, 'rating': note, 'genres': genre
        }])], ignore_index=True)

    user_item_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)

    st.subheader("🔍 Suggestions personnalisées")

    # ======= Méthode SVD =======
    st.markdown("**🔹 SVD (Réduction de dimensions)**")
    svd = TruncatedSVD(n_components=20)
    svd_matrix = svd.fit_transform(user_item_matrix)
    pred_matrix = np.dot(svd_matrix, svd.components_)
    svd_preds = pd.Series(pred_matrix[user_item_matrix.index.get_loc(new_user_id)], index=user_item_matrix.columns)
    top_svd = svd_preds.drop(index=already_rated).sort_values(ascending=False).head(3)
    st.write(top_svd.round(2).to_frame("Score estimé"))

    # ======= Méthode KNN =======
    st.markdown("**🔹 KNN (Utilisateurs similaires)**")
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_item_matrix)
    distances, indices = knn.kneighbors([user_item_matrix.loc[new_user_id]], n_neighbors=5)
    knn_scores = user_item_matrix.iloc[indices[0]].mean().sort_values(ascending=False)
    top_knn = knn_scores.drop(index=already_rated).head(3)
    st.write(top_knn.round(2).to_frame("Score moyen"))

    # ======= Méthode NFM (réseau neuronal) =======
    st.markdown("**🔹 NFM (Réseau neuronal)**")
    user_ids = df["userId"].astype("category").cat.codes.values
    movie_ids = df["title"].astype("category").cat.codes.values
    ratings = df["rating"].values

    X = np.vstack((user_ids, movie_ids)).T
    y = ratings
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_users = len(set(user_ids))
    num_movies = len(set(movie_ids))
    embed_size = 10

    input_user = keras.layers.Input(shape=(1,))
    input_movie = keras.layers.Input(shape=(1,))
    user_emb = keras.layers.Embedding(num_users, embed_size)(input_user)
    movie_emb = keras.layers.Embedding(num_movies, embed_size)(input_movie)
    dot = keras.layers.Dot(axes=1)([user_emb, movie_emb])
    output = keras.layers.Flatten()(dot)

    model = keras.Model([input_user, input_movie], output)
    model.compile(optimizer="adam", loss="mse", run_eagerly=True)
    model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=3, verbose=0)

    # Prédictions personnalisées
    user_cat = df[df['userId'] == new_user_id]["userId"].astype("category").cat.codes.values[0]
    all_movies = df["title"].astype("category").cat.categories
    movies_to_predict = all_movies.difference(already_rated)

    predictions = []
    for movie in movies_to_predict:
        movie_code = all_movies.get_loc(movie)
        pred = model.predict([[user_cat], [movie_code]], verbose=0)[0][0]
        predictions.append((movie, pred))

    top_nfm = sorted(predictions, key=lambda x: x[1], reverse=True)[:3]
    st.write(pd.DataFrame(top_nfm, columns=["Film", "Note prédite"]).set_index("Film").round(2))
