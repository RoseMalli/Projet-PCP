import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Pour les modèles
from surprise import SVD, NMF, KNNBasic, Dataset, Reader

# === Chargement des données ===
df = pd.read_csv("user_ratings_genres_mov.csv")
df['genres'] = df['genres'].str.split('|')
df_exploded = df.explode('genres').drop_duplicates(subset=['title', 'genres'])
movie_cross_tab = pd.crosstab(df_exploded['title'], df_exploded['genres'])

# Jaccard pour contenu
jaccard_distances = pdist(movie_cross_tab, metric='jaccard')
jaccard_similarity_array = 1 - squareform(jaccard_distances)
jaccard_similarity_df = pd.DataFrame(jaccard_similarity_array, index=movie_cross_tab.index, columns=movie_cross_tab.index)

# === Interface utilisateur ===
st.title("🎬 Système de Recommandation de Films")

user_id = st.text_input("🔹 Votre ID utilisateur :", key="user_id")
movies_list = sorted(df['title'].unique())

col1, col2, col3 = st.columns(3)
with col1:
    film1 = st.selectbox("Film 1", movies_list)
    note1 = st.slider("Note", 0.5, 5.0, 3.0, 0.5)
with col2:
    film2 = st.selectbox("Film 2", [m for m in movies_list if m != film1])
    note2 = st.slider("Note ", 0.5, 5.0, 3.0, 0.5)
with col3:
    film3 = st.selectbox("Film 3", [m for m in movies_list if m not in [film1, film2]])
    note3 = st.slider("Note  ", 0.5, 5.0, 3.0, 0.5)

if st.button("📥 Obtenir des recommandations"):

    # Ajouter l'utilisateur
    genre1 = df[df['title'] == film1].iloc[0]['genres']
    genre2 = df[df['title'] == film2].iloc[0]['genres']
    genre3 = df[df['title'] == film3].iloc[0]['genres']
    new_user_id = "user_999"
    new_data = [
        {"userId": new_user_id, "title": film1, "rating": note1, "genres": genre1},
        {"userId": new_user_id, "title": film2, "rating": note2, "genres": genre2},
        {"userId": new_user_id, "title": film3, "rating": note3, "genres": genre3},
    ]
    df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)

    # Matrice utilisateur-film
    user_item = df.pivot_table(index='userId', columns='title', values='rating')
    user_means = user_item.mean(axis=1, skipna=True)
    user_mean = user_means[new_user_id]

    # === Méthode User-User ===
    similarities = {}
    for user in user_item.index:
        if user == new_user_id:
            continue
        common = user_item.loc[new_user_id].notna() & user_item.loc[user].notna()
        if common.sum() >= 2:
            u1 = user_item.loc[new_user_id, common] - user_mean
            u2 = user_item.loc[user, common] - user_means[user]
            num = (u1 * u2).sum()
            den = np.sqrt((u1**2).sum() * (u2**2).sum())
            if den != 0:
                similarities[user] = num / den
    neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]

    user_user_pred = {}
    for movie in user_item.columns:
        if pd.isna(user_item.at[new_user_id, movie]):
            total, weight = 0, 0
            for u, sim in neighbors:
                if not pd.isna(user_item.at[u, movie]):
                    total += sim * (user_item.at[u, movie] - user_means[u])
                    weight += abs(sim)
            if weight > 0:
                user_user_pred[movie] = user_mean + total / weight

    st.subheader("🤝 Recommandation User-User")
    for title, score in sorted(user_user_pred.items(), key=lambda x: x[1], reverse=True)[:5]:
        st.write(f"🎥 **{title}** - {score:.2f}★")

    # === Méthode Item-Item ===
    def item_sim(i1, i2):
        users_i1 = set(df[df['title'] == i1]['userId'])
        users_i2 = set(df[df['title'] == i2]['userId'])
        common_users = users_i1 & users_i2
        if len(common_users) < 2:
            return 0
        d1, d2 = [], []
        for u in common_users:
            r1 = user_item.at[u, i1]
            r2 = user_item.at[u, i2]
            d1.append(r1 - user_means[u])
            d2.append(r2 - user_means[u])
        d1, d2 = np.array(d1), np.array(d2)
        return (d1 * d2).sum() / (np.sqrt((d1**2).sum()) * np.sqrt((d2**2).sum())) if np.sqrt((d1**2).sum()) and np.sqrt((d2**2).sum()) else 0

    item_item_pred = {}
    for movie in user_item.columns:
        if pd.isna(user_item.at[new_user_id, movie]):
            total, sim_sum = 0, 0
            for f, r in zip([film1, film2, film3], [note1, note2, note3]):
                sim = item_sim(movie, f)
                if sim > 0:
                    total += sim * (r - user_mean)
                    sim_sum += sim
            if sim_sum > 0:
                item_item_pred[movie] = user_mean + total / sim_sum

    st.subheader("🎞️ Recommandation Item-Item")
    for title, score in sorted(item_item_pred.items(), key=lambda x: x[1], reverse=True)[:5]:
        st.write(f"🎥 **{title}** - {score:.2f}★")

    # === Méthode par contenu (genres) ===
    best_film = max([(film1, note1), (film2, note2), (film3, note3)], key=lambda x: x[1])[0]
    jaccard_series = jaccard_similarity_df.loc[best_film]
    similar = jaccard_series.drop([film1, film2, film3]).sort_values(ascending=False).head(5)
    avg_ratings = df.groupby("title")["rating"].mean().round(2)
    st.subheader(f"🎭 Recommandation par contenu : similaires à '{best_film}'")
    for title in similar.index:
        st.write(f"🎥 **{title}** - Similarité : {similar[title]*100:.1f}% - Note moy. : {avg_ratings[title]:.2f}★")

    # === Méthodes par modèles (SVD, NMF, KNN) ===
    reader = Reader(rating_scale=(0.5, 5.0))
    data_surprise = Dataset.load_from_df(df[['userId', 'title', 'rating']], reader)
    trainset = data_surprise.build_full_trainset()
    for model_class, model_name in zip([SVD, NMF, KNNBasic], ["SVD", "NMF", "KNN"]):
        model = model_class()
        model.fit(trainset)
        unwatched = [m for m in df['title'].unique() if m not in [film1, film2, film3]]
        predictions = [(m, model.predict(new_user_id, m).est) for m in unwatched]
        top = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
        st.subheader(f"🧠 Recommandation par modèle : {model_name}")
        for title, score in top:
            st.write(f"🎥 **{title}** - {score:.2f}★")
