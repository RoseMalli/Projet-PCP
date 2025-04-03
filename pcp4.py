import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from surprise import SVD, NMF, KNNBasic, Dataset, Reader

# === Chargement des données ===
df = pd.read_csv("user_ratings_genres_mov.csv")
df['genres'] = df['genres'].str.split('|')
df_exploded = df.explode('genres').drop_duplicates(subset=['title', 'genres'])
movie_cross_tab = pd.crosstab(df_exploded['title'], df_exploded['genres'])

# Calcul des similarités Jaccard
jaccard_distances = pdist(movie_cross_tab, metric='jaccard')
jaccard_similarity_array = 1 - squareform(jaccard_distances)
jaccard_similarity_df = pd.DataFrame(jaccard_similarity_array, index=movie_cross_tab.index, columns=movie_cross_tab.index)

# === Interface utilisateur ===
st.set_page_config(page_title="🎬 Recommandation de Films", layout="centered")
st.title("🎬 Système de Recommandation de Films")
st.caption("Notez 3 films vus pour obtenir des recommandations personnalisées (collaboratif, contenu et modèles).")

# === Formulaire utilisateur ===
movies = sorted(df['title'].unique())
with st.form("form_notes"):
    col1, col2, col3 = st.columns(3)
    with col1:
        film1 = st.selectbox("🎥 Film 1", movies)
        note1 = st.slider("⭐ Note 1", 0.5, 5.0, 3.0, 0.5)
    with col2:
        film2 = st.selectbox("🎥 Film 2", [m for m in movies if m != film1])
        note2 = st.slider("⭐ Note 2", 0.5, 5.0, 3.0, 0.5)
    with col3:
        film3 = st.selectbox("🎥 Film 3", [m for m in movies if m not in [film1, film2]])
        note3 = st.slider("⭐ Note 3", 0.5, 5.0, 3.0, 0.5)
    submit = st.form_submit_button("📥 Obtenir des recommandations")

# === Lancement de la recommandation ===
if submit:
    new_user_id = "user_999"
    films_vus = [film1, film2, film3]
    df = pd.concat([
        df,
        pd.DataFrame([
            {"userId": new_user_id, "title": film1, "rating": note1, "genres": df[df['title'] == film1].iloc[0]['genres']},
            {"userId": new_user_id, "title": film2, "rating": note2, "genres": df[df['title'] == film2].iloc[0]['genres']},
            {"userId": new_user_id, "title": film3, "rating": note3, "genres": df[df['title'] == film3].iloc[0]['genres']}
        ])
    ], ignore_index=True)

    # Matrice utilisateur-film
    user_item = df.pivot_table(index='userId', columns='title', values='rating')
    user_means = user_item.mean(axis=1)
    user_mean = user_means[new_user_id]

    st.header("📊 Résultats personnalisés")
    reco_global = []

    # === User-User ===
    st.subheader("🤝 Recommandation User-User")
    similarities = {}
    for user in user_item.index:
        if user == new_user_id: continue
        common = user_item.loc[new_user_id].notna() & user_item.loc[user].notna()
        if common.sum() >= 2:
            u1 = user_item.loc[new_user_id, common] - user_mean
            u2 = user_item.loc[user, common] - user_means[user]
            num = (u1 * u2).sum()
            den = np.sqrt((u1**2).sum() * (u2**2).sum())
            if den: similarities[user] = num / den
    neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
    for movie in user_item.columns.difference(films_vus):
        total = weight = 0
        for user, sim in neighbors:
            if not pd.isna(user_item.at[user, movie]):
                total += sim * (user_item.at[user, movie] - user_means[user])
                weight += abs(sim)
        if weight:
            score = user_mean + total / weight
            reco_global.append(("User-User", movie, round(score, 2)))
    for method, title, score in sorted([r for r in reco_global if r[0] == "User-User"], key=lambda x: x[2], reverse=True)[:5]:
        st.markdown(f"- 🎬 **{title}** – {score}★")

    # === Item-Item ===
    st.subheader("🎞️ Recommandation Item-Item")
    def item_sim(i1, i2):
        users_i1 = set(df[df['title'] == i1]['userId'])
        users_i2 = set(df[df['title'] == i2]['userId'])
        common_users = users_i1 & users_i2
        if len(common_users) < 2: return 0
        d1, d2 = [], []
        for u in common_users:
            d1.append(user_item.at[u, i1] - user_means[u])
            d2.append(user_item.at[u, i2] - user_means[u])
        d1, d2 = np.array(d1), np.array(d2)
        return (d1 * d2).sum() / (np.sqrt((d1**2).sum()) * np.sqrt((d2**2).sum())) if np.sqrt((d1**2).sum()) else 0

    for movie in user_item.columns.difference(films_vus):
        total = sim_sum = 0
        for f, r in zip(films_vus, [note1, note2, note3]):
            sim = item_sim(movie, f)
            if sim > 0:
                total += sim * (r - user_mean)
                sim_sum += sim
        if sim_sum:
            score = user_mean + total / sim_sum
            reco_global.append(("Item-Item", movie, round(score, 2)))
    for method, title, score in sorted([r for r in reco_global if r[0] == "Item-Item"], key=lambda x: x[2], reverse=True)[:5]:
        st.markdown(f"- 🎬 **{title}** – {score}★")

    # === Contenu (Jaccard) ===
    st.subheader("🎭 Recommandation par contenu")
    best_rated = max([(film1, note1), (film2, note2), (film3, note3)], key=lambda x: x[1])[0]
    jaccard_series = jaccard_similarity_df[best_rated].drop(index=films_vus)
    for title, score in jaccard_series.sort_values(ascending=False).head(5).items():
        reco_global.append(("Contenu", title, round(score * 100, 1)))
        st.markdown(f"- 🎬 **{title}** – Similarité : {score * 100:.1f}%")

    # === Modèles SVD, NMF, KNN (Surprise) ===
    st.subheader("🧠 Recommandation par modèles")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[['userId', 'title', 'rating']], reader)
    trainset = data.build_full_trainset()
    for model_class, model_name in zip([SVD, NMF, KNNBasic], ["SVD", "NMF", "KNN"]):
        model = model_class()
        model.fit(trainset)
        unwatched = [m for m in df['title'].unique() if m not in films_vus]
        predictions = [(m, model.predict(new_user_id, m).est) for m in unwatched]
        for title, score in sorted(predictions, key=lambda x: x[1], reverse=True)[:5]:
            reco_global.append((model_name, title, round(score, 2)))
            st.markdown(f"🔸 **{model_name}** → 🎬 **{title}** – {score:.2f}★")

    # === Export CSV ===
    st.subheader("📁 Exporter vos recommandations")
    if reco_global:
        reco_df = pd.DataFrame(reco_global, columns=["Méthode", "Film", "Score"])
        csv = reco_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Télécharger le fichier CSV", data=csv, file_name="recommandations.csv", mime="text/csv")
