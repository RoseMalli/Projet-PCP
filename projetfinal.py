import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

# ----------- Chargement des données ----------- #
df = pd.read_csv("user_ratings_genres_mov.csv")

df['genres'] = df['genres'].str.split('|')
df_exploded = df.explode('genres').drop_duplicates(subset=['title', 'genres'])
movie_cross_tab = pd.crosstab(df_exploded['title'], df_exploded['genres'])
jaccard_distances = pdist(movie_cross_tab, metric='jaccard')
jaccard_similarity_array = 1 - squareform(jaccard_distances)
jaccard_similarity_df = pd.DataFrame(jaccard_similarity_array, index=movie_cross_tab.index, columns=movie_cross_tab.index)

# ----------- Interface Utilisateur ----------- #
st.title("🎬 Système de recommandation de films")

user_id = st.text_input("🔹 Entrez votre ID utilisateur", key="user_id")

movies_list = sorted(df['title'].unique())

col1, col2, col3 = st.columns(3)
with col1:
    film1 = st.selectbox("Film 1", movies_list)
    note1 = st.slider("Votre note", min_value=0.5, max_value=5.0, value=3.0, step=0.5)
with col2:
    options_film2 = [m for m in movies_list if m != film1]
    film2 = st.selectbox("Film 2", options_film2)
    note2 = st.slider("Votre note ", min_value=0.5, max_value=5.0, value=3.0, step=0.5)
with col3:
    options_film3 = [m for m in movies_list if m not in [film1, film2]]
    film3 = st.selectbox("Film 3", options_film3)
    note3 = st.slider("Votre note  ", min_value=0.5, max_value=5.0, value=3.0, step=0.5)

# ----------- Traitement et Recommandations ----------- #
if st.button("Obtenir des recommandations"):

    genre1 = df[df['title'] == film1].iloc[0]['genres']
    genre2 = df[df['title'] == film2].iloc[0]['genres']
    genre3 = df[df['title'] == film3].iloc[0]['genres']

    new_user_id = "user_999"
    new_ratings = [
        {"userId": new_user_id, "title": film1, "rating": note1, "genres": genre1},
        {"userId": new_user_id, "title": film2, "rating": note2, "genres": genre2},
        {"userId": new_user_id, "title": film3, "rating": note3, "genres": genre3},
    ]
    df = pd.concat([df, pd.DataFrame(new_ratings)], ignore_index=True)

    user_item_matrix = df.pivot_table(index='userId', columns='title', values='rating')
    user_means = user_item_matrix.mean(axis=1, skipna=True)
    target_user_mean = user_means[new_user_id]

    # ----------- Méthode User-User ----------- #
    similarities = {}
    for user in user_item_matrix.index:
        if user == new_user_id:
            continue
        both_rated = user_item_matrix.loc[new_user_id].notna() & user_item_matrix.loc[user].notna()
        if both_rated.sum() < 2:
            continue
        new_user_ratings = user_item_matrix.loc[new_user_id, both_rated] - target_user_mean
        other_user_ratings = user_item_matrix.loc[user, both_rated] - user_means[user]
        num = (new_user_ratings * other_user_ratings).sum()
        den = np.sqrt((new_user_ratings**2).sum() * (other_user_ratings**2).sum())
        if den == 0:
            continue
        sim = num / den
        if sim > 0:
            similarities[user] = sim

    top_neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]

    user_user_recommendations = {}
    for movie in user_item_matrix.columns:
        if not pd.isna(user_item_matrix.at[new_user_id, movie]):
            continue
        weighted_sum = 0.0
        sim_sum = 0.0
        for (neighbor, sim) in top_neighbors:
            if not pd.isna(user_item_matrix.at[neighbor, movie]):
                neighbor_rating = user_item_matrix.at[neighbor, movie]
                neighbor_mean = user_means[neighbor]
                weighted_sum += sim * (neighbor_rating - neighbor_mean)
                sim_sum += abs(sim)
        if sim_sum > 0:
            predicted_rating = target_user_mean + (weighted_sum / sim_sum)
            user_user_recommendations[movie] = predicted_rating

    # ----------- Méthode Item-Item ----------- #
    def item_similarity(item1, item2):
        users_item1 = set(df[df['title'] == item1]['userId'])
        users_item2 = set(df[df['title'] == item2]['userId'])
        common_users = users_item1 & users_item2
        if len(common_users) < 2:
            return 0
        diff1, diff2 = [], []
        for u in common_users:
            r1 = user_item_matrix.at[u, item1]
            r2 = user_item_matrix.at[u, item2]
            diff1.append(r1 - user_means[u])
            diff2.append(r2 - user_means[u])
        diff1, diff2 = np.array(diff1), np.array(diff2)
        numerator = (diff1 * diff2).sum()
        denominator = np.sqrt((diff1**2).sum() * (diff2**2).sum())
        return numerator / denominator if denominator else 0

    item_item_recommendations = {}
    for movie in user_item_matrix.columns:
        if not pd.isna(user_item_matrix.at[new_user_id, movie]):
            continue
        weighted_sum = 0.0
        sim_sum = 0.0
        for rated_movie, note in zip([film1, film2, film3], [note1, note2, note3]):
            if rated_movie == movie:
                continue
            sim = item_similarity(movie, rated_movie)
            if sim > 0:
                weighted_sum += sim * (note - target_user_mean)
                sim_sum += sim
        if sim_sum > 0:
            predicted_rating = target_user_mean + (weighted_sum / sim_sum)
            item_item_recommendations[movie] = predicted_rating

    # ----------- Recommandation par le contenu ----------- #
    best_rated_movie = max([(film1, note1), (film2, note2), (film3, note3)], key=lambda x: x[1])[0]
    jaccard_series = jaccard_similarity_df.loc[best_rated_movie]
    ordered_similarity = jaccard_series.sort_values(ascending=False)
    filtered_similarity = ordered_similarity[~ordered_similarity.index.isin([film1, film2, film3])].head(5)

    average_rating_df = df[['title', 'rating']].groupby(['title']).mean().round(2)
    content_df = pd.DataFrame({
        "Film": filtered_similarity.index,
        "Similarité (%)": (filtered_similarity.values * 100).round(1)
    }).merge(average_rating_df, left_on="Film", right_index=True, how="left").rename(columns={'rating': 'Note Moyenne'})

    # ----------- Affichage ----------- #
    st.subheader("🎯 Recommandations - Méthode User-User")
    for title, score in sorted(user_user_recommendations.items(), key=lambda x: x[1], reverse=True)[:5]:
        st.write(f"- **{title}** (score prédictif : {score:.2f}★)")

    st.subheader("🎯 Recommandations - Méthode Item-Item")
    for title, score in sorted(item_item_recommendations.items(), key=lambda x: x[1], reverse=True)[:5]:
        st.write(f"- **{title}** (score prédictif : {score:.2f}★)")

    st.subheader(f"🎯 5 films similaires à '{best_rated_movie}' (basé sur les genres)")
    st.dataframe(content_df)
