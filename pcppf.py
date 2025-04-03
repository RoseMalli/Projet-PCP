import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

# ===== STYLE CUSTOM =====
st.markdown("""
<style>
    .main {
        background-color: #f4f4f8;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: bold;
    }
    .stSlider > div[data-testid="stSlider"] {
        color: #4CAF50;
    }
    .stSelectbox > div[data-baseweb="select"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ===== Navigation =====
page = st.sidebar.radio("Navigation", ["🏠 Accueil", "🎬 Recommandation"])

# ===== Chargement des données =====
df = pd.read_csv("user_ratings_genres_mov.csv")
df['genres'] = df['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else x)
movies_list = sorted(df['title'].unique())

if page == "🏠 Accueil":
    st.title("Bienvenue sur notre système de recommandation de films")
    st.image("https://cdn.pixabay.com/photo/2016/03/09/09/30/popcorn-1246583_1280.jpg", use_column_width=True)
    st.markdown("""
    Ce système vous propose des films en fonction de vos goûts grâce à différentes méthodes :

    - 🔹 **User-User** : utilisateurs ayant des goûts similaires
    - 🔹 **Item-Item** : films similaires à ceux que vous aimez
    - 🔹 **KNN** : utilisateurs les plus proches
    - 🔹 **SVD** : réduction de dimensions
    - 🔹 **NMF** : patterns cachés
    - 🔹 **Genres** : similarité Jaccard

    Sélectionnez l'onglet "🎬 Recommandation" pour commencer !
    """)

elif page == "🎬 Recommandation":
    st.title("🎬 Système de recommandation de films")
    st.caption("Recevez des suggestions personnalisées basées sur vos films préférés.")

    st.markdown("Veuillez sélectionner **3 films** et donner une note pour chacun d'eux :")
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

    if st.button("Obtenir des recommandations"):
        new_user_id = "user_999"
        genre1 = df[df['title'] == film1].iloc[0]['genres']
        genre2 = df[df['title'] == film2].iloc[0]['genres']
        genre3 = df[df['title'] == film3].iloc[0]['genres']

        new_ratings = [
            {"userId": new_user_id, "title": film1, "rating": note1, "genres": genre1},
            {"userId": new_user_id, "title": film2, "rating": note2, "genres": genre2},
            {"userId": new_user_id, "title": film3, "rating": note3, "genres": genre3},
        ]
        df = pd.concat([df, pd.DataFrame(new_ratings)], ignore_index=True)

        user_item_matrix = df.pivot_table(index='userId', columns='title', values='rating')
        user_item_matrix.fillna(user_item_matrix.mean(), inplace=True)
        user_means = user_item_matrix.mean(axis=1, skipna=True)
        target_user_mean = user_means[new_user_id]
        already_rated = [film1, film2, film3]

        # 🔹 KNN
        with st.expander("🔹 Recommandation par KNN"):
            knn = NearestNeighbors(metric='cosine', algorithm='brute')
            knn.fit(user_item_matrix)
            distances, indices = knn.kneighbors([user_item_matrix.loc[new_user_id]], n_neighbors=6)
            knn_scores = user_item_matrix.iloc[indices[0][1:]].mean().sort_values(ascending=False)
            top_knn = knn_scores.drop(index=already_rated).head(5)
            for title, score in top_knn.items():
                st.write(f"- **{title}** (score moyen : {score:.2f}★)")

        # 🔹 SVD
        with st.expander("🔹 Recommandation par SVD"):
            svd = TruncatedSVD(n_components=20, random_state=42)
            svd_matrix = svd.fit_transform(user_item_matrix)
            pred_matrix = np.dot(svd_matrix, svd.components_)
            svd_preds = pd.Series(pred_matrix[user_item_matrix.index.get_loc(new_user_id)], index=user_item_matrix.columns)
            top_svd = svd_preds.drop(index=already_rated).sort_values(ascending=False).head(5)
            for title, score in top_svd.items():
                st.write(f"- **{title}** (score estimé : {score:.2f}★)")

        # 🔹 NMF
        with st.expander("🔹 Recommandation par NMF"):
            nmf = NMF(n_components=20, init='nndsvda', random_state=42, max_iter=500)
            W = nmf.fit_transform(user_item_matrix)
            H = nmf.components_
            pred_matrix_nmf = np.dot(W, H)
            nmf_preds = pd.Series(pred_matrix_nmf[user_item_matrix.index.get_loc(new_user_id)], index=user_item_matrix.columns)
            top_nmf = nmf_preds.drop(index=already_rated).sort_values(ascending=False).head(5)
            for title, score in top_nmf.items():
                st.write(f"- **{title}** (score estimé : {score:.2f}★)")

        # 🔹 Genres (Jaccard)
        with st.expander("🔹 Recommandation par similarité de contenu (genres)"):
            df['genres'] = df['genres'].apply(lambda x: x if isinstance(x, list) else str(x).split('|'))
            df_exploded = df.explode('genres')
            df_exploded = df_exploded.drop_duplicates(subset=['title', 'genres'])
            movie_cross_tab = pd.crosstab(df_exploded['title'], df_exploded['genres'])
            jaccard_distances = pdist(movie_cross_tab, metric='jaccard')
            jaccard_matrix = 1 - squareform(jaccard_distances)
            jaccard_df = pd.DataFrame(jaccard_matrix, index=movie_cross_tab.index, columns=movie_cross_tab.index)
            best_rated = max([(film1, note1), (film2, note2), (film3, note3)], key=lambda x: x[1])[0]
            jaccard_scores = jaccard_df[best_rated].drop(already_rated).sort_values(ascending=False).head(5)
            avg_ratings = df.groupby("title")["rating"].mean().round(2)
            contenu_df = pd.DataFrame({
                "Similarité (%)": (jaccard_scores.values * 100).astype(int),
                "Note Moyenne": avg_ratings[jaccard_scores.index].values
            }, index=jaccard_scores.index)
            st.dataframe(contenu_df)

               # 🔹 User-User
        with st.expander("🔹 Recommandation par User-User"):
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
                if movie in already_rated:
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

            st.subheader("Recommandations - Méthode User-User")
            if user_user_recommendations:
                sorted_user_user_recs = sorted(user_user_recommendations.items(), key=lambda x: x[1], reverse=True)
                for title, score in sorted_user_user_recs[:5]:
                    st.write(f"- **{title}** (score prédictif : {score:.2f}★)")
            else:
                st.write("*(Aucune recommandation - pas de voisins similaires.)*")

        # 🔹 Item-Item
        with st.expander("🔹 Recommandation par Item-Item"):
            def item_similarity(item1, item2):
                users_item1 = set(df[df['title'] == item1]['userId'])
                users_item2 = set(df[df['title'] == item2]['userId'])
                common_users = users_item1 & users_item2
                if len(common_users) < 2:
                    return 0
                diff1 = []
                diff2 = []
                for u in common_users:
                    r1 = user_item_matrix.at[u, item1]
                    r2 = user_item_matrix.at[u, item2]
                    diff1.append(r1 - user_means[u])
                    diff2.append(r2 - user_means[u])
                numerator = np.dot(diff1, diff2)
                denominator = np.sqrt(np.sum(np.square(diff1)) * np.sum(np.square(diff2)))
                return numerator / denominator if denominator != 0 else 0

            item_item_recommendations = {}
            for movie in user_item_matrix.columns:
                if movie in already_rated:
                    continue
                weighted_sum = 0.0
                sim_sum = 0.0
                for rated_movie, rating in zip([film1, film2, film3], [note1, note2, note3]):
                    sim = item_similarity(movie, rated_movie)
                    if sim > 0:
                        weighted_sum += sim * (rating - target_user_mean)
                        sim_sum += sim
                if sim_sum > 0:
                    predicted_rating = target_user_mean + (weighted_sum / sim_sum)
                    item_item_recommendations[movie] = predicted_rating

            st.subheader("Recommandations - Méthode Item-Item")
            if item_item_recommendations:
                sorted_item_item_recs = sorted(item_item_recommendations.items(), key=lambda x: x[1], reverse=True)
                for title, score in sorted_item_item_recs[:5]:
                    st.write(f"- **{title}** (score prédictif : {score:.2f}★)")
            else:
                st.write("*(Aucune recommandation - données insuffisantes.)*")

                return 0
            diff1 = []
            diff2 = []
            for u in common_users:
                r1 = user_item_matrix.at[u, item1]
                r2 = user_item_matrix.at[u, item2]
                diff1.append(r1 - user_means[u])
                diff2.append(r2 - user_means[u])
            numerator = np.dot(diff1, diff2)
            denominator = np.sqrt(np.sum(np.square(diff1)) * np.sum(np.square(diff2)))
            return numerator / denominator if denominator != 0 else 0

        item_item_recommendations = {}
        for movie in user_item_matrix.columns:
            if movie in already_rated:
                continue
            weighted_sum = 0.0
            sim_sum = 0.0
            for rated_movie, rating in zip([film1, film2, film3], [note1, note2, note3]):
                sim = item_similarity(movie, rated_movie)
                if sim > 0:
                    weighted_sum += sim * (rating - target_user_mean)
                    sim_sum += sim
            if sim_sum > 0:
                predicted_rating = target_user_mean + (weighted_sum / sim_sum)
                item_item_recommendations[movie] = predicted_rating

        st.subheader("Recommandations - Méthode Item-Item")
        if item_item_recommendations:
            sorted_item_item_recs = sorted(item_item_recommendations.items(), key=lambda x: x[1], reverse=True)
            for title, score in sorted_item_item_recs[:5]:
                st.write(f"- **{title}** (score prédictif : {score:.2f}★)")
        else:
            st.write("*(Aucune recommandation - données insuffisantes.)*")
