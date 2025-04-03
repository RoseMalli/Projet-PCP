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
""", unsafe_allow_html = True)

# ===== Navigation =====
page = st.sidebar.radio("Navigation", ["🏠 Accueil", "🎬 Recommandation"])

# ===== Chargement des données =====
df = pd.read_csv("user_ratings_genres_mov.csv")
df['genres'] = df['genres'].str.split('|')
movies_list = sorted(df['title'].unique())

if page == "🏠 Accueil":
    st.title("Bienvenue sur notre système de recommandation de films")
    st.markdown("""
    Ce système vous propose des films en fonction de vos goûts grâce à différentes méthodes :

    - 🔹 **User-User** : utilisateurs similaires
    - 🔹 **Item-Item** : films similaires
    - 🔹 **KNN** : utilisateurs ayant des goûts similaires
    - 🔹 **SVD** : réduction des dimensions de la matrice de notation
    - 🔹 **NMF** : factorisation de matrices pour découvrir des patterns cachés
    - 🔹 **Genres** : similarité des genres (Jaccard)

    Sélectionnez l'onglet "🎬 Recommandation" pour commencer !
    """)

elif page == "🎬 Recommandation":
    st.title("🎬 Système de recommandation de films")
    st.caption("Recevez des suggestions personnalisées basées sur vos films préférés.")

    # ===== SIDEBAR POUR SÉLECTION UTILISATEUR =====
    st.sidebar.title("🎯 Vos préférences films")
    
    film1 = st.sidebar.selectbox("Film 1", movies_list)
    genres1 = df[df['title'] == film1].iloc[0]['genres']
    st.sidebar.markdown(f"Genres : {' | '.join(genres1)}")
    note1 = st.sidebar.slider("Note 1", 0.5, 5.0, 4.0, 0.5)
    
    film2 = st.sidebar.selectbox("Film 2", [m for m in movies_list if m != film1])
    genres2 = df[df['title'] == film2].iloc[0]['genres']
    st.sidebar.markdown(f"Genres : {' | '.join(genres2)}")
    note2 = st.sidebar.slider("Note 2", 0.5, 5.0, 4.0, 0.5)
    
    film3 = st.sidebar.selectbox("Film 3", [m for m in movies_list if m not in [film1, film2]])
    genres3 = df[df['title'] == film3].iloc[0]['genres']
    st.sidebar.markdown(f"Genres : {' | '.join(genres3)}")
    note3 = st.sidebar.slider("Note 3", 0.5, 5.0, 4.0, 0.5)
    
    submit = st.sidebar.button("🔍 Obtenir des recommandations")

    if submit:
        new_user_id = "user_999"
        genre1 = df[df['title'] == film1].iloc[0]['genres']
        genre2 = df[df['title'] == film2].iloc[0]['genres']
        genre3 = df[df['title'] == film3].iloc[0]['genres']

        new_ratings = [
            {"userId": new_user_id, "title": film1, "rating": note1, "genres": genre1},
            {"userId": new_user_id, "title": film2, "rating": note2, "genres": genre2},
            {"userId": new_user_id, "title": film3, "rating": note3, "genres": genre3},
        ]
        df = pd.concat([df, pd.DataFrame(new_ratings)], ignore_index = True)


        user_item_matrix = df.pivot_table(index = 'userId', columns = 'title', values = 'rating')
        user_item_matrix.fillna(user_item_matrix.mean(), inplace = True)
        already_rated = [film1, film2, film3]

        with st.expander("🔹 Recommandation par KNN"):
            knn = NearestNeighbors(metric='cosine', algorithm='brute')
            knn.fit(user_item_matrix)
            distances, indices = knn.kneighbors([user_item_matrix.loc[new_user_id]], n_neighbors = 6)
            knn_scores = user_item_matrix.iloc[indices[0][1:]].mean().sort_values(ascending = False)
            top_knn = knn_scores.drop(index = already_rated).head(5)
            for title, score in top_knn.items():
                st.write(f"- **{title}** (score estimé : {score:.2f}★)")


        with st.expander("🔹 Recommandation par SVD"):
            svd = TruncatedSVD(n_components = 20, random_state = 42)
            svd_matrix = svd.fit_transform(user_item_matrix)
            pred_matrix = np.dot(svd_matrix, svd.components_)
            svd_preds = pd.Series(pred_matrix[user_item_matrix.index.get_loc(new_user_id)], index = user_item_matrix.columns)
            top_svd = svd_preds.drop(index = already_rated).sort_values(ascending = False).head(5)
            for title, score in top_svd.items():
                st.write(f"- **{title}** (score estimé : {score:.2f}★)")

        with st.expander("🔹 Recommandation par NMF"):
            nmf = NMF(n_components = 20, init='nndsvda', random_state = 42, max_iter = 500)
            W = nmf.fit_transform(user_item_matrix)
            H = nmf.components_
            pred_matrix_nmf = np.dot(W, H)
            nmf_preds = pd.Series(pred_matrix_nmf[user_item_matrix.index.get_loc(new_user_id)], index = user_item_matrix.columns)
            top_nmf = nmf_preds.drop(index = already_rated).sort_values(ascending = False).head(5)
            for title, score in top_nmf.items():
                st.write(f"- **{title}** (score estimé : {score:.2f}★)")

        with st.expander("🔹 Recommandation par similarité de contenu (genres)"):
            user_ratings = pd.DataFrame({
                'userId': [new_user_id] * 3,
                'title': [film1, film2, film3],
                'rating': [note1, note2, note3],
                'genres': [genre1, genre2, genre3]
            })

            df = pd.concat([df, user_ratings], ignore_index = True)
            df['genres'] = df['genres'].apply(lambda x: x if isinstance(x, list) else str(x).split('|'))

            df_exploded = df.explode('genres')
            df_exploded = df_exploded.drop_duplicates(subset=['title', 'genres'])
            movie_cross_tab = pd.crosstab(df_exploded['title'], df_exploded['genres'])

            jaccard_distances = pdist(movie_cross_tab, metric = 'jaccard')
            jaccard_matrix = 1 - squareform(jaccard_distances)
            jaccard_df = pd.DataFrame(jaccard_matrix, index = movie_cross_tab.index, columns = movie_cross_tab.index)

            best_rated = max([(film1, note1), (film2, note2), (film3, note3)], key = lambda x: x[1])[0]
            jaccard_scores = jaccard_df[best_rated].drop(already_rated).sort_values(ascending = False).head(5)
            avg_ratings = df.groupby("title")["rating"].mean().round(2)
            contenu_df = pd.DataFrame({
                "Similarité (%)": (jaccard_scores.values * 100).astype(int),
                "Note Moyenne": avg_ratings[jaccard_scores.index].values
            }, index = jaccard_scores.index)
            st.dataframe(contenu_df)

        with st.expander("🔹 Recommandation par filtrage collaboratif User-User"):
            similarity_matrix = 1 - pdist(user_item_matrix.fillna(0), metric='cosine')
            similarity_matrix = squareform(similarity_matrix)
            similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

            similar_users = similarity_df[new_user_id].drop(new_user_id).sort_values(ascending=False).head(5).index
            user_based_scores = user_item_matrix.loc[similar_users].mean().sort_values(ascending=False)
            top_user_user = user_based_scores.drop(index=already_rated).head(5)

            for title, score in top_user_user.items():
                st.write(f"- **{title}** (score estimé : {score:.2f}★)")

        def item_similarity(item1, item2, user_item_matrix, user_means):
            users_item1 = set(user_item_matrix.index[user_item_matrix[item1].notna()])
            users_item2 = set(user_item_matrix.index[user_item_matrix[item2].notna()])
            common_users = users_item1 & users_item2

            if len(common_users) < 2:
                return 0

            diff1 = []
            diff2 = []
            for user in common_users:
                diff1.append(user_item_matrix.at[user, item1] - user_means[user])
                diff2.append(user_item_matrix.at[user, item2] - user_means[user])

            diff1, diff2 = np.array(diff1), np.array(diff2)

            numerator = (diff1 * diff2).sum()
            denominator = np.sqrt((diff1**2).sum() * (diff2**2).sum())

            return numerator / denominator if denominator != 0 else 0

        with st.expander("🔹 Recommandation par filtrage collaboratif Item-Item"):
            item_matrix = user_item_matrix.T
            user_means = user_item_matrix.mean(axis=1)

            scores = pd.Series(0, index=item_matrix.index, dtype=float)
            sim_sum = pd.Series(0, index=item_matrix.index, dtype=float)

            for film, note in zip([film1, film2, film3], [note1, note2, note3]):
                for other_film in item_matrix.index:
                    if other_film not in already_rated:
                        sim = item_similarity(film, other_film, user_item_matrix, user_means)
                        scores[other_film] += sim * note
                        sim_sum[other_film] += abs(sim)

            scores = scores.div(sim_sum).dropna()
            top_item_item = scores.sort_values(ascending=False).head(5)

            for title, score in top_item_item.items():
                st.write(f"- **{title}** (score estimé : {score:.2f}★)")
    else:
        st.info("👈 Utilisez la barre latérale pour sélectionner vos films préférés")
