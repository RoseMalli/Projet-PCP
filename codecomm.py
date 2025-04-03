import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

# ===== STYLE CSS PERSONNALISÉ =====
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

# ===== NAVIGATION =====
# Création des deux pages principales
page = st.sidebar.radio("Navigation", ["🏠 Accueil", "🎬 Recommandation"])

# ===== CHARGEMENT DES DONNÉES =====
# Chargement des données initiales (films, genres, évaluations)
df = pd.read_csv("user_ratings_genres_mov.csv")
df['genres'] = df['genres'].str.split('|')
movies_list = sorted(df['title'].unique())

# ===== PAGE D'ACCUEIL =====
if page == "🏠 Accueil":
    st.title("Bienvenue sur notre système de recommandation de films")
    st.markdown("""
    Ce système propose des recommandations personnalisées en utilisant diverses méthodes :

    - 🔹 **User-User** : Utilisateurs aux goûts similaires
    - 🔹 **Item-Item** : Films similaires à ceux que vous aimez déjà
    - 🔹 **KNN** : Algorithme des K voisins les plus proches
    - 🔹 **SVD** : Décomposition de valeurs singulières (réduction dimensionnelle)
    - 🔹 **NMF** : Factorisation matricielle non négative
    - 🔹 **Genres** : Recommandations basées sur la similarité de genres

    Cliquez sur "🎬 Recommandation" pour commencer !
    """)

# ===== PAGE DE RECOMMANDATION =====
elif page == "🎬 Recommandation":
    st.title("🎬 Système de recommandation de films")
    st.caption("Obtenez des suggestions adaptées à vos préférences.")

    # ===== SIDEBAR POUR LES PRÉFÉRENCES UTILISATEUR =====
    st.sidebar.title("🎯 Vos préférences")

    # Sélection des films préférés par l'utilisateur
    film1 = st.sidebar.selectbox("Film 1", movies_list)
    genres1 = df[df['title'] == film1].iloc[0]['genres']
    st.sidebar.markdown(f"Genres : {' | '.join(genres1)}")
    note1 = st.sidebar.slider("Note pour Film 1", 0.5, 5.0, 4.0, 0.5)

    film2 = st.sidebar.selectbox("Film 2", [m for m in movies_list if m != film1])
    genres2 = df[df['title'] == film2].iloc[0]['genres']
    st.sidebar.markdown(f"Genres : {' | '.join(genres2)}")
    note2 = st.sidebar.slider("Note pour Film 2", 0.5, 5.0, 4.0, 0.5)

    film3 = st.sidebar.selectbox("Film 3", [m for m in movies_list if m not in [film1, film2]])
    genres3 = df[df['title'] == film3].iloc[0]['genres']
    st.sidebar.markdown(f"Genres : {' | '.join(genres3)}")
    note3 = st.sidebar.slider("Note pour Film 3", 0.5, 5.0, 4.0, 0.5)

    # Bouton pour lancer les recommandations
    submit = st.sidebar.button("🔍 Obtenir des recommandations")

    if submit:
        # Ajout du nouvel utilisateur et de ses préférences à la matrice utilisateur-item
        new_user_id = "user_999"
        new_ratings = pd.DataFrame([
            {"userId": new_user_id, "title": film1, "rating": note1, "genres": genres1},
            {"userId": new_user_id, "title": film2, "rating": note2, "genres": genres2},
            {"userId": new_user_id, "title": film3, "rating": note3, "genres": genres3}
        ])
        df = pd.concat([df, new_ratings], ignore_index=True)

        # Création de la matrice utilisateur-item
        user_item_matrix = df.pivot_table(index='userId', columns='title', values='rating')
        user_item_matrix.fillna(user_item_matrix.mean(), inplace=True)
        already_rated = [film1, film2, film3]

        # ===== MÉTHODES DE RECOMMANDATION =====

        # Méthode KNN (Nearest Neighbors)
        with st.expander("🔹 Recommandation par KNN"):
            knn = NearestNeighbors(metric='cosine')
            knn.fit(user_item_matrix)
            distances, indices = knn.kneighbors([user_item_matrix.loc[new_user_id]], n_neighbors=6)
            knn_scores = user_item_matrix.iloc[indices[0][1:]].mean().sort_values(ascending=False)
            for title, score in knn_scores.drop(already_rated).head(5).items():
                st.write(f"- **{title}** : {score:.2f}★")

        # Méthode SVD
        with st.expander("🔹 Recommandation par SVD"):
            svd = TruncatedSVD(n_components=20, random_state=42)
            preds_svd = svd.fit_transform(user_item_matrix).dot(svd.components_)
            svd_preds = pd.Series(preds_svd[user_item_matrix.index.get_loc(new_user_id)], index=user_item_matrix.columns)
            for title, score in svd_preds.drop(already_rated).sort_values(ascending=False).head(5).items():
                st.write(f"- **{title}** : {score:.2f}★")

        # Méthode NMF
        with st.expander("🔹 Recommandation par NMF"):
            nmf = NMF(n_components=20, random_state=42)
            preds_nmf = nmf.fit_transform(user_item_matrix).dot(nmf.components_)
            nmf_preds = pd.Series(preds_nmf[user_item_matrix.index.get_loc(new_user_id)], index=user_item_matrix.columns)
            for title, score in nmf_preds.drop(already_rated).sort_values(ascending=False).head(5).items():
                st.write(f"- **{title}** : {score:.2f}★")

        # Similarité de genres (Jaccard)
        with st.expander("🔹 Recommandation par similarité de contenu (genres)"):
            exploded_df = df.explode('genres')
            crosstab = pd.crosstab(exploded_df['title'], exploded_df['genres'])
            jaccard_matrix = 1 - squareform(pdist(crosstab, 'jaccard'))
            best_rated = max([(film1, note1), (film2, note2), (film3, note3)], key=lambda x: x[1])[0]
            jaccard_scores = pd.Series(jaccard_matrix[crosstab.index.get_loc(best_rated)], index=crosstab.index)
            top_jaccard = jaccard_scores.drop(already_rated).sort_values(ascending=False).head(5)
            st.write(top_jaccard)

    else:
        st.info("👈 Sélectionnez des films dans la barre latérale pour obtenir des recommandations.")
