# Recommandation de films avec Streamlit (projet complet)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

# ====== STYLE CUSTOMISÉ ======
st.markdown("""
<style>
    .main { background-color: #f4f4f8; }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ====== BARRE LATERALE ======
with st.sidebar:
    st.title("🎞️ Menu")
    page = st.radio("Aller à :", ["🏠 Accueil", "🎬 Recommandation"])
    st.markdown("---")
    st.markdown("**Projet : Recommandation de films**")
    st.caption("Master Cybersécurité · Avril 2025")

# ====== CHARGEMENT DES DONNÉES ======
df = pd.read_csv("user_ratings_genres_mov.csv")
df['genres'] = df['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else x)
movies_list = sorted(df['title'].unique())
all_genres = sorted(set(g for sub in df['genres'] if isinstance(sub, list) for g in sub))

# ====== PAGE ACCUEIL ======
if page == "🏠 Accueil":
    st.title("Bienvenue sur notre système de recommandation de films")
    st.image("https://cdn.pixabay.com/photo/2016/03/09/09/30/popcorn-1246583_1280.jpg", use_column_width=True)
    st.markdown("""
    Ce système vous propose des films en fonction de vos goûts grâce à différentes méthodes :
    
    - 🔹 **User-User** : utilisateurs similaires
    - 🔹 **Item-Item** : films similaires
    - 🔹 **KNN** : plus proches voisins
    - 🔹 **SVD** / **NMF** : modèles factorisation
    - 🔹 **Genres** : similarité de contenu (Jaccard)
    """)

# ====== PAGE RECOMMANDATION ======
elif page == "🎬 Recommandation":
    st.title("🎬 Système de recommandation de films")
    st.markdown("Veuillez sélectionner **3 films** et donner une note + genres pour chacun :")

    col1, col2, col3 = st.columns(3)
    with col1:
        film1 = st.selectbox("Film 1", movies_list)
        note1 = st.slider("Note", 0.5, 5.0, 3.0, 0.5)
        genre1 = st.multiselect("Genres du film 1", all_genres, key="g1")
    with col2:
        options2 = [m for m in movies_list if m != film1]
        film2 = st.selectbox("Film 2", options2)
        note2 = st.slider("Note", 0.5, 5.0, 3.0, 0.5)
        genre2 = st.multiselect("Genres du film 2", all_genres, key="g2")
    with col3:
        options3 = [m for m in movies_list if m not in [film1, film2]]
        film3 = st.selectbox("Film 3", options3)
        note3 = st.slider("Note", 0.5, 5.0, 3.0, 0.5)
        genre3 = st.multiselect("Genres du film 3", all_genres, key="g3")

    if st.button("Obtenir des recommandations"):
        new_user = "user_999"
        already_rated = [film1, film2, film3]

        df = pd.concat([df, pd.DataFrame([
            {"userId": new_user, "title": film1, "rating": note1, "genres": genre1},
            {"userId": new_user, "title": film2, "rating": note2, "genres": genre2},
            {"userId": new_user, "title": film3, "rating": note3, "genres": genre3},
        ])], ignore_index=True)

        user_item = df.pivot_table(index='userId', columns='title', values='rating')
        user_item.fillna(user_item.mean(), inplace=True)
        user_means = user_item.mean(axis=1)
        mean_target = user_means[new_user]

        # ==== FONCTION SVD ====
        def recommandation_svd():
            svd = TruncatedSVD(n_components=20, random_state=42)
            svd_matrix = svd.fit_transform(user_item)
            pred = np.dot(svd_matrix, svd.components_)
            scores = pd.Series(pred[user_item.index.get_loc(new_user)], index=user_item.columns)
            return scores.drop(index=already_rated).sort_values(ascending=False).head(5)

        # ==== FONCTION NMF ====
        def recommandation_nmf():
            nmf = NMF(n_components=20, init='nndsvda', max_iter=500)
            W = nmf.fit_transform(user_item)
            H = nmf.components_
            pred = np.dot(W, H)
            scores = pd.Series(pred[user_item.index.get_loc(new_user)], index=user_item.columns)
            return scores.drop(index=already_rated).sort_values(ascending=False).head(5)

        # ==== FONCTION KNN ====
        def recommandation_knn():
            knn = NearestNeighbors(metric='cosine', algorithm='brute')
            knn.fit(user_item)
            dists, idx = knn.kneighbors([user_item.loc[new_user]], n_neighbors=6)
            scores = user_item.iloc[idx[0][1:]].mean().sort_values(ascending=False)
            return scores.drop(index=already_rated).head(5)

        # ==== FONCTION USER-USER ====
        def recommandation_user_user():
            sims = {}
            for user in user_item.index:
                if user == new_user:
                    continue
                both = user_item.loc[new_user].notna() & user_item.loc[user].notna()
                if both.sum() < 2:
                    continue
                v1 = user_item.loc[new_user, both] - mean_target
                v2 = user_item.loc[user, both] - user_means[user]
                denom = np.sqrt((v1 ** 2).sum() * (v2 ** 2).sum())
                if denom == 0:
                    continue
                sims[user] = (v1 * v2).sum() / denom
            top_users = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:5]
            recs = {}
            for movie in user_item.columns:
                if movie in already_rated:
                    continue
                num, den = 0, 0
                for u, sim in top_users:
                    if pd.notna(user_item.at[u, movie]):
                        num += sim * (user_item.at[u, movie] - user_means[u])
                        den += abs(sim)
                if den > 0:
                    recs[movie] = mean_target + (num / den)
            return sorted(recs.items(), key=lambda x: x[1], reverse=True)[:5]

        # ==== FONCTION ITEM-ITEM ====
        def recommandation_item_item():
            recs = {}
            for movie in user_item.columns:
                if movie in already_rated:
                    continue
                num, den = 0, 0
                for rated_movie, rating in zip(already_rated, [note1, note2, note3]):
                    common_users = user_item[[movie, rated_movie]].dropna()
                    if len(common_users) < 2:
                        continue
                    diff1 = common_users[movie] - user_means[common_users.index]
                    diff2 = common_users[rated_movie] - user_means[common_users.index]
                    sim = np.dot(diff1, diff2) / (np.linalg.norm(diff1) * np.linalg.norm(diff2))
                    if sim > 0:
                        num += sim * (rating - mean_target)
                        den += sim
                if den > 0:
                    recs[movie] = mean_target + (num / den)
            return sorted(recs.items(), key=lambda x: x[1], reverse=True)[:5]

        # ==== FONCTION CONTENU (Jaccard) ====
        def recommandation_contenu():
            df_ex = df.explode('genres').drop_duplicates(['title', 'genres'])
            cross = pd.crosstab(df_ex['title'], df_ex['genres'])
            dist = pdist(cross, metric='jaccard')
            sim = 1 - squareform(dist)
            sim_df = pd.DataFrame(sim, index=cross.index, columns=cross.index)
            top_film = max([(film1, note1), (film2, note2), (film3, note3)], key=lambda x: x[1])[0]
            scores = sim_df[top_film].drop(already_rated).sort_values(ascending=False).head(5)
            return pd.DataFrame({"Similarité (%)": (scores.values * 100).astype(int)}, index=scores.index)

        st.subheader("🔹 Résultats")
        with st.expander("🔹 KNN"):
            for m, s in recommandation_knn().items():
                st.write(f"- **{m}** ({s:.2f}★)")

        with st.expander("🔹 SVD"):
            for m, s in recommandation_svd().items():
                st.write(f"- **{m}** ({s:.2f}★)")

        with st.expander("🔹 NMF"):
            for m, s in recommandation_nmf().items():
                st.write(f"- **{m}** ({s:.2f}★)")

        with st.expander("🔹 User-User"):
            for m, s in recommandation_user_user():
                st.write(f"- **{m}** ({s:.2f}★)")

        with st.expander("🔹 Item-Item"):
            for m, s in recommandation_item_item():
                st.write(f"- **{m}** ({s:.2f}★)")

        with st.expander("🔹 Basé sur le contenu (genres)"):
            st.dataframe(recommandation_contenu())
