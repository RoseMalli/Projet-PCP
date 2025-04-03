import streamlit as st
import pandas as pd
import numpy as np

# 1. Chargement du dataset existant
df = pd.read_csv("user_ratings_genres_mov.csv")

# Préparation de la liste des films uniques pour les menus déroulants
movies_list = sorted(df['title'].unique())

# Titre de l'application et instructions
st.title("Système de recommandation de films")
st.markdown("Veuillez sélectionner **3 films** et donner une note pour chacun d'eux :")

# 1. Sélection de 3 films distincts et saisie des notes par l'utilisateur via menus déroulants et curseurs
col1, col2, col3 = st.columns(3)
with col1:
    film1 = st.selectbox("Film 1", movies_list)
    note1 = st.slider("Votre note", min_value=0.5, max_value=5.0, value=3.0, step=0.5)
with col2:
    # Mettre à jour la liste des choix pour éviter les doublons
    options_film2 = [m for m in movies_list if m != film1]
    film2 = st.selectbox("Film 2", options_film2)
    note2 = st.slider("Votre note ", min_value=0.5, max_value=5.0, value=3.0, step=0.5)
with col3:
    options_film3 = [m for m in movies_list if m not in [film1, film2]]
    film3 = st.selectbox("Film 3", options_film3)
    note3 = st.slider("Votre note  ", min_value=0.5, max_value=5.0, value=3.0, step=0.5)

# 2. Bouton pour valider les notes et générer les recommandations
if st.button("Obtenir des recommandations"):
    # Ajout des notes de l'utilisateur au dataset existant (comme un nouvel utilisateur)
    new_user_id = "user_999"  # Identifiant unique pour le nouvel utilisateur (doit ne pas exister encore)
    # Récupérer le genre des films sélectionnés (pour compléter toutes les colonnes du dataset)
    genre1 = df[df['title'] == film1].iloc[0]['genres']
    genre2 = df[df['title'] == film2].iloc[0]['genres']
    genre3 = df[df['title'] == film3].iloc[0]['genres']
    # Créer de nouvelles entrées pour le dataset
    new_ratings = [
        {"userId": new_user_id, "title": film1, "rating": note1, "genres": genre1},
        {"userId": new_user_id, "title": film2, "rating": note2, "genres": genre2},
        {"userId": new_user_id, "title": film3, "rating": note3, "genres": genre3},
    ]
    df = pd.concat([df, pd.DataFrame(new_ratings)], ignore_index=True)

    # Créer la matrice utilisateur-film (notes) pour faciliter les calculs
    user_item_matrix = df.pivot_table(index='userId', columns='title', values='rating')
    # Calculer la moyenne de notes de chaque utilisateur (pour la normalisation)
    user_means = user_item_matrix.mean(axis=1, skipna=True)
    # Moyenne de l'utilisateur courant (le nouvel utilisateur)
    target_user_mean = user_means[new_user_id]

    # 3a. Filtrage collaboratif User-User (similitude entre utilisateurs)
    # Calcul de la similarité de Pearson entre le nouvel utilisateur et tous les autres
    similarities = {}
    for user in user_item_matrix.index:
        if user == new_user_id:
            continue
        # Films notés par les deux utilisateurs (intersection)
        both_rated = user_item_matrix.loc[new_user_id].notna() & user_item_matrix.loc[user].notna()
        if both_rated.sum() < 2:
            # Moins de 2 films en commun -> pas assez d'information pour Pearson
            continue
        # Extraire les notes communes et soustraire la moyenne de chaque utilisateur
        new_user_ratings = user_item_matrix.loc[new_user_id, both_rated] - target_user_mean
        other_user_ratings = user_item_matrix.loc[user, both_rated] - user_means[user]
        # Calcul de la corrélation de Pearson (produit scalaire des vecteurs normalisés)
        num = (new_user_ratings * other_user_ratings).sum()
        den = np.sqrt((new_user_ratings**2).sum() * (other_user_ratings**2).sum())
        if den == 0:
            continue
        sim = num / den
        if sim > 0:  # On ne conserve que les similarités positives
            similarities[user] = sim

    # Sélection des utilisateurs les plus similaires (voisins) triés par similarité décroissante
    top_neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]

    # Prédire les notes du nouvel utilisateur pour les films non notés, en utilisant les voisins sélectionnés
    user_user_recommendations = {}
    for movie in user_item_matrix.columns:
        # Ignorer les films déjà notés par le nouvel utilisateur
        if not pd.isna(user_item_matrix.at[new_user_id, movie]):
            continue
        weighted_sum = 0.0
        sim_sum = 0.0
        for (neighbor, sim) in top_neighbors:
            # Si le voisin a noté ce film, utiliser sa note dans le calcul
            if not pd.isna(user_item_matrix.at[neighbor, movie]):
                neighbor_rating = user_item_matrix.at[neighbor, movie]
                neighbor_mean = user_means[neighbor]
                # Ajouter la contribution pondérée de ce voisin (écart par rapport à sa moyenne * similarité)
                weighted_sum += sim * (neighbor_rating - neighbor_mean)
                sim_sum += abs(sim)
        if sim_sum > 0:
            predicted_rating = target_user_mean + (weighted_sum / sim_sum)
            user_user_recommendations[movie] = predicted_rating

    # 3b. Filtrage collaboratif Item-Item (similitude entre films)
    # Fonction utilitaire pour calculer la similarité de Pearson entre deux films
    def item_similarity(item1, item2):
        # Trouver les utilisateurs en commun qui ont noté les deux films
        users_item1 = set(df[df['title'] == item1]['userId'])
        users_item2 = set(df[df['title'] == item2]['userId'])
        common_users = users_item1 & users_item2
        if len(common_users) < 2:
            return 0  # Pas assez d'utilisateurs communs
        # Préparer les vecteurs de notes centrés sur la moyenne de chaque utilisateur
        diff1 = []
        diff2 = []
        for u in common_users:
            # Note de l'utilisateur u pour item1 et item2
            r1 = user_item_matrix.at[u, item1]
            r2 = user_item_matrix.at[u, item2]
            # Soustraire la moyenne de l'utilisateur u
            diff1.append(r1 - user_means[u])
            diff2.append(r2 - user_means[u])
        diff1 = np.array(diff1)
        diff2 = np.array(diff2)
        # Calcul de la corrélation de Pearson entre item1 et item2
        numerator = (diff1 * diff2).sum()
        denominator = np.sqrt((diff1**2).sum() * (diff2**2).sum())
        if denominator == 0:
            return 0
        return numerator / denominator

    # Calcul des prédictions de notes pour les films non notés par le nouvel utilisateur
    item_item_recommendations = {}
    for movie in user_item_matrix.columns:
        if not pd.isna(user_item_matrix.at[new_user_id, movie]):
            continue  # ignorer les films déjà notés
        weighted_sum = 0.0
        sim_sum = 0.0
        # Parcourir les films que le nouvel utilisateur a notés pour estimer la note du film courant
        for rated_movie in [film1, film2, film3]:
            user_rating = None
            if rated_movie == movie:
                # ce cas ne devrait pas arriver car on ignore déjà les films notés
                continue
            # Récupérer la note du nouvel utilisateur pour le film rated_movie
            if rated_movie == film1:
                user_rating = note1
            elif rated_movie == film2:
                user_rating = note2
            elif rated_movie == film3:
                user_rating = note3
            if user_rating is None:
                continue
            # Similarité entre le film courant et ce film noté par l'utilisateur
            sim = item_similarity(movie, rated_movie)
            if sim > 0:
                weighted_sum += sim * (user_rating - target_user_mean)
                sim_sum += sim
        if sim_sum > 0:
            predicted_rating = target_user_mean + (weighted_sum / sim_sum)
            item_item_recommendations[movie] = predicted_rating

    # 4. Affichage des recommandations en excluant les films déjà notés par l'utilisateur
    st.subheader("Recommandations - Méthode User-User")
    if user_user_recommendations:
        # Trier les films recommandés par note prédite décroissante
        sorted_user_user_recs = sorted(user_user_recommendations.items(), key=lambda x: x[1], reverse=True)
        for title, score in sorted_user_user_recs[:5]:
            st.write(f"- **{title}** (score prédictif : {score:.2f}★)")
    else:
        st.write("*(Aucune recommandation calculée - pas de voisins similaires.)*")

    st.subheader("Recommandations - Méthode Item-Item")
    if item_item_recommendations:
        sorted_item_item_recs = sorted(item_item_recommendations.items(), key=lambda x: x[1], reverse=True)
        for title, score in sorted_item_item_recs[:5]:
            st.write(f"- **{title}** (score prédictif : {score:.2f}★)")
    else:
        st.write("*(Aucune recommandation calculée - pas de résultats suffisants.)*")