import streamlit as st
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Charger le dataset
file_path = "user_ratings_genres_mov.csv"
df = pd.read_csv(file_path)

df['genres'] = df['genres'].str.split('|')
df_exploded = df.explode('genres')
df_exploded = df_exploded.drop_duplicates(subset=['title', 'genres'])

movie_cross_tab = pd.crosstab(df_exploded['title'], df_exploded['genres'])

jaccard_distances = pdist(movie_cross_tab, metric='jaccard')
squareform_distances = squareform(jaccard_distances)
jaccard_similarity_array = 1 - squareform_distances

jaccard_similarity_df = pd.DataFrame(jaccard_similarity_array, index=movie_cross_tab.index, columns=movie_cross_tab.index)

# Interface utilisateur
st.title("🎬 Création du Profil Utilisateur")

# Saisie de l'ID utilisateur
user_id = st.text_input("🔹 Entrez votre ID utilisateur", key="user_id")

# Sélection de 3 films
films = df[['title', 'genres']].drop_duplicates()
selected_movies = st.multiselect("🎥 Choisissez 3 films", films['title'].unique(), key="selected_movies", max_selections=3)

# Création d'un dictionnaire pour stocker les notes et genres
ratings = {}
for movie in selected_movies:
    genre = ', '.join(df[df['title'] == movie]['genres'].values[0])  # Convertir la liste en chaîne
  # Récupérer le genre du film
    rating = st.slider(f"⭐ Note pour {movie} ({genre})", 0.5, 5.0, 2.5, 0.5, key=f"rating_{movie}")
    ratings[movie] = {"genre": genre, "rating": rating}

# Boutons d'action
col1, col2 = st.columns(2)

with col1:
    if st.button("✅ Enregistrer le profil"):
        if not user_id:
            st.error("❌ Veuillez entrer votre ID utilisateur.")
        elif len(selected_movies) != 3:
            st.error("❌ Veuillez sélectionner exactement 3 films.")
        else:
            st.success("✅ Profil enregistré avec succès !")
            st.write("**ID utilisateur :**", user_id)
            st.write("**Films notés :**")
            result_df = pd.DataFrame([
                {"🎬 Film": movie, "🎭 Genre": info["genre"], "⭐ Note": info["rating"]}
                for movie, info in ratings.items()
            ])
            st.table(result_df)

# Recommandations
if selected_movies:
    user_ratings = pd.DataFrame.from_dict(ratings, orient='index').reset_index().rename(columns={'index': 'title'})
    films_already_watched = user_ratings['title'].tolist()
    
    best_rated_movie = user_ratings.loc[user_ratings['rating'].idxmax(), 'title']
    jaccard_similarity_series = jaccard_similarity_df.loc[best_rated_movie]
    
    ordered_similarities = jaccard_similarity_series.sort_values(ascending=False)
    ordered_similarities_filtered = ordered_similarities[~ordered_similarities.index.isin(films_already_watched)]
    ordered_similarities_percentage = round(ordered_similarities_filtered * 100)
    
    average_rating_df = round(df[['title', 'rating']].groupby(['title']).mean(), 2)
    recommendations_df = pd.DataFrame({
        'Film': ordered_similarities_percentage.index,
        'Similarité (%)': ordered_similarities_percentage.values
    })
    recommendations_df = recommendations_df.merge(average_rating_df, left_on='Film', right_index=True, how='left')
    recommendations_df.rename(columns={'rating': 'Note Moyenne'}, inplace=True)
    
    st.subheader(f"5 films les plus similaires à '{best_rated_movie}'")
    st.dataframe(recommendations_df.head(5))