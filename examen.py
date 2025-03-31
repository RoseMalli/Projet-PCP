import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity


st.title("Recommandations de films")

data  = pd.read_csv("user_ratings_genres_mov.csv")

user_id = 'user_999'

st.subheader("Premier film")
first_movie_watched = st.selectbox("", list(data['title'].unique()), index = None, placeholder = "Choisissez un film")
first_movie_rating = st.number_input(f"", value = None, placeholder = "Saisissez une note")
first_movie_genres = list(data[data['title'] == first_movie_watched]["genres"].unique())

if first_movie_watched and first_movie_rating in np.arange(0, 5.01, 0.01):
    first_movie_genres_list = first_movie_genres[0].split("|")
    for first_movie_genre in first_movie_genres_list:
        st.write(f"- {first_movie_genre}")
elif first_movie_rating == None:
    pass
else :
    st.error("Saisi incorrect")

st.subheader("Deuxième film")
second_movie_watched = st.selectbox("", list(data['title'].unique()), index = None, placeholder = "Choisissez un film", key = "second")
second_movie_rating = st.number_input(f"", value = None, placeholder = "Saisissez une note", key = 2)
second_movie_genres = list(data[data['title'] == second_movie_watched]["genres"].unique())

if second_movie_watched and second_movie_rating in np.arange(0, 5.01, 0.01):
    second_movie_genres_list = second_movie_genres[0].split("|")
    for second_movie_genre in second_movie_genres_list:
        st.write(f"- {second_movie_genre}")
elif second_movie_rating == None:
    pass
else :
    st.error("Saisi incorrect")


st.subheader("Troisième film")
third_movie_watched = st.selectbox("", list(data['title'].unique()), index = None, placeholder = "Choisissez un film", key = "third")
third_movie_rating = st.number_input(f"", value = None, placeholder = "Saisissez une note", key = 3)
third_movie_genres = list(data[data['title'] == third_movie_watched]["genres"].unique())

if third_movie_watched and third_movie_rating in np.arange(0, 5.01, 0.01):
    third_movie_genres_list = third_movie_genres[0].split("|")
    for third_movie_genre in third_movie_genres_list:
        st.write(f"- {third_movie_genre}")
elif third_movie_rating == None:
    pass
else :
    st.error("Saisi incorrect")



## Recommandation collaborative ##


## Recommandation basée sur le contenu ##

user_ratings = pd.DataFrame({
    'userId': [user_id, user_id, user_id],
    'title': [first_movie_watched, second_movie_watched, third_movie_watched],
    'rating' : [first_movie_rating, second_movie_rating, third_movie_rating],
    'genres': [first_movie_genres, second_movie_genres, third_movie_genres]
})

data = pd.concat([data, user_ratings], ignore_index = True)

data['genres'] = data['genres'].str.split('|')
df_exploded = data.explode('genres')
df_exploded = df_exploded.drop_duplicates(subset = ['title', 'genres'])
movie_cross_tab = pd.crosstab(df_exploded['title'], df_exploded['genres'])

jaccard_distances = pdist(movie_cross_tab, metric = 'jaccard')
squareform_distances = squareform(jaccard_distances)
jaccard_similarity_array = 1 - squareform_distances

jaccard_similarity_df = pd.DataFrame(jaccard_similarity_array, index = movie_cross_tab.index, columns = movie_cross_tab.index)

films_already_watched = user_ratings['title'].tolist()

if first_movie_watched and first_movie_rating and second_movie_watched and second_movie_rating and third_movie_watched and third_movie_rating:
    average_rating_df = round(data[['title', 'rating']].groupby(['title']).mean(), 2)
    best_rated_movie = user_ratings.loc[user_ratings['rating'].idxmax(), 'title']
    jaccard_similarity_series = jaccard_similarity_df.loc[best_rated_movie]
    
    ordered_similarities = jaccard_similarity_series.sort_values(ascending=False)
    ordered_similarities_filtered = ordered_similarities[~ordered_similarities.index.isin(films_already_watched)]
    ordered_similarities_percentage = round(ordered_similarities_filtered * 100)
    recommendations_df = pd.DataFrame({
    'Film': ordered_similarities_percentage.index, 
    'Similarité (%)': ordered_similarities_percentage.values 
    })
    recommendations_df = recommendations_df.merge(average_rating_df, left_on = 'Film', right_index = True, how = 'left')
    recommendations_df.rename(columns={'rating': 'Note Moyenne'}, inplace=True)
    st.subheader(f"5 films les plus similaires à '{best_rated_movie}'")
    st.dataframe(recommendations_df)