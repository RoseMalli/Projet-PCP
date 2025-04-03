# Fonction améliorée pour calculer la similarité cosinus entre films
def item_similarity(item_matrix):
    sim_matrix = 1 - pdist(item_matrix.fillna(0), metric='cosine')
    return pd.DataFrame(squareform(sim_matrix), index=item_matrix.index, columns=item_matrix.index)

with st.expander("🔹 Recommandation par filtrage collaboratif Item-Item"):
    item_matrix = user_item_matrix.T
    similarity_df_items = item_similarity(item_matrix)

    scores = pd.Series(dtype=float)
    sim_sums = pd.Series(dtype=float)

    for film, rating in zip([film1, film2, film3], [note1, note2, note3]):
        sims = similarity_df_items[film].drop(index=already_rated)
        scores = scores.add(sims * rating, fill_value=0)
        sim_sums = sim_sums.add(sims.abs(), fill_value=0)

    scores = (scores / sim_sums).dropna()
    top_recommendations = scores.sort_values(ascending=False).head(5)

    for title, score in top_recommendations.items():
        st.write(f"- **{title}** (score estimé : {score:.2f}★)")


    scores = scores.div(sim_sum).dropna()
    top_item_item = scores.sort_values(ascending=False).head(5)

    for title, score in top_item_item.items():
        st.write(f"- **{title}** (score estimé : {score:.2f}★)")
        st.write(f"- **{title}** (score estimé : {score:.2f}★)")
