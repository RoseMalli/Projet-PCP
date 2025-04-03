with st.expander("🔹 Recommandation par KNN"):
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_item_matrix)
    distances, indices = knn.kneighbors([user_item_matrix.loc[new_user_id]], n_neighbors = 6)
    knn_scores = user_item_matrix.iloc[indices[0][1:]].mean().sort_values(ascending = False)
    top_knn = knn_scores.drop(index = already_rated).head(5)
    for title, score in top_knn.items():
        st.write(f"- **{title}** (score estimé : {score:.2f}★)")

with st.expander("🔹 Recommandation par filtrage collaboratif User-User"):
    similarity_matrix = 1 - pdist(user_item_matrix.fillna(0), metric='cosine')
    similarity_matrix = squareform(similarity_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

    similar_users = similarity_df[new_user_id].drop(new_user_id).sort_values(ascending=False).head(5).index
    user_based_scores = user_item_matrix.loc[similar_users].mean().sort_values(ascending=False)
    top_user_user = user_based_scores.drop(index=already_rated).head(5)

    for title, score in top_user_user.items():
        st.write(f"- **{title}** (score estimé : {score:.2f}★)")

with st.expander("🔹 Recommandation par filtrage collaboratif Item-Item"):
    item_matrix = user_item_matrix.T
    similarity_matrix_items = 1 - pdist(item_matrix.fillna(0), metric='cosine')
    similarity_matrix_items = squareform(similarity_matrix_items)
    similarity_df_items = pd.DataFrame(similarity_matrix_items, index=item_matrix.index, columns=item_matrix.index)

    scores = pd.Series(0, index=item_matrix.index, dtype=float)
    for film, note in zip([film1, film2, film3], [note1, note2, note3]):
        sim_scores = similarity_df_items[film] * note
        scores = scores.add(sim_scores, fill_value=0)

    top_item_item = scores.drop(index=already_rated).sort_values(ascending=False).head(5)
    for title, score in top_item_item.items():
        st.write(f"- **{title}** (score estimé : {score:.2f}★)")
