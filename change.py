def item_similarity(item1, item2, user_item_matrix, user_means):
    users_item1 = set(user_item_matrix.index[user_item_matrix[item1].notna()])
    users_item2 = set(user_item_matrix.index[user_item_matrix[item2].notna()])
    common_users = users_item1 & users_item2

    if len(common_users) < 2:
        return 0

    diff1, diff2 = [], []
    for user in common_users:
        diff1.append(user_item_matrix.at[user, item1] - user_means[user])
        diff2.append(user_item_matrix.at[user, item2] - user_means[user])

    diff1, diff2 = np.array(diff1), np.array(diff2)

    numerator = (diff1 * diff2).sum()
    denominator = np.sqrt((diff1 ** 2).sum() * (diff2 ** 2).sum())

    return numerator / denominator if denominator != 0 else 0

with st.expander("🔹 Recommandation par filtrage collaboratif Item-Item"):
    item_matrix = user_item_matrix.T
    user_means = user_item_matrix.mean(axis=1)
    target_user_mean = user_means[new_user_id]

    scores = pd.Series(dtype=float)

    for other_film in item_matrix.index:
        if other_film in already_rated:
            continue

        sim_sum = 0
        weighted_sum = 0

        for film, note in zip([film1, film2, film3], [note1, note2, note3]):
            sim = item_similarity(film, other_film, user_item_matrix, user_means)
            if sim > 0:
                weighted_sum += sim * (note - target_user_mean)
                sim_sum += sim

        if sim_sum > 0:
            scores[other_film] = target_user_mean + (weighted_sum / sim_sum)

    top_item_item = scores.sort_values(ascending=False).head(5)

    for title, score in top_item_item.items():
        st.write(f"- **{title}** (score estimé : {score:.2f}★)")
