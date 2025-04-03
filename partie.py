# 4. Choix de la méthode d'affichage via une liste déroulante
st.subheader("Choisissez une méthode de recommandation")
choix_methode = st.selectbox(
    "Méthode de recommandation",
    ["User-User", "Item-Item", "Fusion (User-User + Item-Item)"]
)

if choix_methode == "User-User":
    st.subheader("Recommandations - Méthode User-User")
    if user_user_recommendations:
        sorted_user_user_recs = sorted(user_user_recommendations.items(), key=lambda x: x[1], reverse=True)
        for title, score in sorted_user_user_recs[:5]:
            st.write(f"- **{title}** (score prédictif : {score:.2f}★)")
    else:
        st.write("*(Aucune recommandation calculée - pas de voisins similaires.)*")

elif choix_methode == "Item-Item":
    st.subheader("Recommandations - Méthode Item-Item")
    if item_item_recommendations:
        sorted_item_item_recs = sorted(item_item_recommendations.items(), key=lambda x: x[1], reverse=True)
        for title, score in sorted_item_item_recs[:5]:
            st.write(f"- **{title}** (score prédictif : {score:.2f}★)")
    else:
        st.write("*(Aucune recommandation calculée - pas de résultats suffisants.)*")

elif choix_methode == "Fusion (User-User + Item-Item)":
    st.subheader("Recommandations combinées (User-User + Item-Item)")
    fusion_recommendations = {}
    all_movies = set(user_user_recommendations.keys()).union(set(item_item_recommendations.keys()))
    for movie in all_movies:
        score_user = user_user_recommendations.get(movie)
        score_item = item_item_recommendations.get(movie)
        if score_user is not None and score_item is not None:
            fusion_score = (score_user + score_item) / 2
        elif score_user is not None:
            fusion_score = score_user
        elif score_item is not None:
            fusion_score = score_item
        else:
            continue
        fusion_recommendations[movie] = fusion_score

    if fusion_recommendations:
        sorted_fusion_recs = sorted(fusion_recommendations.items(), key=lambda x: x[1], reverse=True)
        for title, score in sorted_fusion_recs[:5]:
            st.write(f"- **{title}** (score moyen : {score:.2f}★)")
    else:
        st.write("*(Pas assez de données pour combiner les recommandations.)*")
