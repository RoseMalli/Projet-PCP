# ----------- Affichage ----------- #
st.subheader("🎯 Recommandations - Méthode User-User")
for title, score in sorted(user_user_recommendations.items(), key=lambda x: x[1], reverse=True)[:5]:
    st.write(f"- **{title}** (score estimé : {score:.2f}★)")

st.subheader("🎯 Recommandations - Méthode Item-Item")
for title, score in sorted(item_item_recommendations.items(), key=lambda x: x[1], reverse=True)[:5]:
    st.write(f"- **{title}** (score estimé : {score:.2f}★)")

st.subheader(f"🎯 5 films similaires à '{best_rated_movie}' (basé sur les genres)")
st.dataframe(content_df)
