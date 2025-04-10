import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Настройки страницы
st.set_page_config(page_title="🎬 MovieShelf", layout="centered")

# Загрузка данных
@st.cache_data
def load_movie_data(path="."):
    df_movies = pd.read_csv(os.path.join(path, "movies_metadata.csv"), low_memory=False)
    df_credits = pd.read_csv(os.path.join(path, "credits.csv"))
    df_keywords = pd.read_csv(os.path.join(path, "keywords.csv"))

    df_movies['id'] = pd.to_numeric(df_movies['id'], errors='coerce')
    df_credits['id'] = pd.to_numeric(df_credits['id'], errors='coerce')
    df_keywords['id'] = pd.to_numeric(df_keywords['id'], errors='coerce')

    df = df_movies.merge(df_credits, on='id').merge(df_keywords, on='id')
    df = df[['title', 'genres', 'keywords', 'crew', 'cast']]
    df = df.dropna(subset=['title'])
    return df

# Загрузка модели
@st.cache_resource
def load_model():
    with open('movie_recommendation_model.pkl', 'rb') as file:
        vectorizer, count_matrix, df = pickle.load(file)
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    return vectorizer, cosine_sim, df

# Рекомендательная функция
def recommend(movie_title, df, cosine_sim):
    movie_title = movie_title.strip().lower()
    titles = df['title'].dropna().str.lower()

    if movie_title not in titles.values:
        return []

    idx = titles[titles == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].values

# Загрузка модели и данных
vectorizer, cosine_sim, df_model = load_model()
df_movies = load_movie_data()

# Интерфейс
st.title("🎬 MovieShelf")

tab1, tab2, tab3 = st.tabs(["🏠 Главная", "🔍 Поиск фильмов", "🎯 Рекомендации"])

with tab1:
    st.header("🏠 Главная")
    st.write("Добро пожаловать в MovieShelf!")
    st.markdown("Ищите фильмы и получайте персональные рекомендации на основе содержимого.")

with tab2:
    st.header("🔍 Поиск фильмов")
    search = st.text_input("Введите название фильма")
    if search:
        results = df_movies[df_movies['title'].str.contains(search, case=False, na=False)].head(10)
        if not results.empty:
            st.write("Найдено:")
            for title in results['title']:
                st.write(f"🎞️ {title}")
        else:
            st.warning("Фильмы не найдены.")

with tab3:
    st.header("🎯 Рекомендации")
    movie_list = sorted(df_model['title'].dropna().unique())
    selected_movie = st.selectbox("Выберите фильм", movie_list)

    if st.button("Показать рекомендации"):
        recs = recommend(selected_movie, df_model, cosine_sim)
        if recs is None or len(recs) == 0:
            st.warning("Фильм не найден или для него нет похожих.")
        else:
            st.write("### Вам может понравиться:")
            for i, title in enumerate(recs, 1):
                st.write(f"{i}. {title}")
