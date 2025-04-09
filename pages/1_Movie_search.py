import streamlit as st
from load_data import load_movie_data

@st.cache_data
def load_movies():
    return load_movie_data()

if 'movies' not in st.session_state:
    st.session_state.movies = load_movies()

df = st.session_state.movies

st.title("🔍 Поиск фильмов")
search = st.text_input("Введите название фильма")

if search:
    results = df[df['title'].str.contains(search, case=False, na=False)].head(10)
    for title in results['title']:
        st.write(f"🎞️ {title}")
