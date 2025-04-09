import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import ast

def get_top_actors(credit_data):
    try:
        actors = ast.literal_eval(credit_data)[:3]
        return ' '.join(actor['name'].replace(" ", "").lower() for actor in actors)
    except:
        return ''

def get_director(crew_data):
    try:
        crew = ast.literal_eval(crew_data)
        director = [member['name'].replace(" ", "").lower() for member in crew if member['job'] == 'Director']
        return ' '.join(director * 3)
    except:
        return ''

def get_genres(genre_data):
    try:
        genres = ast.literal_eval(genre_data)
        return ' '.join(genre['name'].replace(" ", "").lower() for genre in genres)
    except:
        return ''

def get_keywords(keyword_data):
    try:
        keywords = ast.literal_eval(keyword_data)
        return ' '.join(keyword['name'].replace(" ", "").lower() for keyword in keywords)
    except:
        return ''

def prepare_data(path="."):
    df_movies = pd.read_csv(f"{path}/movies_metadata.csv", low_memory=False)
    df_credits = pd.read_csv(f"{path}/credits.csv")
    df_keywords = pd.read_csv(f"{path}/keywords.csv")

    df_movies['id'] = pd.to_numeric(df_movies['id'], errors='coerce')
    df_credits['id'] = pd.to_numeric(df_credits['id'], errors='coerce')
    df_keywords['id'] = pd.to_numeric(df_keywords['id'], errors='coerce')

    df = df_movies.merge(df_credits, on='id').merge(df_keywords, on='id')

    df['metadata'] = (
        df['genres'].apply(get_genres) + ' ' +
        df['keywords'].apply(get_keywords) + ' ' +
        df['crew'].apply(get_director) + ' ' +
        df['cast'].apply(get_top_actors)
    )

    return df[['title', 'metadata']].dropna()

def build_similarity_matrix(df):
    count_vectorizer = CountVectorizer(stop_words='english')
    count_matrix = count_vectorizer.fit_transform(df['metadata'])

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['metadata'])

    combined_matrix = hstack([count_matrix, tfidf_matrix])
    cosine_sim = cosine_similarity(combined_matrix, combined_matrix)

    return cosine_sim

def recommend(title, df, cosine_sim):
    try:
        idx = df[df['title'] == title].index[0]
    except IndexError:
        return f"Фильм '{title}' не найден."

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# Функция, которую ты можешь использовать в Streamlit
def get_recommendations(title, path="."):
    df = prepare_data(path)
    cosine_sim = build_similarity_matrix(df)
    return recommend(title, df, cosine_sim)



def get_recommendations(title):
    result = recommend(title, df, cosine_sim)
    if isinstance(result, str):
        return [result]
    return result.tolist()
