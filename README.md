Book Recommendation System using Levenshtein Distance

Overview

This project builds a Book Recommendation System using the Goodbooks Kaggle dataset. The system applies content-based filtering and collaborative filtering methods, leveraging the Levenshtein distance for improved book matching and recommendations.

Features

Data Cleaning & Exploration: Prepares and analyzes book ratings and metadata.

Content-Based Filtering: Uses cosine similarity and KNN to find similar books based on metadata.

Collaborative Filtering: Employs Nearest Neighbors on user ratings for personalized recommendations.

Levenshtein Distance Matching: Enhances title-matching accuracy for improved recommendations.

Dataset

The system utilizes data from the Goodbooks Kaggle dataset:

books.csv: Metadata for books

ratings.csv: User ratings

book_tags.csv: Book tags

tags.csv: Tag names

Installation

Ensure you have Python 3.8+ installed. Install dependencies using:

pip install pandas numpy matplotlib seaborn scikit-learn scipy Levenshtein

Usage

1. Data Cleaning and Exploration

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

books = pd.read_csv('books.csv', on_bad_lines='skip')
ratings = pd.read_csv('ratings.csv')

print("Oldest book: ", books['original_publication_year'].min())
print("Newer book: ", books['original_publication_year'].max())

2. Content-Based Filtering

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_recs(title, cosine_sim):
    title = title.lower()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    book_indices = [i[0] for i in sim_scores]
    return list(books['original_title'].iloc[book_indices])

# Example usage
test = get_recs('Pride and Prejudice', cosine_sim2)
print(test)

3. Collaborative Filtering with KNN

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

book_user_mat = ratings.pivot(index='book_id', columns='user_id', values='rating').fillna(0)
book_user_mat_sparse = csr_matrix(book_user_mat.values)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(book_user_mat_sparse)

def make_recommendation(model_knn, data, mapper, title, n_recommendations):
    idx = mapper[title]
    indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)[1]
    raw_recommends = sorted(list(indices.squeeze().tolist()))[:n_recommendations]
    reverse_mapper = {v: k for k, v in mapper.items()}
    return [reverse_mapper[idx] for idx in raw_recommends]

# Example usage
recommendations = make_recommendation(model_knn, book_user_mat_sparse, indices, "The Maze Runner", 10)
print(recommendations)

4. Levenshtein Distance for Title Matching

from Levenshtein import ratio

def levenshtein_matching(mapper, fav_book):
    match_tuple = [(title, idx, ratio(title.lower(), fav_book.lower())) for title, idx in mapper.items() if ratio(title.lower(), fav_book.lower()) >= 0.6]
    match_tuple = sorted(match_tuple, key=lambda x: x[2], reverse=True)
    return match_tuple[0][1] if match_tuple else None

# Example usage
matched_index = levenshtein_matching(indices, "Ready Player One")
print(matched_index)

Results

Provides personalized book recommendations.

Ensures robust title matching for user queries.

Future Enhancements

Implement a hybrid recommendation system.

Improve tag selection for content-based filtering.

Expand dataset sources for better accuracy.

License

This project is licensed under the MIT License.

