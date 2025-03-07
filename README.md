# Book Recommendation System

## Overview
This project builds a **book recommendation system** using the **Goodbooks Kaggle Dataset**. It implements multiple recommendation techniques, including **Levenshtein distance, Content-Based Filtering, and Collaborative Filtering** to suggest books.

## Features
- **Data Preprocessing:** Cleaning and filtering missing data
- **Exploratory Data Analysis:** Publication year distribution, rating analysis, and correlation analysis
- **Content-Based Filtering:** Recommends books based on title, author, and tags
- **Collaborative Filtering:** Suggests books based on user ratings
- **Levenshtein Algorithm:** Corrects user-input book titles by finding the closest match

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Levenshtein Distance
- K-Nearest Neighbors (KNN)

## Installation & Usage
### Prerequisites
Install the required libraries using the following command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn python-Levenshtein
```

### Running the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/book-recommendation.git
   ```
2. Navigate to the project directory:
   ```bash
   cd book-recommendation
   ```
3. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook book_recommendation_levenshtein.ipynb
   ```
   or
   ```bash
   python book_recommendation_levenshtein.py
   ```

## Recommendation Methods
### 1. Content-Based Filtering
- Uses **book titles, authors, average ratings, and tags** for recommendations.
- Computes similarity using **Cosine Similarity**.

### 2. Collaborative Filtering
- Applies a **KNN-based recommendation system** using user ratings.
- Identifies similar users and recommends books based on their preferences.

### 3. Levenshtein Distance for Book Matching
- Finds the closest matching book title when the user inputs a misspelled book name.

## Results
- Content-based filtering recommends books based on relevant metadata.
- Collaborative filtering generates personalized suggestions by analyzing similar users' ratings.
- Levenshtein distance improves recommendation accuracy by correcting misspelled book titles.

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to open a pull request.
