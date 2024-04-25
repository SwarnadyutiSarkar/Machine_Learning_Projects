import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample clothing dataset
data = {
    'item_id': [1, 2, 3],
    'item_name': ['Red T-shirt', 'Blue Jeans', 'Black Dress'],
    'style': ['casual', 'casual', 'formal'],
    'color': ['red', 'blue', 'black']
}

df = pd.DataFrame(data)

# Create TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['style'] + ' ' + df['color'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend similar items
def get_recommendations(item_name, cosine_sim, df):
    idx = df[df['item_name'] == item_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Get top 3 similar items
    item_indices = [i[0] for i in sim_scores]
    return df['item_name'].iloc[item_indices]

# Get recommendations for a sample item
item_name = 'Red T-shirt'
recommendations = get_recommendations(item_name, cosine_sim, df)
print(f'Recommendations for {item_name}:')
print(recommendations)
