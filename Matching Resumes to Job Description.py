import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load resumes and job description
resume = "Software engineer with 5 years of experience in web development..."
job_description = "We are looking for a software engineer with expertise in web development..."

# Tokenize and vectorize text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([resume, job_description])

# Compute cosine similarity
cosine_sim = cosine_similarity(X)

# Match score between resume and job description
match_score = cosine_sim[0, 1]
print(f'Match Score: {match_score}')
