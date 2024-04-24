import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Sample dataset (replace with your actual dataset)
data = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
    'item_id': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'],
    'rating': [5, 4, 5, 3, 4, 2, 3, 4]
}

df = pd.DataFrame(data)

# Define the rating scale
reader = Reader(rating_scale=(1, 5))

# Load the dataset
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Initialize the SVD algorithm
algo = SVD()

# Train the algorithm on the trainset
algo.fit(trainset)

# Make predictions on the testset
predictions = algo.test(testset)

# Evaluate the model
rmse = accuracy.rmse(predictions)
print(f'RMSE on test set: {rmse}')

# Generate personalized recommendations for a specific user
user_id = 1
items_to_recommend = ['A', 'B', 'C']  # List of item_ids to recommend

# Predict ratings for items
predicted_ratings = {}
for item_id in items_to_recommend:
    predicted_ratings[item_id] = algo.predict(user_id, item_id).est

# Sort items by predicted rating in descending order
recommended_items = sorted(predicted_ratings, key=predicted_ratings.get, reverse=True)

print(f'Recommended items for user {user_id}: {recommended_items}')
