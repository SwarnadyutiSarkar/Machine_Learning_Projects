from surprise import Dataset
from surprise import SVD
from surprise.model_selection import cross_validate

# Load the dataset
data = Dataset.load_builtin('ml-100k')

# Use the SVD algorithm
algo = SVD()

# Perform cross-validation
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
