import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset (Replace with your actual dataset)
data = {
    'query': ['How do I reset my password?', 'I have a billing issue.', 'How to track my order?', 'Product query', 'Technical support'],
    'label': ['password_reset', 'billing_issue', 'order_tracking', 'product_query', 'technical_support']
}

df = pd.DataFrame(data)

# Convert labels to numerical values
label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
df['label'] = df['label'].map(label_mapping)

# Split data into train and test sets
X = df['query']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize and train Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Chatbot function
def chatbot(query):
    query_vec = vectorizer.transform([query])
    prediction = clf.predict(query_vec)
    predicted_label = [label for label, idx in label_mapping.items() if idx == prediction][0]
    return f'You may need {predicted_label} assistance.'

# Test chatbot
query = 'How do I reset my password?'
response = chatbot(query)
print(response)
