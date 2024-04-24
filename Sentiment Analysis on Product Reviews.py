import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample reviews and labels
reviews = ['This product is great!', 'I hate this product.', 'Amazing product!', 'Terrible product.']
labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# Tokenize the reviews
tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')

# Create a Sequential model
model = Sequential([
    Embedding(1000, 16, input_length=50),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=10, verbose=1)
