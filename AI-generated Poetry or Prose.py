import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample poetry corpus
poetry_corpus = [
    "Roses are red",
    "Violets are blue",
    "Sugar is sweet",
    "And so are you"
]

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(poetry_corpus)
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in poetry_corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Create predictors and labels
X, y = input_sequences[:,:-1],input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build LSTM model
model = tf.keras.Sequential([
    Embedding(total_words, 64, input_length=max_sequence_len-1),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

# Compile and train model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=1)
