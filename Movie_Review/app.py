import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
import os

# Load dataset
file_path = "D:/WebiSoftTech/RNN/Movie_Review/IMDB Dataset.csv"
df = pd.read_csv(file_path, encoding='utf-8', quoting=3, on_bad_lines='skip', engine='python')
df.dropna(inplace=True)

df['sentiment'] = LabelEncoder().fit_transform(df['sentiment'])  # Convert labels to 0 (negative) & 1 (positive)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['cleaned_review'] = df['review'].apply(clean_text)

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42)

# Tokenization and Padding
max_words = 10000
max_len = 200
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# Build Model
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train_pad, y_train, epochs=3, batch_size=64, validation_data=(X_test_pad, y_test))

# Save model
model.save("sentiment_model.h5")

# Load model for prediction
model = load_model("sentiment_model.h5")

# Flask App
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get("review", "")
    cleaned_review = clean_text(data)
    seq = tokenizer.texts_to_sequences([cleaned_review])
    pad_seq = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(pad_seq)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return jsonify({"sentiment": sentiment, "confidence": float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)