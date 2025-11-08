import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os
from flask import Flask, request, render_template, redirect, url_for

# --- 0. Initialize Flask App ---
app = Flask(__name__)

# --- 1. Define Model Parameters (from run_predictor.py) ---
VOCABULARY_SIZE = 10000
MAX_LEN = 500

# --- 2. Load Model and Word Index (from run_predictor.py) ---
print("Loading the saved model...")
# Make sure the 'saved_model' directory is in the same folder as app.py
model = load_model('saved_model/sentiment_lstm_model.h5')
print("Model loaded.")

word_to_id = imdb.get_word_index()
word_to_id = {k:(v+3) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
print("Word index loaded.")

# --- 3. Preprocessing Function (from run_predictor.py) ---
def preprocess_text(text):
    """
    Cleans and converts a single text review into a sequence of word IDs.
    """
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    
    words = text.split()
    review_ids = [word_to_id.get(word, 2) for word in words] # Use 2 for unknown words
    
    # Filter out word IDs outside the vocabulary
    review_ids = [i if i < VOCABULARY_SIZE else 2 for i in review_ids]
    
    padded_review = pad_sequences([review_ids], maxlen=MAX_LEN)
    return padded_review

# --- 4. Define Web Routes ---

@app.route('/')
def home():
    """Renders the main page."""
    # Renders the 'index.html' file from the 'templates' folder
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if request.method == 'POST':
        # Get data from the form
        movie_name = request.form['movie_name']
        review_text = request.form['review_text']
        
        if not review_text:
            # If no review is entered, just reload the page
            return redirect(url_for('home'))

        # Preprocess the text
        processed_review = preprocess_text(review_text)
        
        # Make a prediction
        prediction_score = model.predict(processed_review, verbose=0)[0][0]
        
        # Interpret the result
        if prediction_score >= 0.5:
            sentiment = "Positive"
        else:
            sentiment = "Negative"
            
        # Format the score for display
        score_percent = f"{prediction_score*100:.2f}%"

        # Render the page again, but this time with the prediction results
        return render_template(
            'index.html', 
            sentiment=sentiment, 
            score=score_percent,
            review_submitted=review_text,
            movie_submitted=movie_name
        )

# --- 5. Run the App ---
if __name__ == "__main__":
    # Note: 'debug=True' reloads the server on code changes, 
    # but 'use_reloader=False' is needed to prevent the model from loading twice.
    app.run(debug=True, use_reloader=False)